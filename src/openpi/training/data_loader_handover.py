"""RLT Stage A.6 data: phase-labeled windows of RL tokens read from Stage A.5 caches.

Same trick as `data_loader_rlt.py` -- `z_rl` and the normalized state are precomputed, so a training
sample is a gather of `window` rows out of an array already in RAM. No images, no VLA forward pass,
no LeRobot at training time.

The human labels live in `data/<repo>/annotations.json`, keyed by LeRobot episode index with a
`begin_id`/`end_id` frame span per episode. The cache is a flat frame array, so the two are joined
through `episode_bounds` from the cache. `create_torch_dataset` applies no episode filtering, so
cache episode `e` is LeRobot episode `e` -- `check_cache_alignment` verifies that rather than
assuming it, because a silent off-by-one here would mislabel every frame.

Several caches can be pooled (`TrainConfig.sources`), the way Stage B pools them: once every episode
of a dataset has been hand-labeled and reviewed, the head to deploy is the one trained on all of
them. Their labeled episodes are gathered into a single flat pool up front, so from `LabeledEpisodes`
downward there is exactly one frame array and nothing remembers which cache a frame came from.
"""

from collections.abc import Iterator
import dataclasses
import json
import logging
import pathlib
import random

import jax
import numpy as np
import torch

from openpi.models import rlt_handover
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.data_loader_rlt import RLCache
from openpi.training.data_loader_rlt import check_cache_alignment

logger = logging.getLogger("openpi")


def load_annotations(path: str | pathlib.Path) -> dict[int, tuple[int, int]]:
    """`{episode_index: (begin_id, end_id)}` for every episode with a labeled critical phase."""
    clips = json.loads(pathlib.Path(path).read_text())["clips"]
    annotations = {}
    for episode, spans in clips.items():
        if not spans:
            continue
        if len(spans) > 1:
            # The predictor decodes exactly one contiguous critical span per episode, so a
            # multi-span annotation would be silently truncated.
            raise ValueError(f"episode {episode} has {len(spans)} annotated spans; expected at most 1")
        annotations[int(episode)] = (int(spans[0]["begin_id"]), int(spans[0]["end_id"]))
    return annotations


@dataclasses.dataclass(frozen=True)
class LabeledEpisodes:
    """Every labeled episode of every source, gathered into one flat frame pool.

    Same shape of object as a single cache -- flat frame arrays plus `[E, 2]` bounds -- so windowing,
    Viterbi decoding and the fold split all work on the pool exactly as they worked on one cache.
    """

    z: np.ndarray  # [N, rl_token_dim]
    state: np.ndarray  # [N, state_dim]
    target: np.ndarray  # [N, NUM_PHASES], soft phase targets
    bounds: np.ndarray  # [E, 2] of (from, to) into the pool
    spans: list[tuple[int, int]]  # hand-labeled (begin, end) per episode, in episode-local frames
    names: list[str]  # "<annotations_path>#<episode index>" per episode, for logging

    def __len__(self) -> int:
        return len(self.bounds)


def load_labeled_episodes(sources, ramp: int) -> LabeledEpisodes:
    """Pool the labeled episodes of every source. One cache is read and released at a time."""
    z, state, target, bounds, spans, names = [], [], [], [], [], []
    offset = 0
    for source in sources:
        # The full teleop cache is ~1.8 GB and only its labeled frames survive into the pool, so
        # it is dropped again before the next source is opened.
        cache = RLCache.load(source.cache_dir)
        check_cache_alignment(source.annotations_path, cache.episode_bounds)
        annotations = load_annotations(source.annotations_path)
        for episode in sorted(annotations):
            if episode >= len(cache.episode_bounds):
                raise ValueError(
                    f"{source.annotations_path}: annotated episode {episode} is outside "
                    f"{source.cache_dir} ({len(cache.episode_bounds)} episodes)"
                )
            start, stop = (int(x) for x in cache.episode_bounds[episode])
            begin, end = annotations[episode]
            z.append(np.asarray(cache.z_rl[start:stop], dtype=np.float32))
            state.append(np.asarray(cache.state[start:stop], dtype=np.float32))
            target.append(rlt_handover.phase_targets(stop - start, begin, end, ramp))
            bounds.append((offset, offset + stop - start))
            spans.append((begin, end))
            names.append(f"{source.annotations_path}#{episode}")
            offset += stop - start
        logger.info(
            "  %s: %d labeled episodes of %d cached",
            source.annotations_path,
            len(annotations),
            len(cache.episode_bounds),
        )
        del cache

    if not bounds:
        raise ValueError("no labeled episodes found; check config.sources")
    return LabeledEpisodes(
        np.concatenate(z),
        np.concatenate(state),
        np.concatenate(target),
        np.asarray(bounds, dtype=np.int64),
        spans,
        names,
    )


def fold_split(num_episodes: int, num_folds: int, val_fold: int | None) -> tuple[list[int], list[int]]:
    """Grouped CV split of the pool. Whole episodes go to one side, so no frames leak.

    Striding the pool spreads each source across the folds in proportion to its size, so a fold is
    never all demos or all rollouts.
    """
    episodes = list(range(num_episodes))
    if val_fold is None:
        return episodes, []
    if not 0 <= val_fold < num_folds:
        raise ValueError(f"val_fold={val_fold} must be in [0, {num_folds})")
    val = episodes[val_fold::num_folds]
    return [e for e in episodes if e not in set(val)], val


@dataclasses.dataclass(frozen=True)
class HandoverData:
    episodes: LabeledEpisodes
    train_episodes: list[int]
    val_episodes: list[int]


def load_handover_data(config) -> HandoverData:
    episodes = load_labeled_episodes(config.sources, config.model.label_ramp)
    train_episodes, val_episodes = fold_split(len(episodes), config.num_folds, config.val_fold)
    logger.info(
        "Pooled %d labeled episodes (%d frames) from %d source(s): %d train, %d val",
        len(episodes),
        len(episodes.z),
        len(config.sources),
        len(train_episodes),
        len(val_episodes),
    )
    return HandoverData(episodes, train_episodes, val_episodes)


class HandoverWindowDataset(torch.utils.data.Dataset):
    """One window per frame of the selected episodes.

    Windows reach 375 frames back, so the pool is indexed directly rather than memory-mapped: random
    access over a memmap on network storage would dominate the step time. It is already in RAM
    (~900 MB for all 199 labeled teleop + rollout episodes) and shared with the
    evaluation decode, so this only stores the row indices of the split it covers.
    """

    def __init__(
        self,
        pool: LabeledEpisodes,
        episodes: list[int],
        config: rlt_handover.RLTHandoverConfig,
        *,
        augment: bool,
        seed: int = 0,
    ):
        self.pool = pool
        self.config = config
        self.augment = augment
        self.rng = random.Random(seed)

        rows, starts = [], []
        for episode in episodes:
            start, stop = (int(x) for x in pool.bounds[episode])
            rows.append(np.arange(start, stop))
            starts.append(np.full(stop - start, start, dtype=np.int64))
        self.rows = np.concatenate(rows)
        self.episode_start = np.concatenate(starts)

    def __len__(self) -> int:
        return self.rows.shape[0]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        stride, phase = self.config.window_stride, 0
        if self.augment:
            # On the robot the window is stale by however far the current control step is past the
            # last VLA query, and the query spacing itself drifts with the RTC delay. Train over
            # both rather than only on query-aligned, exactly-spaced windows.
            stride = self.rng.randint(
                self.config.window_stride - self.config.stride_jitter,
                self.config.window_stride + self.config.stride_jitter,
            )
            phase = self.rng.randrange(stride)

        # `rows[index]` and `episode_start[index]` are both pool coordinates, so the window is
        # clamped to the start of its own episode and never reaches into the previous one.
        row = int(self.rows[index])
        idx = rlt_handover.window_indices(
            self.rows[index : index + 1],
            self.episode_start[index : index + 1],
            window=self.config.window,
            stride=stride,
            phase=phase,
        )[0]
        return {
            "z_window": self.pool.z[idx],
            "state_window": self.pool.state[idx],
            "state": self.pool.state[row],
            "target": self.pool.target[row],
        }


class HandoverDataLoader:
    """Infinite shuffled batches of labeled windows, sharded across devices."""

    def __init__(
        self,
        dataset: HandoverWindowDataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        self._dataset = dataset
        self._sharding = sharding or jax.sharding.SingleDeviceSharding(jax.devices()[0])
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=True,
            # Default 0: the dataset already lives in RAM, and every worker would copy all of it.
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=True,
            generator=generator,
            collate_fn=_data_loader._collate_fn,  # noqa: SLF001
        )

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        while True:
            for batch in self._loader:
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)

    def data_config(self) -> _config.DataConfig:
        # Required by checkpoints.save_state. Stage A.6 trains on cached features; the norm stats
        # belong to the Stage A checkpoint that produced the cache.
        return _config.DataConfig()


def create_handover_data_loader(
    config, data: HandoverData, *, sharding: jax.sharding.Sharding | None = None
) -> HandoverDataLoader:
    dataset = HandoverWindowDataset(data.episodes, data.train_episodes, config.model, augment=True, seed=config.seed)
    logger.info("Training on %d labeled frames from %d episodes", len(dataset), len(data.train_episodes))

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size={config.batch_size} must be divisible by device_count={jax.device_count()}")

    return HandoverDataLoader(
        dataset,
        config.batch_size // jax.process_count(),
        sharding=sharding,
        num_workers=config.num_workers,
        seed=config.seed,
    )
