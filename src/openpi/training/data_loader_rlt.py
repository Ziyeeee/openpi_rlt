"""RLT Stage B data: chunk transitions read from one or more Stage A.5 caches.

Because `z_rl` and the VLA reference chunk are precomputed, a transition is just a handful of rows
from arrays already in RAM -- no images, no VLA forward pass, no LeRobot at training time:

    <x_t, a_{t:t+C}, a_ref_t, r_{t:t+C}, x_{t+C}>

Transitions come from *labeled clips*, not whole episodes. Sec. V and Appendix B run RL on a task's
critical phase only, and the clip is that phase: `data/<repo>/clips_by_tag.json` gives one
`(begin_id, end_id, result)` span per rollout episode, hand-reviewed. Two things follow.

  * The reward is real, not assumed. `result == "success"` puts the sparse `+1` on the clip's last
    frame; a failed clip ends with nothing. Both ends are terminal, so a failure bootstraps from 0
    and its value is actively driven down instead of being left unconstrained.
  * The discount finally reaches the reward. A clip is 69-472 frames = 7-47 chunks, so at
    `gamma=0.99` the terminal `+1` is still worth ~0.3 at the start of a median success. Spread over
    a whole ~500-frame episode it was worth ~0.003, i.e. no signal outside the last few chunks.

Because a clip's frames are contiguous, the transitions of one clip are exactly the transitions of a
short episode whose outcome is known -- everything downstream (truncation, `done`, bootstrapping) is
unchanged, it just operates on the clip rather than the episode.

Several caches can be mixed (`TrainConfig.sources`), which is how rollouts and teleop demonstrations
are trained on together. §IV's replay buffer likewise aggregates VLA warmup data, online rollouts
and human interventions; off-policy learning does not care which policy generated a transition.

The paper subsamples chunks 2 control steps apart (Appendix B) because on-robot it stores whichever
intermediate steps it happens to observe. Offline every index is already available, so there is
nothing to subsample and no stride knob here.
"""

from collections.abc import Iterator
import dataclasses
import json
import logging
import multiprocessing
import pathlib
from typing import Any

import jax
import numpy as np
import torch

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class RLCache:
    z_rl: np.ndarray  # [N, rl_token_dim]
    a_ref: np.ndarray  # [N, H, action_dim]
    a_demo: np.ndarray  # [N, H, action_dim]
    state: np.ndarray  # [N, state_dim]
    episode_bounds: np.ndarray  # [num_episodes, 2] of (from, to)

    @classmethod
    def load(cls, cache_dir: str | pathlib.Path) -> "RLCache":
        path = pathlib.Path(cache_dir)
        if not path.exists():
            raise FileNotFoundError(f"RL token cache not found at {path}. Run scripts/cache_rl_token.py first.")
        # Read fully into RAM rather than memmapping. The cache sits on a FUSE-mounted parallel
        # filesystem where a random batch of 256 rows costs ~280 ms in page faults -- 40x slower
        # than the training step it feeds. The whole cache is ~1.8 GB at 84k frames and loads
        # sequentially in ~3 s, after which a batch gather costs 0.3 ms.
        return cls(
            **{name: np.load(path / f"{name}.npy") for name in ("z_rl", "a_ref", "a_demo", "state", "episode_bounds")}
        )


def check_cache_alignment(labels_path: str | pathlib.Path, episode_bounds: np.ndarray) -> None:
    """Verify the cache's episode layout against the LeRobot metadata next to the labels.

    Both the clip labels and the handover annotations are keyed by LeRobot episode index and joined
    to the flat cache through `episode_bounds`. `create_torch_dataset` applies no episode filtering,
    so cache episode `e` is LeRobot episode `e` -- check that rather than assume it, because a
    silent off-by-one here would put every reward on the wrong frame.
    """
    meta = pathlib.Path(labels_path).parent / "meta" / "episodes.jsonl"
    if not meta.exists():
        logger.warning("No %s next to the labels; skipping cache alignment check.", meta)
        return
    lengths = [json.loads(line)["length"] for line in meta.read_text().splitlines() if line.strip()]
    cached = episode_bounds[:, 1] - episode_bounds[:, 0]
    if len(lengths) != len(cached) or not np.array_equal(np.asarray(lengths), cached):
        raise ValueError(
            f"cache episode layout does not match {meta}: {len(cached)} cached episodes vs {len(lengths)} in "
            "metadata (or lengths differ). Episode indices in the labels would not map to cache rows."
        )


@dataclasses.dataclass(frozen=True)
class Clip:
    """One labeled critical-phase span, in global cache frame indices with `end` exclusive."""

    begin: int
    end: int
    success: bool


def load_clips(path: str | pathlib.Path, tag: str, episode_bounds: np.ndarray) -> list[Clip]:
    """Read `clips_by_tag.json` and convert the `tag` clips to global cache frame indices."""
    raw = json.loads(pathlib.Path(path).read_text())
    # The file records its own convention; the +1 below is only correct for the inclusive one.
    if raw.get("frame_range") != "inclusive":
        raise ValueError(f"{path}: expected an inclusive frame_range, got {raw.get('frame_range')!r}")
    if tag not in raw["tags"]:
        raise ValueError(f"{path}: no clips tagged {tag!r}; available tags are {sorted(raw['tags'])}")

    clips = []
    for entry in raw["tags"][tag]:
        episode = int(entry["episode_index"])
        if episode >= len(episode_bounds):
            raise ValueError(f"clip episode {episode} is outside the cache ({len(episode_bounds)} episodes)")
        if entry["result"] not in ("success", "fail"):
            raise ValueError(f"clip {entry} has result {entry['result']!r}; expected 'success' or 'fail'")
        start, stop = (int(x) for x in episode_bounds[episode])
        begin, end = start + int(entry["begin_id"]), start + int(entry["end_id"]) + 1
        if not begin < end <= stop:
            raise ValueError(f"clip {entry} does not fit inside episode {episode} (length {stop - start})")
        clips.append(Clip(begin, end, entry["result"] == "success"))
    return clips


def next_offsets(frames: np.ndarray, chunk_size: int) -> np.ndarray:
    """Rows to advance, per row, to reach the state `chunk_size` frames later inside one clip.

    A row is a stored transition and a frame is a control step, and the two coincide only offline.
    Online (Stage C) rows sit `subsample_stride` frames apart -- Appendix B's "we subsample action
    chunks 2 control steps apart" -- so `<x_0, a_0:C>`'s successor `x_C` is `C / stride` rows later,
    not `C`. Resolving that here, once, against the actual frame numbers keeps the arithmetic exact
    even when a row is missing because its observation was rejected as stale.

    Rows within `chunk_size` frames of the end point at the clip's terminal row: the chunk is
    truncated there, `done` is set, and the bootstrap is masked out anyway.
    """
    frames = np.asarray(frames, dtype=np.int64)
    # searchsorted rather than a loop: frames are sorted and this runs once per clip.
    target = np.searchsorted(frames, frames + chunk_size, side="left")
    # Pick whichever of the bracketing rows is nearer the exact target frame.
    hi = np.clip(target, 0, len(frames) - 1)
    lo = np.clip(hi - 1, 0, len(frames) - 1)
    want = frames + chunk_size
    nearer = np.where(np.abs(frames[lo] - want) <= np.abs(frames[hi] - want), lo, hi)
    # Never step backwards, and never past the terminal row.
    offset = np.maximum(nearer - np.arange(len(frames)), 0)
    to_end = frames[-1] - frames
    return np.where(to_end < chunk_size, len(frames) - 1 - np.arange(len(frames)), offset).astype(np.int64)


def compact_clips(cache: RLCache, clips: list[Clip], chunk_size: int) -> dict[str, np.ndarray]:
    """Gather one cache down to just its clip frames, and just the first `chunk_size` action steps.

    Mixing caches makes this necessary rather than merely tidy: the full teleop and rollout
    caches together are 2230 MB, which every spawned dataloader worker would copy, while the clip
    rows actually sampled come to 333 MB (6.7x less; most of the saving is dropping the 40 unused
    steps of each 50-step cached chunk). Concatenating clips in order also keeps each clip
    contiguous, so the next state stays a local offset and no index remapping is needed.

    A cached row *is* a frame here, so `next_offset` is just `min(chunk_size, to_end)`. Online it
    is not -- see `next_offsets` -- and carrying the offset per row rather than deriving it in
    `__getitem__` is what lets both pools sit in one buffer and share a time unit.
    """
    rows = np.concatenate([np.arange(c.begin, c.end) for c in clips])
    to_end = np.concatenate([np.arange(c.end - c.begin - 1, -1, -1) for c in clips]).astype(np.int64)
    return {
        "z": np.asarray(cache.z_rl[rows], dtype=np.float32),
        "state": np.asarray(cache.state[rows], dtype=np.float32),
        "action": np.asarray(cache.a_demo[rows, :chunk_size], dtype=np.float32),
        "a_ref": np.asarray(cache.a_ref[rows, :chunk_size], dtype=np.float32),
        # Frames remaining to this clip's terminal frame, and the reward waiting there.
        "to_end": to_end,
        "terminal_reward": np.concatenate(
            [np.full(c.end - c.begin, float(c.success), dtype=np.float32) for c in clips]
        ),
        "next_offset": np.minimum(chunk_size, to_end),
    }


class RLTChunkTransitionDataset(torch.utils.data.Dataset):
    """One chunk transition per frame of a labeled clip, over one or more compacted caches."""

    def __init__(self, parts: list[dict[str, np.ndarray]], chunk_size: int):
        self.chunk_size = chunk_size
        cat = lambda key: np.concatenate([p[key] for p in parts])  # noqa: E731
        self.z, self.state = cat("z"), cat("state")
        self.action, self.a_ref = cat("action"), cat("a_ref")
        self.to_end, self.terminal_reward = cat("to_end"), cat("terminal_reward")
        self.next_offset = cat("next_offset")

    def __len__(self) -> int:
        return self.z.shape[0]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        c = self.chunk_size
        # Frames left before this clip's terminal frame. The chunk is truncated there, and the next
        # state is the one actually reached; past the clip the RL policy has handed control back, so
        # there is nothing to bootstrap from and the transition is marked done.
        #
        # `to_end` counts frames while the index counts rows, and the two are the same thing only
        # offline; online, rows sit `subsample_stride` frames apart. `next_offset` carries the
        # conversion per row (see `next_offsets`), so both pools discount by the same gamma^C over
        # the same span of robot time.
        to_end = int(self.to_end[index])
        next_index = index + int(self.next_offset[index])
        done = to_end < c

        # Sparse reward at task completion (Appendix B): +1 on the last frame of a clip labeled
        # `success`, nothing on one labeled `fail`. A failure is terminal all the same, which is what
        # pushes its value down instead of leaving it to bootstrap off whatever follows.
        reward = np.zeros(c, dtype=np.float32)
        if to_end < c:
            reward[to_end] = self.terminal_reward[index]

        return {
            "z": self.z[index],
            "state": self.state[index],
            "action": self.action[index],
            "a_ref": self.a_ref[index],
            "z_next": self.z[next_index],
            "state_next": self.state[next_index],
            "a_ref_next": self.a_ref[next_index],
            "reward": reward,
            "done": np.asarray(done, dtype=np.bool_),
        }


class RLTDataLoader:
    """Infinite shuffled batches of chunk transitions, sharded across devices."""

    def __init__(
        self,
        dataset: RLTChunkTransitionDataset,
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
            num_workers=num_workers,
            # Spawn, not fork: JAX is multithreaded and forking it deadlocks.
            multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
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
        # Required by checkpoints.save_state. Stage B trains on cached features, so there are no norm
        # stats to carry -- they belong to the Stage A checkpoint that produced the cache.
        return _config.DataConfig()


def create_rlt_data_loader(config: Any, *, sharding: jax.sharding.Sharding | None = None) -> RLTDataLoader:
    if not config.sources:
        raise ValueError("config.sources is empty; Stage B needs at least one cache + clip labels.")

    parts = []
    for source in config.sources:
        # One cache is live at a time: the full teleop cache alone is ~1.8 GB, and only its
        # clip rows survive into `parts`.
        cache = RLCache.load(source.cache_dir)
        check_cache_alignment(source.clips_path, cache.episode_bounds)
        clips = load_clips(source.clips_path, source.clip_tag, cache.episode_bounds)
        parts.append(compact_clips(cache, clips, config.model.chunk_size))
        num_success = sum(c.success for c in clips)
        logger.info(
            "  %s: %d %r clips (%d success / %d fail) -> %d transitions of %d cached frames",
            source.cache_dir,
            len(clips),
            source.clip_tag,
            num_success,
            len(clips) - num_success,
            len(parts[-1]["z"]),
            cache.z_rl.shape[0],
        )
        del cache

    dataset = RLTChunkTransitionDataset(parts, config.model.chunk_size)
    logger.info("Stage B dataset: %d transitions from %d cache(s)", len(dataset), len(parts))

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size={config.batch_size} must be divisible by device_count={jax.device_count()}")
    local_batch_size = config.batch_size // jax.process_count()

    return RLTDataLoader(
        dataset,
        local_batch_size,
        sharding=sharding,
        num_workers=config.num_workers,
        seed=config.seed,
    )
