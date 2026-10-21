"""RLT Stage C: the live replay buffer B of Algorithm 1.

The paper's buffer "aggregates VLA warmup data, online RL rollouts, and optional human
interventions" (Sec. V), and off-policy learning does not care which policy produced a transition.
So this is deliberately the *same* container Stage B already trains on -- a list of labeled clips
in the compacted layout of `data_loader_rlt.compact_clips` -- with two differences:

  * clips arrive while training runs, so everything is behind a lock; and
  * an online row is one policy query rather than one frame, which is what the per-row `stride`
    field records (see `RLTChunkTransitionDataset.__getitem__` for the arithmetic it drives).

Offline caches are preloaded through exactly the Stage B path (`RLCache.load` ->
`check_cache_alignment` -> `load_clips` -> `compact_clips`), which is Algorithm 1's `N_warm`
pre-fill: without it the critic is random on the first rollout and the actor has nothing to climb.

A clip only becomes sampleable when the operator labels it. That is not a convenience -- the
sparse success/failure label is the entire reward signal (Sec. V), so an unlabeled episode has no
target to regress toward and must not be trained on.

Everything is held in RAM and written with whole-array `np.save`. The dataset filesystem is
FUSE-mounted OrangeFS, where random-access writeback costs seconds per batch; see the table in
`docs/rlt_wip.md`.
"""

from collections.abc import Iterable
import dataclasses
import logging
import pathlib
import threading

import numpy as np

import openpi.training.data_loader_rlt as _data_loader_rlt

logger = logging.getLogger("openpi")

# The arrays a clip is made of. `z`/`state`/`action`/`a_ref` are per-row features; `to_end` (in
# frames) and `terminal_reward` carry the return; `next_offset` is how many rows ahead the next
# state sits. Kept as one tuple so the buffer, `compact_clips` and the on-disk format cannot drift.
FIELDS = ("z", "state", "action", "a_ref", "to_end", "terminal_reward", "next_offset")


@dataclasses.dataclass
class _Episode:
    """An episode being collected.

    Rows are appended by the post-episode ingest pass, `frames` records each one's control-step
    index inside the critical phase, and nothing is admitted to the pool until the operator labels
    the outcome.
    """

    z: list[np.ndarray] = dataclasses.field(default_factory=list)
    state: list[np.ndarray] = dataclasses.field(default_factory=list)
    action: list[np.ndarray] = dataclasses.field(default_factory=list)
    a_ref: list[np.ndarray] = dataclasses.field(default_factory=list)
    frames: list[int] = dataclasses.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.z)


def _clip_arrays(episode: _Episode, *, success: bool, chunk_size: int) -> dict[str, np.ndarray]:
    """Turn a finished episode into one clip, in the same layout `compact_clips` emits.

    `to_end` is in **frames** so it indexes the `chunk_size`-long reward vector directly, while the
    rows it is attached to are `subsample_stride` frames apart; `next_offset` bridges the two and
    is computed from the actual frame numbers, so a row dropped for a stale observation does not
    shift its neighbours' successors.
    """
    frames = np.asarray(episode.frames, dtype=np.int64)
    return {
        "z": np.stack(episode.z).astype(np.float32),
        "state": np.stack(episode.state).astype(np.float32),
        "action": np.stack(episode.action).astype(np.float32),
        "a_ref": np.stack(episode.a_ref).astype(np.float32),
        "to_end": (frames[-1] - frames).astype(np.int64),
        "terminal_reward": np.full(len(frames), float(success), dtype=np.float32),
        "next_offset": _data_loader_rlt.next_offsets(frames, chunk_size),
    }


class ReplayBuffer:
    """Labeled clips, sampled uniformly as one pool. Thread-safe: one writer, one reader.

    The collector thread calls `open_episode`/`append`/`close_episode`; the learner thread calls
    `sample`. Both take `_lock`, which is held only for the gather itself (~0.3 ms at batch 512),
    so neither meaningfully blocks the other.
    """

    def __init__(self, chunk_size: int, *, capacity: int = 200_000):
        self.chunk_size = chunk_size
        self.capacity = capacity
        self._lock = threading.Lock()
        self._clips: list[dict[str, np.ndarray]] = []
        self._success: list[bool] = []
        self._online: list[bool] = []
        # The flat view the learner samples from, rebuilt whenever a clip is added or evicted.
        self._flat: dict[str, np.ndarray] | None = None
        self._episode: _Episode | None = None
        self.env_steps = 0  # rows collected online; the denominator of the UTD ratio

    # -- collector side --------------------------------------------------------------------------

    def open_episode(self) -> None:
        with self._lock:
            if self._episode is not None and len(self._episode):
                logger.warning("opening an episode over %d unlabeled rows; discarding them", len(self._episode))
            self._episode = _Episode()

    def append_rows(
        self,
        frames: np.ndarray,
        z: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        a_ref: np.ndarray,
    ) -> int:
        """Store a batch of subsampled transition rows. Returns the episode's row count.

        `frames` is each row's control-step index inside the critical phase; it is what makes the
        chunk arithmetic exact when rows are `subsample_stride` apart rather than adjacent.
        """
        with self._lock:
            if self._episode is None:
                return 0
            self._episode.frames.extend(int(f) for f in frames)
            self._episode.z.extend(np.asarray(z, dtype=np.float32))
            self._episode.state.extend(np.asarray(state, dtype=np.float32))
            self._episode.action.extend(np.asarray(action, dtype=np.float32))
            self._episode.a_ref.extend(np.asarray(a_ref, dtype=np.float32))
            self.env_steps += len(frames)
            return len(self._episode)

    def close_episode(self, success: bool | None) -> int:
        """Label the open episode and admit it to the pool. `success=None` discards it.

        Returns the number of rows admitted. A single-row episode is dropped: its only transition
        would be terminal with the reward at frame 0, which teaches the critic nothing about the
        approach and is almost always a mis-key rather than a real outcome.
        """
        with self._lock:
            episode, self._episode = self._episode, None
            if episode is None or success is None or len(episode) < 2:
                if episode is not None and len(episode):
                    logger.info("discarding %d unlabeled/short rows", len(episode))
                return 0
            self._clips.append(_clip_arrays(episode, success=success, chunk_size=self.chunk_size))
            self._success.append(success)
            self._online.append(True)
            self._evict_locked()
            self._flat = None
            return len(episode)

    # -- preload ---------------------------------------------------------------------------------

    def preload(self, sources: Iterable) -> None:
        """Pre-fill from the Stage A.5 caches, i.e. Algorithm 1's N_warm warmup data.

        Exactly the path `create_rlt_data_loader` takes, so the offline and online halves of the
        pool are byte-comparable. One cache is live at a time: the full teleop cache alone
        is ~1.8 GB, and only its clip rows survive.
        """
        for source in sources:
            cache = _data_loader_rlt.RLCache.load(source.cache_dir)
            _data_loader_rlt.check_cache_alignment(source.clips_path, cache.episode_bounds)
            clips = _data_loader_rlt.load_clips(source.clips_path, source.clip_tag, cache.episode_bounds)
            with self._lock:
                for clip in clips:
                    part = _data_loader_rlt.compact_clips(cache, [clip], self.chunk_size)
                    self._clips.append(part)
                    self._success.append(clip.success)
                    self._online.append(False)
                self._flat = None
            num_success = sum(c.success for c in clips)
            logger.info(
                "  preloaded %s: %d %r clips (%d success / %d fail) -> %d transitions",
                source.cache_dir,
                len(clips),
                source.clip_tag,
                num_success,
                len(clips) - num_success,
                sum(c.end - c.begin for c in clips),
            )
            del cache

    # -- learner side ----------------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return sum(len(clip["z"]) for clip in self._clips)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """A uniform batch of chunk transitions, in the layout `train_rlt_ac.train_step` expects.

        The per-sample arithmetic is `RLTChunkTransitionDataset.__getitem__`, vectorized: the
        dataset class stays the definition of a transition and this stays a gather.
        """
        with self._lock:
            flat = self._flatten_locked()
            if flat is None:
                raise ValueError("replay buffer is empty")
            index = rng.integers(0, flat["z"].shape[0], size=batch_size)

            c = self.chunk_size
            to_end = flat["to_end"][index]
            next_index = index + flat["next_offset"][index]
            done = to_end < c

            reward = np.zeros((batch_size, c), dtype=np.float32)
            # Sparse terminal reward: +1 on the last frame of a success clip, nothing on a fail.
            # `np.where` rather than a mask index so the out-of-range `to_end` of non-terminal rows
            # is never used as a column.
            rows = np.flatnonzero(done)
            reward[rows, to_end[rows]] = flat["terminal_reward"][index][rows]

            return {
                "z": flat["z"][index],
                "state": flat["state"][index],
                "action": flat["action"][index],
                "a_ref": flat["a_ref"][index],
                "z_next": flat["z"][next_index],
                "state_next": flat["state"][next_index],
                "a_ref_next": flat["a_ref"][next_index],
                "reward": reward,
                "done": done,
            }

    def stats(self) -> dict[str, float]:
        with self._lock:
            online = [s for s, o in zip(self._success, self._online, strict=True) if o]
            return {
                "buffer_transitions": float(sum(len(clip["z"]) for clip in self._clips)),
                "buffer_clips": float(len(self._clips)),
                "online_clips": float(len(online)),
                "online_success_rate": float(np.mean(online)) if online else 0.0,
                "env_steps": float(self.env_steps),
            }

    # -- persistence -----------------------------------------------------------------------------

    def save(self, directory: str | pathlib.Path) -> None:
        """Write the online clips as flat arrays plus their boundaries.

        Only the online half: the offline caches are already on disk and reloading them from
        `preload` is both cheaper and less likely to go stale. Whole arrays, one sequential
        `np.save` each -- never memmapped, see the module docstring.
        """
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            clips = [clip for clip, online in zip(self._clips, self._online, strict=True) if online]
            success = [s for s, online in zip(self._success, self._online, strict=True) if online]
        if not clips:
            return
        lengths = np.asarray([len(clip["z"]) for clip in clips], dtype=np.int64)
        for name in FIELDS:
            np.save(directory / f"{name}.npy", np.concatenate([clip[name] for clip in clips]))
        np.save(directory / "clip_lengths.npy", lengths)
        np.save(directory / "clip_success.npy", np.asarray(success, dtype=np.bool_))
        logger.info("Wrote %d online clips (%d rows) to %s", len(clips), int(lengths.sum()), directory)

    def load(self, directory: str | pathlib.Path) -> int:
        """Restore online clips written by `save`. Returns the number of rows restored."""
        directory = pathlib.Path(directory)
        if not (directory / "clip_lengths.npy").exists():
            return 0
        arrays = {name: np.load(directory / f"{name}.npy") for name in FIELDS}
        lengths = np.load(directory / "clip_lengths.npy")
        success = np.load(directory / "clip_success.npy")
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        with self._lock:
            for i, ok in enumerate(success):
                lo, hi = int(bounds[i]), int(bounds[i + 1])
                self._clips.append({name: arrays[name][lo:hi] for name in FIELDS})
                self._success.append(bool(ok))
                self._online.append(True)
            self.env_steps += int(lengths.sum())
            self._evict_locked()
            self._flat = None
        logger.info("Restored %d online clips (%d rows) from %s", len(lengths), int(lengths.sum()), directory)
        return int(lengths.sum())

    # -- internals (caller holds _lock) ----------------------------------------------------------

    def _evict_locked(self) -> None:
        """Drop whole oldest clips until the pool fits.

        Whole clips, never a suffix: `to_end` and the `next_index` offset are within-clip, so a
        partially evicted clip would bootstrap off another clip's rows.
        """
        total = sum(len(clip["z"]) for clip in self._clips)
        dropped = 0
        while total > self.capacity and len(self._clips) > 1:
            total -= len(self._clips[0]["z"])
            self._clips.pop(0)
            self._success.pop(0)
            self._online.pop(0)
            dropped += 1
        if dropped:
            logger.info("buffer at capacity %d; evicted %d oldest clips", self.capacity, dropped)

    def _flatten_locked(self) -> dict[str, np.ndarray] | None:
        """Concatenate the clips once per mutation, so sampling is a plain gather.

        Rebuilt on add/evict rather than maintained incrementally: a session adds a clip every few
        minutes, and a 32k-row concatenate costs ~0.1 s against a step that runs thousands of times
        in between.
        """
        if self._flat is None and self._clips:
            self._flat = {name: np.concatenate([clip[name] for clip in self._clips]) for name in FIELDS}
        return self._flat
