#!/usr/bin/python3
"""Run the AgileX policy and record its rollouts at 30 Hz, from one program and one keyboard.

`agilex_inference.py` is imported, not copied: `RolloutNode` subclasses its `AgileXInferenceNode`
and adds a sampler thread, a camera tap and two keys. Homing, the start gate, the publisher
thread, the inference loop, RTC and the velocity clamp all still live over there and are
untouched, so this file only has to answer "what is a frame, and when does an episode begin".

Keyboard (one terminal, one set of keys):

    ENTER  start policy control; opens an episode, or resumes the one 'q' paused
    s      success: hold the arm, save the episode, back to the start gate
    d      failure: hold the arm, discard the episode, back to the start gate
    q      pause: hold the arm and pause recording, leaving the episode open for ENTER
    r      recover (open grippers -> home -> close); discards the episode
    Ctrl-C exits, as in agilex_inference.py; an episode still open is saved, not thrown away

'q' pausing rather than closing is what makes a mid-rollout intervention cheap: the arm holds,
nothing is written, and ENTER picks the same episode back up. Recording follows policy control
exactly -- nothing is sampled while the arm is homing, held, or recovering -- so an episode
contains rollout frames only, with no still segment spliced into the middle of it.

Rows and frames are captured on separate clocks and married at the end:

    the sampler thread   ticks at --record-hz and stores (state, action, tick time)
    CameraTap            stores every camera message that arrives, with its capture stamp
    EpisodeWriter.finish nearest-matches each row to one frame per camera

The tap exists because the control node's ObservationSource keeps only the *latest* frame per
camera. Sampling that store on an independent 30 Hz clock can only ever look backwards, so
whenever two camera frames land inside one tick the first is overwritten and lost -- measured at
9-12% of frames, each one showing up as a row whose image is a repeat of its predecessor. The tap
sees every message, and matching offline can look forwards as well, which is what actually
recovers them. It runs on its own executor thread so recording never queues behind the control
node's viz timer.

The tap subscribes to each camera's `<topic>/compressed`, not to the raw topic the control node
uses, and writes those JPEG bytes to disk untouched. That is not a size optimization, it is what
keeps the policy running. Both nodes live in one process behind one GIL, and the control node's
camera subscriptions are BEST_EFFORT depth 5, so anything that stops the process draining them
promptly makes the middleware *drop* frames rather than queue them. Tapping the raw topics costs a
second 83 MB/s of Image deserialization plus a decode and a re-encode per frame (90/s of each), and
measurably starved that drain: the control node received only ~20 of every 30 published frames, its
`front` sample went stale, `snapshot()` failed --sync-tolerance continuously, and the inference loop
logged "no usable observation". Over a 15 s window, with the tap on the raw topics vs on the
compressed ones:

    frames reaching ObservationSource   290/450  ->  446/450 (i.e. the full 30 Hz)
    snapshot skew p95                     250ms  ->  66ms, the same as with no recorder at all
    snapshots rejected                   45/292  ->  0/292

`front` is always the first to trip the check: it is the one camera published as `image_raw` rather
than `image_rect_raw`, and its header stamps run ~36 ms behind the wrists' ~12 ms, so the baseline
skew is ~48 ms and only ~100 ms of the 150 ms tolerance is actually spare.

Encoder quality therefore belongs to the camera driver now (image_transport's
`<topic>.compressed.jpeg_quality`), not to this script.

What lands in each row:

    puppetLeft/puppetRight   the measured joint state, i.e. the observation
    masterLeft/masterRight   ChunkExecutor's command, i.e. the policy's predicted action after
                             chunk interpolation and the velocity clamp. That is the action that
                             actually produced the next state, so it is the correct behaviour-
                             cloning target; it equals the raw chunk sample except on the ticks
                             where the clamp binds (the inference log reports that as `clamp N%`).
    camera/color/*           the matched jpg, referenced by relative path
    camera/colorTimestamp/*  that frame's capture stamp, so how stale each image really was is a
                             measurable property of the dataset rather than a lost one. `timestamp`
                             is the sampler's own metronome and says nothing about the cameras.

Every frame the tap captured stays on disk, matched or not, so the alignment can be recomputed
later from camera/colorTimestamp without re-running the robot.

`master*`/`puppet*` are dataset field names inherited from the teleop format, not ROS topics --
`/master/joint_*`, the teleop leader arms, is never read. They are the names
`agilex_processor/hdf5_to_lerobot.py` keys on: `puppet*` becomes `observation.state`, `master*`
becomes `action`. The episode layout matches the teleop datasets, so that converter reads these
episodes with no changes (it ignores camera/colorTimestamp):

    <datasetDir>/episodeN/episodeN.hdf5
    <datasetDir>/episodeN/instructions.json
    <datasetDir>/episodeN/camera/color/{front,left,right}/<stamp>.jpg

The instruction stored with the episode is `--prompt`, the one the policy was actually given.

Usage:
.venv/bin/python agilex_recorder.py \
    --datasetDir ~/data/my_task_rollout \
    --execute \
    --host <POLICY_SERVER_IP> \
    --port 8010 \
    --viz
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import contextlib
import json
import logging
from pathlib import Path
import queue
import shutil
import sys
import threading
import time
from typing import Any

import h5py
import numpy as np
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

import agilex_inference as inference

LOGGER = logging.getLogger("agilex_recorder")

ARM_SIDES = ("Left", "Right")  # the dataset splits the 14-vector back into two 7-joint arms
# front, left, right -- already the dataset's camera names. Not inference.CAMERA_NAMES, which is
# the dict mapping these to the server's cam_high/cam_left_wrist/cam_right_wrist keys.
CAMERA_NAMES = inference.IMAGE_KEYS
WRITE_BACKLOG_WARN = 90  # queued jpgs (~1 s of frames) before the disk is called out as too slow
# A row is matched to the nearest frame however far away that is, so a camera that stalls mid-
# episode yields rows holding a stale image rather than no image, which `blind` below cannot see.
# Healthy matching sits within half a row period (~17 ms at 30 Hz); 100 ms means frames went missing.
MAX_MATCH_OFFSET = 0.100
STATUS_PERIOD = 5.0  # s between progress lines while recording

# The tap's own QoS. Same best-effort delivery the cameras publish with, but 1 s of queue instead
# of the 5 frames qos_profile_sensor_data allows. The control node can afford depth 5 because it
# only ever wants the newest frame; the tap wants all of them, so it should ride out a stall of
# more than 167 ms instead of dropping it. At ~60 KB a compressed frame that depth is ~5 MB total.
TAP_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=30)


def nearest(rows: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    """For each row time, the index of the closest capture stamp. Both must be sorted ascending."""
    if len(stamps) == 1:
        return np.zeros(len(rows), dtype=int)
    after = np.clip(np.searchsorted(stamps, rows), 1, len(stamps) - 1)
    before = after - 1
    return np.where(rows - stamps[before] <= stamps[after] - rows, before, after)


class CameraTap(Node):
    """Every camera message, compressed, on its own executor thread. See the module docstring.

    `<topic>/compressed` is image_transport's companion to each raw topic, published by the camera
    driver itself, so the same --*-image-topic flags still name the camera; there is no second set
    of flags to keep in sync.
    """

    def __init__(self, args: argparse.Namespace, sink: Callable[[str, float, CompressedImage], None]) -> None:
        super().__init__("agilex_rollout_camera_tap")
        self._sink = sink
        topics = (args.front_image_topic, args.left_image_topic, args.right_image_topic)
        for name, topic in zip(CAMERA_NAMES, topics, strict=True):
            self.create_subscription(CompressedImage, f"{topic}/compressed", self._callback(name), TAP_QOS)

    def _callback(self, name: str):
        def callback(message: CompressedImage) -> None:
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            if stamp <= 1.0:  # driver left header.stamp unset; arrival time is all we have
                stamp = self.get_clock().now().nanoseconds * 1e-9
            self._sink(name, stamp, message)

        return callback


class EpisodeWriter:
    """One episode: jpgs stream to disk on a writer thread, rows and stamps are held until the end.

    Rows cost ~1 KB each, i.e. under 2 MB per minute at 30 Hz, and the frame index is a float and a
    path each, so buffering both costs nothing and lets a discarded rollout leave nothing behind but
    the directory.
    """

    def __init__(self, directory: Path, task: str) -> None:
        self.directory = directory
        self._task = task
        self._queue: queue.Queue[tuple[Path, Any] | None] = queue.Queue()
        self._failures: list[str] = []

        self.timestamps: list[float] = []
        self.state: list[np.ndarray] = []  # (14,) measured
        self.action: list[np.ndarray] = []  # (14,) commanded
        self.frames: dict[str, list[tuple[float, str]]] = {name: [] for name in CAMERA_NAMES}
        self.out_of_order = 0

        for name in CAMERA_NAMES:
            (directory / "camera" / "color" / name).mkdir(parents=True, exist_ok=True)
        # One writer is plenty now that it only copies bytes to disk: 90 frames/s of ~60 KB is
        # ~5 MB/s. Its job is to keep disk latency off the tap's callback, not to encode anything.
        self._thread = threading.Thread(target=self._write_loop, name="recorder-jpg", daemon=True)
        self._thread.start()

    def __len__(self) -> int:
        return len(self.timestamps)

    def append_row(self, state: np.ndarray, action: np.ndarray, timestamp: float) -> None:
        self.timestamps.append(timestamp)
        self.state.append(state)
        self.action.append(action)

    def add_frame(self, name: str, stamp: float, message: CompressedImage) -> None:
        frames = self.frames[name]
        if frames and frames[-1][0] >= stamp:
            # A repeated or out-of-order stamp would collide on the filename. Counted rather than
            # ignored: this file exists to record every frame, so a dropped one should be visible.
            self.out_of_order += 1
            return
        relative = f"camera/color/{name}/{stamp:.6f}.jpg"
        frames.append((stamp, relative))
        # `data` is an array.array of the already-deserialized JPEG; queue it rather than the whole
        # ROS message so the writer's backlog doesn't pin a Message per queued frame. write_bytes
        # takes a memoryview of it, so there is no reason to copy it to bytes first.
        self._queue.put((self.directory / relative, message.data))

    def counts(self) -> str:
        counted = ", ".join(f"{name} {len(frames)}" for name, frames in self.frames.items())
        return f"{counted}, {self.out_of_order} out of order" if self.out_of_order else counted

    def backlog(self) -> int:
        return self._queue.qsize()

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            path, jpeg = item
            try:
                # The driver already encoded this as BGR JPEG, which is what the converters get
                # back from cv2.imread -- see the note in hdf5_to_lerobot.py's populate_episode.
                # So the bytes go straight down: no decode, no re-encode, no generation loss.
                path.write_bytes(jpeg)
            except OSError as exc:
                self._failures.append(str(exc))

    def finish(self, *, save: bool) -> Path | None:
        """Drain the writer thread, match rows to frames, then write episodeN.hdf5 or delete."""
        self._queue.put(None)
        self._thread.join()
        if self._failures:
            LOGGER.error("%d frames failed to write, first: %s", len(self._failures), self._failures[0])
        blind = [name for name, frames in self.frames.items() if not frames]
        if save and blind:
            # No frame at all from a camera means every row is missing an image; a half-written
            # episode would be silently unusable, so refuse it here instead.
            LOGGER.error("no frames captured from %s; discarding the episode", ", ".join(blind))
        if not save or blind or not self.timestamps:
            shutil.rmtree(self.directory, ignore_errors=True)
            return None

        rows = np.asarray(self.timestamps, dtype=np.float64)
        matched = {}
        for name, frames in self.frames.items():
            stamps = np.array([stamp for stamp, _ in frames])
            index = nearest(rows, stamps)
            matched[name] = ([frames[i][1] for i in index], stamps[index])
            offset = np.abs(rows - stamps[index])
            LOGGER.info(
                "%s: %d frames captured, %d matched to rows, offset ms median %.1f max %.1f",
                name,
                len(frames),
                len(np.unique(index)),
                np.median(offset) * 1e3,
                offset.max() * 1e3,
            )
            stalled = int((offset > MAX_MATCH_OFFSET).sum())
            if stalled:
                # Kept, not discarded: the episode is still usable and colorTimestamp records
                # exactly which rows are affected, but they should not be trusted as-is.
                LOGGER.error(
                    "%s: %d of %d rows hold a frame more than %.0f ms away (worst %.0f ms); "
                    "the camera stalled mid-episode",
                    name,
                    stalled,
                    len(rows),
                    MAX_MATCH_OFFSET * 1e3,
                    offset.max() * 1e3,
                )

        h5_path = self.directory / f"{self.directory.name}.hdf5"
        text = h5py.string_dtype()
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("size", data=len(rows))
            f.create_dataset("timestamp", data=rows)
            for role, values in (("puppet", self.state), ("master", self.action)):
                joints = np.stack(values).astype(np.float64)
                for index, side in enumerate(ARM_SIDES):
                    columns = slice(index * inference.ARM_DOF, (index + 1) * inference.ARM_DOF)
                    f.create_dataset(f"arm/jointStatePosition/{role}{side}", data=joints[:, columns])
            for name, (paths, stamps) in matched.items():
                f.create_dataset(f"camera/color/{name}", data=np.asarray(paths, dtype=object), dtype=text)
                f.create_dataset(f"camera/colorTimestamp/{name}", data=stamps)
            f.create_dataset(
                "instructions/full_instructions/text", data=np.asarray([self._task], dtype=object), dtype=text
            )
        (self.directory / "instructions.json").write_text(
            json.dumps({"full-instructions": [self._task], "segment-instructions": []}, indent=1)
        )
        return h5_path


class RolloutNode(inference.AgileXInferenceNode):
    """AgileXInferenceNode plus a 30 Hz recorder driven by the same keyboard.

    Only four of the parent's methods are extended, and none of its control logic is repeated:
    `_run_policy` brackets a rollout with the episode's open/pause, `_next_command` adds the two
    recording keys, and `start`/`shutdown` own the extra threads. `_episode` and `_sampling` are
    written by the controller thread and read by both the sampler and the tap, so every access
    goes through `_episode_lock`.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._episode_lock = threading.Lock()
        self._episode: EpisodeWriter | None = None
        self._sampling = False
        self._dropped = 0
        self._tap = CameraTap(args, self._add_frame)
        self._tap_thread = threading.Thread(target=self._spin_tap, name="recorder-tap", daemon=True)
        self._sampler = threading.Thread(target=self._record_loop, name="recorder-sampler", daemon=True)
        if not args.execute:
            self.get_logger().warning("--execute is off: the arm will not move, so any recording is a dry run")
        self.get_logger().info(
            f"recording to {args.datasetDir} at {args.record_hz} Hz, "
            f"instruction {args.prompt!r}; 's' saves an episode, 'd' discards it"
        )

    def start(self) -> None:
        super().start()
        self._tap_thread.start()
        self._sampler.start()

    def shutdown(self) -> None:
        self.stop()
        self._tap_thread.join(timeout=5.0)  # stop capturing before the episode is closed
        self._sampler.join(timeout=30.0)  # it still has to flush jpgs and close an open episode
        if self._sampler.is_alive():
            # It is a daemon, so the process is about to kill it mid-write and leave an hdf5 the
            # converter cannot read. Name the episode rather than let it look like a good one.
            self.get_logger().error("the sampler did not finish in 30s; the last episode may be truncated")
        with contextlib.suppress(Exception):
            self._tap.destroy_node()
        super().shutdown()

    # -- controller thread ---------------------------------------------------------------------

    def _next_command(self, timeout: float) -> str | None:
        """Act on the recording keys, then hand the control keys to the parent unchanged.

        's' and 'd' are reported to it as 'q', so it does what it already does for a pause: hold
        the arm and fall back to the start gate. The arm is held here first so that it stops the
        instant the key is pressed rather than after the episode has finished flushing to disk.
        """
        command = super()._next_command(timeout)
        if command in ("s", "d", "r"):
            self._executor.hold()
            self._finish_episode(save=command == "s")
        return "q" if command in ("s", "d") else command

    def _run_policy(self) -> None:
        self._open_episode()
        with self._episode_lock:
            self._sampling = True
        try:
            super()._run_policy()
        finally:
            # Every exit -- 's', 'd', 'q', 'r', a dropped connection, shutdown -- lands here, and
            # none of them should leave the recorder capturing an arm that is no longer driven.
            with self._episode_lock:
                self._sampling = False

    def _next_episode_dir(self) -> Path:
        root = Path(self.args.datasetDir)
        root.mkdir(parents=True, exist_ok=True)
        used = [int(n) for p in root.glob("episode*") if (n := p.name.removeprefix("episode")).isdigit()]
        return root / f"episode{max(used) + 1 if used else 0}"

    def _open_episode(self) -> None:
        with self._episode_lock:
            if self._episode is not None:
                self.get_logger().info(f"resuming {self._episode.directory.name} at {len(self._episode)} rows")
                return
            self._episode = EpisodeWriter(self._next_episode_dir(), self.args.prompt)
            self._dropped = 0
            self.get_logger().info(f"recording {self._episode.directory.name} ('s' saves, 'd' discards)")

    def _finish_episode(self, *, save: bool) -> None:
        # Swap the episode out under the lock, then write it outside: the sampler and the tap are
        # free again immediately, and neither can be mid-append while the jpg queue drains.
        with self._episode_lock:
            episode, self._episode, self._sampling = self._episode, None, False
        if episode is None:
            return
        self.get_logger().info(f"{'saving' if save else 'discarding'} {episode.directory.name}...")
        h5_path = episode.finish(save=save)
        if h5_path is None:
            self.get_logger().info(f"discarded {episode.directory.name}")
        else:
            self.get_logger().info(f"saved {h5_path} ({len(episode)} rows, {self._dropped} dropped)")

    # -- camera tap thread ---------------------------------------------------------------------

    def _spin_tap(self) -> None:
        executor = SingleThreadedExecutor()
        executor.add_node(self._tap)
        try:
            while not self._stop.is_set():
                executor.spin_once(timeout_sec=0.05)
        finally:
            with contextlib.suppress(Exception):
                executor.remove_node(self._tap)

    def _add_frame(self, name: str, stamp: float, message: CompressedImage) -> None:
        with self._episode_lock:
            if self._episode is not None and self._sampling:
                self._episode.add_frame(name, stamp, message)

    # -- sampler thread ------------------------------------------------------------------------

    def _record_loop(self) -> None:
        """Store a (state, action) row at --record-hz for as long as the policy drives the arm."""
        period = 1.0 / self.args.record_hz
        next_tick = status = time.monotonic()
        while not self._stop.is_set():
            with self._episode_lock:
                active = self._episode is not None and self._sampling
            if active:
                self._sample_once()
                if time.monotonic() - status >= STATUS_PERIOD:
                    status = time.monotonic()
                    self._log_progress()

            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep < -0.5 * period:  # big overrun: resync instead of burning down the backlog
                next_tick, sleep = time.monotonic(), 0.0
            if sleep > 0 and self._stop.wait(sleep):
                break

        # Ctrl-C with an episode open: keep it. A rollout is expensive to redo and cheap to delete.
        # No-ops when there is none; _finish_episode does that check under the lock, unlike here.
        self._finish_episode(save=True)

    def _sample_once(self) -> None:
        measured = self._obs.latest_state()
        command = self._executor.command()
        if measured is None or command is None:
            self._dropped += 1
            return
        state, captured = measured
        age = time.monotonic() - captured
        if age > self.args.max_observation_age:
            self._dropped += 1
            self.get_logger().warning(f"state is {age:.3f}s old; dropping the row", throttle_duration_sec=2.0)
            return
        timestamp = self.get_clock().now().nanoseconds * 1e-9
        with self._episode_lock:
            # Re-checked, not redundant: the lock was down for the reads above, and an 's' or 'd' in
            # that window hands this episode to finish(), which stacks these three lists together.
            if self._episode is not None and self._sampling:
                self._episode.append_row(state, command, timestamp)

    def _log_progress(self) -> None:
        with self._episode_lock:
            episode = self._episode
        if episode is None:
            return
        backlog = episode.backlog()
        self.get_logger().info(
            f"{episode.directory.name}: {len(episode)} rows, {self._dropped} dropped, "
            f"frames [{episode.counts()}], {backlog} jpgs queued"
        )
        if backlog > WRITE_BACKLOG_WARN:
            self.get_logger().warning(f"jpg writer is {backlog} frames behind; the disk cannot keep up")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every agilex_inference.py flag is also accepted; run `agilex_inference.py --help` for those.",
    )
    parser.add_argument("--datasetDir", required=True, help="Episodes are written as <datasetDir>/episodeN/")
    parser.add_argument("--record-hz", type=float, default=30.0, help="Row rate; must equal the training fps")
    args, rest = parser.parse_known_args()

    if args.record_hz <= 0:
        parser.error("--record-hz must be positive")

    # agilex_inference.parse_args() reads sys.argv and owns every control-side flag and its
    # validation. Hand it exactly what this parser did not claim, rather than restating any of it.
    sys.argv = [sys.argv[0], *rest]
    control = inference.parse_args()
    for key, value in vars(args).items():
        setattr(control, key, value)
    return control


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # rclpy setup, the SIGINT handling and the spin loop are all the parent's, unchanged. The first
    # Ctrl-C reaches _record_loop's tail, which is what saves an episode that is still open.
    inference.main(RolloutNode, parse_args())
