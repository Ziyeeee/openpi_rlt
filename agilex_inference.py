#!/usr/bin/python3
"""ROS 2 adapter for the remote OpenPI AgileX policy.

The policy server owns checkpoint loading, normalization and image/model transforms. This node
transports raw Piper observations, shapes the returned absolute 14-dimensional joint action chunk
into a velocity-limited command stream, and publishes it.

    ObservationSource   latest-frame storage, header-stamp freshness, lazy image decode
    ChunkExecutor       the only place motion is shaped: interpolation + per-joint velocity clamp
    AgileXInferenceNode homing, start gate, publisher thread, paced inference loop

Threads: rclpy callbacks on the main thread (cheap stores only), a publisher thread that owns the
clamp integrator, a controller thread that homes, connects and drives snapshot -> infer -> submit,
and a keyboard thread that reads line-buffered stdin commands into a queue the controller polls at
chunk boundaries: blank ENTER starts/resumes policy control, 'q' pauses it (hold in place, stay
connected), 'r' recovers (open the grippers, carry them open to the home pose, then close them)
and always leaves the operator back at the start gate so resuming is a deliberate ENTER, not
automatic.

The returned chunk's 12 arm dimensions are absolute but *anchored to the state that was sent*
(use_delta_joint_actions + AbsoluteActions server-side), while dims 6 and 13 (the grippers) are
genuinely absolute. That anchoring is why chunk seams need a clamp.

It is also why chunk playback runs off the wall clock rather than a tick counter. actions[k] is
the command for `anchor + k / policy_hz`, where `anchor` is the instant the state was captured, so
a chunk that took 120ms to come back enters at step ~3.6 -- the steps that elapsed in flight are
skipped, not replayed. Starting every chunk at index 0 would drag the arm back to where it stood
at request time, once per replan.

For RTC configs (e.g. `pi05_rtc_agilex_right`, see openpi.models.pi0_rtc), each request also sends
`action_prefix`: the previous chunk's own predicted actions, resampled at the *new* request's
anchor and passed back in as `--delay-steps` steps of ground truth the model must reproduce exactly
before it's free to plan the rest of the chunk. This is what makes consecutive chunks agree on their
overlap instead of merely being clamp-smoothed into agreement after the fact.

Usage:
.venv/bin/python agilex_inference.py \
    --execute \
    --host <POLICY_SERVER_IP> \
    --port 8010 \
    --viz
"""

from __future__ import annotations

import argparse
import array
import contextlib
from dataclasses import dataclass
import enum
import logging
import os
import queue
import signal
import sys
import threading
import time
from typing import Any

import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

LOGGER = logging.getLogger("agilex_inference")


ARM_DOF = 7
ACTION_DIM = 14
# Not a knob: the server's model transform resizes to 224 regardless, so sending any other size
# just pads the image twice. Matching it here keeps the request small without degrading the input.
IMAGE_SIZE = 224
# The 12 arm joints are delta-anchored server-side and get interpolated between chunk steps; the
# two grippers are absolute and near-binary, so they get a zero-order hold and their own limit.
ARM_IDX = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIP_IDX = np.array([6, 13])
JOINT_NAMES = [f"joint_{index + 1}" for index in range(ARM_DOF)]

STATE_KEYS = ("left_state", "right_state")
IMAGE_KEYS = ("front", "left", "right")
ALL_KEYS = STATE_KEYS + IMAGE_KEYS
# ROS topic role -> the camera name the server expects (a subset of AlohaInputs.EXPECTED_CAMERAS).
CAMERA_NAMES = {"front": "cam_high", "left": "cam_left_wrist", "right": "cam_right_wrist"}
VIZ_TITLE = "AgileX cameras"
VIZ_ORDER = ("left", "front", "right")  # [left | head | right], left to right on screen
VIZ_LABELS = {"left": "left", "front": "head", "right": "right"}
VIZ_CELL = (480, 360)  # (w, h) each camera is resized to before concatenation

BOUNDARY_JUMP_LIMIT = 0.60  # rad; reject a chunk starting further than this from the command
STALE_CHUNK_GRACE = 0.5  # s to hold an exhausted chunk's final target before freezing
OBS_RETRY = 0.02  # s between snapshot attempts
OBS_TIMEOUT = 1.0  # s without a usable observation -> hold position
OBS_STOP_TIMEOUT = 3.0  # s without a usable observation -> stop
MAX_STAMP_LATENCY = 0.5  # s; a header.stamp further than this from our clock is not believed
HOME_GRIPPER_VEL = 0.2  # gripper units/s during the home ramp
HOME_TIMEOUT = 20.0  # s for homing to converge
RETRY_DELAY = 1.0  # s between reconnect attempts
VIZ_HZ = 10.0

# States and commands are low-rate and must not be dropped; the camera topics use sensor-data QoS.
RELIABLE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)


def action_chunk(result: dict[str, Any]) -> np.ndarray:
    """Validate the policy response and return its (H, 14) chunk."""
    if "actions" not in result:
        raise ValueError(f"policy response has no 'actions' key: {tuple(result)}")
    actions = np.asarray(result["actions"], dtype=np.float64)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] == 0:
        raise ValueError(f"expected action chunk [H, {ACTION_DIM}], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("policy returned NaN or Inf")
    return actions


def rtc_action_prefix(
    prev_actions: np.ndarray | None, prev_anchor: float | None, anchor: float, policy_hz: float, delay_steps: int
) -> tuple[np.ndarray | None, int]:
    """Resample the previous chunk's own prediction at `anchor` for use as an RTC action_prefix.

    `prev_actions[k]` commands `prev_anchor + k / policy_hz`, the same convention ChunkExecutor plays
    it back with, so the step that lands on the new anchor is `round((anchor - prev_anchor) * policy_hz)`.
    Returns (None, 0) if there is no previous chunk, or the round trip ran past its horizon. Otherwise
    an array shaped like `prev_actions` with the aligned steps front-loaded into the first `delay` rows
    -- the rest is unused padding (see the shape requirement on Pi0RTC.sample_actions's action_prefix).
    """
    if prev_actions is None or delay_steps == 0:
        return None, 0
    start = round((anchor - prev_anchor) * policy_hz)
    delay = min(delay_steps, len(prev_actions) - start) if 0 <= start < len(prev_actions) else 0
    if delay == 0:
        return None, 0
    action_prefix = np.zeros_like(prev_actions)
    action_prefix[:delay] = prev_actions[start : start + delay]
    return action_prefix, delay


# --------------------------------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    payload: Any  # a 7-vector for states; the raw Image message for cameras
    captured: float  # capture time on the monotonic clock


@dataclass(frozen=True)
class Snapshot:
    observation: dict
    skew: float
    captured: float  # oldest sample across ALL_KEYS; drives the staleness check
    anchor: float  # oldest state sample; the instant the returned actions[0] refers to


class ObservationSource:
    """Latest-frame observation store. Subscription callbacks only do a cheap dict store."""

    def __init__(self, node: Node, args: argparse.Namespace) -> None:
        self._node = node
        self._args = args
        self._cv = threading.Condition()
        self._latest: dict[str, Sample] = {}
        self._bridge = CvBridge()  # only ever touched by the controller thread

        # The Node owns every subscription it creates, so there is nothing to keep alive here.
        for key, topic in zip(STATE_KEYS, (args.left_state_topic, args.right_state_topic), strict=True):
            node.create_subscription(JointState, topic, self._state_callback(key), RELIABLE_QOS)
        images = (args.front_image_topic, args.left_image_topic, args.right_image_topic)
        for key, topic in zip(IMAGE_KEYS, images, strict=True):
            node.create_subscription(Image, topic, self._image_callback(key), qos_profile_sensor_data)

    # -- callbacks (main thread) --------------------------------------------------------------------

    def _state_callback(self, key: str):
        def callback(msg: JointState) -> None:
            if len(msg.position) < ARM_DOF:
                self._warn(f"{key} has {len(msg.position)} positions; expected >= {ARM_DOF}")
                return
            value = np.asarray(msg.position[:ARM_DOF], dtype=np.float64)
            if not np.isfinite(value).all():
                self._warn(f"{key} contains NaN or Inf")
                return
            self._store(key, value, msg.header)

        return callback

    def _image_callback(self, key: str):
        return lambda msg: self._store(key, msg, msg.header)

    def _store(self, key: str, payload: Any, header: Any) -> None:
        received = time.monotonic()
        stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        # ROS time and monotonic have different epochs, so project the driver's stamp onto our
        # timeline via its transport latency rather than comparing the two clocks directly.
        latency = self._node.get_clock().now().nanoseconds * 1e-9 - stamp
        # The bound is symmetric and generous: a small negative latency is ordinary cross-process
        # clock jitter, not a bad stamp. Only unset stamps and wrong epochs should fall back.
        if stamp > 1.0 and abs(latency) <= MAX_STAMP_LATENCY:
            captured = received - latency
        else:
            captured = received
            self._warn(f"{key}: unusable header.stamp ({latency:.3f}s); using arrival time for this topic")
        with self._cv:
            self._latest[key] = Sample(payload=payload, captured=captured)
            self._cv.notify_all()

    # -- readers (controller / viz threads) --------------------------------------------------------

    def latest_state(self) -> tuple[np.ndarray, float] | None:
        """The measured 14-joint state and the capture time of its older half, or None if unset.

        The capture time is the same quantity `snapshot()` uses as its action anchor; returning it
        here too is what lets a caller judge how stale the state is without reading `_latest`.
        """
        with self._cv:
            if any(key not in self._latest for key in STATE_KEYS):
                return None
            samples = [self._latest[key] for key in STATE_KEYS]
        return np.concatenate([s.payload for s in samples]), min(s.captured for s in samples)

    def raw_frames(self) -> dict[str, Any]:
        with self._cv:
            return {key: (self._latest[key].payload if key in self._latest else None) for key in IMAGE_KEYS}

    def snapshot(self, timeout: float) -> Snapshot | None:
        """Build the wire observation, or return None with the reason logged."""
        with self._cv:
            if not self._cv.wait_for(lambda: len(self._latest) == len(ALL_KEYS), timeout=timeout):
                missing = [key for key in ALL_KEYS if key not in self._latest]
                self._warn(f"waiting for first message on: {', '.join(missing)}")
                return None
            samples = dict(self._latest)

        oldest = min(ALL_KEYS, key=lambda key: samples[key].captured)
        newest = max(ALL_KEYS, key=lambda key: samples[key].captured)
        skew = samples[newest].captured - samples[oldest].captured
        age = time.monotonic() - samples[oldest].captured
        if skew > self._args.sync_tolerance:
            self._warn(f"skew {skew:.3f}s > {self._args.sync_tolerance:.3f}s: {oldest} trails {newest}")
            return None
        if age > self._args.max_observation_age:
            self._warn(f"observation is {age:.3f}s old (oldest: {oldest})")
            return None

        try:
            images = {CAMERA_NAMES[key]: self._encode_image(samples[key].payload) for key in IMAGE_KEYS}
        except (CvBridgeError, ValueError) as exc:
            self._warn(f"image decode failed: {exc}")
            return None

        state = np.concatenate([samples[key].payload for key in STATE_KEYS])
        return Snapshot(
            observation={"state": state.astype(np.float32), "images": images, "prompt": self._args.prompt},
            skew=skew,
            captured=samples[oldest].captured,
            # The arm actions come back delta-anchored to *this state*, so the action timeline
            # starts when the state was captured -- not at the oldest camera frame, which
            # --sync-tolerance allows to sit several policy steps earlier.
            anchor=min(samples[key].captured for key in STATE_KEYS),
        )

    def _encode_image(self, msg: Image) -> np.ndarray:
        rgb = np.asarray(self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected HWC RGB image, got {rgb.shape}")
        # resize_with_pad is HWC-only, so resize before transposing. The server reads CHW uint8 and
        # does not validate the layout, so getting this order wrong fails silently and much later.
        resized = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, IMAGE_SIZE, IMAGE_SIZE))
        return np.ascontiguousarray(np.transpose(resized, (2, 0, 1)))

    def _warn(self, message: str) -> None:
        """Warn at most once every few seconds per call site."""
        self._node.get_logger().warning(message, throttle_duration_sec=2.0)


# --------------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------------


class Mode(enum.Enum):
    GOTO = "goto"  # ramp to a fixed goal; the clamp is the ramp
    CHUNK = "chunk"  # play an action chunk
    HOLD = "hold"  # freeze in place


class ChunkExecutor:
    """Turns action chunks into a velocity-limited command stream.

    Playback is a function of the wall clock, not of a tick counter: a chunk carries the monotonic
    time its step 0 refers to, so ``phase = (now - anchor) * policy_hz``. Inference latency
    therefore *skips into* a freshly arrived chunk instead of replaying the steps that elapsed
    while it was in flight.

    Every field is guarded by ``_lock``, but only the publisher thread ever writes ``_command`` --
    it is a rate-limiter integrator, so a second writer would corrupt the ramp.
    """

    def __init__(self, publish_hz: float, policy_hz: float, arm_vel: float, gripper_vel: float) -> None:
        self._publish_hz = publish_hz
        self._policy_hz = policy_hz
        self._limit = self._limits(arm_vel, gripper_vel)
        self._lock = threading.Lock()
        self._command: np.ndarray | None = None
        self._mode = Mode.HOLD
        self._goal: np.ndarray | None = None
        self._goal_limit: np.ndarray | None = None
        self._chunk: np.ndarray | None = None
        self._anchor: float | None = None  # monotonic time that _chunk[0] commands
        self._exhausted_since: float | None = None
        self._ticks = 0
        self._saturated = 0

    def _limits(self, arm_vel: float, gripper_vel: float) -> np.ndarray:
        limit = np.empty(ACTION_DIM)
        limit[ARM_IDX] = arm_vel / self._publish_hz
        limit[GRIP_IDX] = gripper_vel / self._publish_hz
        return limit

    # -- controller thread -------------------------------------------------------------------------

    def seed(self, measured: np.ndarray) -> None:
        """Start the integrator at the measured state, so the first command is not a jump."""
        with self._lock:
            self._command = np.asarray(measured, dtype=np.float64).copy()

    def goto(self, goal: np.ndarray, arm_vel: float, gripper_vel: float) -> None:
        with self._lock:
            self._goal = np.asarray(goal, dtype=np.float64).copy()
            self._goal_limit = self._limits(arm_vel, gripper_vel)
            self._mode = Mode.GOTO

    def hold(self) -> None:
        with self._lock:
            self._mode = Mode.HOLD

    def submit(self, actions: np.ndarray, anchor: float, now: float) -> str | None:
        """Install a chunk anchored to the observation instant it was predicted from.

        ``actions[k]`` commands ``anchor + k / policy_hz``, and the round trip has already burned
        the first ``entry`` steps, so playback enters there. Returns a rejection reason if the
        chunk is too old to be useful or enters too far from the current command.
        """
        entry = (now - anchor) * self._policy_hz
        if entry >= len(actions) - 1:
            return f"latency {now - anchor:.3f}s consumed the whole {len(actions)}-step chunk"
        with self._lock:
            if self._command is not None:
                # Compare the sample we will actually command, not actions[0], which is in the
                # past. Grippers are excluded: they are genuinely absolute and legitimately step.
                jump = float(np.abs(self._sample(actions, entry)[ARM_IDX] - self._command[ARM_IDX]).max())
                if jump > BOUNDARY_JUMP_LIMIT:
                    return f"entry sample (step {entry:.1f}) is {jump:.3f} rad from the current command"
            self._chunk, self._anchor = actions, anchor
            self._mode, self._exhausted_since = Mode.CHUNK, None
            return None

    def command(self) -> np.ndarray | None:
        with self._lock:
            return None if self._command is None else self._command.copy()

    def saturation(self) -> float:
        """Fraction of ticks where the velocity clamp bound, then reset the counters."""
        with self._lock:
            fraction = self._saturated / self._ticks if self._ticks else 0.0
            self._ticks = self._saturated = 0
            return fraction

    # -- publisher thread --------------------------------------------------------------------------

    def step(self, now: float) -> np.ndarray | None:
        with self._lock:
            if self._command is None:
                return None  # not seeded yet: never publish an unknown command
            target = self._target(now)
            limit = self._goal_limit if self._mode is Mode.GOTO else self._limit
            delta = target - self._command
            clipped = np.clip(delta, -limit, limit)
            self._command += clipped
            self._ticks += 1
            # np.clip returns delta unchanged when it is already in range, so any difference here
            # is exactly the velocity limit binding.
            self._saturated += bool((clipped != delta).any())
            return self._command.copy()

    def _target(self, now: float) -> np.ndarray:
        # _goal and _chunk are always set before their mode is, under this same lock.
        if self._mode is Mode.GOTO:
            return self._goal
        if self._mode is not Mode.CHUNK:
            return self._command  # HOLD: zero delta, so the clamp is a no-op
        last = len(self._chunk) - 1
        phase = (now - self._anchor) * self._policy_hz
        if phase >= last:
            if self._exhausted_since is None:
                self._exhausted_since = now
                LOGGER.warning("chunk exhausted; holding its final target")
            elif now - self._exhausted_since > STALE_CHUNK_GRACE:
                self._mode = Mode.HOLD
                LOGGER.error("no new chunk for %.1fs after exhaustion; freezing command", STALE_CHUNK_GRACE)
        return self._sample(self._chunk, phase)

    @staticmethod
    def _sample(chunk: np.ndarray, phase: float) -> np.ndarray:
        """The command a chunk asks for at fractional step ``phase``, clamped to its horizon."""
        last = len(chunk) - 1
        phase = min(max(phase, 0.0), float(last))
        index = min(int(phase), last)
        target = chunk[index].copy()
        if index < last:  # lerp the arms; the grippers keep their zero-order hold
            frac = phase - index
            target[ARM_IDX] += frac * (chunk[index + 1][ARM_IDX] - chunk[index][ARM_IDX])
        return target


# --------------------------------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------------------------------


class AgileXInferenceNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("agilex_openpi_inference")
        self.args = args
        self._stop = threading.Event()  # stop closed-loop control
        self._exit = threading.Event()  # tear the process down
        self._commands: queue.Queue[str] = queue.Queue()  # stdin lines: "" start/resume, "q", "r"
        self._policy: websocket_client_policy.WebsocketClientPolicy | None = None
        self._viz_bridge = CvBridge() if args.viz else None
        self._threads: list[threading.Thread] = []

        self._obs = ObservationSource(self, args)
        self._executor = ChunkExecutor(args.publish_hz, args.policy_hz, args.max_joint_vel, args.max_gripper_vel)

        # Not `self._publishers`: rclpy's Node already owns an attribute by that name and walks it
        # during destroy_node().
        self._command_publishers = tuple(
            self.create_publisher(JointState, topic, RELIABLE_QOS)
            for topic in (args.left_action_topic, args.right_action_topic)
        )
        # Built once so the 200 Hz tick only writes the stamp and the positions: rclpy re-validates
        # all seven joint names on every `name` assignment, and skips validating `position` only
        # when it is handed an array.array("d").
        self._command_messages = tuple(JointState(name=JOINT_NAMES) for _ in self._command_publishers)

        if args.viz:
            width, height = VIZ_CELL
            cv2.namedWindow(VIZ_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(VIZ_TITLE, width * len(VIZ_ORDER), height)
            self.create_timer(1.0 / VIZ_HZ, self._viz_tick)

        self.get_logger().info(
            f"policy={args.host}:{args.port} execute={args.execute} publish_hz={args.publish_hz} "
            f"policy_hz={args.policy_hz} chunk_steps={args.chunk_steps} replan_chunk_step={args.replan_chunk_step} "
            f"delay_steps={args.delay_steps} max_joint_vel={args.max_joint_vel}rad/s"
        )

    @property
    def exit_requested(self) -> bool:
        return self._exit.is_set()

    def stop(self) -> None:
        """Stop control and exit. The executor holds, so the arm keeps its last command."""
        self._stop.set()
        self._exit.set()

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._publisher_loop, name="openpi-publisher", daemon=True),
            threading.Thread(target=self._controller_loop, name="openpi-controller", daemon=True),
            threading.Thread(target=self._read_keyboard, name="openpi-keyboard", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _read_keyboard(self) -> None:
        """Feed stdin lines to the controller thread as commands: "" start/resume, 'q' pause, 'r' recover."""
        while not self._exit.is_set():
            line = None
            with contextlib.suppress(Exception):
                line = sys.stdin.readline()
            if not line:
                return  # EOF: stdin closed, nothing more to read
            self._commands.put(line.strip().lower())

    def _next_command(self, timeout: float) -> str | None:
        try:
            return self._commands.get(timeout=timeout)
        except queue.Empty:
            return None

    # -- publisher thread --------------------------------------------------------------------------

    def _publisher_loop(self) -> None:
        period = 1.0 / self.args.publish_hz
        next_tick = time.monotonic()
        while not self._stop.is_set():
            command = self._executor.step(time.monotonic())
            if self._stop.is_set():
                break  # the last command sent is the last one computed: no snap back, no zeros
            if command is not None and self.args.execute:
                self._publish(command)
            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep < -0.5 * period:  # big overrun: resync instead of burning down the backlog
                next_tick, sleep = time.monotonic(), 0.0
            if sleep > 0 and self._stop.wait(sleep):
                break

    def _publish(self, command: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        halves = (command[:ARM_DOF], command[ARM_DOF:])
        # publish() serializes before it returns, so the two messages are safe to reuse each tick.
        for publisher, message, part in zip(self._command_publishers, self._command_messages, halves, strict=True):
            message.header.stamp = stamp
            message.position = array.array("d", part)
            publisher.publish(message)

    # -- controller thread -------------------------------------------------------------------------

    def _controller_loop(self) -> None:
        try:
            measured = self._wait_for_state()
            if measured is None:
                return
            self._executor.seed(measured)
            self.get_logger().info(f"seeded command from measured state: {np.round(measured, 4).tolist()}")
            if not self._goto_home(self.args.home_pose):
                return
            while not self._stop.is_set():
                # The one place a connection is established. It happens after homing and on its own
                # thread on purpose: _connect blocks and retries forever, and a down server must
                # never stop the arm from reaching a safe pose or answering the keyboard.
                if self._policy is None:
                    threading.Thread(target=self._connect, name="openpi-connect", daemon=True).start()
                if not self._await_start():
                    break
                self._run_policy()
        except Exception:
            self.get_logger().error("controller thread crashed", exc_info=True)
        finally:
            self._executor.hold()
            self.get_logger().info("controller stopped; holding the last command")

    def _wait_for_state(self) -> np.ndarray | None:
        while not self._stop.is_set():
            measured = self._obs.latest_state()
            if measured is not None:
                return measured[0]
            self.get_logger().warning(
                f"waiting for {self.args.left_state_topic} and {self.args.right_state_topic}",
                throttle_duration_sec=1.0,
            )
            self._stop.wait(0.1)
        return None

    def _goto(self, target: np.ndarray) -> bool:
        """Ramp the executor to `target` and block until it converges or times out."""
        self._executor.goto(target, self.args.home_vel, HOME_GRIPPER_VEL)
        deadline = time.monotonic() + HOME_TIMEOUT
        while not self._stop.wait(0.02):
            error = float(np.abs(self._executor.command() - target).max())
            if error < 1e-3:
                return True
            if time.monotonic() > deadline:
                self.get_logger().error(f"did not converge in {HOME_TIMEOUT}s (error {error:.4f})")
                return False
        return False

    def _goto_home(self, target: np.ndarray) -> bool:
        """Ramp to a home pose. `target` is always derived from --home-pose-*, never from wherever
        the arm happens to be, so startup and recover retract to the same place."""
        self.get_logger().info(f"homing to {np.round(target, 4).tolist()} ({self.args.home_vel} rad/s)")
        done = self._goto(target)
        if done:
            self.get_logger().info("homing complete")
        return done

    def _open_gripper(self) -> None:
        current = self._executor.command()
        if current is None:
            return
        target = current.copy()
        target[GRIP_IDX] = (self.args.gripper_open_left, self.args.gripper_open_right)
        self.get_logger().info(f"opening gripper to {target[GRIP_IDX].tolist()}")
        self._goto(target)

    def _recover(self) -> None:
        """'r': open the grippers, carry them open to the home pose, then close them.

        The gripper dims are pinned open for the traverse because `_goto` ramps all 14 dims at
        once: homing straight to the home pose would close the grippers *while* the arms retract,
        dragging whatever is still held along the path. Always leaves control at the start gate.
        """
        self.get_logger().info("recovering: open gripper -> home -> close gripper")
        self._open_gripper()
        home_open = self.args.home_pose.copy()
        home_open[GRIP_IDX] = (self.args.gripper_open_left, self.args.gripper_open_right)
        if not self._goto_home(home_open):
            return
        self.get_logger().info("closing gripper")
        self._goto(self.args.home_pose)  # the arms are already home; only the gripper dims move

    def _await_start(self) -> bool:
        while self._policy is None and not self._stop.is_set():
            self._stop.wait(0.1)
        if self._stop.is_set():
            return False
        self.get_logger().info("press ENTER to start policy control, or 'r' to recover first")
        while not self._stop.is_set():
            command = self._next_command(0.1)
            if command == "":
                return True
            if command == "r":
                self._recover()
                self.get_logger().info("press ENTER to start policy control")
        return False

    def _connect(self) -> None:
        """Connect, then warm up: the first pi05 call pays seconds of compile cost."""
        while not self._stop.is_set():
            try:
                policy = websocket_client_policy.WebsocketClientPolicy(self.args.host, self.args.port)
                self.get_logger().info(f"connected to policy server: {policy.get_server_metadata()}")
                while not self._stop.is_set():
                    snapshot = self._obs.snapshot(timeout=1.0)
                    if snapshot is None:
                        continue
                    started = time.monotonic()
                    action_chunk(policy.infer(snapshot.observation))
                    self.get_logger().info(f"warm-up inference took {time.monotonic() - started:.2f}s")
                    self._policy = policy
                    return
            except Exception as exc:
                self.get_logger().error(f"connect or warm-up failed: {exc}")
                self._stop.wait(RETRY_DELAY)

    def _run_policy(self) -> None:
        """Drive the inference loop until stopped, paused ('q'), or sent to recover ('r').

        Every exit returns to the caller's start gate, so resuming is always a deliberate ENTER.
        """
        self.get_logger().info("running ('q' pauses, 'r' recovers)")
        args = self.args
        # Replan once --replan-chunk-step steps of the current chunk have played: the publisher
        # keeps interpolating the remaining steps through the whole blocking infer(), so the arm
        # never stalls waiting for the next prediction.
        replan_after = args.replan_chunk_step / args.policy_hz
        next_request = last_chunk = time.monotonic()
        # RTC prefix source: the previous chunk's own (untruncated) prediction and the anchor it was
        # predicted from. Reset on every call, since a pause/recover moves the arm out from under it.
        prev_actions: np.ndarray | None = None
        prev_anchor: float | None = None

        while not self._stop.is_set():
            command = self._next_command(0.0)
            if command in ("q", "r"):
                self._executor.hold()
                self.get_logger().info("paused; holding position")
                if command == "r":
                    self._recover()
                return

            sleep = next_request - time.monotonic()
            if sleep > 0 and self._stop.wait(sleep):
                return

            snapshot = self._obs.snapshot(timeout=OBS_RETRY)
            if snapshot is None:
                gap = time.monotonic() - last_chunk
                if gap > OBS_STOP_TIMEOUT:
                    self.get_logger().error(f"no usable observation for {gap:.1f}s; stopping")
                    self._stop.set()
                elif gap > OBS_TIMEOUT:
                    self._executor.hold()
                    self.get_logger().error(
                        f"no usable observation for {gap:.1f}s; holding position", throttle_duration_sec=1.0
                    )
                continue

            action_prefix, delay = rtc_action_prefix(
                prev_actions, prev_anchor, snapshot.anchor, args.policy_hz, args.delay_steps
            )
            try:
                started = time.monotonic()
                result = self._policy.infer(snapshot.observation, action_prefix=action_prefix, delay=delay or None)
                actions_full = action_chunk(result)
                actions = actions_full[: args.chunk_steps]
                round_trip = time.monotonic() - started
            except ValueError as exc:
                self.get_logger().error(f"bad action chunk: {exc}")
                continue
            except Exception as exc:
                # Drop the connection and fall back to the start gate rather than reconnecting
                # here: _connect blocks and retries forever, and running it inline would park this
                # loop -- and with it the keyboard and the observation watchdog -- for its duration.
                self.get_logger().error(f"inference failed: {exc}")
                self._executor.hold()
                self._policy = None
                return

            # The untruncated chunk is what the RTC prefix for the *next* request resamples from --
            # keeping it regardless of whether the executor accepts this chunk for playback below.
            prev_actions, prev_anchor = actions_full, snapshot.anchor

            last_chunk = time.monotonic()
            rejected = self._executor.submit(actions, snapshot.anchor, last_chunk)
            if rejected:
                self.get_logger().error(f"rejected chunk: {rejected}")
            # Both clocks run off the observation instant: playback is anchored there, so the
            # request cadence must be too, or latency creeps in and eats the remaining steps.
            next_request = snapshot.anchor + replan_after

            # implied is the speed this chunk asks for: if it exceeds the limit the clamp will bind
            # and the arm will lag, which usually means --policy-hz or --max-joint-vel is wrong.
            implied = np.abs(np.diff(actions[:, ARM_IDX], axis=0)).max() * args.policy_hz if len(actions) > 1 else 0.0
            grippers = snapshot.observation["state"][GRIP_IDX]
            self.get_logger().info(
                f"chunk: {len(actions)} steps, rtt {round_trip * 1e3:.0f}ms, "
                f"align +{(last_chunk - snapshot.anchor) * args.policy_hz:.1f} steps, "
                f"prefix {delay}/{args.delay_steps} steps, "
                f"obs age {time.monotonic() - snapshot.captured:.3f}s, skew {snapshot.skew:.3f}s, "
                f"implied {implied:.2f} rad/s, clamp {self._executor.saturation():.0%}, "
                f"gripper {grippers[0]:.3f}/{grippers[1]:.3f}"
            )
            if implied > args.max_joint_vel:
                self.get_logger().warning(
                    f"velocity limit binds: chunk needs {implied:.2f} rad/s, limit {args.max_joint_vel:.2f} rad/s"
                )

    # -- viz (main thread, where the GUI belongs) ---------------------------------------------------

    def _viz_tick(self) -> None:
        frames = self._obs.raw_frames()
        width, height = VIZ_CELL
        cells = []
        for key in VIZ_ORDER:
            cell = self._viz_cell(frames.get(key))
            cell = np.zeros((height, width, 3), dtype=np.uint8) if cell is None else cv2.resize(cell, (width, height))
            cv2.putText(cell, VIZ_LABELS[key], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cells.append(cell)
        mosaic = np.hstack(cells)

        cv2.imshow(VIZ_TITLE, mosaic)
        cv2.waitKey(1)

    def _viz_cell(self, message: Image | None) -> np.ndarray | None:
        if message is None:
            return None
        try:
            return self._viz_bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError:
            return None

    def shutdown(self) -> None:
        self.stop()
        with contextlib.suppress(Exception):
            if self._policy is not None:
                self._policy.close()  # unblocks a wedged infer(); see the client's ping_timeout note
        for thread in self._threads:
            if thread.name != "openpi-keyboard":  # blocked on stdin; a daemon, so let it be reaped
                thread.join(timeout=2.0)
        if self.args.viz:
            cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost", help="OpenPI policy server host")
    parser.add_argument("--port", type=int, default=8010, help="OpenPI policy server port")
    parser.add_argument("--prompt", help="Sent with every observation; the server requires it")
    parser.add_argument("--publish-hz", type=float, default=200.0, help="Command output rate")
    parser.add_argument("--policy-hz", type=float, default=30.0, help="Chunk STEP rate; must equal the dataset fps")
    parser.add_argument(
        "--chunk-steps", type=int, default=30, help="Steps kept from each 50-step chunk, counted from the observation"
    )
    parser.add_argument(
        "--replan-chunk-step",
        type=int,
        default=20,
        help="Request the next chunk after this many steps have played; the rest cover the round trip",
    )
    parser.add_argument(
        "--delay-steps",
        type=int,
        default=4,
        help="RTC only: steps of the previous chunk's own prediction to send back as action_prefix ground truth",
    )
    parser.add_argument("--max-joint-vel", type=float, default=1.0, help="Arm joint velocity limit (rad/s)")
    parser.add_argument("--max-gripper-vel", type=float, default=0.5, help="Gripper velocity limit (units/s)")
    home_default = [0.0] * ARM_DOF
    # home_default[1] = 0.1
    # home_default[2] = -0.1
    parser.add_argument(
        "--home-pose-left", type=float, nargs=ARM_DOF, default=home_default, help="Left home pose, ROS/dataset units"
    )
    parser.add_argument(
        "--home-pose-right", type=float, nargs=ARM_DOF, default=home_default, help="Right home pose, ROS/dataset units"
    )
    parser.add_argument("--home-vel", type=float, default=0.25, help="Home ramp arm velocity (rad/s)")
    parser.add_argument(
        "--gripper-open-left", type=float, default=0.1, help="Left gripper 'open' value (wire units) for 'r' recover"
    )
    parser.add_argument(
        "--gripper-open-right",
        type=float,
        default=0.1,
        help="Right gripper 'open' value (wire units) for 'r' recover",
    )
    parser.add_argument("--sync-tolerance", type=float, default=0.15, help="Max inter-topic capture skew (s)")
    parser.add_argument("--max-observation-age", type=float, default=0.30, help="Reject observations older than this")
    parser.add_argument("--left-state-topic", default="/puppet/joint_left")
    parser.add_argument("--right-state-topic", default="/puppet/joint_right")
    parser.add_argument("--front-image-topic", default="/camera_f/color/image_raw")
    parser.add_argument("--left-image-topic", default="/camera_l/color/image_rect_raw")
    parser.add_argument("--right-image-topic", default="/camera_r/color/image_rect_raw")
    parser.add_argument("--left-action-topic", default="/joint_left_states")
    parser.add_argument("--right-action-topic", default="/joint_right_states")
    parser.add_argument("--execute", action="store_true", help="Publish actions; without this the node is dry-run")
    parser.add_argument("--viz", action="store_true", help="Show camera windows")
    args = parser.parse_args()

    if min(args.publish_hz, args.policy_hz) <= 0:
        parser.error("--publish-hz and --policy-hz must be positive")
    if args.publish_hz < args.policy_hz:
        parser.error(f"--publish-hz {args.publish_hz} < --policy-hz {args.policy_hz}: that is decimation")
    if not 1 <= args.chunk_steps <= 50:
        parser.error("--chunk-steps must be in [1, 50]; the server returns a 50-step chunk")
    if not 1 <= args.replan_chunk_step <= args.chunk_steps:
        parser.error(f"--replan-chunk-step must be in [1, --chunk-steps={args.chunk_steps}]")
    if args.delay_steps < 0:
        parser.error("--delay-steps must be non-negative")
    if min(args.max_joint_vel, args.max_gripper_vel, args.home_vel) <= 0:
        parser.error("velocity limits must be positive; a zero limit would freeze the command")
    if args.sync_tolerance < 0 or args.max_observation_age <= 0:
        parser.error("--sync-tolerance must be non-negative and --max-observation-age positive")
    if not args.prompt.strip():
        parser.error("--prompt must be non-empty; the server rejects a request without one")

    args.home_pose = np.asarray(args.home_pose_left + args.home_pose_right, dtype=np.float64)
    return args


def main(node_cls: type[AgileXInferenceNode] = AgileXInferenceNode, args: argparse.Namespace | None = None) -> None:
    """Run `node_cls` until it exits. agilex_recorder.py passes its own subclass and flags."""
    args = parse_args() if args is None else args

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = node_cls(args)
    interrupts = 0

    def handle_sigint(_signum: int, _frame: Any) -> None:
        # First Ctrl-C stops control and holds the last command; a second one is the escape hatch
        # for a wedged websocket recv() (the client sets ping_timeout=None, so a dead server
        # blocks forever and shutdown() could hang).
        nonlocal interrupts
        interrupts += 1
        if interrupts > 1:
            os._exit(1)
        node.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    node.start()
    try:
        while not node.exit_requested and rclpy.ok():
            executor.spin_once(timeout_sec=0.05)
    finally:
        node.shutdown()
        with contextlib.suppress(Exception):
            executor.remove_node(node)
        with contextlib.suppress(Exception):
            node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
