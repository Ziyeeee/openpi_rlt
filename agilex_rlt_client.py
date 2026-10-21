#!/usr/bin/python3
"""Run RLT Stage C on the AgileX: collect rollouts of the online RL policy and label them.

This is the robot half of Algorithm 1, and it is deliberately thin. The server owns the frozen
VLA, the RL token, the handover head, the actor/critic, the replay buffer and the training loop;
it sees every observation it is queried on and every action it returns, which is already a
transition stream. So the only things this file has to supply are the two the server cannot
observe: **where an episode begins**, and **whether it succeeded**.

Nothing about control changes. `agilex_recorder.RolloutNode` is subclassed, which already brings
the sampler thread, the camera tap and the episode keys, and through it `AgileXInferenceNode`
keeps owning homing, the start gate, the publisher thread, RTC, replanning and the velocity clamp.
Four things are added and nothing is overridden that shapes motion:

    _RLPolicyProxy      wraps the websocket client so each request carries the episode's RL state
                        and each response's `rl` block gets read back
    RL row buffering    once the handover latches, the existing 30 Hz sampler also stores an
                        observation every `subsample_stride` ticks, plus every executed command
    control connection  a second websocket to the same server for begin / end / abort, and for
                        shipping the buffered rows
    'g' and 'v'         operator override of the handover decision

Why the client buffers at all. Appendix B stores a transition every 2 control steps ("<x_0, a_0:C>,
<x_2, a_2:C+2>, ..."), but forming one needs a VLA prefix forward for `z_rl` and the denoiser for
`a_ref` -- there is no way to do that at 15 Hz inside a 30 Hz control loop. So the raw material is
buffered here through the critical phase and shipped at episode close, when the arm is held and the
operator is resetting the scene. The server then runs one batched Stage A.5 pass over it.

What that buys, beyond density: the actions stored are this node's own command stream, i.e. what
the arm *executed* after interpolation and the velocity clamp, rather than the chunk the server
emitted and only partly ran. And the RL step length `C` stops being tied to `--replan-chunk-step`,
so the controller replans at whatever rate suits control.

Keyboard (one terminal, the recorder's keys plus two):

    ENTER  start policy control; opens an episode, or resumes the one 'q' paused
    s      success: hold, save the episode, and label the RL clip a success (+1 reward)
    d      failure: hold, discard the episode, and label the RL clip a failure (terminal, no reward)
    q      pause: hold and pause recording, leaving the episode open for ENTER
    r      recover (open grippers -> home -> close); discards and aborts the episode
    g      force the RL policy to take over now, without waiting for the handover head
    v      VLA only for the rest of this episode; the actor is never queried
    Ctrl-C exits; an episode still open is saved, and its RL clip is aborted rather than mislabeled

Why 's' and 'd' both matter. The sparse label is the entire reward signal (Sec. V), and
`docs/rlt_wip.md` measures that held-out discrimination scales with the number of **failure**
clips and is flat in successes. A failed rollout is the more valuable of the two -- press 'd', do
not press 'r'.

Why the second connection. `_finish_episode` runs from the controller thread after the executor is
already held, between inference calls, and at Ctrl-C there is no next request at all. Buffering the
label until the next `infer()` would lose exactly the episodes that end a session.

The sampling grid -- stride, chunk length, message size, row cap -- comes from the server's
handshake frame, so there are no flags here to keep in sync with it.

Usage (the control flags are `agilex_inference.py`'s own defaults; nothing about the replan rate
is dictated by RL any more):
.venv/bin/python agilex_rlt_client.py \
    --datasetDir ~/data/my_task_online \
    --prompt "Pull out the charging gun and plug it back in" \
    --execute --host <POLICY_SERVER_IP> --port 8010 --viz
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy

import agilex_inference as inference
import agilex_recorder as recorder

LOGGER = logging.getLogger("agilex_rlt_client")

# Keys that only change which policy drives the arm, so the parent's control loop must not see
# them: `_run_policy` acts on 'q'/'r' and `_await_start` acts on '', and both ignore None.
OVERRIDE_KEYS = {"g": "force_rl", "v": "force_vla"}


class _RLPolicyProxy:
    """`WebsocketClientPolicy` plus the RL request/response fields.

    The wire protocol carries one arbitrary dict each way, and the server already pops
    `action_prefix`/`delay` out of the observation for RTC -- an `rl` key rides along the same way,
    so neither the protocol nor `agilex_inference._run_policy` needs to change.
    """

    def __init__(self, inner: websocket_client_policy.WebsocketClientPolicy, node: RLTClientNode) -> None:
        self.inner = inner
        self._node = node

    def infer(self, obs: dict, *, action_prefix: Any = None, delay: int | None = None) -> dict:
        payload = dict(obs)
        payload["rl"] = {"control": self._node.control}
        result = self.inner.infer(payload, action_prefix=action_prefix, delay=delay)
        self._node.on_rl_response(result.pop("rl", None))
        return result

    def get_server_metadata(self) -> dict:
        return self.inner.get_server_metadata()

    def close(self) -> None:
        self.inner.close()


class RLTClientNode(recorder.RolloutNode):
    """RolloutNode plus episode signalling to the Stage C server."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.control = "auto"
        self.latched = False
        self._control_lock = threading.Lock()
        self._control_client: websocket_client_policy.WebsocketClientPolicy | None = None
        # Filled from the server's handshake frame on the first control message, so the sampling
        # grid has exactly one definition and it lives on the server.
        self._grid: dict[str, int] = {}
        # The critical phase, buffered. `_exec_commands` is every 30 Hz executed command since the
        # latch; `_rows` is (index into it, observation, state) every `subsample_stride` ticks.
        # Not `_commands`: that name is taken by `AgileXInferenceNode`'s stdin key queue.
        self._exec_commands: list[np.ndarray] = []
        self._rows: list[tuple[int, dict, np.ndarray]] = []
        self._rl_lock = threading.Lock()
        self._rl_dropped = 0
        self._rl_truncated = False
        self.get_logger().info(
            f"RLT online client -> {args.host}:{args.port}; 'g' forces the RL policy, 'v' forces VLA only"
        )

    # -- control connection ----------------------------------------------------------------------

    def _connect_control(self) -> websocket_client_policy.WebsocketClientPolicy:
        if self._control_client is None:
            self._control_client = websocket_client_policy.WebsocketClientPolicy(self.args.host, self.args.port)
            self._grid = dict(self._control_client.get_server_metadata().get("rl", {}))
            self.get_logger().info(f"rl grid from the server: {self._grid}")
        return self._control_client

    def _send_control(self, event: str, result: str | None = None) -> None:
        """Tell the server about an episode boundary. Best effort: a lost label is not worth
        stopping the arm for, and the episode is on disk either way."""
        with self._control_lock:
            try:
                client = self._connect_control()
                if event in ("end", "abort"):
                    self._ship_rows(client)
                message = {"event": event, "control": self.control}
                if result is not None:
                    message["result"] = result
                ack = client.infer({"rl_control": message})
                self.get_logger().info(f"rl {event}{'/' + result if result else ''}: {ack}")
            except Exception as exc:
                self.get_logger().error(f"rl control {event!r} failed: {exc}")
                self._control_client = None
            finally:
                self._reset_rows()

    def _ship_rows(self, client: websocket_client_policy.WebsocketClientPolicy) -> None:
        """Send the critical phase's subsampled transitions, in server-batch-sized messages.

        This is the whole reason the client buffers anything. Appendix B stores a transition every
        `subsample_stride` control steps, but forming one needs a VLA prefix forward for `z_rl` and
        the denoiser for `a_ref` -- impossible at 15 Hz inside the control loop. Deferring it to
        here costs nothing: the arm is already held and the operator is resetting the scene.

        The actions sent are this node's own 30 Hz command stream, i.e. what the arm *executed*
        after `ChunkExecutor`'s interpolation and velocity clamp. That is the correct
        behaviour-cloning target, and it is why the server does not have to guess from the chunk it
        emitted.
        """
        with self._rl_lock:
            rows, commands = list(self._rows), list(self._exec_commands)
        if not rows:
            return
        chunk = int(self._grid.get("chunk_size", 16))
        batch = int(self._grid.get("ingest_batch_size", 32))
        if not commands:
            return
        # Pad the command stream so the last rows still have a full chunk, repeating the final
        # command. This is what LeRobot's `delta_timestamps` does at an episode's end, so an online
        # clip's tail matches an offline one's.
        padded = commands + [commands[-1]] * chunk

        started = time.monotonic()
        sent = 0
        for lo in range(0, len(rows), batch):
            part = rows[lo : lo + batch]
            message = {
                "frames": np.asarray([index for index, _, _ in part], dtype=np.int64),
                "state": np.stack([state for _, _, state in part]).astype(np.float32),
                "actions": np.stack([np.stack(padded[i : i + chunk]) for i, _, _ in part]).astype(np.float32),
                "images": {
                    name: np.stack([obs["images"][name] for _, obs, _ in part]) for name in part[0][1]["images"]
                },
                "prompt": self.args.prompt,
            }
            ack = client.infer({"rl_batch": message})
            sent += int(ack.get("ingested", 0))
        self.get_logger().info(
            f"shipped {sent} transitions ({len(rows)} sampled, {self._rl_dropped} dropped"
            f"{', truncated' if self._rl_truncated else ''}) in {time.monotonic() - started:.1f}s"
        )

    def _reset_rows(self) -> None:
        with self._rl_lock:
            self._exec_commands.clear()
            self._rows.clear()
            self._rl_dropped = 0
            self._rl_truncated = False
        self.latched = False

    # -- controller thread -----------------------------------------------------------------------

    def _next_command(self, timeout: float) -> str | None:
        """Close the episode *and* label its RL clip, which is one decision made from one key.

        This deliberately calls the grandparent rather than chaining through
        `RolloutNode._next_command`: that method also closes the episode, but by the time it
        returns, all that survives of the key is `save`, which cannot separate 'd' from 'r'. Those
        two must not produce the same label. 'd' is a task failure -- terminal, no reward -- and
        `docs/rlt_wip.md` measures that failures are the scarce class carrying all of the critic's
        held-out discrimination. 'r' is a scene reset, so its clip is aborted rather than blamed on
        the policy; a fabricated failure is the one label that would actively teach the wrong thing.
        """
        command = inference.AgileXInferenceNode._next_command(self, timeout)  # noqa: SLF001
        if command in OVERRIDE_KEYS:
            self.control = OVERRIDE_KEYS[command]
            self.get_logger().info(f"operator override: {self.control}")
            self._send_control("control")
            return None
        if command in ("s", "d", "r"):
            self._executor.hold()
            self._finish_episode(save=command == "s")
            if command == "r":
                self._send_control("abort")
            else:
                self._send_control("end", "success" if command == "s" else "fail")
        return "q" if command in ("s", "d") else command

    def _run_policy(self) -> None:
        # `_connect` assigns a bare WebsocketClientPolicy, and `_run_policy` drops it back to None
        # on a failed inference, so re-wrap on every entry rather than once at startup.
        if self._policy is not None and not isinstance(self._policy, _RLPolicyProxy):
            self._policy = _RLPolicyProxy(self._policy, self)
        super()._run_policy()

    def _open_episode(self) -> None:
        # RolloutNode resumes an already-open episode after 'q' without reopening it; only tell the
        # server when a genuinely new one starts, or its buffered rows would be thrown away.
        fresh = self._episode is None
        super()._open_episode()
        if fresh:
            self.control = "auto"
            self._send_control("begin")

    # -- sampler thread --------------------------------------------------------------------------

    def _sample_once(self) -> None:
        """The recorder's 30 Hz row, plus the RL buffering once the handover has latched.

        Piggy-backing on the existing sampler rather than adding a thread keeps one clock: the RL
        rows are a strict subsample of the recorded rows, so `frames` indexes both.

        `snapshot()` re-runs the decode and resize the control loop already does, ~7% of a core at
        stride 2. That is affordable in a way the camera tap's raw-topic experiment was not (see
        `agilex_recorder`'s module docstring): there is no second subscription and no re-encode
        here, only conversion of a message `ObservationSource` has already received.
        """
        super()._sample_once()
        if not self.latched:
            return
        with self._rl_lock:
            index = len(self._exec_commands)
            command = self._executor.command()
            if command is None:
                return
            self._exec_commands.append(np.asarray(command, dtype=np.float64))
            stride = int(self._grid.get("subsample_stride", 2))
            if index % stride or self._rl_truncated:
                return
            if len(self._rows) >= int(self._grid.get("max_rows_per_episode", 600)):
                # Announced, not silent: a truncated clip would otherwise look like a short
                # critical phase and its terminal reward would land in the wrong place.
                self._rl_truncated = True
                self.get_logger().error(
                    f"critical phase exceeded max_rows_per_episode ({len(self._rows)}); "
                    "no further transitions will be stored for this episode"
                )
                return

        snapshot = self._obs.snapshot(timeout=0.0)
        if snapshot is None:
            # Rejected for skew or staleness by the same checks the control loop uses. The row is
            # simply absent; `next_offsets` works off the frame numbers, so the gap costs one
            # transition and shifts nothing.
            with self._rl_lock:
                self._rl_dropped += 1
            return
        with self._rl_lock:
            self._rows.append((index, snapshot.observation, snapshot.observation["state"]))

    def on_rl_response(self, block: dict | None) -> None:
        if not block:
            return
        # The latch is the server's decision (the Stage A.6 head, or an operator override) and it
        # brackets the critical phase: it rises when the RL policy takes the arm and falls when the
        # head calls `post`. Buffering follows it on both edges, so the stored clip is exactly that
        # span -- the same one `clips_by_tag.json` carries offline. The buffered rows are kept
        # across the falling edge and shipped at episode close; only appending stops.
        latched = bool(block.get("latched"))
        if latched != self.latched:
            self.latched = latched
            self.get_logger().info(
                "handover latched; buffering RL transitions"
                if latched
                else f"handover released at `post`; {len(self._rows)} rows buffered, VLA has the arm back"
            )
        self.get_logger().info(
            "rl: phase {phase} mode {mode} latched {latched} "
            "q {q:.3f} |a-a_ref| {action_delta:.3f} learner {learner_step} buffer {buffer}".format(**block)
            + f" rows {len(self._rows)}",
            throttle_duration_sec=1.0,
        )

    def shutdown(self) -> None:
        # Ctrl-C keeps an open episode on disk (RolloutNode's sampler saves it) but must not label
        # its RL clip: "the operator stopped the program" is not "the task succeeded", and a
        # fabricated `+1` would be trained on. Unconditional because the sampler has already
        # cleared `self._episode` by the time this runs; the server no-ops if nothing is open.
        super().shutdown()
        self._send_control("abort")
        with contextlib.suppress(Exception):
            if self._control_client is not None:
                self._control_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every agilex_recorder.py and agilex_inference.py flag is also accepted.",
    )
    parser.add_argument("--datasetDir", required=True, help="Episodes are written as <datasetDir>/episodeN/")
    parser.add_argument("--record-hz", type=float, default=30.0, help="Row rate; must equal the training fps")
    args, rest = parser.parse_known_args()
    if args.record_hz <= 0:
        parser.error("--record-hz must be positive")

    sys.argv = [sys.argv[0], *rest]
    control = inference.parse_args()
    for key, value in vars(args).items():
        setattr(control, key, value)
    Path(control.datasetDir).mkdir(parents=True, exist_ok=True)
    return control


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    inference.main(RLTClientNode, parse_args())
