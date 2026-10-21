"""Drive a Stage C server without a robot, and check what it did.

`agilex_rlt_client.py` needs ROS, two arms and three cameras. This speaks the same wire protocol
with synthetic frames, so the whole server -- the two-checkpoint load, the handover latch, the
actor and its safety filter, the replay buffer's reward placement, the learner's UTD throttle --
can be exercised before anything is allowed to move a real arm.

It is a validation tool first and a smoke test second: run it after any change to the Stage C
files, and once against the machine that will actually serve the robot.

The observation it sends is exactly `ObservationSource.snapshot`'s
(`agilex_inference.py`): CHW uint8 images under the three `cam_*` keys, a raw 14-dim state, and a
prompt, plus the RTC `action_prefix`/`delay` a real client resamples from its previous chunk.

Usage:
    python scripts/mock_rlt_client.py --episodes=4 --queries_per_episode=10 --port=8010
"""

import dataclasses
import logging
import math
import time

import numpy as np
from openpi_client import websocket_client_policy
import tyro

logger = logging.getLogger("mock_rlt_client")

ACTION_DIM = 14
IMAGE_SIZE = 224
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8010
    prompt: str = "Pull out the charging gun and plug it back in"

    episodes: int = 4
    queries_per_episode: int = 10
    # Force the handover on the first query. The handover head is trained on real `z_rl`, and
    # random images do not reliably land in `critical`, so leaving this off would test the latch
    # rather than everything downstream of it.
    force_rl: bool = True
    # Set when the server was started with --collect_only. The assertions invert: the actor must
    # never be engaged, so no clip is recorded and every chunk is the VLA's own.
    collect_only: bool = False
    # RTC prefix length, matching `agilex_inference.py --delay-steps`.
    delay_steps: int = 4
    # Pace requests like the robot does, so the server's measured query stride is realistic.
    query_period: float = 0.5
    seed: int = 0


def observation(rng: np.random.Generator, prompt: str) -> dict:
    return {
        "images": {name: rng.integers(0, 256, (3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) for name in CAMERAS},
        "state": rng.normal(0.0, 0.1, ACTION_DIM).astype(np.float32),
        "prompt": prompt,
    }


def main(args: Args) -> None:
    rng = np.random.default_rng(args.seed)
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    control = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info("server metadata: %s", policy.get_server_metadata())

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:  # noqa: FBT001
        (logger.info if condition else logger.error)("%s %s", "PASS" if condition else "FAIL", message)
        if not condition:
            failures.append(message)

    # The sampling grid is the server's, delivered in the handshake frame -- same as the real client.
    grid = policy.get_server_metadata()["rl"]
    check(
        grid["chunk_size"] % grid["subsample_stride"] == 0,
        f"chunk_size {grid['chunk_size']} is a multiple of subsample_stride {grid['subsample_stride']}",
    )

    before = control.infer({"rl_control": {"event": "control", "control": "auto"}})
    check(before.get("ok") is True, "control channel acks")
    check(before["buffer_transitions"] > 0, f"buffer is preloaded ({before['buffer_transitions']:.0f} transitions)")
    expected_clips = before["buffer_clips"]

    # A fixed observation, sent before and after collection with no episode open (so it is never
    # stored). Q on it must move: that is the learner actually changing the networks the robot is
    # being served from, not just incrementing a counter.
    probe = observation(np.random.default_rng(12345), args.prompt)
    q_before = policy.infer(probe)["rl"]["q"]

    prev_chunk: np.ndarray | None = None
    for episode in range(args.episodes):
        # Alternate the outcome: a run of successes alone would leave the critic with no negative
        # to separate them from, which is the axis `docs/rlt_wip.md` measures as the binding one.
        success = episode % 2 == 0
        control.infer({"rl_control": {"event": "begin", "control": "force_rl" if args.force_rl else "auto"}})
        prev_chunk = None
        blocks = []
        for query in range(args.queries_per_episode):
            started = time.monotonic()
            obs = observation(rng, args.prompt)
            kwargs = {}
            if prev_chunk is not None and args.delay_steps:
                # What `agilex_inference.rtc_action_prefix` builds: the previous chunk's own
                # prediction, front-loaded into the first `delay` rows of a full-horizon array.
                prefix = np.zeros_like(prev_chunk)
                prefix[: args.delay_steps] = prev_chunk[: args.delay_steps]
                kwargs = {"action_prefix": prefix, "delay": args.delay_steps}
            result = policy.infer(obs, **kwargs)

            actions = np.asarray(result["actions"])
            if query == 0 and episode == 0:
                check(actions.ndim == 2 and actions.shape[1] == ACTION_DIM, f"actions are [H, 14], got {actions.shape}")
                check(np.isfinite(actions).all(), "actions are finite")
                check("rl" in result, "response carries an rl block")
            blocks.append(result["rl"])
            prev_chunk = actions
            time.sleep(max(0.0, args.query_period - (time.monotonic() - started)))

        # Ship the critical phase, exactly as `agilex_rlt_client._ship_rows` does: rows every
        # `subsample_stride` control steps, actions taken from a 30 Hz executed-command stream.
        n_frames = args.queries_per_episode * grid["chunk_size"]
        commands = rng.normal(0.0, 0.1, (n_frames + grid["chunk_size"], ACTION_DIM)).astype(np.float32)
        frames = np.arange(0, n_frames, grid["subsample_stride"], dtype=np.int64)
        shipped = 0
        if not args.collect_only:
            for lo in range(0, len(frames), grid["ingest_batch_size"]):
                part = frames[lo : lo + grid["ingest_batch_size"]]
                ack = control.infer(
                    {
                        "rl_batch": {
                            "frames": part,
                            "state": np.stack([commands[f] for f in part]),
                            "actions": np.stack([commands[f : f + grid["chunk_size"]] for f in part]),
                            "images": {
                                name: np.stack(
                                    [rng.integers(0, 256, (3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) for _ in part]
                                )
                                for name in CAMERAS
                            },
                            "prompt": args.prompt,
                        }
                    }
                )
                shipped += ack["ingested"]
            check(shipped == len(frames), f"all {len(frames)} subsampled rows ingested (got {shipped})")

        phases = {b["phase"] for b in blocks}
        modes = {b["mode"] for b in blocks}
        logger.info(
            "episode %d (%s): phases %s, modes %s, %d rows shipped, last q %.3f, |a-a_ref| max %.3f",
            episode,
            "success" if success else "fail",
            sorted(phases),
            sorted(modes),
            shipped,
            blocks[-1]["q"],
            max(b["action_delta"] for b in blocks),
        )
        if args.collect_only:
            # Even with the handover forced, --collect_only must keep the actor out of the loop:
            # this is the paper's warmup collection phase, and the chunk has to be the unmodified
            # VLA reference or the rollouts are not warmup data at all.
            check(modes == {"vla"}, f"--collect_only never engages the actor (saw {sorted(modes)})")
            check(
                max(b["action_delta"] for b in blocks) == 0.0,
                "--collect_only returns the VLA chunk untouched (|a - a_ref| == 0)",
            )
        elif args.force_rl:
            check(modes <= {"rl", "warmup"}, f"forced episodes never fall back to the VLA (saw {sorted(modes)})")
            check(all(b["latched"] for b in blocks), "the client is told to buffer from the first query")
        ack = control.infer(
            {"rl_control": {"event": "end", "result": "success" if success else "fail", "control": "auto"}}
        )
        expected_clips += 0 if args.collect_only else 1
        check(
            ack["buffer_clips"] == expected_clips,
            f"episode {episode} admitted {'no' if args.collect_only else 'exactly one'} clip "
            f"(buffer now {ack['buffer_clips']:.0f})",
        )

    final = control.infer({"rl_control": {"event": "control", "control": "auto"}})
    if not args.collect_only:
        # Density is the point of the subsampling: one row every `subsample_stride` control steps,
        # not one per policy query.
        per_episode = math.ceil(args.queries_per_episode * grid["chunk_size"] / grid["subsample_stride"])
        expected_rows = args.episodes * per_episode
        check(
            final["env_steps"] == expected_rows,
            f"env_steps counted {expected_rows} subsampled rows "
            f"({per_episode}/episode = {grid['chunk_size'] // grid['subsample_stride']}x one-per-query)",
        )
        check(
            abs(final["online_success_rate"] - 0.5) < 1e-6,
            f"success labels round-tripped (rate {final['online_success_rate']:.2f}, expected 0.50)",
        )

        # The learner must have moved. It is throttled to `updates_per_env_step * env_steps`, so
        # give it a moment rather than expecting the count current the instant collection stops.
        time.sleep(5.0)
        after = control.infer({"rl_control": {"event": "control", "control": "auto"}})
        check(after["learner_step"] > 0, f"learner is training (step {after['learner_step']})")

        q_after = policy.infer(probe)["rl"]["q"]
        check(
            q_before != q_after,
            f"trained parameters reached the served policy (Q on a fixed probe {q_before:.4f} -> {q_after:.4f})",
        )
    else:
        check(final["env_steps"] == 0, f"--collect_only collects no transitions (env_steps {final['env_steps']:.0f})")
        check(final["learner_step"] == 0, f"--collect_only leaves the learner idle (step {final['learner_step']})")

    if failures:
        logger.error("%d check(s) failed:\n  %s", len(failures), "\n  ".join(failures))
        raise SystemExit(1)
    logger.info("all checks passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
