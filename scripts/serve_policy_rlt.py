"""RLT Stage C: serve the online RL policy and train it from the rollouts it produces.

This is Algorithm 1's loop with the robot on the other end of a websocket. One process holds all
four pieces, because they all have to see the same `z_rl`:

    frozen Stage A VLA + RL token   one prefix forward per query gives a_ref, z_rl and the
    + Stage A.6 handover head       pre/critical/post logits (`sample_actions_and_handover`)
    OnlineRLTPolicy                 lines 7-12: choose the action, store the transition
    ReplayBuffer                    B, pre-filled with the offline clips (the N_warm warmup data)
    OnlineLearner                   lines 13-18, on its own thread at an update-to-data ratio of 5

The robot client (`agilex_rlt_client.py`) only executes chunks and reports two things this process
cannot observe: where an episode begins, and whether it succeeded.

`--collect_only` reproduces what this script used to do unconditionally -- serve the unmodified VLA
reference policy and never engage the actor. That is Algorithm 1's warmup collection phase, and it
is also how to watch the handover head's predictions on the robot before trusting them.

Usage:
    python scripts/serve_policy_rlt.py rlt_pi05_online --exp_name=run0 --overwrite \
        --ac_params_path=checkpoints/rlt_pi05_ac_c16/c16/49999/params
"""

import dataclasses
import logging
import socket

import jax
import wandb

from openpi.policies import policy_config as _policy_config
from openpi.policies import rlt_online_policy as _rlt_policy
from openpi.serving import websocket_policy_server
from openpi.shared import normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.config_rlt as _config_rlt
import openpi.training.config_rlt_online as _config
import openpi.training.replay_buffer_rlt as _replay
import openpi.training.rlt_online_learner as _learner
import openpi.training.sharding as sharding


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool) -> None:
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if resuming and (ckpt_dir / "wandb_id.txt").exists():
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name, mode="offline")
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name, mode="offline")
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def build_policy(config: _config.TrainConfig, buffer, *, asset_path: str | None, default_prompt: str | None):
    """Load Stage A + the handover head, and wire them to the transform chain and the buffer."""
    train_config = _config_rlt.get_config(config.stage_a)
    # The head is only built when RLTConfig.handover is set; with it left None the parameter tree
    # has no `handover` key to graft onto and `sample_actions_and_handover` refuses to run.
    model_config = dataclasses.replace(train_config.model, handover=config.handover)
    model = _rlt_policy.load_stage_a_with_handover(model_config, config.stage_a_dir, config.handover_dir)

    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    if asset_path is not None:
        norm_stats = _normalize.load(asset_path)
        logging.info("Loaded customized norm stats from %s", asset_path)
    else:
        # The checkpoint's own copy, so the policy normalizes exactly as the run that produced
        # every `z_rl` in the preloaded buffer did.
        norm_stats = _checkpoints.load_norm_stats(f"{config.stage_a_dir}/assets", data_config.asset_id)
    # The Stage A data config's own `default_prompt` is already inside `model_transforms`, so this
    # argument is purely the command-line override.
    inputs, outputs = _policy_config.make_transforms(data_config, norm_stats, default_prompt=default_prompt)

    actor_critic = config.ac.model.create(jax.random.key(config.seed))
    return _rlt_policy.OnlineRLTPolicy(
        model,
        model_config,
        actor_critic,
        buffer,
        transforms=inputs,
        output_transforms=outputs,
        chunk_size=config.chunk_size,
        ingest_batch_size=config.ingest_batch_size,
        reference_delay=config.reference_delay,
        num_denoise_steps=config.num_denoise_steps,
        handover_patience=config.handover_patience,
        warmup_transitions=config.warmup_transitions,
        max_action_delta=config.max_action_delta,
        deterministic=config.deterministic,
        collect_only=config.collect_only,
        policy_hz=config.policy_hz,
        # The handshake frame is how the client learns the sampling grid, so the stride, the chunk
        # length and the message size live in exactly one place. `agilex_rlt_client.py` needs no
        # flags of its own for them, and they cannot drift out of sync with the server.
        metadata={
            # `policy_metadata` is None unless a config sets it (the agilex ones do not).
            **(train_config.policy_metadata or {}),
            "rl": {
                "chunk_size": config.chunk_size,
                "subsample_stride": config.subsample_stride,
                "ingest_batch_size": config.ingest_batch_size,
                "max_rows_per_episode": config.max_rows_per_episode,
                "collect_only": config.collect_only,
            },
        },
    )


def main(config: _config.TrainConfig) -> None:
    sharding.init_distributed()
    logging.info("Stage C server: %s / %s", config.name, config.exp_name)

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled and jax.process_index() == 0)

    buffer = _replay.ReplayBuffer(config.chunk_size, capacity=config.buffer_capacity)
    buffer.preload(config.preload)
    if resuming:
        buffer.load(config.buffer_dir)
    logging.info("Replay buffer: %s", buffer.stats())

    policy = build_policy(config, buffer, asset_path=config.asset_path, default_prompt=config.default_prompt)
    # Before the socket is opened, not on the first request: see OnlineRLTPolicy.warmup.
    policy.warmup()
    learner = _learner.OnlineLearner(config, buffer, policy, checkpoint_manager=checkpoint_manager, resuming=resuming)
    if config.collect_only:
        # Nothing to learn from until the actor is in the loop, and a learner running against a
        # policy that never queries it would just burn the offline data down.
        logging.info("--collect_only: serving the VLA reference policy; the learner stays idle")
    else:
        learner.start()

    hostname = socket.gethostname()
    logging.info("Creating server (host: %s, ip: %s, port: %d)", hostname, socket.gethostbyname(hostname), config.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=config.port,
        metadata=policy.metadata,
    )
    try:
        server.serve_forever()
    finally:
        learner.stop()
        learner.save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(_config.cli())
