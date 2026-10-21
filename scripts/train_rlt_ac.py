"""RLT Stage B: train the actor and twin critics on cached RL tokens.

Everything the VLA contributes -- the RL token `z_rl` and the reference chunk `a_ref` -- comes from
the Stage A.5 cache, so this loop touches only small MLPs and runs at thousands of steps/second.

Two things make the update non-standard and are worth reading carefully:

1. The actor loss reads Q_psi. A single combined gradient would therefore push the critic to inflate
   its own values. The two objectives are differentiated separately under complementary parameter
   filters, so critic gradients come only from L_Q and actor gradients only from L_pi.
2. Critic Q warmup. `actor_gate` zeroes the actor gradient until the critic has been trained for
   `critic_warmup_steps`, and thereafter only lets it through every
   `critic_updates_per_actor_update` steps (Appendix B: 2 critic updates per actor update). Target
   critics are soft-updated on every step the critic moves.

Usage:
    python scripts/train_rlt_ac.py rlt_pi05_ac --exp_name=debug --overwrite
"""

import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

from openpi.models import rlt_actor_critic
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config_rlt_ac as _config
import openpi.training.data_loader_rlt as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

# The two objectives own disjoint parameter subtrees. Target critics are updated by Polyak
# averaging, never by the optimizer, so they are excluded from both.
_CRITIC_FILTER = nnx.All(nnx.Param, nnx_utils.PathRegex(r"critic_\d.*"))
_ACTOR_FILTER = nnx.All(nnx.Param, nnx_utils.PathRegex(r"actor.*"))
_TRAINABLE_FILTER = nnx.Any(_CRITIC_FILTER, _ACTOR_FILTER)


def configure_runtime():
    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get("JAX_COMPILATION_CACHE_DIR", str(epath.Path("~/.cache/jax").expanduser())),
    )


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name, mode="offline")
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name, mode="offline")
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike) -> training_utils.TrainState:
        model = config.model.create(rng)
        params = nnx.state(model)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            # Separate optimizer state per objective. The same `tx` drives both -- AdamW does not
            # care which subtree it is applied to -- but the moments must not be shared, because the
            # actor is updated on only a subset of the steps the critic is.
            opt_state=(tx.init(params.filter(_CRITIC_FILTER)), tx.init(params.filter(_ACTOR_FILTER))),
            ema_decay=None,
            ema_params=None,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(init, in_shardings=replicated_sharding, out_shardings=state_sharding)(init_rng)
    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: dict[str, at.Array],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    critic_rng, actor_rng = jax.random.split(jax.random.fold_in(rng, state.step))

    def critic_loss_fn(model, rng):
        return model.critic_loss(rng, batch)

    def actor_loss_fn(model, rng):
        return model.actor_loss(rng, batch)

    critic_grads, critic_info = nnx.grad(critic_loss_fn, argnums=nnx.DiffState(0, _CRITIC_FILTER), has_aux=True)(
        model, critic_rng
    )
    actor_grads, actor_info = nnx.grad(actor_loss_fn, argnums=nnx.DiffState(0, _ACTOR_FILTER), has_aux=True)(
        model, actor_rng
    )

    # Critic Q warmup, then 2 critic updates per actor update.
    open_gate = jnp.logical_and(
        state.step >= config.critic_warmup_steps,
        state.step % config.critic_updates_per_actor_update == 0,
    )

    critic_params = state.params.filter(_CRITIC_FILTER)
    actor_params = state.params.filter(_ACTOR_FILTER)
    critic_opt_state, actor_opt_state = state.opt_state

    critic_updates, new_critic_opt_state = state.tx.update(critic_grads, critic_opt_state, critic_params)
    actor_updates, new_actor_opt_state = state.tx.update(actor_grads, actor_opt_state, actor_params)

    # A closed gate must be a genuine no-op. Zeroing only the gradient is not enough: Adam's stored
    # moments would keep nudging the actor on the suppressed steps, so the warmup would leak and the
    # 2:1 ratio would not hold. Roll the actor's optimizer state back as well.
    actor_updates = jax.tree.map(lambda u: u * open_gate.astype(u.dtype), actor_updates)
    new_actor_opt_state = jax.tree.map(
        lambda new, old: jnp.where(open_gate, new, old), new_actor_opt_state, actor_opt_state
    )

    new_params = nnx.State.merge(
        optax.apply_updates(critic_params, critic_updates),
        optax.apply_updates(actor_params, actor_updates),
    )
    new_opt_state = (new_critic_opt_state, new_actor_opt_state)

    nnx.update(model, new_params)
    # psi' <- tau * psi + (1 - tau) * psi'. The critic moves every step, so the targets do too.
    rlt_actor_critic.soft_update_targets(model, config.model.target_tau)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    info = {
        **critic_info,
        **actor_info,
        "actor_gate": open_gate.astype(jnp.float32),
        "critic_grad_norm": optax.global_norm(critic_grads),
        "actor_grad_norm": optax.global_norm(actor_grads),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    sharding.init_distributed()
    init_logging()
    configure_runtime()
    logging.info(f"Running on: {platform.node()}")

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled and jax.process_index() == 0)

    data_loader = _data_loader.create_rlt_data_loader(config, sharding=data_sharding)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=jax.process_index() != 0,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            logging.info(f"Step {step}: " + ", ".join(f"{k}={float(v):.4f}" for k, v in reduced_info.items()))
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
