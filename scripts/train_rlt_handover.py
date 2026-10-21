"""RLT Stage A.6: train the handover predictor on cached RL tokens.

The head reads only `z_rl` and the normalized state, both precomputed by Stage A.5, so this loop
touches a ~1.2M-parameter MLP and runs at thousands of steps per second. What it costs is
supervision: hand-labeled critical-phase spans, pooled over one or more Stage A.5 caches
(`TrainConfig.sources`).

Two numbers decide whether the head is worth deploying, and both are reported on the held-out fold
rather than on the training loss:

  * the boundary error, `|begin_hat - begin|` in frames, from the Viterbi decode; and
  * the same error for a clock-only trigger that always fires at the training folds' mean
    `begin_id`. Handover times on this task are stereotyped (std 32 frames), so that baseline is
    not weak and the head has to beat it clearly to be carrying any observational signal.

Errors are also split into early and late. They are not symmetric on the robot: firing early only
spends extra RL runtime, firing late means the base VLA has already entered the precision phase.

Usage:
    # Cross-validation, to get the error bars before trusting the head. `rlt_pi05_handover_all` pools
    # every labeled dataset; `rlt_pi05_handover` is the teleop dataset only.
    for fold in 0 1 2 3 4; do
        python scripts/train_rlt_handover.py rlt_pi05_handover_all \
            --exp_name=fold$fold --val_fold=$fold --overwrite
    done

    # The head to ship: trained on every labeled episode of every source, no held-out fold.
    python scripts/train_rlt_handover.py rlt_pi05_handover_all --exp_name=all --overwrite
"""

import dataclasses
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm
import wandb

from openpi.models import rlt_handover
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config_rlt_handover as _config
import openpi.training.data_loader_handover as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


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
            opt_state=tx.init(params.filter(nnx.Param)),
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
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: dict[str, at.Array],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    dropout_rng = jax.random.fold_in(rng, state.step)
    grads, info = nnx.grad(lambda m: m.loss(dropout_rng, batch), argnums=nnx.DiffState(0, nnx.Param), has_aux=True)(
        model
    )

    params = state.params.filter(nnx.Param)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))

    new_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=new_opt_state)
    return new_state, {**info, "grad_norm": optax.global_norm(grads)}


def boundary_errors(
    head_fn,
    data: _data_loader.HandoverData,
    episodes: list[int],
    config: _config.TrainConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Signed `(begin, end)` errors in frames per episode, plus the count that failed to decode."""
    begin_errors, end_errors, undecoded = [], [], 0
    for episode in episodes:
        start, stop = (int(x) for x in data.episodes.bounds[episode])
        log_probs = rlt_handover.episode_log_probs(
            head_fn,
            data.episodes.z[start:stop],
            data.episodes.state[start:stop],
            config.model,
        )
        span = rlt_handover.viterbi_decode(log_probs, config.model.transition_penalty)
        if span is None:
            undecoded += 1
            continue
        begin_true, end_true = data.episodes.spans[episode]
        begin_errors.append(span[0] - begin_true)
        end_errors.append(span[1] - end_true)
    return np.asarray(begin_errors), np.asarray(end_errors), undecoded


def _error_summary(prefix: str, errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {}
    return {
        f"{prefix}_median_abs": float(np.median(np.abs(errors))),
        f"{prefix}_p90_abs": float(np.percentile(np.abs(errors), 90)),
        f"{prefix}_frac_early": float(np.mean(errors < 0)),
        f"{prefix}_frac_late": float(np.mean(errors > 0)),
    }


def evaluate(state: training_utils.TrainState, data: _data_loader.HandoverData, config: _config.TrainConfig) -> dict:
    """Held-out boundary metrics, plus the clock-only baseline computed on the same episodes."""
    if not data.val_episodes:
        return {}
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    head_fn = nnx_utils.module_jit(model.__call__)

    begin_errors, end_errors, undecoded = boundary_errors(head_fn, data, data.val_episodes, config)
    metrics = {
        **_error_summary("val_begin", begin_errors),
        **_error_summary("val_end", end_errors),
        "val_undecoded": float(undecoded),
    }

    # Clock-only trigger: always fire at the mean handover frame of the *training* episodes.
    clock = np.mean([data.episodes.spans[e][0] for e in data.train_episodes])
    clock_errors = np.asarray([clock - data.episodes.spans[e][0] for e in data.val_episodes])
    metrics.update(_error_summary("clock_begin", clock_errors))
    return metrics


def merge_into_stage_a(state: training_utils.TrainState, config: _config.TrainConfig) -> None:
    """Graft the trained head onto a Stage A params tree, so deployment loads one checkpoint."""
    params = _model.restore_params(config.merge_into, restore_type=np.ndarray)
    # `RLT.handover` is the attribute name the head is bound to inside the model.
    params["handover"] = jax.device_get(state.params.filter(nnx.Param).to_pure_dict())

    out_dir = config.checkpoint_dir / "params_with_handover"
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(out_dir, {"params": params})
    logging.info("Wrote merged Stage A + handover params to %s", out_dir)


def main(config: _config.TrainConfig):
    sharding.init_distributed()
    init_logging()
    configure_runtime()
    logging.info(f"Running on: {platform.node()}")

    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))

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

    data = _data_loader.load_handover_data(config)
    data_loader = _data_loader.create_handover_data_loader(config, data, sharding=data_sharding)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        train_step,
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

        if step % config.eval_interval == 0 or step == config.num_train_steps - 1:
            metrics = evaluate(train_state, data, config)
            if metrics:
                logging.info(f"Step {step} eval: " + ", ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
                wandb.log(metrics, step=step)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            checkpoint_manager.wait_until_finished()

    if config.merge_into:
        merge_into_stage_a(train_state, config)


if __name__ == "__main__":
    main(_config.cli())
