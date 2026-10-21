"""RLT Stage C: the learner half of Algorithm 1 (lines 13-18), running beside the robot.

    for g = 1..G:  sample b ~ B;  train the critic (Eq. 3);  train the policy (Eq. 5)

The update itself is `scripts/train_rlt_ac.train_step`, imported unchanged -- the same separate
actor/critic gradients under complementary parameter filters, the same critic warmup gate with its
optimizer-state rollback, the same Polyak target update. Nothing about a gradient step depends on
whether the batch came from a cache or from a robot two seconds ago, and forking it would leave
two places to keep the TD backup correct.

What *is* new is the pacing. "To remain compute and time-efficient during training, we perform the
rollouts and learning asynchronously... We use a high update-to-data ratio of 5, which is essential
in the low-data online regime" (Sec. V). So this runs on its own thread and throttles against the
collector's transition count instead of free-running: at 5 updates per environment step and a query
every half second that is ~10 MLP steps/s, some 3 ms/s of GPU, which does not contend with the VLA
prefix forward the same device is serving.

Freshly trained parameters reach the collector by `policy.actor_params = ...`, a plain attribute
swap. The collector's actor apply takes params as an ordinary argument, so a publish needs no lock
and can never hand out a half-updated tree.
"""

import contextlib
import dataclasses
import functools
import importlib.util
import logging
import pathlib
import threading
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import wandb

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.sharding as sharding

logger = logging.getLogger(__name__)


def load_train_rlt_ac():
    """Import `scripts/train_rlt_ac.py` as a module.

    `scripts/` is a directory of entry points, not a package, so there is no import path to it.
    Loading it by file keeps Stage C on the *same* `train_step` Stage B is verified against
    (`docs/rlt_wip.md`: gradient isolation, the warmup gate, the exact Polyak update) rather than
    on a copy that can silently diverge from it.
    """
    path = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "train_rlt_ac.py"
    if not path.exists():
        raise FileNotFoundError(f"expected scripts/train_rlt_ac.py at {path}; run Stage C from the repo root")
    spec = importlib.util.spec_from_file_location("train_rlt_ac", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _AssetlessLoader:
    """`checkpoints.save_state` asks the loader for a data config to store beside the params.

    Stage C has no loader -- the buffer *is* the data -- and no norm stats of its own: those belong
    to the Stage A checkpoint that produced every `z_rl` in the buffer. Same empty stand-in
    `RLTDataLoader.data_config` returns for Stage B.
    """

    def data_config(self):
        import openpi.training.config as _config

        return _config.DataConfig()


class OnlineLearner:
    """A background thread that trains the actor and critic off the live replay buffer."""

    def __init__(self, config, buffer, policy, *, checkpoint_manager=None, resuming: bool = False):
        self._config = config
        self._buffer = buffer
        self._policy = policy
        self._checkpoint_manager = checkpoint_manager

        train_rlt_ac = load_train_rlt_ac()
        self._mesh = sharding.make_mesh(1)
        self._replicated = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
        self._np_rng = np.random.default_rng(config.seed)

        init_rng, self._train_rng = jax.random.split(jax.random.key(config.seed))
        ac = config.ac
        self.state, state_sharding = train_rlt_ac.init_train_state(ac, init_rng, self._mesh)
        if resuming and checkpoint_manager is not None:
            self.state = _checkpoints.restore_state(checkpoint_manager, self.state, None)
            logger.info("Resumed the actor/critic at step %d", int(self.state.step))
        elif config.ac_params_path is not None:
            self.state = self._warm_start(config.ac_params_path)
            # The gate exists to stop the actor climbing a *random* critic. A warm-started critic
            # is not random -- it has already been fit to the offline returns -- and the step
            # counter restarts at 0 here, so keeping the offline warmup would freeze the actor for
            # the first few hundred seconds of robot time, which is most of a session.
            ac = dataclasses.replace(ac, critic_warmup_steps=0)
            logger.info("Warm-started critic: actor gate opens immediately (critic_warmup_steps=0)")

        # No `donate_argnums=(1,)`, unlike the offline `train_rlt_ac.main`. Donation frees the input
        # train state's buffers, and the collector thread is holding a reference to `params` from a
        # previous publish -- reading it after the next step would hit "Buffer has been deleted or
        # donated" mid-rollout. Offline nothing else reads the state, so donation is free there;
        # here it would trade a correctness hazard for ~14 MB on a 30-tensor MLP.
        self._step_fn = jax.jit(
            functools.partial(train_rlt_ac.train_step, ac),
            in_shardings=(self._replicated, state_sharding, self._replicated),
            out_shardings=(state_sharding, self._replicated),
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="rlt-learner", daemon=True)
        self.publish()

    def _warm_start(self, params_path: str):
        """Load an offline Stage B checkpoint's parameters into the online train state.

        Parameters only. The optimizer moments belong to a cosine schedule that has already decayed
        to `decay_lr`, so carrying them across would apply an end-of-run learning rate to the first
        online data -- exactly when the critic most needs to move.
        """
        params = _model.restore_params(params_path, restore_type=np.ndarray)
        state = self.state
        target = state.params.filter(nnx.Param).to_pure_dict()
        expected = {path: leaf.shape for path, leaf in _leaves(target)}
        got = {path: np.shape(leaf) for path, leaf in _leaves(params)}
        if expected != got:
            bad = {k: (expected.get(k), got.get(k)) for k in set(expected) | set(got) if expected.get(k) != got.get(k)}
            raise ValueError(
                f"{params_path} does not match this Stage C actor/critic (mismatched: {bad}). It was most "
                f"likely trained at a different chunk_size -- this config uses C={self._config.chunk_size}, "
                "so retrain with `python scripts/train_rlt_ac.py rlt_pi05_ac_c16 --exp_name=c16`."
            )
        new_params = state.params
        new_params.replace_by_pure_dict(_model._intify_pytree_keys(params))  # noqa: SLF001
        logger.info("Warm-started the actor/critic from %s", params_path)
        return dataclasses.replace(state, params=new_params)

    # -- lifecycle -------------------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30.0)

    @property
    def step(self) -> int:
        return int(self.state.step)

    def publish(self) -> None:
        self._policy.actor_params = self.state.params
        self._policy.learner_step = self.step

    # -- the loop --------------------------------------------------------------------------------

    def _loop(self) -> None:
        config = self._config
        infos: list[dict] = []
        last_log = time.monotonic()
        start_step = self.step
        while not self._stop.is_set():
            if len(self._buffer) < config.learning_starts:
                self._stop.wait(1.0)
                continue
            # UTD throttle. `env_steps` counts rows the robot actually produced, so the learner runs
            # flat out while behind and idles while ahead -- which is what makes a ratio of 5 an
            # accounting fact rather than an aspiration.
            if self.step - start_step >= config.updates_per_env_step * max(self._buffer.env_steps, 1):
                self._stop.wait(0.05)
                continue

            batch = self._buffer.sample(config.ac.batch_size, self._np_rng)
            batch = jax.device_put(jax.tree.map(jnp.asarray, batch), self._replicated)
            with sharding.set_mesh(self._mesh):
                self.state, info = self._step_fn(self._train_rng, self.state, batch)
            infos.append(info)

            if self.step % config.publish_interval == 0:
                self.publish()
            if time.monotonic() - last_log >= 10.0:
                last_log = time.monotonic()
                self._log(infos, start_step)
                infos = []
            if self._checkpoint_manager is not None and self.step % config.save_interval == 0:
                self.save()
        self.publish()

    def _log(self, infos: list[dict], start_step: int) -> None:
        if not infos:
            return
        reduced = {k: float(np.mean([float(i[k]) for i in infos])) for k in infos[0]}
        stats = self._buffer.stats()
        reduced.update(stats)
        # The measured update-to-data ratio, which is the knob Appendix B actually names.
        reduced["utd"] = (self.step - start_step) / max(stats["env_steps"], 1.0)
        logger.info("Learner step %d: %s", self.step, ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
        with contextlib.suppress(Exception):
            wandb.log(reduced, step=self.step)

    def save(self) -> None:
        if self._checkpoint_manager is None:
            return
        _checkpoints.save_state(self._checkpoint_manager, self.state, _AssetlessLoader(), self.step)
        self._checkpoint_manager.wait_until_finished()
        # The online clips go beside the checkpoint, so a session can be resumed and -- more
        # usefully -- replayed into an ordinary Stage B run afterwards without touching the robot.
        self._buffer.save(self._config.buffer_dir)


def _leaves(tree, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten a pure-dict param tree to (path, leaf), for a readable shape diff."""
    out = []
    for key, value in sorted(tree.items()):
        path = f"{prefix}/{key}" if prefix else str(key)
        out.extend(_leaves(value, path) if isinstance(value, dict) else [(path, value)])
    return out
