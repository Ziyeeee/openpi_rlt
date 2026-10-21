"""RLT Stage C: the on-robot policy of Algorithm 1, lines 7-12.

One query does the acting half of a rollout step:

    line 7   sample the VLA reference chunk    a_ref ~ pi_vla(s)
    line 8   form the RL state                 x = (z_rl(s), s_p)
    line 9   choose the action                 a_ref while warming up or before the handover,
                                               a ~ pi_theta(.| x, a_ref) once the RL policy owns
                                               the critical phase

Line 12 -- storing transitions in B -- happens in `_ingest` after the episode instead; see below.

All three of `a_ref`, `z_rl` and the handover logits come off a *single* PaliGemma prefix forward
(`RLTRTC.sample_actions_and_handover`), so adding RL costs one small MLP on top of the VLA pass the
robot was already paying for.

Two checkpoints, one model. Stage A (`rlt_pi05`) and the Stage A.6 handover head
(`rlt_pi05_handover_all`) are trained separately and are grafted here at load time, which is the
same graft `scripts/train_rlt_handover.py:merge_into_stage_a` writes to disk as
`params_with_handover` -- done in memory so neither checkpoint has to be rewritten.

Where the RL span comes from. Sec. V applies RL to a task's *critical phase* only. Online that
span is decided by the handover head: the server latches the first time `critical` is the argmax
for `handover_patience` consecutive queries, and never un-latches within an episode (the offline
counterpart is `rlt_handover.viterbi_decode`, which can use an episode's future and so is not
available here). The operator can override in both directions -- the head is new enough that its
first robot sessions should be watched, not trusted.

The latched span *is* the clip: the client starts buffering when a response first reports
`latched`, which makes the stored span exactly the `clips_by_tag.json` schema Stage B trains on.

Transitions are *not* formed here. Serving a query and storing a transition are on different
clocks: a query fires when the controller needs a new chunk, which is a control-latency decision,
while Appendix B wants a row every `subsample_stride` control steps ("Concretely we pick a stride
of 2 and save transitions corresponding to <x_0, a_0:C>, <x_2, a_2:C+2>, ..."). Those intermediate
rows each need a VLA prefix forward for `z_rl` plus the denoiser for `a_ref`, which is far too much
to do at 15 Hz inside the control loop. So the client buffers raw observations through the critical
phase and ships them to `_ingest` once the episode is over, with the arm held and the GPU idle.

Two things follow, both improvements over forming rows at query time:

  * `C` is decoupled from the controller's replan rate. The client runs whatever
    `--replan-chunk-step` is good for control; the RL step length is a server-side constant.
  * The stored action is the client's own 30 Hz command stream -- what the arm *executed*, after
    interpolation and the velocity clamp -- rather than the chunk this file emitted, only the first
    `--replan-chunk-step` steps of which are ever run.
"""

from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import threading
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import rlt_actor_critic
from openpi.models import rlt_config as _rlt_config
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger(__name__)

# The five submodules `RLTHandoverHead` is built from. Checked at load so a wrong --handover_dir
# fails here rather than as a shape error inside the first prefix forward.
_HANDOVER_MODULES = ("slot_proj", "state_proj", "layers", "out")

PHASE_NAMES = ("pre", "critical", "post")


def load_stage_a_with_handover(
    model_config: _rlt_config.RLTConfig,
    stage_a_dir: str | pathlib.Path,
    handover_dir: str | pathlib.Path,
):
    """Graft a Stage A.6 head onto a Stage A parameter tree and build the model.

    `RLTConfig.handover` must already be set: it is what makes `RLT.__init__` build the head, and
    with it left `None` the tree has no `handover` key to fill and `sample_actions_and_handover`
    raises.
    """
    if model_config.handover is None:
        raise ValueError("model_config.handover must be set before grafting a handover checkpoint")

    params = _model.restore_params(pathlib.Path(stage_a_dir) / "params", dtype=jnp.bfloat16)
    if "handover" in params:
        raise ValueError(
            f"{stage_a_dir} already carries a handover head (it looks like a params_with_handover "
            "tree); point --stage_a_dir at a plain Stage A checkpoint instead"
        )
    # Left in its own dtype, not cast to bf16 with the VLA: the head is ~1.7M params, it is the
    # thing deciding when RL takes the arm, and f32 costs nothing at this size.
    head = _model.restore_params(pathlib.Path(handover_dir) / "params")
    missing = [name for name in _HANDOVER_MODULES if name not in head]
    if missing:
        raise ValueError(f"{handover_dir} is not a handover checkpoint: missing {missing}, has {sorted(head)}")
    params["handover"] = head

    num_head = len(jax.tree.leaves(head))
    logger.info(
        "Loaded %d Stage A tensors from %s and %d handover tensors from %s",
        len(jax.tree.leaves(params)) - num_head,
        stage_a_dir,
        num_head,
        handover_dir,
    )
    model = model_config.load(params)
    model.eval()
    return model


def _actor_functions(model: rlt_actor_critic.RLTActorCritic):
    """Jitted actor/critic applies that take the params as an argument.

    `nnx_utils.module_jit` freezes module state, which is right for the frozen VLA and wrong here:
    the learner thread republishes actor parameters every few dozen steps. Threading `params`
    through as an ordinary pytree argument instead means a publish is a single atomic attribute
    swap, with no lock on the inference path.
    """
    graphdef = nnx.graphdef(model)

    @jax.jit
    def sample(params, rng, z, state, a_ref):
        return nnx.merge(graphdef, params).sample(rng, z, state, a_ref)[0]

    @jax.jit
    def mean(params, z, state, a_ref):
        return nnx.merge(graphdef, params).deterministic_action(z, state, a_ref)

    @jax.jit
    def q_value(params, z, state, chunk):
        return nnx.merge(graphdef, params)._q_min(z, state, chunk, target=False)  # noqa: SLF001

    return sample, mean, q_value


@dataclasses.dataclass
class _EpisodeState:
    """Per-episode collection state. Reset by an `rl_control` `begin`."""

    open: bool = False
    latched: bool = False
    released: bool = False  # the critical phase has ended; back to the VLA, recording stopped
    critical_run: int = 0  # consecutive queries the head called `critical`
    post_run: int = 0  # consecutive queries the head called `post`
    override: str = "auto"  # "auto" | "force_rl" | "force_vla"
    queries: int = 0
    last_query_time: float | None = None
    stride_samples: list[float] = dataclasses.field(default_factory=list)


class OnlineRLTPolicy(_base_policy.BasePolicy):
    """The collector half of Stage C. The learner half is `training/rlt_online_learner.py`."""

    def __init__(
        self,
        model,
        model_config: _rlt_config.RLTConfig,
        actor_critic: rlt_actor_critic.RLTActorCritic,
        buffer,
        *,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        chunk_size: int,
        ingest_batch_size: int = 32,
        reference_delay: int = 4,
        num_denoise_steps: int = 10,
        handover_patience: int = 2,
        warmup_transitions: int = 0,
        max_action_delta: float | None = 0.5,
        deterministic: bool = False,
        collect_only: bool = False,
        policy_hz: float = 30.0,
        metadata: dict[str, Any] | None = None,
        rng: at.KeyArrayLike | None = None,
    ):
        self._model = model
        self._window = model.handover.config.window
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._chunk_size = chunk_size
        self._ingest_batch_size = ingest_batch_size
        self._reference_delay = reference_delay
        self._num_denoise_steps = num_denoise_steps
        self._handover_patience = handover_patience
        self._warmup_transitions = warmup_transitions
        self._max_action_delta = max_action_delta
        self._deterministic = deterministic
        self._collect_only = collect_only
        self._policy_hz = policy_hz
        self._metadata = metadata or {}

        self._buffer = buffer
        self._sample_actor, self._mean_actor, self._q_value = _actor_functions(actor_critic)
        # Swapped wholesale by the learner; read without a lock, which is safe because the
        # assignment itself is atomic and a query only ever needs one self-consistent snapshot.
        self.actor_params = nnx.state(actor_critic)
        self.learner_step = 0

        # Same freeze-the-state jit the caching script uses on the frozen VLA; the RL-token branch,
        # the action expert and the handover head are all constant here.
        self._infer = nnx_utils.module_jit(model.sample_actions_and_handover)
        # The post-episode ingest pass. Same entry point `cache_rl_token.py` uses, so an online
        # clip and an offline one are produced by literally the same computation.
        self._reference = nnx_utils.module_jit(model.sample_rl_state_and_reference)
        self._rng = rng if rng is not None else jax.random.key(0)
        self._lock = threading.Lock()
        self._episode = _EpisodeState()
        # `window - 1` past queries, newest first, as `sample_actions_and_handover` expects: this
        # call's own z_rl and state become the newest slot inside the model.
        self._z_history: list[np.ndarray] = []
        self._state_history: list[np.ndarray] = []
        self._model_config = model_config

    # -- warmup ----------------------------------------------------------------------------------

    def warmup(self) -> None:
        """Compile every jitted path before the first robot query arrives.

        Not an optimization -- a correctness fix. The VLA graph takes minutes to compile, and
        `websockets` closes a connection whose keepalive ping goes unanswered for 20 s, so paying
        that cost inside the first `infer()` drops the connection instead of returning a chunk.
        `agilex_inference._connect` does issue a throwaway warm-up call, but it would hit the same
        20 s ceiling; compiling here, off any connection, is what makes that call cheap.

        Driven from `inputs_spec` rather than a synthetic robot observation so this stays free of
        any assumption about camera names or joint counts -- the transforms in front of the model
        are numpy and cost nothing to run cold.
        """
        started = time.monotonic()
        observation, _ = self._model_config.inputs_spec(batch_size=1)
        observation = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), observation)
        z_window, state_window = self._history_arrays()
        rng = jax.random.key(0)

        # Two distinct traces: the first query of an episode has no chunk in flight, every later
        # one does, and RTC arguments change the signature.
        horizon, dim = self._model_config.action_horizon, self._model_config.action_dim
        variants = [{}, {"action_prefix": jnp.zeros((1, horizon, dim)), "delay": jnp.zeros((1,), jnp.int32)}]
        for kwargs in variants:
            actions, z_rl, _ = self._infer(
                rng, observation, z_window, state_window, num_steps=self._num_denoise_steps, **kwargs
            )
        a_ref = actions[:, : self._chunk_size]
        action = self._sample_actor(self.actor_params, rng, z_rl, observation.state, a_ref)
        self._mean_actor(self.actor_params, z_rl, observation.state, a_ref)
        self._q_value(self.actor_params, z_rl, observation.state, action)

        # The ingest pass is a third trace: a different method at a different batch size. Compiling
        # it now costs startup time; compiling it lazily would stall the first episode close for
        # minutes with the operator waiting and the client's keepalive running out.
        batch = self._ingest_batch_size
        wide = jax.tree.map(lambda x: jnp.repeat(x, batch, axis=0), observation)
        self._reference(
            rng,
            wide,
            num_steps=self._num_denoise_steps,
            action_prefix=jnp.zeros((batch, horizon, dim)),
            delay=jnp.full((batch,), self._reference_delay, jnp.int32),
        )
        jax.block_until_ready(action)
        logger.info("Warmed up every jitted path in %.1fs", time.monotonic() - started)

    # -- control channel -------------------------------------------------------------------------

    def _control(self, message: dict) -> dict:
        """Episode boundaries and outcome labels, the two things the server cannot observe.

        These arrive on a second websocket connection because they are produced between inference
        calls -- `agilex_recorder._finish_episode` runs after the executor is already held, and at
        Ctrl-C there is no next request to piggyback on at all.
        """
        event = message.get("event")
        with self._lock:
            if event == "begin":
                self._buffer.open_episode()
                self._episode = _EpisodeState(open=True, override=message.get("control", "auto"))
                self._z_history.clear()
                self._state_history.clear()
                logger.info("episode begin (control=%s)", self._episode.override)
            elif event in ("end", "abort"):
                result = message.get("result") if event == "end" else None
                success = {"success": True, "fail": False}.get(result) if result else None
                rows = self._buffer.close_episode(success)
                stride = self._episode.stride_samples
                logger.info(
                    "episode %s (result=%s): %d rows admitted, %d queries at a measured %.1f-frame cadence, latched=%s",
                    event,
                    result,
                    rows,
                    self._episode.queries,
                    float(np.mean(stride)) if stride else float("nan"),
                    self._episode.latched,
                )
                self._episode = _EpisodeState()
            elif event == "control":
                self._episode.override = message.get("control", "auto")
                logger.info("operator override: %s", self._episode.override)
            else:
                raise ValueError(f"unknown rl_control event {event!r}")
            return {"ok": True, **self._buffer.stats(), "learner_step": self.learner_step}

    # -- ingest ----------------------------------------------------------------------------------

    def _ingest(self, message: dict) -> dict:
        """Turn a batch of subsampled control steps into replay transitions (Alg. 1 line 12).

        This is Appendix B's chunk subsampling: "While the RL policy uses an action chunk length of
        C, we obtain observations for each intermediate step. We can thus increase data and improve
        learning efficiency by storing intermediate steps into the replay buffer. Concretely we pick
        a stride of 2 and save transitions corresponding to <x_0, a_0:C>, <x_2, a_2:C+2>, ..."

        Those intermediate rows are not free the way they are offline: forming `x = (z_rl(s), s_p)`
        needs a VLA prefix forward, and `a_ref` needs the denoiser on top. Doing that at 15 Hz
        inside the control loop is impossible. So the client buffers the raw observations during
        the critical phase and ships them here once the episode is over -- the arm is held, the
        operator is resetting the scene, and the GPU is idle. It is the Stage A.5 caching pass
        (`sample_rl_state_and_reference`), run on one episode at a time.

        The actions are the client's own 30 Hz command stream, i.e. what the arm *executed* after
        interpolation and the velocity clamp -- not the chunk the server emitted. That is both the
        correct behaviour-cloning target and a strictly better answer than the query-time storage
        this replaced, whose "stored action != executed action" gap `docs/rlt_wip.md` had to flag.
        """
        frames = np.asarray(message["frames"], dtype=np.int64)
        states = np.asarray(message["state"], dtype=np.float32)  # [n, 14] raw robot space
        actions = np.asarray(message["actions"], dtype=np.float32)  # [n, C, 14] executed commands
        images = message["images"]  # dict of [n, 3, 224, 224] uint8
        prompt = message.get("prompt")
        n = len(frames)

        # Every row through the training transform chain, so the cached state and actions land in
        # exactly the normalized space the actor and critic operate in -- the same guarantee
        # `cache_rl_token.py` makes about the offline cache.
        rows = [
            self._input_transform(
                {
                    "images": {name: images[name][i] for name in images},
                    "state": states[i],
                    "actions": actions[i],
                    **({"prompt": prompt} if prompt is not None else {}),
                }
            )
            for i in range(n)
        ]
        # Pad up to a whole batch: a short trailing batch is a new input shape, and would force XLA
        # to recompile the whole VLA graph while the operator waits (`cache_rl_token.py` hit the
        # same thing). Sliced back off below.
        batch = self._ingest_batch_size
        if n > batch:
            raise ValueError(f"ingest message carries {n} rows, more than the batch size {batch}")
        padded = rows + [rows[-1]] * (batch - n)
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *padded)

        observation = _model.Observation.from_dict(stacked)
        # The RTC prefix is the executed chunk's own leading steps, matching both how Stage A was
        # trained and how `cache_rl_token.py` draws the offline `a_ref`. Only the first `delay`
        # entries are read, so padding the horizon with the last step is irrelevant.
        horizon = self._model_config.action_horizon
        committed = stacked["actions"]
        prefix = jnp.concatenate(
            [committed, jnp.repeat(committed[:, -1:], horizon - committed.shape[1], axis=1)], axis=1
        )
        with self._lock:
            self._rng, step_rng = jax.random.split(self._rng)
            z_rl, a_ref = self._reference(
                step_rng,
                observation,
                num_steps=self._num_denoise_steps,
                action_prefix=prefix,
                delay=jnp.full((batch,), self._reference_delay, jnp.int32),
            )
            total = self._buffer.append_rows(
                frames,
                np.asarray(z_rl)[:n],
                np.asarray(observation.state)[:n],
                np.asarray(stacked["actions"])[:n, : self._chunk_size],
                np.asarray(a_ref)[:n, : self._chunk_size],
            )
        return {"ok": True, "rows": total, "ingested": n}

    # -- inference -------------------------------------------------------------------------------

    @override
    def infer(  # type: ignore[misc]
        self,
        obs: dict,
        *,
        action_prefix: np.ndarray | None = None,
        delay: int | np.ndarray | None = None,
    ) -> dict:
        if "rl_control" in obs:
            return self._control(obs["rl_control"])
        if "rl_batch" in obs:
            return self._ingest(obs["rl_batch"])

        started = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)
        request = inputs.pop("rl", None) or {}
        if action_prefix is not None:
            # Through the same DeltaActions -> MaskActions -> Normalize -> pad chain as training
            # actions, exactly as `Policy.infer` does. The prefix arrives in raw robot space.
            inputs["actions"] = action_prefix
        inputs = self._input_transform(inputs)
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        rtc_kwargs = {}
        if action_prefix is not None and delay is not None:
            rtc_kwargs = {
                "action_prefix": batched["actions"],
                "delay": jnp.asarray(delay).reshape(1).astype(jnp.int32),
            }

        with self._lock:
            self._apply_override(request.get("control"))
            self._rng, step_rng, actor_rng = jax.random.split(self._rng, 3)
            observation = _model.Observation.from_dict(batched)

            z_window, state_window = self._history_arrays()
            actions_ref, z_rl, logits = self._infer(
                step_rng,
                observation,
                z_window,
                state_window,
                num_steps=self._num_denoise_steps,
                **rtc_kwargs,
            )
            state = observation.state
            self._push_history(np.asarray(z_rl[0]), np.asarray(state[0]))

            phase, mode = self._decide(np.asarray(logits[0]))
            a_ref = actions_ref[:, : self._chunk_size]
            if mode == "rl":
                action = self._actor_chunk(actor_rng, z_rl, state, a_ref)
                actions = jnp.concatenate([action, actions_ref[:, self._chunk_size :]], axis=1)
            else:
                action, actions = a_ref, actions_ref

            info = self._record(z_rl, state, action, a_ref, mode=mode, phase=phase)

        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), {"state": batched["state"], "actions": actions})
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - started) * 1000}
        outputs["rl"] = info
        return outputs

    # -- decision --------------------------------------------------------------------------------

    def _apply_override(self, control: str | None) -> None:
        if control and control != self._episode.override:
            self._episode.override = control
            logger.info("operator override: %s", control)

    def _decide(self, logits: np.ndarray) -> tuple[str, str]:
        """Latch the handover, then pick `rl` or `vla` for this query (Alg. 1 line 9)."""
        episode = self._episode
        episode.queries += 1
        now = time.monotonic()
        if episode.last_query_time is not None:
            episode.stride_samples.append((now - episode.last_query_time) * self._policy_hz)
        episode.last_query_time = now

        phase = PHASE_NAMES[int(np.argmax(logits))]
        episode.critical_run = episode.critical_run + 1 if phase == "critical" else 0
        episode.post_run = episode.post_run + 1 if phase == "post" else 0
        if episode.critical_run >= self._handover_patience:
            # `pre` never un-latches: the causal head sees only the past, so a single `pre` reading
            # mid-insertion is far more likely to be noise than a genuine return to the approach.
            if not episode.latched:
                logger.info("handover latched at query %d", episode.queries)
            episode.latched = True
        if episode.latched and episode.post_run >= self._handover_patience and episode.override != "force_rl":
            # `post` does release it. Sec. V scopes RL to the critical phase, and the head's
            # left-to-right chain makes critical -> post the one legitimate exit: `post` means the
            # task is done, so the retreat belongs to the VLA. It also puts the terminal `+1` on
            # the last recorded row, i.e. at completion, rather than wherever the operator got
            # around to pressing `s`. `force_rl` holds through it -- the override exists precisely
            # for when the head is not to be trusted -- but cannot undo a release that already
            # fired, because re-latching would leave a time gap between two adjacent rows and
            # `next_offset` assumes they are `subsample_stride` control steps apart.
            if not episode.released:
                logger.info(
                    "handover released at query %d: `post` for %d queries; back to the VLA",
                    episode.queries,
                    episode.post_run,
                )
            episode.released = True

        if self._collect_only or episode.override == "force_vla" or episode.released:
            return phase, "vla"
        if episode.override == "force_rl":
            episode.latched = True
        elif not episode.latched:
            return phase, "vla"
        if self._buffer.env_steps < self._warmup_transitions:
            # Alg. 1 line 9: below N_warm the reference policy collects, so the critic gets its
            # initial signal from competent VLA behaviour rather than from an untrained actor.
            return phase, "warmup"
        return phase, "rl"

    def _actor_chunk(self, rng, z_rl, state, a_ref):
        """a ~ pi(.|x, a_ref), then the safety filter.

        Exploration noise is not incidental here: it is the only thing that ever visits a state
        under more than one action, and therefore the only source of the action contrast the critic
        needs (`docs/rlt_wip.md`, "collect rollouts with action diversity"). `--deterministic` turns
        it off for evaluation, where the point is to measure the learned policy, not improve it.
        """
        if self._deterministic:
            action = self._mean_actor(self.actor_params, z_rl, state, a_ref)
        else:
            action = self._sample_actor(self.actor_params, rng, z_rl, state, a_ref)
        if self._max_action_delta is not None:
            # A pre-execution bound in normalized space: the actor may edit the VLA's chunk, not
            # replace it. The client's velocity clamp and BOUNDARY_JUMP_LIMIT are the second line
            # of defence, but they act after the fact and cannot tell a bad plan from a fast one.
            action = a_ref + jnp.clip(action - a_ref, -self._max_action_delta, self._max_action_delta)
        return action

    def _record(self, z_rl, state, action, a_ref, *, mode: str, phase: str) -> dict:
        """Build the block the client logs, and tell it whether to be buffering rows.

        Nothing is stored here. Transitions are formed by `_ingest` after the episode, at
        `subsample_stride` control steps rather than at query boundaries -- a query is a poor
        sampling grid (it is chosen for control latency, not for RL) and it throws away the
        intermediate steps Appendix B is explicit about keeping.

        `latched` is the one bit the client needs back, and it means "buffer this row": it rises on
        the first frame of the critical phase and falls when `post` releases it, so the span the
        client buffers is exactly the critical phase -- the same span `clips_by_tag.json` carries
        offline. `released` is reported separately only so the log can tell "not yet" from "done".
        """
        episode = self._episode
        delta = float(jnp.max(jnp.abs(action - a_ref)))
        q = float(self._q_value(self.actor_params, z_rl, state, action)[0])
        return {
            "phase": phase,
            "mode": mode,
            "latched": episode.latched and not episode.released,
            "released": episode.released,
            "override": episode.override,
            "q": q,
            "action_delta": delta,
            "learner_step": self.learner_step,
            "buffer": int(self._buffer.env_steps),
        }

    # -- handover window -------------------------------------------------------------------------

    def _history_arrays(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The `window - 1` past queries, newest first, oldest repeated to fill.

        Repeating the oldest is `rlt_handover.window_indices`' clamp to the episode start, which is
        what the head was trained with -- early frames simply see their first observation in every
        slot that reaches before the episode began.
        """
        slots = self._window - 1
        if not self._z_history:
            zeros_z = np.zeros((slots, self._model.handover.config.rl_token_dim), dtype=np.float32)
            zeros_s = np.zeros((slots, self._model.handover.config.state_dim), dtype=np.float32)
            return jnp.asarray(zeros_z)[None], jnp.asarray(zeros_s)[None]
        pad = slots - len(self._z_history)
        z = self._z_history + [self._z_history[-1]] * pad
        s = self._state_history + [self._state_history[-1]] * pad
        return jnp.asarray(np.stack(z[:slots]))[None], jnp.asarray(np.stack(s[:slots]))[None]

    def _push_history(self, z: np.ndarray, state: np.ndarray) -> None:
        self._z_history.insert(0, z.astype(np.float32))
        self._state_history.insert(0, state.astype(np.float32))
        del self._z_history[self._window :]
        del self._state_history[self._window :]

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
