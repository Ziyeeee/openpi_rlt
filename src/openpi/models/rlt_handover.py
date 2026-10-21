"""RLT Stage A.6: the handover predictor that marks the task's critical phase.

Sec. V of the RLT paper applies RL only to the *critical phase* of a task -- during data collection
a human decides when to hand control from the base VLA to the RL policy -- and closes by training
the VLA to predict that handover point so the switch fires automatically at test time. This file is
that predictor.

The prediction cannot be made from the current frame alone. In the charging task the robot first pulls
the charging gun out and then plugs it back in; the two stages look nearly identical, so a
single-frame classifier is guessing. The head therefore reads a *window* of past RL tokens:

    slot j (j = 0 is newest) is the frame t - phase - stride * j

with `stride` set to one policy query, because `z_rl` only changes when the VLA is queried. What
disambiguates pull-out from plug-in is simply that the window reaches back far enough to contain the
pull-out.

Three classes, not two: `pre / critical / post`. `post` is what makes the ambiguity representable --
"after the plug-in" is a different state from "before the pull-out" even when the images agree -- it
yields both segment boundaries, and it admits the left-to-right chain decode in `viterbi_decode`.

The head is a submodule of `RLT`/`RLTRTC` (see `RLTConfig.handover`), so at deployment one call
returns the action chunk, `z_rl`, and the handover logits off a single prefix forward. It is trained
separately, in Stage A.6, against the Stage A.5 cache -- it never needs an image or a VLA forward
pass of its own.
"""

from collections.abc import Callable
import dataclasses
import itertools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.shared import array_typing as at

# The left-to-right phase chain. `pre` is before the critical phase begins, `critical` is the span
# the RL policy owns, `post` is after it ends.
PRE, CRITICAL, POST = 0, 1, 2
NUM_PHASES = 3


@dataclasses.dataclass(frozen=True)
class RLTHandoverConfig:
    """Shapes and hyperparameters of the handover head. `None` on `RLTConfig` means no head."""

    # Matches `RLTConfig.rl_token_dim` (gemma_rlt width) and the padded action/state dim.
    rl_token_dim: int = 2048
    state_dim: int = 32

    # K: how many past policy queries the head sees.
    window: int = 16
    # Frames between window slots. Re-predict after running out the first 20 actions during inference.
    window_stride: int = 20
    # Train-time augmentation: the stride is drawn from `window_stride +/- stride_jitter` so the head
    # does not become brittle to the RTC delay jitter that shifts query times on the robot.
    stride_jitter: int = 10

    slot_dim: int = 256
    hidden_dim: int = 256
    depth: int = 2
    dropout: float = 0.1

    # Half-width, in frames, of the linear ramp between adjacent phases in the training target.
    # Human boundary annotations are noisy at roughly this scale; a hard step would spend most of
    # the gradient on frames nobody can label consistently.
    label_ramp: int = 10

    # Log-probability charged for each phase switch in `viterbi_decode`. Higher means the offline
    # decode needs more evidence before it accepts a boundary.
    transition_penalty: float = 4.0

    def create(self, rng: at.KeyArrayLike) -> "RLTHandoverHead":
        return RLTHandoverHead(self, rngs=nnx.Rngs(rng))

    def load(self, params: at.Params) -> "RLTHandoverHead":
        """Restore a head from a Stage A.6 checkpoint's `params` directory.

        Same dance as `BaseModelConfig.load`; the head is a plain `nnx.Module`, not a `BaseModel`,
        so it does not inherit that method.
        """
        head = nnx.eval_shape(lambda: self.create(jax.random.key(0)))
        graphdef, state = nnx.split(head)
        state.replace_by_pure_dict(_model._intify_pytree_keys(params))  # noqa: SLF001
        return nnx.merge(graphdef, state)


class RLTHandoverHead(nnx.Module):
    """Window of (z_rl, state) pairs -> pre/critical/post logits."""

    def __init__(self, config: RLTHandoverConfig, rngs: nnx.Rngs):
        self.config = config
        # One shared projection over window slots, so the slot count can change without retraining
        # the bulk of the head.
        self.slot_proj = nnx.Linear(config.rl_token_dim + config.state_dim, config.slot_dim, rngs=rngs)
        # The current state enters separately: on the robot it updates every control step while the
        # window is stale until the next VLA query.
        self.state_proj = nnx.Linear(config.state_dim, config.slot_dim, rngs=rngs)

        dims = [(config.window + 1) * config.slot_dim] + [config.hidden_dim] * config.depth
        self.layers = [nnx.Linear(a, b, rngs=rngs) for a, b in itertools.pairwise(dims)]
        self.out = nnx.Linear(config.hidden_dim, NUM_PHASES, rngs=rngs)
        # Defaults to eval, like the RLT models; `nnx.Module.train()` flips it. The rngs are passed
        # per call rather than stored, so the head's checkpointed state is params only.
        self.dropout = nnx.Dropout(config.dropout, deterministic=True)

    @at.typecheck
    def __call__(
        self,
        z_window: at.Float[at.Array, "b k d"],
        state_window: at.Float[at.Array, "b k s"],
        state: at.Float[at.Array, "b s"],
        *,
        rngs: nnx.Rngs | None = None,
    ) -> at.Float[at.Array, "b 3"]:
        slots = self.slot_proj(jnp.concatenate([z_window, state_window], axis=-1))
        x = jnp.concatenate([slots.reshape(slots.shape[0], -1), self.state_proj(state)], axis=-1)
        for layer in self.layers:
            x = self.dropout(nnx.swish(layer(x)), rngs=rngs)
        return self.out(x)

    def loss(self, rng: at.KeyArrayLike, batch: dict) -> tuple[jnp.ndarray, dict]:
        """Cross entropy against the ramped soft targets."""
        logits = self(batch["z_window"], batch["state_window"], batch["state"], rngs=nnx.Rngs(dropout=rng))
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = jnp.mean(-jnp.sum(batch["target"] * log_probs, axis=-1))

        label = jnp.argmax(batch["target"], axis=-1)
        predicted = jnp.argmax(logits, axis=-1)
        return loss, {
            "loss": loss,
            "accuracy": jnp.mean(predicted == label),
            # Accuracy is dominated by the phase interiors, which are easy. This is the fraction of
            # frames the head puts on the wrong side of the pre/critical divide -- the quantity the
            # handover trigger actually depends on.
            "critical_confusion": jnp.mean((predicted == CRITICAL) != (label == CRITICAL)),
        }


@at.typecheck
def phase_targets(length: int, begin: int, end: int, ramp: int) -> np.ndarray:
    """Soft per-frame targets `[length, 3]` for a critical span `[begin, end]`.

    Linear ramps of half-width `ramp` replace the two hard boundaries. The rows always sum to 1.
    """
    if not 0 <= begin <= end < length:
        raise ValueError(f"critical span ({begin}, {end}) does not fit in an episode of {length} frames")
    t = np.arange(length, dtype=np.float32)
    # Fraction of the way past each boundary, saturating outside the ramp.
    entered = np.clip((t - begin) / (2 * ramp) + 0.5, 0.0, 1.0)
    exited = np.clip((t - end) / (2 * ramp) + 0.5, 0.0, 1.0)
    return np.stack([1.0 - entered, entered - exited, exited], axis=-1)


@at.typecheck
def window_indices(
    frames: np.ndarray, episode_start: np.ndarray, *, window: int, stride: int, phase: int = 0
) -> np.ndarray:
    """Frame indices `[n, window]` of each query frame's window slots, newest first.

    Slot j sits `phase + stride * j` frames back, clamped to the episode start so early frames
    simply repeat their first observation.
    """
    offsets = phase + stride * np.arange(window)
    return np.maximum(frames[:, None] - offsets[None, :], episode_start[:, None])


def episode_log_probs(
    head_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    z: np.ndarray,
    state: np.ndarray,
    config: RLTHandoverConfig,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-frame log probabilities `[T, 3]` for one episode's cached `z_rl` and `state`.

    Offline every frame has a window available, so this evaluates at full frame resolution with
    `phase = 0`; on the robot the same head runs once per query instead.
    """
    length = z.shape[0]
    idx = window_indices(
        np.arange(length), np.zeros(length, dtype=np.int64), window=config.window, stride=config.window_stride
    )
    # Pad to a whole number of batches: a short trailing batch is a new input shape and would force
    # XLA to recompile the head once per distinct episode length.
    padded = int(np.ceil(length / batch_size)) * batch_size
    idx = np.concatenate([idx, np.repeat(idx[-1:], padded - length, axis=0)])
    frames = np.concatenate([np.arange(length), np.full(padded - length, length - 1)])

    out = np.empty((padded, NUM_PHASES), dtype=np.float32)
    for lo in range(0, padded, batch_size):
        rows = idx[lo : lo + batch_size]
        logits = head_fn(
            jnp.asarray(z[rows], dtype=jnp.float32),
            jnp.asarray(state[rows], dtype=jnp.float32),
            jnp.asarray(state[frames[lo : lo + batch_size]], dtype=jnp.float32),
        )
        out[lo : lo + batch_size] = np.asarray(jax.nn.log_softmax(logits, axis=-1))
    return out[:length]


@at.typecheck
def viterbi_decode(log_probs: np.ndarray, transition_penalty: float) -> tuple[int, int] | None:
    """Best `pre -> critical -> post` path over a whole episode; returns `(begin, end)` or None.

    The offline counterpart of the online latch. Only three transitions are legal, so the decoded
    path has exactly one contiguous critical span -- which is the shape of the annotation schema --
    and the decision at every frame can use the whole episode, including its future. That is the
    extra information worth spending when labeling, and it is unavailable to the causal head at
    deployment time.
    """
    length = log_probs.shape[0]
    neg_inf = -1e30

    # Every episode starts before the critical phase.
    score = np.array([log_probs[0, PRE], neg_inf, neg_inf], dtype=np.float64)
    backpointer = np.zeros((length, NUM_PHASES), dtype=np.int8)
    # Where a switch into each phase can come from; `pre` can only ever be entered at frame 0.
    switch_from = np.array([PRE, PRE, CRITICAL])

    for t in range(1, length):
        switch = np.array([neg_inf, score[PRE], score[CRITICAL]]) - transition_penalty
        take_switch = switch > score
        backpointer[t] = np.where(take_switch, switch_from, np.arange(NUM_PHASES))
        score = np.where(take_switch, switch, score) + log_probs[t]

    path = np.empty(length, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(length - 1, 0, -1):
        path[t - 1] = backpointer[t, path[t]]

    critical = np.flatnonzero(path == CRITICAL)
    if critical.size == 0:
        return None
    return int(critical[0]), int(critical[-1])
