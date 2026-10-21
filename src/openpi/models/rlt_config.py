import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import rlt_handover
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.rlt import RLT
    from openpi.models.rlt_rtc import RLTRTC


@dataclasses.dataclass(frozen=True)
class RLTConfig(_model.BaseModelConfig):
    """Stage A of RLT: a pi0.5 VLA with an RL-token encoder/decoder attached.

    The encoder compresses the VLA's final-layer prefix embeddings into a single token `z_rl`; the
    decoder must be able to autoregressively reconstruct those embeddings from `z_rl` alone, which
    makes `z_rl` an information bottleneck. Trained jointly with the VLA's own flow-matching loss
    (RLT paper Alg. 1 line 3: argmin L_ro + alpha * L_vla).

    `create()` returns `RLTRTC` by default (Real-Time Chunking) and `RLT` when `rtc=False`; the two
    differ only in the action expert, following the same split as `Pi0`/`Pi0RTC`.
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    # Backbone for both the RL-token encoder and decoder. Width must match the PaliGemma width so
    # that prefix embeddings can be fed in without a projection.
    rlt_variant: _gemma.Variant = "gemma_rlt"

    # Set the model specific defaults. RLT is pi0.5 only: adaRMS timestep injection and the
    # discretized state carried in the language tokens (hence max_token_len 200).
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    # Not used by the model itself; read by the ModelTransformFactory to tokenize state into the prompt.
    discrete_state_input: bool = True

    # alpha in Alg. 1 line 3: weight on the VLA's own action loss. > 0 means the VLA is co-trained
    # with the RL token; 0 means the VLA is frozen and only the RL-token branch is learned.
    vla_loss_weight: float = 1.0
    # Which prefix embeddings the decoder must reconstruct. The paper drops language embeddings
    # (footnote 1) because their tasks use a fixed instruction, but in pi0.5 the discretized robot
    # state lives in the language tokens, so reconstructing the whole prefix is the better default.
    recon_target: Literal["prefix", "images"] = "images"
    # Fraction of the decoder's teacher-forced context tokens to zero out during training. With full
    # teacher forcing the autoregressive decoder can predict z_i from sg(z_1:i-1) alone and ignore
    # z_rl, which leaves the RL token uninformative. Raise this if z_rl collapses.
    decoder_context_dropout: float = 0.0

    # RL chunk length C used downstream by the actor and critic (Appendix B: C = 10). The VLA itself
    # still predicts action_horizon = H = 50; C < H makes the RL policy more reactive.
    rl_chunk_size: int = 10

    # Stage A.6 handover predictor. `None` -- the default -- builds no head, which is what Stage A
    # trains: the head is fit separately against the Stage A.5 cache and never sees a gradient from
    # the VLA objective, so leaving it out keeps the Stage A parameter tree unchanged. Set it (and
    # load `params_with_handover`) for deployment, where one call then returns the action chunk,
    # `z_rl`, and the handover logits off a single prefix forward.
    handover: rlt_handover.RLTHandoverConfig | None = None

    # Real-Time Chunking, on by default for RLT. RTC only affects the action expert (the first
    # `delay` actions of a chunk enter as already-executing ground truth), which is the regime the
    # robot actually runs in -- so the reference chunks the actor learns to refine should be produced
    # the same way. The RL token is read off the PaliGemma prefix and is unaffected either way.
    # Set rtc=False for an ablation against a plain pi0.5 VLA.
    rtc: bool = True
    max_delay: int = 6  # matches the `pi05_rtc_agilex_right` default

    def __post_init__(self):
        if self.rl_chunk_size > self.action_horizon:
            raise ValueError(f"rl_chunk_size={self.rl_chunk_size} must be <= action_horizon={self.action_horizon}")
        if self.max_delay > self.action_horizon:
            raise ValueError(f"max_delay={self.max_delay} must be <= action_horizon={self.action_horizon}")
        if self.handover is not None and (
            self.handover.rl_token_dim != self.rl_token_dim or self.handover.state_dim != self.action_dim
        ):
            raise ValueError(
                f"handover head expects ({self.handover.rl_token_dim}, {self.handover.state_dim}) but this config "
                f"produces z_rl of {self.rl_token_dim} and states of {self.action_dim}"
            )

    @property
    def rl_token_dim(self) -> int:
        return _gemma.get_config(self.rlt_variant).width

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "RLT | RLTRTC":
        if self.rtc:
            from openpi.models.rlt_rtc import RLTRTC

            return RLTRTC(self, rngs=nnx.Rngs(rng))
        from openpi.models.rlt import RLT

        return RLT(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the VLA when alpha == 0, so only the RL-token branch trains."""
        if self.vla_loss_weight == 0.0:
            return nnx.All(nnx.Not(nnx_utils.PathRegex(".*rlt_.*|.*e_rl.*|.*rl_out_proj.*")))
        return nnx.Nothing
