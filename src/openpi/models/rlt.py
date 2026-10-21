"""RLT (RL Token) Stage A: pi0.5 with an RL-token encoder/decoder attached.

Implements Sec. IV-A of "RL Token: Bootstrapping Online RL with Vision-Language-Action Models".
A small encoder transformer reads the VLA's final-layer prefix embeddings plus a learned `<rl>`
query and emits a single token `z_rl` (Eq. 1). A small causal decoder must reconstruct those same
embeddings from `z_rl` alone (Eq. 2), which forces `z_rl` to act as an information bottleneck that
summarizes what the downstream actor and critic need to know about the observation.

The reconstruction targets are stop-gradient'd: the VLA is frozen *with respect to* L_ro. The VLA
still receives gradient from its own flow-matching loss, weighted by `vla_loss_weight` (alpha in
Alg. 1 line 3), which is what makes this a co-training rather than a two-phase recipe.

This file is a fork of `pi0.py`, with the RTC variant kept in `rlt_rtc.py`.
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import rlt_config
from openpi.models import rlt_handover as _rlt_handover
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    input_mask = jnp.asarray(input_mask, dtype=jnp.bool_)
    mask_ar = jnp.asarray(mask_ar, dtype=jnp.bool_)

    # Most call sites pass a static 1D autoregressive mask. Evaluate its cumsum at
    # trace time before broadcasting so XLA does not spend time constant-folding a
    # batch-sized reduce-window during compilation.
    if mask_ar.ndim == 1:
        with jax.ensure_compile_time_eval():
            block_id = jnp.cumsum(mask_ar.astype(jnp.int32), axis=0)
        block_id = jnp.broadcast_to(block_id[None, :], input_mask.shape)
    else:
        mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
        block_id = jnp.cumsum(mask_ar.astype(jnp.int32), axis=1)

    attn_mask = block_id[:, :, None] >= block_id[:, None, :]
    valid_mask = input_mask[:, :, None] & input_mask[:, None, :]
    return attn_mask & valid_mask


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@at.typecheck
def _effective_rank(z: at.Float[at.Array, "b d"]) -> at.Float[at.Array, ""]:
    """Participation ratio of the batch-centered tokens: how many directions the batch spans.

    A mean pairwise cosine similarity -- the obvious diagnostic -- does not work here. Uncentered
    it is dominated by the constant offset shared by every frame (it reads ~0.99 whether or not the
    bottleneck is healthy), and centered it averages to ~0 in *every* case, because a rank-1 batch
    has residuals +/-d whose cosines cancel by sign. The participation ratio has no such blind spot.

    Ranges over [1, b], except for an exactly constant batch where the epsilon dominates and it
    reads 0; that case is the one `z_rl_batch_std` already flags.
    """
    centered = z - jnp.mean(z, axis=0, keepdims=True)
    gram = centered @ centered.T
    return jnp.trace(gram) ** 2 / (jnp.sum(jnp.square(gram)) + 1e-8)


class RLT(_model.BaseModel):
    def __init__(self, config: rlt_config.RLTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.recon_target = config.recon_target
        self.decoder_context_dropout = config.decoder_context_dropout
        self.vla_loss_weight = config.vla_loss_weight

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # --- RL token branch (Sec. IV-A) ---
        rlt_cfg = _gemma.get_config(config.rlt_variant)
        if rlt_cfg.width != paligemma_config.width:
            raise ValueError(
                f"rlt_variant width {rlt_cfg.width} must match paligemma width {paligemma_config.width}; "
                "the RLT transformers consume PaliGemma prefix embeddings directly."
            )
        # `use_embedder=False`: these transformers only ever see embeddings, never token ids.
        rlt_encoder = nnx_bridge.ToNNX(_gemma.Module(configs=[rlt_cfg], embed_dtype=config.dtype, use_embedder=False))
        rlt_encoder.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        rlt_decoder = nnx_bridge.ToNNX(_gemma.Module(configs=[rlt_cfg], embed_dtype=config.dtype, use_embedder=False))
        rlt_decoder.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        self.rlt_encoder = rlt_encoder
        self.rlt_decoder = rlt_decoder
        # The learned `<rl>` query embedding e_rl. The encoder output at its position is the RL token.
        self.e_rl = nnx.Param(jax.random.normal(rngs.params(), (1, 1, rlt_cfg.width)) * 0.02)
        # h_theta: linear readout mapping decoder states back to VLA embedding space.
        self.rl_out_proj = nnx.Linear(rlt_cfg.width, rlt_cfg.width, rngs=rngs)

        # --- Handover predictor (Stage A.6, Sec. V) ---
        # Trained separately against the cached RL tokens; present here only so that deployment gets
        # actions, `z_rl` and the handover logits out of one call. `None` in Stage A.
        self.handover = (
            _rlt_handover.RLTHandoverHead(config.handover, rngs=rngs) if config.handover is not None else None
        )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        # pi0.5 puts the discretized state in the language tokens, so there is no continuous state token.
        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # time MLP (for adaRMS)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # --- RL token (Sec. IV-A) ---

    @at.typecheck
    def encode_rl_token(
        self, prefix_out: at.Float[at.Array, "b s emb"], prefix_mask: at.Bool[at.Array, "b s"]
    ) -> at.Float[at.Array, "b emb"]:
        """Eq. 1: z_rl = g_theta([z_1:M, e_rl])_{M+1}, with the VLA frozen w.r.t. this branch."""
        z = jax.lax.stop_gradient(prefix_out)
        batch_size, seq_len, _ = z.shape

        e_rl = jnp.broadcast_to(self.e_rl.value, (batch_size, 1, z.shape[-1])).astype(z.dtype)
        tokens = jnp.concatenate([z, e_rl], axis=1)
        mask = jnp.concatenate([prefix_mask, jnp.ones((batch_size, 1), dtype=jnp.bool_)], axis=1)
        # Bidirectional: the readout query and every VLA embedding see each other.
        attn_mask = make_attn_mask(mask, jnp.zeros(seq_len + 1, dtype=jnp.bool_))
        positions = jnp.cumsum(mask.astype(jnp.int32), axis=1) - 1

        (encoded,), _ = self.rlt_encoder([tokens], mask=attn_mask, positions=positions)
        return encoded[:, -1].astype(jnp.float32)

    def _recon_targets(
        self, prefix_out: jnp.ndarray, prefix_mask: jnp.ndarray, obs: _model.Observation
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Which prefix embeddings the decoder must reconstruct."""
        if self.recon_target == "images":
            # The paper drops language embeddings (footnote 1). Image tokens are the leading block.
            n_image = prefix_out.shape[1] - (obs.tokenized_prompt.shape[1] if obs.tokenized_prompt is not None else 0)
            return prefix_out[:, :n_image], prefix_mask[:, :n_image]
        return prefix_out, prefix_mask

    @at.typecheck
    def decode_from_rl_token(
        self,
        rng: at.KeyArrayLike,
        z_rl: at.Float[at.Array, "b emb"],
        targets: at.Float[at.Array, "b s emb"],
        target_mask: at.Bool[at.Array, "b s"],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "b s emb"]:
        """Teacher-forced decode: position i sees [z_rl, sg(z_1), ..., sg(z_{i-1})] and predicts z_i."""
        batch_size, seq_len, _ = targets.shape

        context = targets[:, :-1]
        if train and self.decoder_context_dropout > 0.0:
            # Without this the decoder can reconstruct z_i from its context alone and ignore z_rl,
            # which leaves the bottleneck -- and therefore the RL state -- uninformative.
            keep = jax.random.bernoulli(rng, 1.0 - self.decoder_context_dropout, (batch_size, seq_len - 1, 1))
            context = context * keep

        tokens = jnp.concatenate([z_rl[:, None].astype(targets.dtype), context], axis=1)
        # The z_rl slot is always valid; the rest inherit the validity of the token they carry.
        mask = jnp.concatenate([jnp.ones((batch_size, 1), dtype=jnp.bool_), target_mask[:, :-1]], axis=1)
        attn_mask = make_attn_mask(mask, jnp.ones(seq_len, dtype=jnp.bool_))  # strictly causal
        positions = jnp.cumsum(mask.astype(jnp.int32), axis=1) - 1

        (decoded,), _ = self.rlt_decoder([tokens], mask=attn_mask, positions=positions)
        return self.rl_out_proj(decoded).astype(jnp.float32)

    @at.typecheck
    def reconstruction_loss(
        self,
        rng: at.KeyArrayLike,
        prefix_out: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        z_rl: at.Float[at.Array, "b emb"],
        obs: _model.Observation,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, " b"]:
        """Eq. 2: autoregressively reconstruct sg(z_1:M) from z_rl through the decoder."""
        targets, target_mask = self._recon_targets(prefix_out, prefix_mask, obs)
        targets = jax.lax.stop_gradient(targets)
        pred = self.decode_from_rl_token(rng, z_rl, targets, target_mask, train=train)

        # Mean (not sum) over tokens and features, so the scale is comparable to the action loss and
        # independent of the prefix length.
        sq_err = jnp.mean(jnp.square(pred - targets.astype(jnp.float32)), axis=-1)
        valid = target_mask.astype(jnp.float32)
        return jnp.sum(sq_err * valid, axis=-1) / jnp.maximum(jnp.sum(valid, axis=-1), 1.0)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, " b"], dict]:
        preprocess_rng, noise_rng, time_rng, recon_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask.astype(jnp.int32), axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=[1, 2])  # (b,)

        # The RL token branch reuses `prefix_out`, which the plain pi0.5 loss discards -- no extra
        # VLA forward pass.
        z_rl = self.encode_rl_token(prefix_out, prefix_mask)
        recon_loss = self.reconstruction_loss(recon_rng, prefix_out, prefix_mask, z_rl, observation, train=train)

        # Counterfactual for the autoregressive leak: reconstruct each frame from *another* frame's
        # RL token. `recon_loss` falling proves the decoder improved, not that `z_rl` is what carried
        # the information -- with teacher forcing the decoder can predict z_i from sg(z_1:i-1) alone.
        # Same rng and `train` flag as above, so the two numbers are a paired comparison. Logging
        # only: this is returned in `info`, which the trainer does not differentiate.
        recon_loss_shuffled = self.reconstruction_loss(
            recon_rng, prefix_out, prefix_mask, jnp.roll(z_rl, 1, axis=0), observation, train=train
        )

        # Alg. 1 line 3: argmin L_ro(theta) + alpha * L_vla(pi_vla).
        loss = recon_loss + self.vla_loss_weight * action_loss
        info = {
            "action_loss": jnp.mean(action_loss),
            "recon_loss": jnp.mean(recon_loss),
            # Should sit well above `recon_loss`. If the two converge, `z_rl` is vestigial and the
            # bottleneck has leaked -- raise `decoder_context_dropout`.
            "recon_loss_shuffled": jnp.mean(recon_loss_shuffled),
            # Spread of the RL token across the batch. `z_rl` leaves the encoder through a RMSNorm,
            # so ||z_rl|| == sqrt(width) by construction and its norm says nothing; this std is
            # therefore already a *fraction* of the token's fixed energy budget. Toward 0 means
            # every frame maps to the same token and downstream RL has nothing to condition on.
            "z_rl_batch_std": jnp.mean(jnp.std(z_rl, axis=0)),
            # Effective rank (participation ratio) of the batch-centered tokens, in [1, batch].
            # `z_rl_batch_std` cannot see the other collapse mode -- every frame varying along a
            # single shared direction still shows a healthy std. Computed from the b x b Gram
            # matrix, so no eigendecomposition: PR = tr(G)^2 / ||G||_F^2.
            "z_rl_eff_rank": _effective_rank(z_rl),
        }
        return loss, info

    def _denoise(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_mask: jnp.ndarray,
        prefix_len: int,
        kv_cache,
        num_steps: int | at.Int[at.Array, ""],
        noise: at.Float[at.Array, "b ah ad"] | None,
    ) -> _model.Actions:
        """Euler-integrate the flow from t=1 (noise) to t=0 (actions) against a cached prefix."""
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (batch_size, suffix_tokens.shape[1], prefix_len + suffix_tokens.shape[1])
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask.astype(jnp.int32), axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask.astype(jnp.int32), axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return self._denoise(rng, observation, prefix_mask, prefix_tokens.shape[1], kv_cache, num_steps, noise)

    @at.typecheck
    def sample_rl_state_and_reference(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> tuple[at.Float[at.Array, "b emb"], _model.Actions]:
        """Return (z_rl, reference action chunk) from a single shared prefix forward pass.

        This is what the Stage A.5 caching script calls: the prefix forward dominates the cost, and
        both the RL state and the VLA reference chunk are derived from it.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask.astype(jnp.int32), axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        z_rl = self.encode_rl_token(prefix_out, prefix_mask)
        actions = self._denoise(rng, observation, prefix_mask, prefix_tokens.shape[1], kv_cache, num_steps, None)
        return jax.lax.stop_gradient(z_rl), jax.lax.stop_gradient(actions)

    @at.typecheck
    def sample_actions_and_handover(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        z_history: at.Float[at.Array, "b k_1 emb"],
        state_history: at.Float[at.Array, "b k_1 ad"],
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> tuple[_model.Actions, at.Float[at.Array, "b emb"], at.Float[at.Array, "b 3"]]:
        """One call, three outputs: the action chunk, `z_rl`, and the pre/critical/post logits.

        `z_history`/`state_history` are the previous `window - 1` policy queries, newest first; this
        call's own `z_rl` and state become the newest window slot. The caller keeps them in a ring
        buffer, the same way it already keeps the in-flight action chunk for RTC.
        """
        if self.handover is None:
            raise ValueError("this model was built with RLTConfig.handover=None, so it has no handover head")
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask.astype(jnp.int32), axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        z_rl = self.encode_rl_token(prefix_out, prefix_mask)
        actions = self._denoise(rng, observation, prefix_mask, prefix_tokens.shape[1], kv_cache, num_steps, None)

        logits = self.handover(
            jnp.concatenate([z_rl[:, None], z_history], axis=1),
            jnp.concatenate([observation.state[:, None], state_history], axis=1),
            observation.state,
        )
        return jax.lax.stop_gradient(actions), jax.lax.stop_gradient(z_rl), jax.lax.stop_gradient(logits)
