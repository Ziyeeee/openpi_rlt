"""RLT Stage B: the lightweight actor and critic that refine the VLA's action chunks.

Implements Sec. IV-B of the RLT paper. Both networks are small MLPs over the RL state
`x = (z_rl, s_p)`; neither touches the VLA, which is frozen and whose outputs (`z_rl` and the
reference chunk `a_ref`) arrive precomputed from the Stage A.5 cache.

Critic (Eq. 3):
    L_Q = E[(Qhat - Q_psi(x, a_1:C))^2]
    Qhat = sum_{t'=1..C} gamma^(t'-1) r_t'  +  gamma^C (1 - done) * min_i Q_psi'_i(x', a'),  a' ~ pi

Actor (Eq. 4/5):
    pi_theta(a | x, a_ref) = N(mu_theta(x, a_ref), sigma^2 I)
    L_pi = E[-Q_psi(x, a) + beta * ||a - a_ref||^2],  a = mu + sigma * eps   (reparameterized)

Note the actor is *not* a residual head: the paper conditions on `a_ref` directly and pulls the
action toward it with the beta term, which is what makes reference-action dropout possible.
"""

import dataclasses
import itertools

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class RLTActorCriticConfig:
    # Dimensions of the cached RL state and of the action chunk.
    rl_token_dim: int = 2048
    state_dim: int = 32
    action_dim: int = 32
    chunk_size: int = 16  # C

    # Appendix B: two-layer MLP, hidden 256 (three layers / 512 for the hardest task).
    hidden_dim: int = 512
    depth: int = 3

    # Not stated in the paper; standard values, exposed so they can be swept.
    gamma: float = 0.99
    actor_std: float = 0.1  # sigma, "a small fixed standard deviation"
    # beta in Eq. 5. Offline (no fresh rollouts) this term is the only thing keeping the critic from
    # being queried far off the data distribution: with beta=0 the value estimate diverges outright.
    # 1.0 is a stable starting point, not a tuned value -- it should be revisited once a critic
    # trained on real rollouts is actually providing useful guidance.
    bc_beta: float = 0.2
    target_tau: float = 0.001

    # Appendix B: the reference chunk is masked out for 50% of training transitions and always
    # provided at inference.
    reference_dropout: float = 0.5

    def create(self, rng: at.KeyArrayLike) -> "RLTActorCritic":
        return RLTActorCritic(self, rngs=nnx.Rngs(rng))


class MLP(nnx.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int, *, rngs: nnx.Rngs):
        dims = [in_dim] + [hidden_dim] * depth
        self.layers = [nnx.Linear(a, b, rngs=rngs) for a, b in itertools.pairwise(dims)]
        self.out = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = nnx.swish(layer(x))
        return self.out(x)


class RLTActorCritic(nnx.Module):
    def __init__(self, config: RLTActorCriticConfig, rngs: nnx.Rngs):
        self.config = config
        chunk_dim = config.chunk_size * config.action_dim
        in_dim = config.rl_token_dim + config.state_dim + chunk_dim

        self.actor = MLP(in_dim, config.hidden_dim, chunk_dim, config.depth, rngs=rngs)
        self.critic_1 = MLP(in_dim, config.hidden_dim, 1, config.depth, rngs=rngs)
        self.critic_2 = MLP(in_dim, config.hidden_dim, 1, config.depth, rngs=rngs)
        # Twin targets, initialized to exactly match the online critics.
        self.target_critic_1 = MLP(in_dim, config.hidden_dim, 1, config.depth, rngs=rngs)
        self.target_critic_2 = MLP(in_dim, config.hidden_dim, 1, config.depth, rngs=rngs)
        nnx.update(self.target_critic_1, nnx.state(self.critic_1, nnx.Param))
        nnx.update(self.target_critic_2, nnx.state(self.critic_2, nnx.Param))

    def _features(self, z: jnp.ndarray, state: jnp.ndarray, chunk: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([z, state, chunk.reshape(chunk.shape[0], -1)], axis=-1)

    @at.typecheck
    def actor_mean(
        self,
        z: at.Float[at.Array, "b d"],
        state: at.Float[at.Array, "b s"],
        a_ref: at.Float[at.Array, "b c a"],
    ) -> at.Float[at.Array, "b c a"]:
        mu = self.actor(self._features(z, state, a_ref))
        return mu.reshape(a_ref.shape)

    @at.typecheck
    def sample(
        self,
        rng: at.KeyArrayLike,
        z: at.Float[at.Array, "b d"],
        state: at.Float[at.Array, "b s"],
        a_ref: at.Float[at.Array, "b c a"],
        *,
        reference_dropout: bool = False,
    ) -> tuple[at.Float[at.Array, "b c a"], at.Float[at.Array, "b c a"]]:
        """Reparameterized sample a = mu + sigma * eps, so critic gradients reach the actor.

        Returns both `a` and `mu`: the Q term needs the sample (Q is nonlinear, so Q(x, mu) is not
        E_a[Q(x, a)]), while the BC term wants the mean -- see `actor_loss`.
        """
        dropout_rng, noise_rng = jax.random.split(rng)
        conditioning = a_ref
        if reference_dropout and self.config.reference_dropout > 0.0:
            # Zero the reference for a random subset of the batch. Without this the actor can just
            # copy a_ref -- both the conditioning and the beta term push it that way -- and never
            # builds an independent action pathway of its own.
            keep = jax.random.bernoulli(dropout_rng, 1.0 - self.config.reference_dropout, (a_ref.shape[0], 1, 1))
            conditioning = a_ref * keep
        mu = self.actor_mean(z, state, conditioning)
        return mu + self.config.actor_std * jax.random.normal(noise_rng, mu.shape), mu

    @at.typecheck
    def deterministic_action(
        self,
        z: at.Float[at.Array, "b d"],
        state: at.Float[at.Array, "b s"],
        a_ref: at.Float[at.Array, "b c a"],
    ) -> at.Float[at.Array, "b c a"]:
        """The actor mean, with no exploration noise added.

        Appendix B: the reference chunk is "always provided during inference", which is why there is
        no dropout here. Collection on the robot instead executes `sample`, whose `actor_std` noise
        is the only thing that ever visits a state under more than one action -- and therefore the
        only thing that can teach the critic to depend on the action at all.
        """
        return self.actor_mean(z, state, a_ref)

    def _q_min(self, z: jnp.ndarray, state: jnp.ndarray, chunk: jnp.ndarray, *, target: bool) -> jnp.ndarray:
        feats = self._features(z, state, chunk)
        c1, c2 = (self.target_critic_1, self.target_critic_2) if target else (self.critic_1, self.critic_2)
        return jnp.minimum(c1(feats)[..., 0], c2(feats)[..., 0])

    def td_target(self, rng: at.KeyArrayLike, batch: dict) -> jnp.ndarray:
        """Eq. 3, chunk-level n-step target. Fully stop-gradient'd."""
        c = self.config.chunk_size
        # sum_{t'=1..C} gamma^(t'-1) r_t'
        discounts = self.config.gamma ** jnp.arange(c, dtype=jnp.float32)
        chunk_return = jnp.sum(batch["reward"] * discounts, axis=-1)

        # a' ~ pi(.|x', a_ref'). No reference dropout here: this is an inference-time query of the
        # policy, and Appendix B says the reference is always provided at inference.
        a_next, _ = self.sample(rng, batch["z_next"], batch["state_next"], batch["a_ref_next"])
        # Clipped double-Q over the twin targets, which is what the ensemble is for.
        q_next = self._q_min(batch["z_next"], batch["state_next"], a_next, target=True)

        bootstrap = self.config.gamma**c * (1.0 - batch["done"].astype(jnp.float32)) * q_next
        return jax.lax.stop_gradient(chunk_return + bootstrap)

    def critic_loss(self, rng: at.KeyArrayLike, batch: dict) -> tuple[jnp.ndarray, dict]:
        target = self.td_target(rng, batch)
        feats = self._features(batch["z"], batch["state"], batch["action"])
        q1 = self.critic_1(feats)[..., 0]
        q2 = self.critic_2(feats)[..., 0]
        loss = jnp.mean(jnp.square(q1 - target)) + jnp.mean(jnp.square(q2 - target))

        td_error = 0.5 * (q1 + q2) - target
        info = {
            "critic_loss": loss,
            "mean_q": jnp.mean(0.5 * (q1 + q2)),
            "target_q_mean": jnp.mean(target),
            "td_error_mean": jnp.mean(td_error),
            "td_error_std": jnp.std(td_error),
        }
        return loss, info

    def actor_loss(self, rng: at.KeyArrayLike, batch: dict) -> tuple[jnp.ndarray, dict]:
        a_ref = batch["a_ref"]
        action, mu = self.sample(rng, batch["z"], batch["state"], a_ref, reference_dropout=True)
        # Eq. 5 writes a single Q; with twin critics the min is the conservative choice and keeps the
        # actor from exploiting whichever critic happens to overestimate.
        q = self._q_min(batch["z"], batch["state"], action, target=False)

        # The regularizer always targets the true reference, even where the actor's input was masked.
        # Measured from `mu`, not the sample: E_eps ||mu + sigma*eps - a_ref||^2 = ||mu - a_ref||^2 +
        # sigma^2*C*A.
        delta = mu - a_ref
        q_term = -jnp.mean(q)
        bc_term = jnp.mean(jnp.sum(jnp.square(delta), axis=(-2, -1)))
        loss = q_term + self.config.bc_beta * bc_term

        info = {
            "actor_loss": loss,
            "actor_q_term": q_term,
            "actor_bc_term": bc_term,
            "mean_action_delta": jnp.mean(jnp.abs(delta)),
            "max_action_delta": jnp.max(jnp.abs(delta)),
        }
        return loss, info


def soft_update_targets(model: RLTActorCritic, tau: float) -> None:
    """psi' <- tau * psi + (1 - tau) * psi', in place on the merged model."""
    for online, target in ((model.critic_1, model.target_critic_1), (model.critic_2, model.target_critic_2)):
        online_state = nnx.state(online, nnx.Param)
        target_state = nnx.state(target, nnx.Param)
        nnx.update(target, jax.tree.map(lambda t, o: (1.0 - tau) * t + tau * o, target_state, online_state))
