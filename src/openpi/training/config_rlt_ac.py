"""Stage B training configs for RLT: the online actor and critic.

Stage B never touches the VLA or a LeRobot dataset -- it reads the Stage A.5 cache -- so it has its
own small TrainConfig rather than reusing the observation/action-shaped one from `config.py`.
"""

import dataclasses
import difflib
import pathlib

import tyro

from openpi.models import rlt_actor_critic
import openpi.training.optimizer as _optimizer


@dataclasses.dataclass(frozen=True)
class CacheSource:
    """One Stage A.5 cache plus the clip labels that select and score its frames."""

    # Directory written by scripts/cache_rl_token.py.
    cache_dir: str
    # Hand-reviewed critical-phase clips with a success/fail label per clip. RL runs on these spans
    # only (Sec. V), and they are what supplies the reward -- Stage B has no other outcome signal.
    clips_path: str
    clip_tag: str = "plug in"


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi_rlt"
    exp_name: str = tyro.MISSING

    model: rlt_actor_critic.RLTActorCriticConfig = dataclasses.field(
        default_factory=rlt_actor_critic.RLTActorCriticConfig
    )
    # Caches to train on, sampled uniformly as one pool. Off-policy learning does not care which
    # policy produced a transition (§IV), so rollouts and demonstrations can be mixed freely.
    # Change the mixture by adding a config below rather than overriding this on the command line.
    sources: tuple[CacheSource, ...] = ()

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)

    # Critic Q warmup: the actor is held at its initialization until the critic carries signal.
    # Starting policy improvement from a random critic just pushes the actor into garbage.
    critic_warmup_steps: int = 2_000
    # Appendix B: two critic updates for each actor update.
    critic_updates_per_actor_update: int = 2

    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 512
    num_workers: int = 4
    num_train_steps: int = 50_000
    log_interval: int = 1000
    save_interval: int = 10_000
    keep_period: int | None = 20_000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_CONFIGS = [
    TrainConfig(
        name="rlt_pi05_ac",
        model=rlt_actor_critic.RLTActorCriticConfig(
            rl_token_dim=2048,  # gemma_rlt width
            state_dim=32,
            action_dim=32,
            chunk_size=10,
            hidden_dim=512,
            depth=3,
        ),
        sources=(
            # Rollouts of the Stage A policy, not teleoperated demos: `a_demo` is what
            # the policy actually executed and the clips carry its real outcome (33 success /
            # 17 fail), which is what makes the critic's targets meaningful. All of Stage B's
            # discriminative signal comes from here.
            CacheSource(
                cache_dir="data/agilex/my_task_rollout/rl_cache",
                clips_path="data/agilex/my_task_rollout/clips_by_tag.json",
            ),
            # Teleop demonstrations, 149 clips and all of them successes. Measured against
            # rollout-only on a held-out split, this leaves the ceiling alone (AUC@late 0.856 ->
            # 0.917, within noise) but cuts seed variance 4.6x (sd 0.139 -> 0.030): rollout-only has
            # seeds that collapse to 0.56, and with this mixed in every seed clears 0.86. It
            # regularizes V(s); it adds no outcome signal, because held-out discrimination scales
            # with the number of *failure* clips and is flat in successes. `rlt_pi05_ac_rollout` is
            # the ablation twin without it.
            CacheSource(
                cache_dir="data/agilex/my_task/rl_cache",
                clips_path="data/agilex/my_task/clips_by_tag.json",
            ),
        ),
        # 32037 labeled transitions, so 20k steps at batch 256 is ~160 epochs of a 2-layer MLP.
        # The original 50k was sized for the 84k-frame full-episode cache.
        num_train_steps=20_000,
        save_interval=10_000,
        keep_period=25_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
    ),
]

_CONFIGS.append(
    dataclasses.replace(
        _CONFIGS[0],
        name="rlt_pi05_ac_c12",
        model=dataclasses.replace(_CONFIGS[0].model, chunk_size=12),
    )
)

# Same mixture at C = 16, the chunk length Stage C runs online -- about half a second at 30 Hz, and
# an exact multiple of Appendix B's subsampling stride of 2 so that `x_(t+C)` is itself a stored
# row. This exists to produce a warm start the online server can load: the actor/critic input is
# 2048 + 32 + C * 32, so a C = 10 checkpoint is a different shape (2400 vs 2560) and
# `rlt_pi05_online --ac_params_path` would reject it.
_CONFIGS.append(
    dataclasses.replace(
        _CONFIGS[0],
        name="rlt_pi05_ac_c16",
        model=dataclasses.replace(_CONFIGS[0].model, chunk_size=16),
    )
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
