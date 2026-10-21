"""Stage A.6 training configs for RLT: the handover predictor.

Like Stage B, this stage never touches the VLA or a LeRobot dataset -- it reads the Stage A.5 cache
plus a hand-labeled `annotations.json` -- so it has its own small TrainConfig rather than reusing
the observation/action-shaped one from `config.py`.
"""

import dataclasses
import difflib
import pathlib

import tyro

from openpi.models import rlt_handover
import openpi.training.optimizer as _optimizer


@dataclasses.dataclass(frozen=True)
class AnnotationSource:
    """One Stage A.5 cache plus the hand-labeled critical-phase spans over its episodes."""

    # Directory written by scripts/cache_rl_token.py.
    cache_dir: str
    # Hand-labeled critical-phase spans, keyed by LeRobot episode index.
    annotations_path: str


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi_rlt"
    exp_name: str = tyro.MISSING

    model: rlt_handover.RLTHandoverConfig = dataclasses.field(default_factory=rlt_handover.RLTHandoverConfig)
    # Caches to train on, pooled into one set of labeled episodes. The head reads only `z_rl` and the
    # state, both of which live in whatever normalization the Stage A checkpoint that produced the
    # cache used, so caches may be mixed exactly when they share that checkpoint -- which teleop demos
    # and rollouts of the same policy do. Change the mixture by adding a config below.
    sources: tuple[AnnotationSource, ...] = ()

    # Grouped cross-validation over the pooled labeled episodes: whole episodes go to one side, so no
    # frames leak. `val_fold=None` trains on every labeled episode -- that is the model to ship to
    # scripts/label_handover.py, once the folds have established the error bars.
    num_folds: int = 5
    val_fold: int | None = 0

    # Stage A params directory. When set, training ends by writing
    # `<checkpoint_dir>/params_with_handover`: the Stage A tree with the trained head grafted in, so
    # deployment loads exactly one checkpoint.
    merge_into: str | None = None

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=200, peak_lr=1e-3, decay_steps=5_000, decay_lr=1e-5
        )
    )
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)

    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 512
    # 0 by default: the dataset is materialized in RAM, and every worker would copy all of it.
    num_workers: int = 0
    num_train_steps: int = 5_000
    log_interval: int = 100
    # Boundary metrics need a Viterbi decode over whole episodes, so they are not free per step.
    eval_interval: int = 1_000
    save_interval: int = 2_500
    keep_period: int | None = None
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
        name="rlt_pi05_handover",
        model=rlt_handover.RLTHandoverConfig(
            rl_token_dim=2048,  # gemma_rlt width
            state_dim=32,
        ),
        sources=(
            AnnotationSource(
                cache_dir="data/agilex/my_task/rl_cache",
                annotations_path="data/agilex/my_task/annotations.json",
            ),
        ),
    ),
]

# Every labeled episode there is: the 149 teleop demos plus the 50 rollouts, both cached
# from the same Stage A checkpoint. This is the config to train the head that Stage B's online RL and
# scripts/label_handover.py actually use -- the rollouts matter because that is the visual
# distribution the head runs on, and the demos are two thirds of the supervision. Defaults to
# `val_fold=None` (train on everything); pass `--val_fold=0..4` to get the pooled error bars first.
_CONFIGS.append(
    dataclasses.replace(
        _CONFIGS[0],
        name="rlt_pi05_handover_all",
        sources=(
            *_CONFIGS[0].sources,
            AnnotationSource(
                cache_dir="data/agilex/my_task_rollout/rl_cache",
                annotations_path="data/agilex/my_task_rollout/annotations.json",
            ),
        ),
        val_fold=None,
    )
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def all_configs() -> dict[str, TrainConfig]:
    return dict(_CONFIGS_DICT)


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
