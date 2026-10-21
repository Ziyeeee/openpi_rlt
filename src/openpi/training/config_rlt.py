"""Stage A training configs for RLT: co-train the RL token with the pi0.5 VLA.

Stage A needs nothing beyond observations and actions, so it reuses `openpi.training.config`
wholesale (TrainConfig, the LeRobot data configs, the standard PI05 model transforms) and only
adds its own `_CONFIGS` registry. Stage B lives in `config_rlt_ac.py`.
"""

import difflib

import tyro

from openpi.models import rlt_config
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import LeRobotAlohaDataConfig
from openpi.training.config import TrainConfig
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

_AGILEX_REPACK = _transforms.Group(
    inputs=[
        _transforms.RepackTransform(
            {
                "images": {
                    "cam_high": "observation.images.front",
                    "cam_left_wrist": "observation.images.left",
                    "cam_right_wrist": "observation.images.right",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            }
        )
    ]
)

_CONFIGS = [
    TrainConfig(
        name="rlt_pi05",
        # RTC is on by default (RLTConfig.rtc): the robot executes chunks under real-time chunking,
        # so the reference chunks the Stage B actor refines must be produced the same way.
        # alpha = 1.0: the VLA is fine-tuned on the task alongside the RL token (Alg. 1 line 3).
        model=rlt_config.RLTConfig(vla_loss_weight=0.5, decoder_context_dropout=0.2),
        data=LeRobotAlohaDataConfig(
            # `agilex/my_task` is the dataset with hand-labeled critical-phase spans
            # (`data/agilex/my_task/annotations.json`), which Stage A.6 trains the handover
            # predictor on. Keeping every RLT stage on one dataset keeps the caches consistent.
            repo_id="agilex/my_task",
            assets=AssetsConfig(assets_dir="assets", asset_id="agilex/my_task"),
            default_prompt="Pull out the charging gun and plug it back in",
            gripper_type="agilex",
            arm_action_mask=_transforms.make_bool_mask(7, -7),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_AGILEX_REPACK,
        ),
        weight_loader=weight_loaders.RLTWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=32,
        num_workers=2,
        # Appendix B: the RL token is trained for 2000-10000 gradient steps on single-task data.
        num_train_steps=10_000,
        save_interval=2_000,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2e-5,
            decay_steps=10_000,
            decay_lr=1e-6,
        ),
    ),
]

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
