"""Stage C training configs for RLT: online RL on the robot (Algorithm 1, lines 6-18).

Stage C is a Stage A checkpoint, a Stage A.6 handover head and a Stage B actor/critic running in
one server process, so its config is mostly references to the other three registries plus the
knobs that only exist once a robot is in the loop: the update-to-data ratio, the warmup length,
the handover latch, and the safety bound on how far the actor may edit the VLA's chunk.

The Stage B config is *embedded* rather than restated, so `scripts/train_rlt_ac.py:train_step`
can be imported and used unchanged by the online learner -- it is `@at.typecheck`'d against that
exact type, and one train step is one train step whether the batch came from a cache or a robot.
"""

import dataclasses
import difflib
import pathlib

import tyro

from openpi.models import rlt_handover
import openpi.training.config_rlt_ac as _config_rlt_ac
import openpi.training.optimizer as _optimizer


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi_rlt"
    exp_name: str = tyro.MISSING

    # -- the frozen half -------------------------------------------------------------------------
    # Stage A registry name; supplies the model shape, the data transforms and the norm stats.
    stage_a: str = "rlt_pi05"
    stage_a_dir: str = "checkpoints/rlt_pi05/my_run/9999"
    # Stage A.6 head, grafted onto the Stage A tree at load time. `rlt_pi05_handover_all` is the
    # one to deploy: it pools all 199 hand-labeled episodes and, unlike the CV folds, includes the
    # rollouts, which are the visual distribution the head actually runs on.
    handover_dir: str = "checkpoints/rlt_pi05_handover_all/all/4999"
    handover: rlt_handover.RLTHandoverConfig = dataclasses.field(default_factory=rlt_handover.RLTHandoverConfig)

    # -- the learning half -----------------------------------------------------------------------
    # Two edits to the Stage B config it is taken from:
    #   `exp_name` is filled in because Stage B's is `tyro.MISSING` and would otherwise be demanded
    #   on the command line; Stage C checkpoints under its own `exp_name`, so the value is unused.
    #   `lr_schedule` becomes a flat 1e-4. Stage B cosine-decays over a known `num_train_steps`,
    #   but an online session has no horizon -- it runs as long as the robot collects -- so a decay
    #   would anneal to `decay_lr` at an arbitrary wall-clock and stay there for the rest of the
    #   session, with the actor and critic frozen exactly when the fresh data starts arriving.
    ac: _config_rlt_ac.TrainConfig = dataclasses.field(
        default_factory=lambda: dataclasses.replace(
            _config_rlt_ac.get_config("rlt_pi05_ac_c16"),
            exp_name="online",
            lr_schedule=_optimizer.ConstantSchedule(lr=1e-4),
        )
    )
    # Optional Stage B warm start. Must have been trained at the same `chunk_size`: the actor input
    # is rl_token_dim + state_dim + C * action_dim, so a C = 10 checkpoint is a different shape.
    ac_params_path: str | None = None
    # Alg. 1's N_warm pre-fill, taken from the Stage B mixture. Without it the critic is random on
    # the first rollout and there is nothing for the actor to climb.
    preload: tuple[_config_rlt_ac.CacheSource, ...] = ()

    # -- collection ------------------------------------------------------------------------------
    # C, the RL action chunk length. Independent of the controller's replan rate: rows come from
    # the post-episode ingest pass, not from queries, so the client can replan at whatever cadence
    # suits control.
    chunk_size: int = 16
    # Appendix B: "we pick a stride of 2 and save transitions corresponding to <x_0, a_0:C>,
    # <x_2, a_2:C+2>, ...". Control steps between stored rows, so at 30 Hz this is 15 rows/s from
    # each second of critical phase -- an 8x denser buffer than one row per policy query.
    # `chunk_size` must be a multiple of it, or `x_{t+C}` is not itself a stored row.
    subsample_stride: int = 2
    # Rows per ingest message. Also the XLA batch size of the post-episode VLA pass, so that one
    # message is one compiled shape and one bounded round trip (~1.5 s at 32).
    ingest_batch_size: int = 32
    # RTC delay when drawing `a_ref` during ingest, using the executed chunk's own leading steps as
    # the committed prefix. Matches `cache_rl_token.py --reference_delay`, so online and offline
    # `a_ref` are produced identically.
    reference_delay: int = 4
    # Cap on rows buffered per episode, ~40 s of critical phase at stride 2. Truncation is logged,
    # never silent.
    max_rows_per_episode: int = 600
    # Alg. 1 line 9: execute the VLA reference until this many online transitions exist. 0 because
    # `preload` already gives the critic warmup data; raise it to reproduce the paper's cold start.
    warmup_transitions: int = 0
    # Consecutive `critical` predictions before the handover latches. 2 at a ~0.5 s query period is
    # a ~1 s commitment -- enough to reject a single noisy reading, short next to a critical phase.
    handover_patience: int = 2
    # Safety filter: L-inf bound on |a - a_ref| in normalized space, so the actor edits the VLA's
    # chunk rather than replacing it. `None` disables it. 0.5 is ~2.5 exploration sigmas.
    max_action_delta: float | None = 0.5
    # Evaluation mode: `deterministic_action` instead of a sample. Off during collection, because
    # the exploration noise is the only source of action contrast in the data (`docs/rlt_wip.md`).
    deterministic: bool = False
    # Never engage the actor: pure VLA rollout collection, which is Alg. 1's warmup phase and what
    # `serve_policy_rlt.py` used to do unconditionally. Also how to watch the handover head alone.
    collect_only: bool = False
    num_denoise_steps: int = 10
    policy_hz: float = 30.0
    # A norm_stats.json to use instead of the one bundled with the Stage A checkpoint.
    asset_path: str | None = None
    # Used when the client omits "prompt". The Stage A data config's own `default_prompt` is
    # already applied further down the transform chain, so this is only an override.
    default_prompt: str | None = None

    # -- learner ---------------------------------------------------------------------------------
    # Appendix B: "a high update-to-data ratio of 5, which is essential in the low-data online
    # regime". The learner sleeps whenever it is ahead of this.
    updates_per_env_step: float = 5.0
    # Transitions in the buffer before the learner starts. Cleared instantly by `preload`.
    learning_starts: int = 1_000
    # Learner steps between actor-parameter publishes to the collector. At ~10 updates/s this is a
    # fresh policy every few seconds, which is far finer than the episode cadence.
    publish_interval: int = 50
    buffer_capacity: int = 200_000

    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    save_interval: int = 5_000
    keep_period: int | None = 20_000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    port: int = 8010

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def buffer_dir(self) -> pathlib.Path:
        """Online clips, next to the checkpoints. Reloaded on --resume, and usable as a Stage B
        cache afterwards without re-running the robot."""
        return self.checkpoint_dir / "buffer"

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.ac.model.chunk_size != self.chunk_size:
            raise ValueError(
                f"chunk_size={self.chunk_size} but the embedded Stage B config uses "
                f"{self.ac.model.chunk_size}; the actor input width depends on it"
            )
        if self.chunk_size % self.subsample_stride != 0:
            # Rows sit `subsample_stride` frames apart, so a transition's successor `x_{t+C}` is a
            # stored row only when the stride divides C. Otherwise the TD backup's gamma^C would be
            # charged over a gap that is off by up to one control step.
            raise ValueError(
                f"chunk_size={self.chunk_size} must be a multiple of subsample_stride="
                f"{self.subsample_stride}, or x_(t+C) is not one of the stored rows"
            )
        if self.reference_delay >= self.chunk_size:
            raise ValueError(f"reference_delay={self.reference_delay} must be < chunk_size={self.chunk_size}")


_CONFIGS = [
    TrainConfig(
        name="rlt_pi05_online",
        # The same mixture Stage B trains on: 50 labeled rollout clips carry all the outcome
        # signal, and the 149 all-success teleop clips cut seed variance 4.6x by regularizing V(s).
        preload=_config_rlt_ac.get_config("rlt_pi05_ac_c16").sources,
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
