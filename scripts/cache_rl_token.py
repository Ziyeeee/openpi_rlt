"""RLT Stage A.5: cache the frozen VLA's RL token and reference action chunk for every frame.

Sec. IV-B of the RLT paper freezes the VLA and the RL-token module before any RL happens, so
`z_rl` and the reference chunk `a_ref` are constants of the dataset. Precomputing them once turns
Stage B into a pure MLP training loop over small numpy arrays -- no images, no `Observation`, no
LeRobot at training time.

Every frame is pushed through the *same* transform pipeline the trainer uses, so the cached
actions and states are in exactly the normalized space the actor and critic operate in.

When the Stage A model is RTC (the default), `a_ref` is drawn with prefix conditioning: the first
`--reference_delay` actions enter as already-committed ground truth, taken from the demonstration.
That matches both how Stage A was trained and how the policy will draw chunks on the robot.

Usage (same registry as scripts/train_rlt.py; the train config is nested under `--train.`):

    python scripts/cache_rl_token.py rlt_pi05 --train.exp_name=cache \
        --params_path=checkpoints/rlt_pi05/my_run/9999/params \
        --reference_delay=4

    python scripts/cache_rl_token.py rlt_pi05 --train.exp_name=cache \
        --train.data.repo_id=agilex/my_task --train.data.assets.asset_id=agilex/my_task \
        --params_path=checkpoints/rlt_pi05/my_run/9999/params \
        --reference_delay=4
"""

import dataclasses
import logging
import multiprocessing
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from lerobot.datasets import lerobot_dataset
import numpy as np
import torch
import tqdm_loggable.auto as tqdm
import tyro

from openpi.models import rlt_config
import openpi.models.model as _model
from openpi.shared import download
from openpi.shared import nnx_utils
import openpi.training.config as _config
import openpi.training.config_rlt as _config_rlt
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.transforms as _transforms

logger = logging.getLogger("openpi")


class SequentialVideoLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """LeRobotDataset that decodes each episode's videos once instead of re-opening them per frame.

    `LeRobotDataset._query_videos` constructs a fresh `VideoDecoder` and re-reads the video file for
    every frame of every camera. Training tolerates that because it shuffles and is GPU-bound, but a
    caching pass is dataloader-bound and touches *every* frame: on the teleop dataset that is 3 x 551 file
    opens per episode against network storage, measured at 155 ms/frame (~6 frames/s single-threaded,
    and it does not scale with workers because they contend on the same NFS reads).

    A caching pass reads every frame of an episode anyway, so decode each video in one sequential
    pass and serve frames from memory: 3 file opens per episode instead of 1653.

    Correct only for an in-order scan: the loader must not shuffle.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decoded_episode: int | None = None
        self._decoded: dict[str, torch.Tensor] = {}

    def _decode_episode(self, ep_idx: int) -> None:
        from torchcodec.decoders import VideoDecoder

        # Drop the previous episode first: one episode of uint8 frames is ~0.5 GB per camera.
        self._decoded = {}
        for vid_key in self.meta.video_keys:
            path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            decoder = VideoDecoder(str(path), device="cpu", seek_mode="approximate")
            self._decoded[vid_key] = decoder[:]  # uint8 [N, C, H, W]
        self._decoded_episode = ep_idx

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        if ep_idx != self._decoded_episode:
            self._decode_episode(ep_idx)
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            frames = self._decoded[vid_key]
            # Same index arithmetic as `decode_video_frames_torchcodec`; timestamps restart at 0 for
            # each episode, so they index straight into that episode's video.
            indices = [min(round(ts * self.meta.fps), frames.shape[0] - 1) for ts in query_ts]
            item[vid_key] = (frames[indices].type(torch.float32) / 255).squeeze(0)
        return item


def _create_sequential_dataset(data_config: _config.DataConfig, action_horizon: int) -> torch.utils.data.Dataset:
    """`data_loader.create_torch_dataset`, but with the sequential-decode dataset underneath."""
    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = SequentialVideoLeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    if data_config.prompt_from_task:
        return _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(meta.tasks)])
    return dataset


@dataclasses.dataclass
class CacheConfig:
    """The Stage A train config (so `--train.data.*` overrides carry over) plus caching options."""

    train: _config.TrainConfig
    # Stage A checkpoint to freeze, e.g. checkpoints/rlt_pi05/<exp>/<step>/params.
    params_path: str = tyro.MISSING
    # Defaults to <dataset root>/rl_cache, i.e. next to the LeRobot dataset it was extracted from.
    output_dir: str | None = None
    # Inference batch size for the frozen VLA pass, split across devices. Raising it does not help:
    # the VLA saturates the GPU at 32.
    batch_size: int = 32
    # Torch dataloader workers. Each holds its own decoded episode (~1.5 GB) and, because torch
    # hands out batches round-robin, every worker decodes every episode. Two is already ~125
    # frames/s, ~3x what the frozen VLA pass consumes on 2 GPUs, so more workers only buy redundant
    # decodes and memory.
    num_workers: int = 4
    # Flow-matching denoising steps used to draw the reference chunk (matches serving).
    num_denoise_steps: int = 10
    # RTC delay used when drawing the reference chunk: how many leading actions enter as
    # already-committed ground truth. Offline there is no chunk in flight, so the demonstration's
    # own leading actions stand in for it -- exactly what `compute_loss` does during Stage A
    # training. Deliberately smaller than `max_delay` so most of the chunk is still denoised.
    # Ignored when the model is non-RTC.
    reference_delay: int = 4
    seed: int = 0
    # Cache only the first N frames, rounded down to a whole number of episodes. Quick sanity pass.
    max_frames: int | None = None


def _unwrap(dataset):
    """The underlying LeRobotDataset, past any TransformedDataset wrappers."""
    while isinstance(dataset, _data_loader.TransformedDataset):
        dataset = dataset._dataset  # noqa: SLF001
    return dataset


def _episode_bounds(dataset) -> np.ndarray:
    """[num_episodes, 2] of (from, to) global frame indices."""
    index = _unwrap(dataset).episode_data_index
    return np.stack([np.asarray(index["from"]), np.asarray(index["to"])], axis=-1).astype(np.int64)


def main(args: CacheConfig) -> None:
    config = args.train
    if not isinstance(config.model, rlt_config.RLTConfig):
        raise ValueError(f"cache_rl_token expects an RLTConfig, got {type(config.model)}")

    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = _create_sequential_dataset(data_config, config.model.action_horizon)
    bounds = _episode_bounds(raw_dataset)
    dataset = _data_loader.transform_dataset(raw_dataset, data_config)

    # Truncate on an episode boundary. Cutting mid-episode would make Stage B treat the cut point as
    # a task completion and hand out a spurious terminal reward there.
    num_frames = len(dataset)
    if args.max_frames is not None and args.max_frames < num_frames:
        kept = bounds[bounds[:, 1] <= args.max_frames]
        if len(kept) == 0:
            raise ValueError(f"--max_frames={args.max_frames} is shorter than the first episode ({bounds[0, 1]}).")
        bounds = kept
        num_frames = int(kept[:, 1].max())
    logger.info("Caching RL tokens for %d frames across %d episodes", num_frames, len(bounds))

    # Next to the dataset it was extracted from, not under `assets/` (which holds norm stats). The
    # cache is only meaningful for this dataset, `annotations.json` already lives there, and deriving
    # the path from `dataset.root` makes it follow HF_LEROBOT_HOME automatically.
    output_dir = pathlib.Path(args.output_dir or (_unwrap(raw_dataset).root / "rl_cache"))
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = download.maybe_download(args.params_path)
    model = config.model.load(_model.restore_params(params_path, dtype=jnp.bfloat16))
    model.eval()
    logger.info("Loaded frozen Stage A params from %s", params_path)

    # Data parallel across every local GPU: the frozen params are replicated and each device gets a
    # slice of the batch. Pure replication rather than the FSDP that Stage A training uses -- there
    # is no optimizer state here and the bf16 params fit on one card, so sharding them would only
    # add all-gathers. Measured 1.96x on 2x H20 (21.8 -> 42.8 frames/s), i.e. linear.
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size={args.batch_size} must be divisible by {jax.device_count()} devices")
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    graphdef, state = nnx.split(model)
    model = nnx.merge(graphdef, jax.device_put(state, replicated_sharding))
    logger.info("Running data parallel over %d device(s)", jax.device_count())

    infer = nnx_utils.module_jit(model.sample_rl_state_and_reference)

    # Held in RAM, not memmapped. The dataset lives on a FUSE-mounted parallel filesystem, where
    # every dirtied mmap page is written back through the FUSE protocol individually: measured at
    # 16 s per batch of 32 frames, versus 0.3 ms on local disk. The whole cache is only ~1.8 GB at
    # 84k frames, so buffer it and write each array once, sequentially, at the end.
    horizon, action_dim = config.model.action_horizon, config.model.action_dim
    out = {
        "z_rl": np.empty((num_frames, config.model.rl_token_dim), dtype=np.float32),
        "a_ref": np.empty((num_frames, horizon, action_dim), dtype=np.float32),
        "a_demo": np.empty((num_frames, horizon, action_dim), dtype=np.float32),
        "state": np.empty((num_frames, action_dim), dtype=np.float32),
    }

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(num_frames)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # Spawn, not fork: JAX is multithreaded and forking it deadlocks (same reason TorchDataLoader
        # sets this).
        multiprocessing_context=multiprocessing.get_context("spawn") if args.num_workers > 0 else None,
        collate_fn=_data_loader._collate_fn,  # noqa: SLF001
        drop_last=False,
    )

    # RTC: condition the reference chunk on the actions that would already be executing, so the
    # cached `a_ref` is drawn the same way the policy will draw it on the robot. `RLT` (non-RTC)
    # has no such argument, hence the guard.
    if config.model.rtc:
        if args.reference_delay >= config.model.action_horizon:
            raise ValueError(f"reference_delay={args.reference_delay} must be < action_horizon")
        if args.reference_delay >= config.model.max_delay:
            # Stage A samples delay ~ U[0, max_delay), so a larger delay here asks the model to
            # condition on a longer prefix than it ever saw in training.
            logger.warning(
                "reference_delay=%d is outside the training distribution (max_delay=%d).",
                args.reference_delay,
                config.model.max_delay,
            )
        delay = jnp.full((args.batch_size,), args.reference_delay, dtype=jnp.int32)
        logger.info("Sampling a_ref with RTC prefix conditioning, delay=%d", args.reference_delay)
    else:
        delay = None
        logger.info("Model is non-RTC; sampling a_ref unconditioned.")

    rng = jax.random.key(args.seed)
    cursor = 0
    for batch in tqdm.tqdm(loader, total=(num_frames + args.batch_size - 1) // args.batch_size, desc="RL tokens"):
        n = np.asarray(batch["actions"]).shape[0]
        # Pad the trailing partial batch back up to batch_size. A short batch is a new input shape,
        # which would force XLA to recompile the whole VLA graph -- minutes of work for a few frames.
        pad = args.batch_size - n
        padded = batch if pad == 0 else jax.tree.map(lambda x: np.concatenate([x, np.repeat(x[-1:], pad, 0)]), batch)  # noqa: B023

        observation = _model.Observation.from_dict(padded)
        rng, step_rng = jax.random.split(rng)
        rtc_kwargs = {} if delay is None else {"action_prefix": jnp.asarray(padded["actions"]), "delay": delay}
        # Split the batch across devices. The rng is a scalar key and stays replicated.
        observation, rtc_kwargs = jax.device_put((observation, rtc_kwargs), data_sharding)
        z_rl, a_ref = infer(step_rng, observation, num_steps=args.num_denoise_steps, **rtc_kwargs)

        # `[:n]` drops the padding rows added above.
        out["z_rl"][cursor : cursor + n] = np.asarray(z_rl, dtype=np.float32)[:n]
        out["a_ref"][cursor : cursor + n] = np.asarray(a_ref, dtype=np.float32)[:n]
        out["a_demo"][cursor : cursor + n] = np.asarray(batch["actions"], dtype=np.float32)
        out["state"][cursor : cursor + n] = np.asarray(observation.state, dtype=np.float32)[:n]
        cursor += n

    assert cursor == num_frames, f"wrote {cursor} of {num_frames} frames"
    for name, array in out.items():
        np.save(output_dir / f"{name}.npy", array)
    np.save(output_dir / "episode_bounds.npy", bounds)

    logger.info("Wrote cache to %s", output_dir)
    # Measure on the denoised postfix only: RTC returns the prefix bit-exactly, so including it
    # would report a difference of zero over those steps and flatter the model.
    d = 0 if delay is None else args.reference_delay
    logger.info(
        "mean |a_ref - a_demo| over the denoised postfix (steps %d:) = %.4f (should be small on demonstration data)",
        d,
        float(np.mean(np.abs(out["a_ref"][:, d:] - out["a_demo"][:, d:]))),
    )
    if d:
        prefix_err = float(np.max(np.abs(out["a_ref"][:, :d] - out["a_demo"][:, :d])))
        logger.info("max |a_ref - a_demo| over the RTC prefix (steps :%d) = %.2e (must be 0)", d, prefix_err)
    logger.info(
        "z_rl per-dim std across frames = %.4f (near 0 means a collapsed bottleneck)",
        float(np.mean(np.std(out["z_rl"][:], axis=0))),
    )


def cli() -> CacheConfig:
    # Same registry as scripts/train_rlt.py, so the cache is built from the exact dataset and
    # normalization the RL stage will train on.
    return tyro.extras.overridable_config_cli(
        {name: (name, CacheConfig(train=cfg)) for name, cfg in _config_rlt.all_configs().items()}
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(cli())
