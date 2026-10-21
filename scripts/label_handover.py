"""RLT Stage A.6: auto-annotate critical-phase spans and rank them for human review.

Labeling and triggering are the same head but not the same problem. On the robot the decision is
causal and one-shot. Offline the whole episode is available, so this decodes with
`rlt_handover.viterbi_decode`: the best `pre -> critical -> post` path, which by construction yields
exactly one contiguous span per episode and lets a frame's phase be settled by evidence that arrives
after it.

Predictions are written to a **separate** `annotations_pred.json` in the same schema, never merged
into the hand-labeled `annotations.json`. Two reasons: a human's spans must not be silently
overwritten, and the head must never be retrained on its own output. Move a span across by hand once
you have reviewed it.

The review queue is ordered by seed disagreement -- pass several Stage A.6 checkpoints trained with
different `--seed` and the spread of their decoded boundaries, in frames, is the uncertainty to
review first. With one checkpoint the spread is 0 and the queue is unordered.

Usage:
    python scripts/label_handover.py rlt_pi05_handover --train.exp_name=label \
        --params_paths=checkpoints/rlt_pi05_handover/fold0/4999/params,\
checkpoints/rlt_pi05_handover/fold1/4999/params,\
checkpoints/rlt_pi05_handover/fold2/4999/params,\
checkpoints/rlt_pi05_handover/fold3/4999/params,\
checkpoints/rlt_pi05_handover/fold4/4999/params
"""

import dataclasses
import json
import logging
import pathlib

import jax
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro

from openpi.models import rlt_handover
import openpi.models.model as _model
from openpi.shared import download
from openpi.shared import nnx_utils
import openpi.training.config_rlt_handover as _config
import openpi.training.data_loader_handover as _data_loader

logger = logging.getLogger("openpi")


@dataclasses.dataclass
class LabelConfig:
    """The Stage A.6 train config (for the caches, annotations and head shape) plus labeling options."""

    train: _config.TrainConfig
    # Comma-separated Stage A.6 params directories. More than one gives per-boundary uncertainty.
    params_paths: str = tyro.MISSING
    # Tag written on every predicted span; must match a tag the labeling tool knows.
    tag: str = "plug in"


def _decode(head_fns, z: np.ndarray, state: np.ndarray, config) -> tuple[tuple[int, int] | None, list]:
    """Ensemble span from the mean log-probabilities, plus each member's own span."""
    per_seed_log_probs = [rlt_handover.episode_log_probs(fn, z, state, config.model) for fn in head_fns]
    penalty = config.model.transition_penalty
    ensemble = rlt_handover.viterbi_decode(np.mean(per_seed_log_probs, axis=0), penalty)
    members = [rlt_handover.viterbi_decode(lp, penalty) for lp in per_seed_log_probs]
    return ensemble, [m for m in members if m is not None]


def label_source(source, head_fns, args: LabelConfig) -> None:
    """Decode every episode of one Stage A.5 cache and write its `annotations_pred.json`."""
    config = args.train
    cache = _data_loader.RLCache.load(source.cache_dir)
    if pathlib.Path(source.annotations_path).exists():
        _data_loader.check_cache_alignment(source.annotations_path, cache.episode_bounds)
        human = _data_loader.load_annotations(source.annotations_path)
    else:
        # A fresh dataset has no hand labels yet -- that's exactly what this script is for. `human`
        # is only used below as a sanity check against existing labels and to skip them in the
        # review queue, so an empty dict is the correct "nothing labeled yet" state.
        logger.info("No annotations at %s; treating as unlabeled dataset", source.annotations_path)
        human = {}

    clips, uncertainty = {}, {}
    for episode in tqdm.tqdm(range(len(cache.episode_bounds)), desc=source.cache_dir):
        start, stop = (int(x) for x in cache.episode_bounds[episode])
        span, members = _decode(
            head_fns,
            np.asarray(cache.z_rl[start:stop], dtype=np.float32),
            np.asarray(cache.state[start:stop], dtype=np.float32),
            config,
        )
        if span is None:
            # No critical phase survived the decode. Left out of the predictions so it surfaces as a
            # gap for a human rather than as a confident span.
            logger.warning("Episode %d: no critical phase decoded", episode)
            continue
        begin_std = float(np.std([m[0] for m in members])) if len(members) > 1 else 0.0
        end_std = float(np.std([m[1] for m in members])) if len(members) > 1 else 0.0
        clips[str(episode)] = [
            {
                "begin_id": span[0],
                "end_id": span[1],
                "tag": args.tag,
                "source": "pred",
                "begin_std": round(begin_std, 1),
                "end_std": round(end_std, 1),
            }
        ]
        uncertainty[episode] = begin_std

    # Always beside that source's hand labels, never on top of them.
    output_path = pathlib.Path(source.annotations_path).parent / "annotations_pred.json"
    output_path.write_text(json.dumps({"tags": [args.tag], "clips": clips}, indent=2))
    logger.info("Wrote %d predicted spans to %s", len(clips), output_path)

    # Agreement with the human labels. These episodes were (mostly) in training, so this is a
    # sanity check that the decode is well-formed -- the honest error bars come from the held-out
    # folds in scripts/train_rlt_handover.py, not from here.
    overlap = [e for e in human if str(e) in clips]
    if overlap:
        errors = np.asarray([clips[str(e)][0]["begin_id"] - human[e][0] for e in overlap])
        logger.info(
            "On %d hand-labeled episodes: median |begin error| = %.1f frames, p90 = %.1f",
            len(overlap),
            float(np.median(np.abs(errors))),
            float(np.percentile(np.abs(errors), 90)),
        )

    queue = sorted((e for e in uncertainty if e not in human), key=lambda e: -uncertainty[e])
    logger.info("Review queue for %s (most uncertain first): %s", source.annotations_path, queue[:20])


def main(args: LabelConfig) -> None:
    config = args.train

    head_fns = []
    for path in args.params_paths.split(","):
        params = _model.restore_params(download.maybe_download(path.strip()), restore_type=np.ndarray)
        head = config.model.load(jax.tree.map(np.asarray, params))
        head.eval()
        head_fns.append(nnx_utils.module_jit(head.__call__))
    logger.info("Loaded %d handover head(s)", len(head_fns))

    # One cache at a time, one `annotations_pred.json` per source. To label a dataset the training
    # config does not list, override the sources: `--train.sources.0.cache-dir=... --train.sources.0.annotations-path=...`.
    for source in config.sources:
        label_source(source, head_fns, args)


def cli() -> LabelConfig:
    # Same registry as scripts/train_rlt_handover.py, so the cache, annotations and head shape are
    # guaranteed to match the ones the head was trained on.
    return tyro.extras.overridable_config_cli(
        {name: (name, LabelConfig(train=cfg)) for name, cfg in _config.all_configs().items()}
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(cli())
