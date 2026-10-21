"""Merge several LeRobot v2.1 datasets into a single dataset directory.

The sources must agree on fps, robot type, and feature schema. Everything that identifies an
episode is re-indexed: episodes are concatenated in the order the sources are given, the task
table becomes the union of the source tasks (ordered by first appearance), and each frame's
`episode_index` / `index` / `task_index` is rewritten to match. Nothing else is touched --
parquet files are rewritten with only those three columns replaced, and videos are hardlinked,
so a merge costs no video re-encoding and essentially no extra disk.

Norm stats are per-dataset, so recompute them for the merged dataset before training:
    python scripts/compute_norm_stats.py <config-name>

Usage:
    python scripts/merge_lerobot_datasets.py \
        data/agilex/my_task_part1 data/agilex/my_task_part2 \
        --out data/agilex/my_task
"""

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Fields that must be identical across sources: they describe the robot and the columns
# themselves, so a mismatch would produce a dataset whose frames don't mean the same thing.
SHARED_INFO_FIELDS = ("codebase_version", "robot_type", "fps", "chunks_size", "data_path", "video_path", "features")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def place_file(src: Path, dst: Path, mode: str) -> None:
    """Put `src` at `dst` as a hardlink, symlink, or copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different filesystems, or a backend without hardlinks -- fall back to a copy
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def column_stats(values: np.ndarray) -> dict:
    """Per-column stats in the layout LeRobot writes to meta/episodes_stats.jsonl."""
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [len(values)],
    }


def rewrite_episode_parquet(src_path: Path, dst_path: Path, columns: dict[str, np.ndarray]) -> None:
    """Copy one episode's parquet, replacing the re-indexed columns and keeping the schema."""
    table = pq.read_table(src_path)
    for name, values in columns.items():
        i = table.schema.get_field_index(name)
        table = table.set_column(i, table.schema.field(i), pa.array(values))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", type=Path, nargs="+", help="LeRobot dataset roots to merge, in order")
    parser.add_argument("--out", type=Path, required=True, help="output dataset root")
    parser.add_argument("--overwrite", action="store_true", help="delete --out if it already exists")
    parser.add_argument(
        "--copy-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="how to place the source videos in the merged dataset (default: hardlink)",
    )
    args = parser.parse_args()

    infos = [json.loads((src / "meta" / "info.json").read_text()) for src in args.sources]
    ref = infos[0]
    for src, info in zip(args.sources[1:], infos[1:], strict=True):
        for field in SHARED_INFO_FIELDS:
            if info[field] != ref[field]:
                raise SystemExit(f"{src}: {field} differs from {args.sources[0]} -- these datasets cannot be merged")

    video_keys = [key for key, feature in ref["features"].items() if feature["dtype"] == "video"]
    chunks_size = ref["chunks_size"]

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} already exists -- pass --overwrite to replace it")
        shutil.rmtree(args.out)
    (args.out / "meta").mkdir(parents=True)

    tasks: dict[str, int] = {}  # task string -> merged task index, ordered by first appearance
    episodes_out: list[dict] = []
    stats_out: list[dict] = []
    total_frames = 0

    for src in args.sources:
        src_tasks = {row["task_index"]: row["task"] for row in read_jsonl(src / "meta" / "tasks.jsonl")}
        for task in src_tasks.values():
            tasks.setdefault(task, len(tasks))
        task_map = {old: tasks[task] for old, task in src_tasks.items()}
        src_stats = {row["episode_index"]: row["stats"] for row in read_jsonl(src / "meta" / "episodes_stats.jsonl")}

        for episode in tqdm(read_jsonl(src / "meta" / "episodes.jsonl"), desc=src.name):
            old_ep = episode["episode_index"]
            new_ep = len(episodes_out)
            length = episode["length"]

            src_parquet = src / ref["data_path"].format(episode_chunk=old_ep // chunks_size, episode_index=old_ep)
            old_task_index = pq.read_table(src_parquet, columns=["task_index"])["task_index"].to_numpy()
            columns = {
                "episode_index": np.full(length, new_ep, dtype=np.int64),
                "index": np.arange(total_frames, total_frames + length, dtype=np.int64),
                "task_index": np.array([task_map[i] for i in old_task_index], dtype=np.int64),
            }
            rewrite_episode_parquet(
                src_parquet,
                args.out / ref["data_path"].format(episode_chunk=new_ep // chunks_size, episode_index=new_ep),
                columns,
            )

            for key in video_keys:
                place_file(
                    src
                    / ref["video_path"].format(
                        episode_chunk=old_ep // chunks_size, video_key=key, episode_index=old_ep
                    ),
                    args.out
                    / ref["video_path"].format(
                        episode_chunk=new_ep // chunks_size, video_key=key, episode_index=new_ep
                    ),
                    args.copy_mode,
                )

            episodes_out.append({"episode_index": new_ep, "tasks": episode["tasks"], "length": length})
            stats = dict(src_stats[old_ep]) | {name: column_stats(values) for name, values in columns.items()}
            stats_out.append({"episode_index": new_ep, "stats": stats})
            total_frames += length

    write_jsonl(args.out / "meta" / "episodes.jsonl", episodes_out)
    write_jsonl(args.out / "meta" / "episodes_stats.jsonl", stats_out)
    write_jsonl(args.out / "meta" / "tasks.jsonl", [{"task_index": i, "task": t} for t, i in tasks.items()])

    info = dict(ref) | {
        "total_episodes": len(episodes_out),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes_out) * len(video_keys),
        "total_chunks": (len(episodes_out) - 1) // chunks_size + 1,
        "splits": {"train": f"0:{len(episodes_out)}"},
    }
    (args.out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(
        f"merged {len(args.sources)} datasets into {args.out}: "
        f"{len(episodes_out)} episodes, {total_frames} frames, {len(tasks)} tasks"
    )


if __name__ == "__main__":
    main()
