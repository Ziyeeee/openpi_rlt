# openpi_rlt

RLT (RL Token) on top of [OpenPI](https://github.com/Physical-Intelligence/openpi): lightweight
online reinforcement learning for a frozen pi0.5 policy, with Real-Time Chunking (RTC) enabled by
default. A small actor/critic learns to refine the VLA's action chunks during the *critical phase*
of a task (e.g. the precise insertion at the end of a plug-in motion), while the VLA handles
everything else.

This guide walks through the full workflow, from a LeRobot dataset to online RL on a real robot.
The robot client shipped here targets a bimanual AgileX setup (two 7-DoF arms, three cameras);
the training stages work for any robot whose data you can express in the same LeRobot format.

## Workflow at a glance

| Step | What you do | Script | Config | Produces |
|---|---|---|---|---|
| 0 | Prepare the dataset and norm stats | `merge_lerobot_datasets.py`, `compute_norm_stats.py` | `pi05_agilex_right` | `assets/<asset_id>/norm_stats.json` |
| A | Train pi0.5 together with the RL token | `train_rlt.py` | `rlt_pi05` | `checkpoints/rlt_pi05/<run>/` |
| A.5 | Cache RL tokens and reference chunks | `cache_rl_token.py` | `rlt_pi05` | `<dataset>/rl_cache/` |
| A.6 | Label the critical phase, train the handover head | `train_rlt_handover.py`, `label_handover.py` | `rlt_pi05_handover[_all]` | `checkpoints/rlt_pi05_handover_all/all/` |
| B | Train the actor/critic offline (warm start) | `train_rlt_ac.py` | `rlt_pi05_ac_c16` | `checkpoints/rlt_pi05_ac_c16/<run>/` |
| C | Online RL on the robot | `serve_policy_rlt.py` + `agilex_rlt_client.py` | `rlt_pi05_online` | improved actor/critic, online clips |

Every stage reads the previous one's output by path, and the defaults in the configs follow the
layout used in this guide. If you use different names, override the paths on the command line
(every config field is a CLI flag) or edit the config files listed in
[Adapting to your own task](#adapting-to-your-own-task).

## Installation

```bash
git clone --recurse-submodules <this-repo> openpi_rlt
cd openpi_rlt
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
```

Set these in every shell you run the scripts from (training machine and Stage C server):

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=$(pwd)/data          # LeRobot datasets live under ./data/<repo_id>
export CACHE_HOME=$HOME/.cache
export OPENPI_DATA_HOME="$CACHE_HOME/openpi" # downloaded pi0.5 base weights
export HF_HOME="$CACHE_HOME/huggingface"
```

Stage A fine-tunes the full pi0.5 model, so it needs the same GPU resources as an OpenPI full
fine-tune. Stages A.6 and B train small MLPs on cached features and are fast. The Stage C server
needs one GPU that can hold pi0.5 for inference.

The robot side (`agilex_*.py`) additionally needs ROS 2 and the AgileX drivers on the robot
computer. It talks to the GPU server over a websocket, so the two can be different machines.

## Step 0: Prepare the data

### Dataset format

Datasets are LeRobot v2.1 directories under `$HF_LEROBOT_HOME/<repo_id>`. The RLT configs expect
these features:

| Feature | Content |
|---|---|
| `observation.images.front` | top / front camera |
| `observation.images.left` | left wrist camera |
| `observation.images.right` | right wrist camera |
| `observation.state` | 14-dim joint state (2 x [6 joints + gripper]) |
| `action` | 14-dim joint command, same layout |
| task | the language instruction |

Different keys are fine: edit the repack mapping (`_AGILEX_REPACK` in
`src/openpi/training/config_rlt.py`). The examples below use `agilex/my_task` for teleoperated
demonstrations and `agilex/my_task_rollout` for policy rollouts.

To collect data on the AgileX, `agilex_recorder.py` records policy rollouts at 30 Hz (see its
`--help`). Teleop demonstrations come from your usual teleop stack. Convert both to LeRobot.

### Merge recordings (optional)

If one task was recorded over several sessions, merge them into one dataset. The sources must
share fps, robot type, and features. Videos are hardlinked, so this needs very little extra disk.

```bash
python scripts/merge_lerobot_datasets.py \
    data/agilex/my_task_part1 data/agilex/my_task_part2 \
    --out data/agilex/my_task
```

### Normalization statistics

Compute them once per dataset. They are written to `assets/<asset_id>/norm_stats.json`, which is
where every RLT config reads them from.

```bash
python scripts/compute_norm_stats.py pi05_agilex_right --exp_name=norm_stats \
    --data.repo_id=agilex/my_task --data.assets.asset_id=agilex/my_task
```

## Stage A: Train the VLA with the RL token

This fine-tunes pi0.5 (starting from the public `pi05_base` weights) on your task, together with
the RL token that later stages use as their compact state representation. Training takes 10k steps.

```bash
python scripts/train_rlt.py rlt_pi05 --exp_name=my_run --overwrite
```

Checkpoints go to `checkpoints/rlt_pi05/my_run/<step>/`. The final one, `9999`, is the one the
later stages use. Pass `--resume` instead of `--overwrite` to continue an interrupted run.
To train on another dataset, add `--data.repo_id=... --data.assets.asset_id=...`.

## Stage A.5: Cache RL tokens

Runs the frozen Stage A model once over every frame of a dataset and stores what Stages A.6 and B
need. After this, neither stage touches images or the VLA again.

```bash
# Teleop demonstrations (the dataset in the rlt_pi05 config)
python scripts/cache_rl_token.py rlt_pi05 --train.exp_name=cache \
    --params_path=checkpoints/rlt_pi05/my_run/9999/params \
    --reference_delay=4

# Any other dataset of the same task, e.g. rollouts
python scripts/cache_rl_token.py rlt_pi05 --train.exp_name=cache \
    --train.data.repo_id=agilex/my_task_rollout --train.data.assets.asset_id=agilex/my_task \
    --params_path=checkpoints/rlt_pi05/my_run/9999/params \
    --reference_delay=4
```

The cache is written to `<dataset>/rl_cache/` (override with `--output_dir`). Keep
`--train.data.assets.asset_id` pointing at the norm stats Stage A was trained with.

> Re-run this step whenever you retrain Stage A. Caches from an older checkpoint silently mismatch.

## Stage A.6: Critical phase and handover head

RL only runs during the task's critical phase. The handover head watches the RL token stream and
decides when to hand control from the VLA to the RL actor, and when to hand it back.

### 1. Hand-label critical-phase spans

Create `annotations.json` next to each dataset (e.g. `data/agilex/my_task/annotations.json`). Each
episode gets at most one span, with frame indices relative to the episode start:

```json
{
  "tags": ["plug in"],
  "clips": {
    "0": [{"begin_id": 212, "end_id": 340, "tag": "plug in"}],
    "1": [{"begin_id": 198, "end_id": 351, "tag": "plug in"}]
  }
}
```

Leave episodes out (or give them an empty list) if you haven't labeled them yet.

### 2. Train the head and check it

Cross-validate first to see how accurate the head is before you trust it:

```bash
for fold in 0 1 2 3 4; do
    python scripts/train_rlt_handover.py rlt_pi05_handover --exp_name=fold$fold --val_fold=$fold --overwrite
done
```

Each run reports the held-out boundary error in frames, next to a clock-only baseline that always
fires at the average handover time. The head is worth deploying only if it clearly beats that
baseline.

### 3. Auto-label the remaining episodes (optional)

Use the fold checkpoints to propose spans for unlabeled episodes:

```bash
python scripts/label_handover.py rlt_pi05_handover --train.exp_name=label \
    --params_paths=checkpoints/rlt_pi05_handover/fold0/4999/params,\
checkpoints/rlt_pi05_handover/fold1/4999/params,\
checkpoints/rlt_pi05_handover/fold2/4999/params,\
checkpoints/rlt_pi05_handover/fold3/4999/params,\
checkpoints/rlt_pi05_handover/fold4/4999/params
```

Predictions go to a separate `annotations_pred.json` beside each `annotations.json`. Your hand
labels are never overwritten. The log prints a review queue, most uncertain episodes first. Check
those, then copy the spans you accept into `annotations.json` by hand.

### 4. Train the head to deploy

`rlt_pi05_handover_all` pools the teleop and rollout datasets and trains on every labeled episode,
with no held-out fold:

```bash
python scripts/train_rlt_handover.py rlt_pi05_handover_all --exp_name=all --overwrite
```

Stage C loads `checkpoints/rlt_pi05_handover_all/all/4999` by default.

## Stage B: Offline actor/critic

Stage B trains the actor and twin critics on the labeled critical-phase clips. Its main use is as a
warm start for Stage C, so the online critic does not start from random.

### Clip outcomes

Stage B needs each clip's **outcome**, which is the only reward signal. Write
`clips_by_tag.json` next to each dataset you train on:

```json
{
  "frame_range": "inclusive",
  "tags": {
    "plug in": [
      {"episode_index": 0, "begin_id": 206, "end_id": 356, "result": "success"},
      {"episode_index": 1, "begin_id": 190, "end_id": 402, "result": "fail"}
    ]
  }
}
```

Spans are the same critical-phase spans as in `annotations.json`. `result` is `"success"` or
`"fail"`. Teleop demonstrations are usually all successes. Policy rollouts that include **failures**
are what teach the critic to tell good actions from bad ones, so record and keep failed rollouts
too.

### Train

```bash
python scripts/train_rlt_ac.py rlt_pi05_ac_c16 --exp_name=c16 --overwrite
```

Use `rlt_pi05_ac_c16`, not `rlt_pi05_ac`: the suffix is the action chunk length, and Stage C runs
with 16. A checkpoint trained at a different chunk length is a different shape and the server will
reject it. The final checkpoint is `checkpoints/rlt_pi05_ac_c16/c16/19999/params`.

## Stage C: Online RL on the robot

Two processes:

- **Server** (GPU machine): `serve_policy_rlt.py` holds the frozen VLA, the handover head, the
  actor/critic, the replay buffer and the learner, and keeps training while the robot runs.
- **Client** (robot computer): `agilex_rlt_client.py` executes action chunks, and you tell it,
  by key press, when each episode starts and whether it succeeded.

### 1. Start the server

```bash
python scripts/serve_policy_rlt.py rlt_pi05_online --exp_name=online0 --overwrite \
    --ac_params_path=checkpoints/rlt_pi05_ac_c16/c16/19999/params
```

By default it loads Stage A from `checkpoints/rlt_pi05/my_run/9999` and the handover head from
`checkpoints/rlt_pi05_handover_all/all/4999`. Override them with `--stage_a_dir` and
`--handover_dir`. Point `--stage_a_dir` at the plain Stage A checkpoint; the server attaches the
head itself. The replay buffer is pre-filled with the Stage B clips.

Startup takes a couple of minutes (loading weights and compiling). **Wait for
`server listening on 0.0.0.0:8010`** before starting a client. Useful lines to check on the way:

```
Loaded ... Stage A tensors from ... and ... handover tensors from ...
Replay buffer: {'buffer_transitions': ..., 'buffer_clips': ..., ...}
Warm-started critic: actor gate opens immediately (critic_warmup_steps=0)
server listening on 0.0.0.0:8010
```

### 2. Validate without the robot

Always do this before the server drives a real arm, and after changing any Stage C code. It speaks
the robot's protocol with synthetic frames:

```bash
python scripts/mock_rlt_client.py --episodes=4 --queries_per_episode=10 --port=8010
```

Expect `all checks passed`. If the server runs with `--collect_only`, pass `--collect_only` here
too. The mock leaves synthetic clips in the buffer, so restart the server with `--overwrite`
before a real session.

### 3. Run the robot client

On the robot computer:

```bash
python agilex_rlt_client.py \
    --datasetDir ~/data/my_task_online \
    --prompt "<your task instruction>" \
    --execute --host <SERVER_IP> --port 8010 --viz
```

Drop `--execute` for a dry run: everything runs but the arm does not move. All the RL settings come
from the server, so the client has no RL flags. The control flags of `agilex_inference.py` and
`agilex_recorder.py` still apply. Run `--help` to list them.

Successful episodes are also saved to `--datasetDir`, so you can reuse them as training data later.

**Keys**

| Key | Effect |
|---|---|
| `ENTER` | start policy control: open an episode, or resume one paused with `q` |
| `s` | **success**: hold, save the episode, label the RL clip `+1` |
| `d` | **failure**: hold, discard the episode, label the RL clip as failed |
| `q` | pause: hold and stop recording; `ENTER` resumes the same episode |
| `r` | recover (open grippers, go home, close); **aborts** the episode with no label |
| `g` | force the RL actor to take over now, without waiting for the handover head |
| `v` | VLA only for the rest of this episode |
| `Ctrl-C` | exit; an open episode is saved but its RL clip is aborted, not labeled |

**When the policy fails, press `d`, not `r`.** Failures are the rare, valuable label. `r` means
"reset the scene" and records nothing.

### 4. Suggested session order

1. **Check the handover head.** Start the server with `--collect_only`: it serves the plain VLA and
   never engages the actor, and nothing is learned. Watch `phase` in the client log and confirm
   the head switches to `critical` where you expect.
2. **Online RL.** Restart without `--collect_only`. The head now hands over on its own. Keep `g`
   and `v` ready to override it while you build confidence.
3. **Evaluate.** Restart with `--deterministic` to measure the learned policy without exploration
   noise. Don't collect training data in this mode: the exploration noise is what the critic
   learns from.

Continue a session later with `--resume` instead of `--overwrite`. The learner state and the
online clips are both restored.

### 5. What to watch

The client prints once per second:

```
rl: phase critical mode rl latched True q 0.312 |a-a_ref| 0.184 learner 4210 buffer 1840 rows 96
```

- `phase` / `latched`: the handover head's prediction, and whether RL control is active.
- `mode`: `vla`, `warmup` (below `--warmup_transitions`), or `rl`.
- `q`: the critic's value for the current chunk.
- `|a-a_ref|`: how far the actor moved from the VLA's chunk. If it stays pinned at
  `--max_action_delta`, the safety bound is active on every step; investigate before continuing.
- `rows`: transitions buffered this episode, sent to the server when the episode ends.

Every 10 s the server logs `critic_loss`, `mean_q`, `utd` (should settle near 5), buffer counts and
`online_success_rate`.

### 6. Useful server flags

| Flag | Default | Meaning |
|---|---|---|
| `--collect_only` | off | serve the plain VLA; never engage the actor |
| `--deterministic` | off | evaluation: no exploration noise |
| `--max_action_delta` | 0.5 | max deviation of the actor from the VLA chunk (normalized units); `0.0` = pure VLA |
| `--warmup_transitions` | 0 | execute the VLA until this many online transitions exist |
| `--handover_patience` | 2 | consecutive `critical` predictions before RL takes over |
| `--updates_per_env_step` | 5.0 | learner updates per collected transition |
| `--port` | 8010 | websocket port |

### 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Client hangs at "Waiting for server" | Server is still loading; wait for `server listening`. |
| `does not match this Stage C actor/critic` | Warm start used another chunk length; train `rlt_pi05_ac_c16`. |
| `is not a handover checkpoint` | `--handover_dir` points at the wrong checkpoint. |
| `already carries a handover head` | `--stage_a_dir` must be the plain Stage A checkpoint. |
| `0 rows admitted` at episode end | The handover never fired. Check `phase`, or press `g`. |
| `critical phase exceeded max_rows_per_episode` | Critical phase was too long; raise `--max_rows_per_episode`. |

## Adapting to your own task

| What | Where |
|---|---|
| Dataset, norm stats, task prompt, camera/state keys | `rlt_pi05` in `src/openpi/training/config_rlt.py` |
| Datasets with critical-phase labels for the handover head | `sources` in `src/openpi/training/config_rlt_handover.py` |
| Datasets and clip files for the actor/critic, and the clip tag (`clip_tag`, default `"plug in"`) | `sources` in `src/openpi/training/config_rlt_ac.py` |
| Default checkpoint paths for Stage C | `stage_a_dir`, `handover_dir` in `src/openpi/training/config_rlt_online.py` |
| Robot I/O (ROS topics, rates, velocity limits) | flags of `agilex_inference.py` |

For a different robot, write a client that speaks the same websocket protocol.
`scripts/mock_rlt_client.py` is a minimal working example.

## Repository layout

```
scripts/
  train_rlt.py              Stage A
  cache_rl_token.py         Stage A.5
  train_rlt_handover.py     Stage A.6 training
  label_handover.py         Stage A.6 auto-labeling
  train_rlt_ac.py           Stage B
  serve_policy_rlt.py       Stage C server
  mock_rlt_client.py        Stage C validation without a robot
  merge_lerobot_datasets.py, compute_norm_stats.py
agilex_inference.py         AgileX policy client (ROS 2)
agilex_recorder.py          AgileX client + rollout recorder
agilex_rlt_client.py        AgileX Stage C client
src/openpi/models/rlt*.py           RLT models
src/openpi/training/config_rlt*.py  RLT configs
```

## License

See [`LICENSE`](LICENSE) and [`LICENSE_GEMMA.txt`](LICENSE_GEMMA.txt).
