# RoboSynChallenge — Preliminary (Simulation) Submission

**Team:** `solidsnail`
**Artifact / experiment name:** `solidsnail-smolvla-10tasks`
**Policy:** SmolVLA (LeRobot), fine-tuned **one checkpoint per task**
**Task coverage:** **all 10 tasks**

| Item | Value |
| --- | --- |
| Code repository | this repository (`main`) — policy adapter entry points: `policy/smolvla/eval.sh`, `policy/smolvla/deploy_policy.yml`, `policy/smolvla/deploy_policy.py` |
| Checkpoints (Hugging Face) | `https://huggingface.co/solidsnail/robosyn-smolvla-10tasks` |
| Checkpoint repo visibility | **private** — read access granted to the official HF account **`RoboSynChallenge`** |
| Base code revision | `EDEM-AI/RoboSynChallenge` @ `8818c82` (Configure deterministic top camera randomization) |
| Training base weights | `lerobot/smolvla_base` |

> The policy adapter is **task-agnostic**: a single `eval.sh` + `deploy_policy.yml` covers all ten
> tasks. The task is selected by the `<task_name>` argument; the checkpoint is selected by the
> `<checkpoint_path>` argument. Nothing in the adapter is task-specific.

---

## 1. Task → checkpoint mapping

All ten checkpoints live in a **single** Hugging Face repository,
`solidsnail/robosyn-smolvla-10tasks`, one sub-directory per task. Each sub-directory is a
complete LeRobot `pretrained_model/` folder (7 files), i.e. it can be passed to `eval.sh`
directly as `checkpoint_path`.

| # | `task_name` (eval.sh arg 1) | HF path inside the checkpoint repo | `checkpoint_path` after download |
| --- | --- | --- | --- |
| 1 | `click_bell` | `click_bell/` | `checkpoints/click_bell` |
| 2 | `drawer_open_place` | `drawer_open_place/` | `checkpoints/drawer_open_place` |
| 3 | `handle_basket` | `handle_basket/` | `checkpoints/handle_basket` |
| 4 | `item_assembly` | `item_assembly/` | `checkpoints/item_assembly` |
| 5 | `items_handover` | `items_handover/` | `checkpoints/items_handover` |
| 6 | `manipulate_pipette` | `manipulate_pipette/` | `checkpoints/manipulate_pipette` |
| 7 | `mixer_operating` | `mixer_operating/` | `checkpoints/mixer_operating` |
| 8 | `sample_loading` | `sample_loading/` | `checkpoints/sample_loading` |
| 9 | `table_rearrangement` | `table_rearrangement/` | `checkpoints/table_rearrangement` |
| 10 | `water_pouring` | `water_pouring/` | `checkpoints/water_pouring` |

Each `checkpoints/<task>/` directory contains a `pretrained_model/` sub-folder; `eval.sh`
detects this automatically and uses it as the `--checkpoint_dir`.

---

## 2. Get the code

```bash
git clone https://github.com/solidsnailz36-cloud/robosyn-submission.git
cd robosyn-submission
```

The repository is a snapshot of `EDEM-AI/RoboSynChallenge` @ `8818c82` with this submission
documentation and a convenience runner added. No source file of the evaluation pipeline was
modified.

## 3. Get the checkpoints

```bash
mkdir -p checkpoints
# private repo: authenticate as an account that has been granted access
# (the official HF account RoboSynChallenge has been granted read access)
hf download solidsnail/robosyn-smolvla-10tasks --repo-type model --local-dir checkpoints
```

Result: `checkpoints/<task>/pretrained_model/{config.json, model.safetensors, ...}` for all 10 tasks.

## 4. Environment

Two Python contexts are involved (see `policy/smolvla/README.md` for the upstream description):

1. **Simulator context** — RoboSynChallenge + EmbodiChain. Install per the official guide:
   <https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html>.
   Expected workspace layout:
   ```
   RoboSynChallenge_ws/
     EmbodiChain/
     RoboSynChallenge/
   ```
2. **SmolVLA worker context** — a Python environment that can `import lerobot`.
   Use `policy/smolvla/setup_lerobot.sh`, or point the adapter at an existing LeRobot checkout:

```bash
# Option A: install LeRobot from inside this repo
bash policy/smolvla/setup_lerobot.sh
export SMOLVLA_LEROBOT_ROOT="$PWD/policy/smolvla/lerobot"

# Option B: an existing LeRobot checkout
export SMOLVLA_LEROBOT_ROOT=/path/to/lerobot

# If the same environment also runs the SmolVLA worker:
export SMOLVLA_PYTHON="$(which python)"
```

Python-level dependencies: see [`requirements.txt`](requirements.txt).

## 5. Run the evaluation (single task)

```bash
task_name=click_bell
checkpoint_path=checkpoints/${task_name}

bash policy/smolvla/eval.sh ${task_name} random ${checkpoint_path} 0 \
  --pytorch_device cuda \
  --headless true \
  --renderer auto \
  --max_episodes 20 \
  --eval_video_log true \
  --smolvla_steps 10 \
  --smolvla_rescale_gripper true
```

`<setting>` (arg 2) is supplied by the official evaluation protocol. All ten tasks use the
**same** command, changing only `task_name` and `checkpoint_path`.

## 6. Run all ten tasks

```bash
bash submission/run_all.sh            # gpu_id=0
bash submission/run_all.sh 0 20       # gpu_id=0, max_episodes=20
```

## 7. Dependencies

- `requirements.txt` in this repository lists the Python packages used by the SmolVLA worker.
- Simulator dependencies are those of the official **EmbodiChain** installer; we did not add or
  pin anything there.
- No external network access is required at evaluation time; no online API is called. The policy
  consumes **only** RGB images and robot proprioception.

## 8. Provenance / notes

- **Data:** trained on synthetic RoboSynChallenge data (official released `cobotmagic_Sim_*`
  datasets) plus our own self-collected trajectories; no real-robot data at inference time.
- **Adapter:** unmodified official SmolVLA adapter (`policy/smolvla/`).
- **Inference inputs:** 3 RGB views (`cam_high`, `cam_left_wrist`, `cam_right_wrist`) and the
  robot joint state (`observation.qpos`). No ground-truth object pose, no contact force, no
  reward, and no simulator privileged information is used.
- The `is_task_success` return value is only used by the official evaluation loop, not by the
  controller.
