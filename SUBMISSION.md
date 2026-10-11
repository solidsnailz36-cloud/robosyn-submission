# RoboSynChallenge — Preliminary (Simulation) Submission

**Team:** `solidsnail`
**Artifact / experiment name:** `solidsnail-smolvla-10tasks`
**Policy:** SmolVLA (LeRobot), fine-tuned **one checkpoint per task**
**Task coverage:** **all 10 tasks**

| Item | Value |
| --- | --- |
| Code repository | this repository (`main`) — policy adapter entry points: `policy/smolvla/eval.sh`, `policy/smolvla/deploy_policy.yml`, `policy/smolvla/deploy_policy.py` |
| Checkpoints (Hugging Face) | `https://huggingface.co/solidsnail/robosyn-smolvla-10tasks` |
| Checkpoint repo visibility | **public** — no collaborator invite or token is needed; anyone (including the organizers) can read it |
| Base code revision | `EDEM-AI/RoboSynChallenge` @ `8818c82` (Configure deterministic top camera randomization) |
| Training base weights | `lerobot/smolvla_base` (VLM backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) |
| Runtime Python | 3.10.12 — `torch 2.10.0+cu128`, `torchvision 0.25.0`, `lerobot 0.4.4`, `transformers 4.57.6` |

> The policy adapter is **task-agnostic**: a single `eval.sh` + `deploy_policy.yml` covers all ten
> tasks. The task is selected by the `<task_name>` argument; the checkpoint is selected by the
> `<checkpoint_path>` argument. Nothing in the adapter is task-specific.
>
> **The only upstream source file we changed is `policy/smolvla/deploy_policy.py`** — a one-line
> return-value contract fix. See [section 9](#9-adapter-change-1-line-contract-fix).

---

## 1. Task → checkpoint mapping

All ten checkpoints live in a **single public** Hugging Face repository,
`solidsnail/robosyn-smolvla-10tasks`, one sub-directory per task. Each sub-directory holds the
complete LeRobot `pretrained_model` contents (7 files) **directly at its root** — the same
layout as the officially released `RoboSynChallenge/SmolVLA_sim_<task>` repositories — so the
downloaded directory can be passed to `eval.sh` unchanged as `checkpoint_path`.

| # | `task_name` (eval.sh arg 1) | HF sub-directory | `checkpoint_path` after download |
| --- | --- | --- | --- |
| 1 | `table_rearrangement` | `table_rearrangement/` | `checkpoints/table_rearrangement` |
| 2 | `click_bell` | `click_bell/` | `checkpoints/click_bell` |
| 3 | `water_pouring` | `water_pouring/` | `checkpoints/water_pouring` |
| 4 | `handle_basket` | `handle_basket/` | `checkpoints/handle_basket` |
| 5 | `items_handover` | `items_handover/` | `checkpoints/items_handover` |
| 6 | `drawer_open_place` | `drawer_open_place/` | `checkpoints/drawer_open_place` |
| 7 | `mixer_operating` | `mixer_operating/` | `checkpoints/mixer_operating` |
| 8 | `item_assembly` | `item_assembly/` | `checkpoints/item_assembly` |
| 9 | `manipulate_pipette` | `manipulate_pipette/` | `checkpoints/manipulate_pipette` |
| 10 | `sample_loading` | `sample_loading/` | `checkpoints/sample_loading` |

Per-task checkpoint URLs (as submitted in the policy-submission form):

```
https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/table_rearrangement
https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/click_bell
https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/water_pouring
https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/handle_basket
https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/items_handover
https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/drawer_open_place
https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/mixer_operating
https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/item_assembly
https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/manipulate_pipette
https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/sample_loading
```

Each `checkpoints/<task>/` directory contains the checkpoint files directly
(`config.json`, `model.safetensors`, `policy_preprocessor.json`,
`policy_preprocessor_step_5_normalizer_processor.safetensors`, `policy_postprocessor.json`,
`policy_postprocessor_step_0_unnormalizer_processor.safetensors`, `train_config.json`).
`eval.sh` treats a `checkpoint_path` that has no `pretrained_model/` sub-folder as the checkpoint
directory itself, so `checkpoints/<task>` is accepted as-is.

## 2. Get the code

```bash
git clone https://github.com/solidsnailz36-cloud/robosyn-submission.git
cd robosyn-submission
```

The repository is a snapshot of `EDEM-AI/RoboSynChallenge` @ `8818c82` with this submission
documentation and a convenience runner added. The only source file changed is
`policy/smolvla/deploy_policy.py` (see section 9).

## 3. Get the checkpoints

```bash
mkdir -p checkpoints
hf download solidsnail/robosyn-smolvla-10tasks --repo-type model --local-dir checkpoints
```

The repository is **public** — no authentication is required.
Result: `checkpoints/<task>/{config.json, model.safetensors, policy_preprocessor.json, ...}` for all 10 tasks.

## 4. Environment

Two Python contexts are involved (see `policy/smolvla/README.md`):

1. **Simulator context** — RoboSynChallenge + EmbodiChain. Install per the official guide:
   <https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html>.
   Expected workspace layout:
   ```
   RoboSynChallenge_ws/
     EmbodiChain/
     RoboSynChallenge/
   ```
2. **SmolVLA worker context** — a Python environment that can `import lerobot`.

```bash
# Option A: install LeRobot from inside this repo (recommended)
bash policy/smolvla/setup_lerobot.sh
export SMOLVLA_LEROBOT_ROOT="$PWD/policy/smolvla/lerobot"

# Option B: an existing LeRobot checkout
export SMOLVLA_LEROBOT_ROOT=/path/to/lerobot

# If the same environment also runs the SmolVLA worker:
export SMOLVLA_PYTHON="$(which python)"

# Python packages:
pip install -r requirements.txt
```

## 5. GPU / CUDA requirement

The official evaluation host is a **single RTX 5090 (Blackwell, compute capability sm_120)** with
**NVIDIA driver 580.95**. Blackwell requires a **CUDA 12.8+** PyTorch build; a CUDA 12.6 wheel —
such as the `torch 2.7.1` pinned by `policy/act/uv.lock` — can **not** run on it.

This submission was trained and evaluated on:

```
Python 3.10.12 | torch 2.10.0+cu128 | torchvision 0.25.0 | lerobot 0.4.4 | transformers 4.57.6
```

If your resolver does not pick the cu128 build by default:

```bash
pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

## 6. Run the evaluation (single task)

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

`<setting>` (arg 2) is supplied by the official evaluation protocol — use `random`. All ten tasks
use the **same** command, changing only `task_name` and `checkpoint_path`.

## 7. Run all ten tasks

```bash
bash submission/run_all.sh            # gpu_id=0
bash submission/run_all.sh 0 20       # gpu_id=0, max_episodes=20
```

## 8. Dependencies

| File | Contents |
| --- | --- |
| `requirements.txt` | Curated SmolVLA worker dependencies, pinned to the versions we used (CUDA 12.8 stack) |
| `requirements-freeze.txt` | The exact 134-package `pip freeze` of the training / evaluation environment |

- Simulator dependencies are those of the official **EmbodiChain** installer; we did not add or pin
  anything there. If that environment already provides compatible `torch` / `torchvision` /
  `transformers`, keep the official ones and only add what is missing (`lerobot`, `torchcodec`,
  `num2words`).
- `train_config.json` inside each checkpoint records `"video_backend": "torchcodec"`, hence the
  `torchcodec` pin.
- The checkpoints are self-contained, but the SmolVLA worker loads the SmolVLM2 tokenizer /
  processor assets by name (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`). On an offline host,
  pre-populate the HF cache once (`hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct`).
- No external network access is required at evaluation time; no online API is called. The policy
  consumes **only** RGB images and robot proprioception.

## 9. Adapter change (1-line contract fix)

The only file we modified relative to upstream `8818c82` is `policy/smolvla/deploy_policy.py`.

`scripts/eval_policy.py` documents and uses the adapter contract

```python
eval(env, model, obs) -> (obs, info, truncated, inference_times)
```

and unpacks four values at the call site (`obs, info, truncated, new_times = policy_pkg.eval(...)`),
as do **all four other adapters in this repository** (`act`, `dp`, `pi0`, `pi05` — each
`return final_obs, info, truncated, inference_times_s`).

The upstream SmolVLA adapter returned **three** values (`return final_obs, info, truncated`),
which raises `ValueError: not enough values to unpack (expected 4, got 3)` on the very first
policy query. We fixed the arity and report the per-inference time using the shared helper
`policy/inference_timing.py`. No other evaluation-pipeline file was touched —
`scripts/eval_policy.py`, the task configs, `deploy_policy.yml` and `eval.sh` are byte-identical
to upstream.

## 10. Provenance / notes

- **Data:** trained on synthetic RoboSynChallenge data (official released `cobotmagic_Sim_*`
  datasets) plus our own self-collected trajectories; no real-robot data, no third-party datasets.
- **Inference inputs:** 3 RGB views (`cam_high`, `cam_left_wrist`, `cam_right_wrist`) and the
  robot joint state (`observation.qpos`). No ground-truth object pose, no contact force, no
  reward, and no `info` / simulator privileged information is used.
- The `is_task_success` return value is only used by the official evaluation loop, not by the
  controller.
- **Checkpoint integrity:** every `model.safetensors` was verified by SHA-256 against the source
  artifact the checkpoints were uploaded from.
