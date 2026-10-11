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
| Runtime Python | 3.10.12 — `torch 2.10.0+cu128`, `torchvision 0.25.0`, `lerobot 0.6.2`, `transformers 4.57.6` |

> The policy adapter is **task-agnostic**: a single `eval.sh` + `deploy_policy.yml` covers all ten
> tasks. The task is selected by the `<task_name>` argument; the checkpoint is selected by the
> `<checkpoint_path>` argument. Nothing in the adapter is task-specific.
>
> **We changed two upstream source files, both under `policy/smolvla/`** — `deploy_policy.py`
> (return-value contract) and `smolvla_worker.py` (input image dtype + per-episode RNG seeding).
> Without these, the upstream SmolVLA path does not run at all. See [section 9](#9-adapter-changes).

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
documentation and a convenience runner added. Two source files changed, both under
`policy/smolvla/` (see section 9).

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
2. **SmolVLA worker context** — a Python environment that can `import lerobot` **>= 0.6**
   (`requirements.txt` pins `lerobot==0.6.2`; see section 9.4).

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
Python 3.10.12 | torch 2.10.0+cu128 | torchvision 0.25.0 | lerobot 0.6.2 | transformers 4.57.6
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
  --smolvla_steps 10 \
  --smolvla_rescale_gripper auto
```

> Keep `--smolvla_rescale_gripper auto` (also the default in `deploy_policy.yml`).
> Do **not** pass `true`: `auto` inspects each checkpoint's `action.max` statistics and,
> for these checkpoints, correctly resolves to *no* rescaling (their gripper channels
> already carry physical values, max ~0.05). Forcing `true` rescales the gripper incorrectly.

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
| `requirements-freeze.txt` | The exact 134-package `pip freeze` of the training / evaluation environment, with `lerobot` bumped to the 0.6.2 the worker needs |

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

## 9. Adapter changes

Two files under `policy/smolvla/` differ from upstream `8818c82`. Every other evaluation-pipeline
file (`scripts/eval_policy.py`, the task configs, `deploy_policy.yml`, `eval.sh`) is
byte-identical to upstream.

### 9.1 `deploy_policy.py` — 3-value → 4-value return contract

`scripts/eval_policy.py` unpacks **four** values at the call site
(`obs, info, truncated, new_times = policy_pkg.eval(...)`, line 802), and so do **all five other
adapters** in this repository (`act`, `dp`, `pi0`, `pi05`, `Your_Policy` — each
`return final_obs, info, truncated, inference_times_s`). The upstream SmolVLA adapter returned
**three**:

```python
return final_obs, info, truncated            # upstream 8818c82
```

which raises `ValueError: not enough values to unpack (expected 4, got 3)` on the very first
policy query. We fixed the arity and now report the per-inference wall-clock time, which feeds
the 5% inference-efficiency score.

### 9.2 `smolvla_worker.py` — image dtype (uint8 → float)

The adapter's `encode_obs()` delivers images as **`uint8` in `[0, 255]`**. The SmolVLM2 vision
tower upsamples with a bilinear resize, which has no `uint8` kernel, so the worker dies inside the
**first** `model.infer()`:

```
RuntimeError: NotImplementedError: "upsample_bilinear2d_out_frame" not implemented for 'Byte'
```

The worker now rescales any `uint8` tensor to `float [0, 1]` before the preprocessor (whose visual
step is `IDENTITY`, i.e. it does not rescale on its own).

### 9.3 `smolvla_worker.py` — per-episode RNG seeding (reproducibility)

SmolVLA samples its flow-matching noise from the global torch RNG. `eval_policy.py` seeds only the
simulator process; the policy worker is a separate process and was never seeded, so two runs of
the same episode produced different action sequences (measured per-episode flip rate 20-35%). The
worker now re-seeds `random` / `numpy` / `torch` (CPU + CUDA) from the episode seed published on
`env.unwrapped._robosyn_episode_seed`, falling back to upstream behaviour when it is absent.

### 9.4 LeRobot version

`requirements.txt` pins **`lerobot==0.6.2`**. The worker imports `policy_action_to_transition` /
`transition_to_policy_action` from `lerobot.processor`; those are re-exported from that module only
from 0.6 on (on 0.4.x they live only under `lerobot.processor.converters`, which the worker also
falls back to).

### 9.5 Verification

All of the above was verified end-to-end on an RTX 4090 + EmbodiChain by running
`bash policy/smolvla/eval.sh table_rearrangement random <ckpt> 0`: the adapter reaches its
`model.infer()` loop with no exception, the expert-feasibility check reproduces the reference
seed, and the per-episode outcomes match the reference run.

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
