<div align="center">
<h1>RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills</h1>

<h2 align="center"> 👉<a href="https://edem-ai.github.io/robosynchallenge.github.io/">Webpage</a> | <a href="https://edem-ai.github.io/RoboSynChallenge/html/">Document</a> | <a href="">Paper</a> | <a href="https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard">Leaderboard</a></h2>

![image](misc/robosynchallenge-pipeline.png)

</div>

---

# 📦 Preliminary submission — team `solidsnail`

**Policy:** SmolVLA (LeRobot), fine-tuned **one checkpoint per task** ·
**Task coverage: all 10 tasks** · **Artifact/experiment name:** `solidsnail-smolvla-10tasks`

| Item | Value |
| --- | --- |
| **Checkpoints (Hugging Face)** | **<https://huggingface.co/solidsnail/robosyn-smolvla-10tasks>** — **public**, 10 task sub-directories × 7 files, 8.44 GB total |
| **Adapter entry points** | [`policy/smolvla/eval.sh`](policy/smolvla/eval.sh) · [`policy/smolvla/deploy_policy.yml`](policy/smolvla/deploy_policy.yml) · [`policy/smolvla/deploy_policy.py`](policy/smolvla/deploy_policy.py) |
| **Base code revision** | snapshot of upstream `EDEM-AI/RoboSynChallenge` @ `8818c82` |
| **Training base weights** | `lerobot/smolvla_base` (VLM backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) |
| **Runtime requirements** | [`requirements.txt`](requirements.txt) (curated) · [`requirements-freeze.txt`](requirements-freeze.txt) (exact 134-package freeze) |

This repository is a snapshot of the upstream repository plus the submission documentation and a
convenience runner. **Two SmolVLA adapter files carry fixes without which the upstream SmolVLA
path cannot run at all** — see [Adapter changes](#5-adapter-changes-three-fixes) below.

---

## 1. Task → checkpoint mapping

All ten checkpoints live in a **single public** Hugging Face repository,
`solidsnail/robosyn-smolvla-10tasks`, one sub-directory per task. Each sub-directory holds the
complete LeRobot `pretrained_model` contents (7 files) **directly at its root** — the same layout
as the officially released `RoboSynChallenge/SmolVLA_sim_<task>` repositories — so the downloaded
directory can be passed to `eval.sh` unchanged as `checkpoint_path`.

| # | `task_name` (arg 1 of `eval.sh`) | Checkpoint URL | `checkpoint_path` after download |
| --- | --- | --- | --- |
| 1 | `table_rearrangement` | <https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/table_rearrangement> | `checkpoints/table_rearrangement` |
| 2 | `click_bell` | <https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/click_bell> | `checkpoints/click_bell` |
| 3 | `water_pouring` | <https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/water_pouring> | `checkpoints/water_pouring` |
| 4 | `handle_basket` | <https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/handle_basket> | `checkpoints/handle_basket` |
| 5 | `items_handover` | <https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/items_handover> | `checkpoints/items_handover` |
| 6 | `drawer_open_place` | <https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/drawer_open_place> | `checkpoints/drawer_open_place` |
| 7 | `mixer_operating` | <https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/mixer_operating> | `checkpoints/mixer_operating` |
| 8 | `item_assembly` | <https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/item_assembly> | `checkpoints/item_assembly` |
| 9 | `manipulate_pipette` | <https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/manipulate_pipette> | `checkpoints/manipulate_pipette` |
| 10 | `sample_loading` | <https://huggingface.co/solidsnail/robosyn-smolvla-10tasks/tree/main/sample_loading> | `checkpoints/sample_loading` |

Every `checkpoints/<task>/` directory contains `config.json`, `model.safetensors`,
`policy_preprocessor.json`, `policy_preprocessor_step_5_normalizer_processor.safetensors`,
`policy_postprocessor.json`, `policy_postprocessor_step_0_unnormalizer_processor.safetensors`
and `train_config.json`. `eval.sh` treats a `checkpoint_path` without a `pretrained_model/`
sub-folder as the checkpoint directory itself, so `checkpoints/<task>` is accepted as-is.

## 2. Quick start

```bash
# 1) get the code
git clone https://github.com/solidsnailz36-cloud/robosyn-submission.git
cd robosyn-submission

# 2) get all ten checkpoints (single command, ~8.4 GB; the repo is public)
mkdir -p checkpoints
hf download solidsnail/robosyn-smolvla-10tasks --repo-type model --local-dir checkpoints

# 3) run one task
bash policy/smolvla/eval.sh click_bell random checkpoints/click_bell 0 \
  --pytorch_device cuda --headless true --renderer auto

# 4) or run all ten tasks sequentially
bash submission/run_all.sh          # gpu_id=0
bash submission/run_all.sh 0 20     # gpu_id=0, max_episodes=20
```

## 3. Environment

Two Python contexts are involved (see [`policy/smolvla/README.md`](policy/smolvla/README.md)).
They may be the same environment, but they do not have to be.

**A. Simulator context — RoboSynChallenge + EmbodiChain.**
Install per the official guide:
<https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html>.
Expected workspace layout:

```
RoboSynChallenge_ws/
  EmbodiChain/
  RoboSynChallenge/     <- this repository
```

**B. SmolVLA worker context — a Python environment that can `import lerobot`.**

> **Pinned to LeRobot 0.4.4** -- the exact release this submission was trained and evaluated
> with, and the newest one installable in this stack. LeRobot 0.5.0 raised its own
> `Requires-Python` to `>=3.12`, while this environment is Python 3.10 (the official
> repository declares the same: `requires-python = ">=3.10"`). Asking pip for a 0.5+
> version therefore aborts with "Package 'lerobot' requires a different Python".
> A >= 0.6 host is still fine: the worker reads `policy_action_to_transition` /
> `transition_to_policy_action` from `lerobot.processor.converters` on 0.4.x and falls back
> the other way on newer releases.

```bash
# Option A: install LeRobot from inside this repository (recommended)
bash policy/smolvla/setup_lerobot.sh          # clones LeRobot and pip-installs it
export SMOLVLA_LEROBOT_ROOT="$PWD/policy/smolvla/lerobot"

# Option B: reuse an existing LeRobot checkout / venv
export SMOLVLA_LEROBOT_ROOT=/path/to/lerobot

# If the same environment also runs the SmolVLA worker (it may be the same one):
export SMOLVLA_PYTHON="$(which python)"

# Python packages (pins lerobot==0.4.4, torch cu128, transformers 4.57.6):
pip install -r requirements.txt
```

<details>
<summary><b>GPU / CUDA requirement (please read)</b></summary>

The official evaluation host is a **single RTX 5090 (Blackwell, sm_120)** with **NVIDIA driver
580.95**. Blackwell requires a **CUDA 12.8+** PyTorch build. A CUDA 12.6 wheel — such as the
`torch 2.7.1` pinned by `policy/act/uv.lock` in this repository — can **not** run on it.

Our stack is CUDA 12.8:

```
Python 3.10.12 | torch 2.10.0+cu128 | torchvision 0.25.0 | lerobot 0.4.4 | transformers 4.57.6
```

If your resolver does not pick the cu128 build by default:

```bash
pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

The exact 134-package freeze we trained and evaluated with is in
[`requirements-freeze.txt`](requirements-freeze.txt).
</details>

<details>
<summary><b>Model / tokenizer download</b></summary>

The checkpoints (`model.safetensors` + the `policy_*` processor configs) are **self-contained**:
all policy weights are inside the downloaded directories. The SmolVLA worker additionally loads
the SmolVLM2 text tokenizer / processor assets by name (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
recorded in each checkpoint's `config.json`). On an offline host, pre-populate the Hugging Face
cache once:

```bash
export HF_HOME=<writable cache dir>
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```
</details>

### 3.1 Pre-flight check (run this before the full evaluation)

Skipping this is how a submission silently scores **zero on all ten tasks**: the harness
starts, the simulator comes up, and the worker dies at the *very first* policy query.
The command below reproduces the worker's own startup path — **no simulator required,
~30 s**:

```bash
CK=checkpoints/click_bell        # any of the ten
echo "" | timeout 420 python policy/smolvla/smolvla_worker.py \
    --checkpoint-dir "$CK" --device cuda ; echo "rc=$?"
```

`rc=0`, no `ImportError`, and no repeated `Network is unreachable` means the environment
is ready. The three failure modes that abort a run at the first rollout:

| Symptom in the worker output | Cause | Fix |
| --- | --- | --- |
| `ImportError: Package 'num2words' is required to run SmolVLM processor` | `num2words` is missing from the worker environment — `pip install -e .` for LeRobot does **not** pull it in | `pip install num2words==0.5.14` |
| `Loading HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...` followed by `Network is unreachable` retries (1/2/4/8/8 s) | the SmolVLM2 tokenizer/processor assets are fetched **by name** on first use and the host has no route to the Hub | pre-populate the cache (panel above), or point `HF_ENDPOINT` at a reachable mirror |
| `RuntimeError: SmolVLA worker exited unexpectedly.` | always a *consequence*, never the cause — read the worker's own traceback printed above it | fix the underlying error (the two rows above) |

Note that the checkpoints themselves are self-contained; only the SmolVLM2
tokenizer/processor assets come from the Hub by name.

## 4. Run the evaluation

Single task — the signature is
`bash policy/smolvla/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]`:

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

> Keep `--smolvla_rescale_gripper auto` (it is also the default in
> `deploy_policy.yml`). Do **not** pass `true`: `auto` inspects each checkpoint's
> `action.max` statistics, and for these checkpoints it correctly resolves to *no*
> rescaling (their gripper channels already carry physical values, max ~0.05),
> so forcing `true` would rescale the gripper incorrectly.

All ten tasks use the **same** command, changing only `task_name` and `checkpoint_path`:

```bash
for t in table_rearrangement click_bell water_pouring handle_basket items_handover \
         drawer_open_place mixer_operating item_assembly manipulate_pipette sample_loading; do
  bash policy/smolvla/eval.sh "$t" random "checkpoints/$t" 0 \
    --pytorch_device cuda --headless true --renderer auto
done
```

Or simply:

```bash
bash submission/run_all.sh
```

Results, including videos, are written under
`eval_result/<task_name>/smolvla/<setting>/<train_config_name>/<model_name>/<timestamp>/`.

`<setting>` (arg 2) is supplied by the official evaluation protocol — use `random`.

## 5. Adapter changes (three fixes)

Two files under `policy/smolvla/` differ from upstream `8818c82`. Every other evaluation-pipeline
file (`scripts/eval_policy.py`, the environments, the task configs, `deploy_policy.yml`,
`eval.sh`) is byte-identical to upstream.

### 5.1 `deploy_policy.py` — 3-value → 4-value return contract

`scripts/eval_policy.py` unpacks **four** values at the call site
(`obs, info, truncated, new_times = policy_pkg.eval(eval_env, model, obs)`, line 802), and so do
**all five other adapters** in this repository (`act`, `dp`, `pi0`, `pi05`, `Your_Policy` — each
`return final_obs, info, truncated, inference_times_s`). The upstream SmolVLA adapter returned
**three**:

```python
return final_obs, info, truncated            # upstream 8818c82
```

which raises `ValueError: not enough values to unpack (expected 4, got 3)` on the very first
policy query. The adapter now also reports the (synchronous, out-of-process) inference
wall-clock time, which feeds the 5% inference-efficiency score:

```python
_t0 = time.time()
actions = model.infer(encoded_obs, task=instruction)
inference_times_s = time.time() - _t0
...
return final_obs, info, truncated, [inference_times_s]
```

### 5.2 `smolvla_worker.py` — image dtype (uint8 → float)

The adapter's `encode_obs()` delivers images as **`uint8` in `[0, 255]`** (see
`_extract_image()`, lines 49-54). The SmolVLM2 vision tower upsamples with a bilinear resize,
which has no `uint8` kernel, so the worker dies inside the **first** `model.infer()`:

```
RuntimeError: NotImplementedError: "upsample_bilinear2d_out_frame" not implemented for 'Byte'
```

The worker now rescales any `uint8` tensor to `float [0, 1]` before the preprocessor (whose
visual step is `IDENTITY`, i.e. it does not rescale on its own).

### 5.3 `smolvla_worker.py` — per-episode RNG seeding (reproducibility)

SmolVLA samples its flow-matching noise from the global torch RNG. `eval_policy.py` seeds only
the simulator process; the policy worker is a separate process and was never seeded, so two runs
of the same episode produced different action sequences (measured per-episode flip rate
20-35%) — the success rate was noisy and per-episode A/B pairing was impossible. The worker now
re-seeds `random` / `numpy` / `torch` (CPU + CUDA) from the episode seed that `eval_policy.py`
publishes on `env.unwrapped._robosyn_episode_seed`, transparently falling back to upstream
behaviour when that attribute is absent.

### 5.4 LeRobot version

`requirements.txt` pins **`lerobot==0.4.4`** -- the release this submission was trained and
evaluated with, and the newest one installable here: LeRobot 0.5.0+ requires Python >=3.12,
while this stack is Python 3.10 (matching the official repository's
`requires-python = ">=3.10"`). `smolvla_worker.py` reads
`policy_action_to_transition` / `transition_to_policy_action` from
`lerobot.processor.converters` (the 0.4.x location) and falls back the other way on newer
releases, so it runs on either 0.4.x or >= 0.6.

### 5.5 Verification

All three fixes were verified on a 4090 + `EmbodiChain`, running
`policy/smolvla/eval.sh table_rearrangement random <ckpt> 0`: the adapter reaches its
`model.infer()` loop with no exception, the expert-feasibility check reproduces the reference
seed, and per-episode behaviour matches the reference run.

## 6. Notes on the policy

- **Inputs at inference:** 3 RGB views (`cam_high`, `cam_left_wrist`, `cam_right_wrist`) and the
  robot joint state (`observation.qpos`). No ground-truth object pose, no contact force, no reward,
  no `info`, and no other simulator-privileged information.
- `is_task_success` is only consumed by the official evaluation loop; the controller never uses it.
- The policy is **task-agnostic in the adapter** — task selection is purely the `task_name`
  argument plus the selected checkpoint. All ten checkpoints share the same SmolVLA architecture.
- No network access and no external API are required at evaluation time.

## 7. Provenance

- **Training data:** official released RoboSynChallenge simulated datasets (`cobotmagic_Sim_*`)
  plus our own self-collected EmbodiChain trajectories. No real-robot data; no third-party datasets.
- **Training base weights:** `lerobot/smolvla_base`.
- **Checkpoint integrity:** every `model.safetensors` was verified by SHA-256 against the source
  artifact the checkpoints were uploaded from.

---

# Contents

- [📦 Preliminary submission — team `solidsnail`](#-preliminary-submission--team-solidsnail)
- [1. Task → checkpoint mapping](#1-task--checkpoint-mapping)
- [2. Quick start](#2-quick-start)
- [3. Environment](#3-environment)
- [4. Run the evaluation](#4-run-the-evaluation)
- [5. Adapter changes (three fixes)](#5-adapter-changes-three-fixes)
- [6. Notes on the policy](#6-notes-on-the-policy)
- [7. Provenance](#7-provenance)
- [Upstream: Installation](#installtion)
- [Upstream: Datasets](#datasets)
- [Upstream: Training and Evaluation](#training-and-evaluation)
- [Upstream: Released Checkpoint Results](#released-checkpoint-results)
- [Upstream: LeaderBoard](#leaderboard)

# Installtion
Based on the [**EmbodiChain**](https://dexforce.github.io/EmbodiChain/main/quick_start/install.html), we offer both `docker` and `local` installation methods. For detailed installation instructions, please refer to [**Installation Document**](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html).

# Datasets
We provide 1,000 pre-collected trajectories per task as part of the open-source release **RoboSynChallenge** Dataset. The datasets hosted on HuggingFace are available at [here](https://edem-ai.github.io/robosynchallenge.github.io/#/data).

However, we still strongly recommend users to perform data collection themselves. For detailed data collection instructions, please refer to [**Data Collection Document**](https://edem-ai.github.io/RoboSynChallenge/html/tutorials/collect_data.html).


# Training and Evaluation
Currently, RoboSynChallenge integrates training and evaluation for <a href="https://github.com/Physical-Intelligence/openpi">PI0</a>, <a href="https://github.com/Physical-Intelligence/openpi">PI0.5</a>, and <a href="https://github.com/thu-ml/Motus">Motus</a>. Detailed procedures can be found in the documentation for the corresponding strategies: 👉<a href="https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/index.html">Webpage</a>.
In addition, you can easily extend your own policys for training and evaluation by following the documentation 👉<a href="https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html.html">Webpage</a>.

# Released Checkpoint Results

100-episode simulation evaluations for the released ACT and Diffusion Policy checkpoints are published in [`evaluation_results`](evaluation_results/README.md). The machine-readable result file pins success rate, action steps, millisecond inference time, Hugging Face checkpoint revisions, and the `random` protocol configuration.

# LeaderBoard
The full leaderboard and setting can be found in: https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard.
