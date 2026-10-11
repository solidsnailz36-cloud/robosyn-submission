#!/bin/bash
# ----------------------------------------------------------------------------
# Evaluate all 10 RoboSynChallenge tasks with this submission's SmolVLA checkpoints.
#
#   bash submission/run_all.sh [gpu_id] [max_episodes] [setting]
#
# Defaults: gpu_id=0, max_episodes=20, setting=random
#
# Prerequisites:
#   - checkpoints/<task>/  (7 files directly inside; see SUBMISSION.md section 3)
#   - the SmolVLA worker environment (see SUBMISSION.md section 4)
# ----------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID="${1:-0}"
MAX_EPISODES="${2:-20}"
SETTING="${3:-random}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints}"

TASKS=(
  click_bell
  drawer_open_place
  handle_basket
  item_assembly
  items_handover
  manipulate_pipette
  mixer_operating
  sample_loading
  table_rearrangement
  water_pouring
)

echo "==================================================================="
echo "  RoboSynChallenge submission: solidsnail-smolvla-10tasks"
echo "  gpu_id=$GPU_ID  max_episodes=$MAX_EPISODES  setting=$SETTING"
echo "  checkpoint_root=$CHECKPOINT_ROOT"
echo "==================================================================="

rc_total=0
for task in "${TASKS[@]}"; do
    ckpt="$CHECKPOINT_ROOT/$task"
    if [[ ! -d "$ckpt/pretrained_model" && ! -f "$ckpt/config.json" ]]; then
        echo "!! SKIP $task : checkpoint not found at $ckpt"
        rc_total=1
        continue
    fi
    echo ""
    echo ">>> [$task] ==============================================="
    bash policy/smolvla/eval.sh "$task" "$SETTING" "$ckpt" "$GPU_ID" \
        --pytorch_device cuda \
        --headless true \
        --renderer auto \
        --max_episodes "$MAX_EPISODES" \
        --smolvla_steps 10 \
        --smolvla_rescale_gripper auto
    rc=$?
    echo "<<< [$task] exit=$rc"
    [[ $rc -ne 0 ]] && rc_total=1
done

echo ""
echo "==================================================================="
echo "  done. results under eval_result/<task>/smolvla/<setting>/..."
echo "==================================================================="
exit $rc_total
