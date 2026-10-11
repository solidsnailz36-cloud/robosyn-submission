#!/bin/bash
# ----------------------------------------------------------------------------
# Optional helper for users who only have RoboSynChallenge checked out.
# It clones LeRobot inside policy/smolvla/lerobot and installs it editable in
# the current Python environment.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-$SCRIPT_DIR/lerobot}"
LEROBOT_REPO="${LEROBOT_REPO:-https://github.com/huggingface/lerobot.git}"
# Pinned to the release the checkpoints were validated against. The worker's
# processor pipelines are version sensitive, so do NOT default this to a
# floating branch (e.g. `main`) -- the environment must be reproducible.
LEROBOT_REF="${LEROBOT_REF:-v0.6.1}"

if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo "LeRobot already exists: $INSTALL_DIR"
else
    git clone "$LEROBOT_REPO" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
git fetch --tags origin || true
if ! git checkout "$LEROBOT_REF" 2>/dev/null; then
    echo "!! ref '$LEROBOT_REF' not found; falling back to the PyPI release pinned in requirements.txt"
    python -m pip install -U "lerobot==0.6.1"
fi
python -m pip install -U pip
python -m pip install -e . || true

# ---------------------------------------------------------------------------
# HARD requirement that is easy to miss.
#
# The SmolVLM text normalizer imports `num2words` when the processor is built.
# Without it the worker dies on the very first rollout with:
#   ImportError: Package 'num2words' is required to run SmolVLM processor.
# `pip install -e .` does NOT pull it in, so install it explicitly here.
# ---------------------------------------------------------------------------
python -m pip install "num2words==0.5.14"

# --- self-check: print what actually got installed ---------------------------
python - <<'PY'
import importlib
for m in ("lerobot", "num2words", "transformers"):
    try:
        mod = importlib.import_module(m)
        print("ok  %-14s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("ERR %-14s %s: %s" % (m, type(e).__name__, e))
PY

echo "LeRobot is ready at: $INSTALL_DIR"
echo "For eval, set:"
echo "  export SMOLVLA_LEROBOT_ROOT=$INSTALL_DIR"
