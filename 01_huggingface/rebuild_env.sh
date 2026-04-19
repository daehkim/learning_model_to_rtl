#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv311"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 is required but not found. Install Python 3.11 first." >&2
  exit 1
fi

echo "Recreating ${VENV_DIR}"
rm -rf "${VENV_DIR}"
python3.11 -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/01_huggingface/requirements.txt"

echo "Done. Activate with:"
echo "  source \"${ROOT_DIR}/activate_venv.sh\""
