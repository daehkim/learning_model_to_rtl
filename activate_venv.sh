#!/usr/bin/env bash
# 가상환경 생성(없을 때) 후 활성화.
# 반드시 현재 셸에서 소스해야 합니다:
#   source ./activate_venv.sh
#
# bash / zsh 모두에서 repo 루트를 기준으로 venv를 둡니다.
# 가능한 경우 Python 3.11(.venv311)을 우선 사용합니다.

if [[ -n "${BASH_VERSION:-}" ]]; then
  _L2R_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _L2R_ROOT="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
  echo "Use bash or zsh: source ./activate_venv.sh" >&2
  return 2 2>/dev/null || exit 2
fi

if command -v python3.11 >/dev/null 2>&1; then
  _PY_BIN="python3.11"
  _VENV="${_L2R_ROOT}/.venv311"
else
  _PY_BIN="python3"
  _VENV="${_L2R_ROOT}/.venv"
fi

if [[ ! -d "${_VENV}" ]]; then
  echo "Creating venv at ${_VENV} with ${_PY_BIN}"
  "${_PY_BIN}" -m venv "${_VENV}" || return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "${_VENV}/bin/activate"

echo "Virtualenv active: ${_VENV} (python: $(command -v python))"
unset _L2R_ROOT _VENV _PY_BIN
