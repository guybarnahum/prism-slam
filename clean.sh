#!/usr/bin/env bash
# PRISM-SLAM uninstall / cleanup helper
# - Uninstalls editable package (if installed)
# - Removes .venv and common build/test caches
# - Optional Hugging Face logout
# Flags:
#   -y|--yes          : non-interactive (assume "yes")
#   --venv-only       : only remove .venv (and uninstall within it)
#   --artifacts-only  : only remove artifacts/caches (keep .venv)
#   --hf-logout       : run `huggingface-cli logout`
#   --dry-run         : print actions without deleting

set -euo pipefail

AUTO_YES=""
VENV_ONLY=""
ARTIFACTS_ONLY=""
HF_LOGOUT=""
DRY_RUN=""
VENV_DIR=".venv"
PKG_NAME="prismslam"

for arg in "$@"; do
  case "$arg" in
    -y|--yes) AUTO_YES="1" ;;
    --venv-only) VENJ_ONLY="1"; VENJ_ONLY="1"; VENJ_ONLY="1" ;; # guard
  esac
done
# Fix a small typo guard above
VENJ_ONLY=""; : "${VENJ_ONLY:=}"
for arg in "$@"; do
  case "$arg" in
    --venv-only) VENV_ONLY="1" ;;
    --artifacts-only) ARTIFACTS_ONLY="1" ;;
    --hf-logout) HF_LOGOUT="1" ;;
    --dry-run) DRY_RUN="1" ;;
    -y|--yes) ;; # already handled
    *) ;;
  esac
done

ask_yes_no() {
  local prompt="$1"
  if [[ -n "${AUTO_YES}" ]]; then
    echo "Auto-yes: $prompt -> yes"; return 0
  fi
  read -p "$prompt " -n 1 -r; echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

run() {
  if [[ -n "${DRY_RUN}" ]]; then
    echo "[dry-run] $*"
  else
    eval "$@"
  fi
}

exists() { [[ -e "$1" ]]; }

echo "PRISM-SLAM clean: ${PWD}"

# --- 1) Uninstall package (prefer inside venv) ---
uninstall_pkg() {
  if [[ -d "${VENV_DIR}" && -x "${VENV_DIR}/bin/activate" ]]; then
    echo "Using ${VENV_DIR} to uninstall ${PKG_NAME}…"
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
    run "pip uninstall -y ${PKG_NAME} || true"
    deactivate || true
  else
    # fallback to current python if user is already in an env
    if command -v python &>/dev/null; then
      echo "Attempting to uninstall ${PKG_NAME} from current Python env…"
      run "python -m pip uninstall -y ${PKG_NAME} || true"
    fi
  fi
}

# --- 2) Remove .venv (optional) ---
remove_venv() {
  if [[ -d "${VENV_DIR}" ]]; then
    if [[ -n "${AUTO_YES}" ]] || ask_yes_no "Remove virtualenv '${VENV_DIR}'? [y/N]"; then
      run "rm -rf '${VENV_DIR}'"
      echo "Removed ${VENV_DIR}"
    else
      echo "Skipped ${VENV_DIR}"
    fi
  fi
}

# --- 3) Remove artifacts & caches ---
remove_artifacts() {
  echo "Removing build artifacts, caches, and checkpoints…"

  # Directories to zap if present
  DIRS=(
    build dist .eggs
    .pytest_cache .mypy_cache .ruff_cache .tox .nox
    .ipynb_checkpoints .jupyter/lab/workspaces
    logs tmp temp cache .cache
    wandb mlruns tensorboard tb_logs runs experiments
    outputs results checkpoints
  )
  # Data directories you may want to keep—comment out to always delete
  MAYBE_DATA_DIRS=( data datasets )

  for d in "${DIRS[@]}"; do
    [[ -d "$d" ]] && run "rm -rf '$d'" && echo "  rm -rf $d"
  done

  for d in "${MAYBE_DATA_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
      if [[ -n "${AUTO_YES}" ]] || ask_yes_no "Remove data dir '$d'? [y/N]"; then
        run "rm -rf '$d'"; echo "  rm -rf $d"
      else
        echo "  kept $d"
      fi
    fi
  done

  # __pycache__ and *.egg-info anywhere
  run "find . -type d -name '__pycache__' -prune -exec rm -rf {} +"
  run "find . -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +"
}

# --- 4) Optional HF logout ---
hf_logout() {
  if [[ -n "${HF_LOGOUT}" ]]; then
    if command -v huggingface-cli &>/dev/null; then
      echo "Logging out of Hugging Face CLI…"
      run "huggingface-cli logout || true"
    else
      echo "huggingface-cli not found; skipping logout."
    fi
  fi
}

# --- Orchestration based on flags ---
if [[ -n "${ARTIFACTS_ONLY}" ]]; then
  remove_artifacts
  hf_logout
  echo "Done (artifacts-only)."
  exit 0
fi

if [[ -n "${VENV_ONLY}" ]]; then
  uninstall_pkg
  remove_venv
  echo "Done (venv-only)."
  exit 0
fi

# Default: full clean
uninstall_pkg
remove_artifacts
remove_venv
hf_logout

echo "✅ Clean complete."
[[ -n "${DRY_RUN}" ]] && echo "(No files were deleted due to --dry-run)"

