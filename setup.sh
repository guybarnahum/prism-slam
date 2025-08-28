#!/usr/bin/env bash
#
# PRISM-SLAM setup (pyproject-only)
# - Sources .env (optional)
# - Finds Python 3.11/3.12
# - Installs build deps (Linux/macOS)
# - Creates/activates .venv
# - Installs Torch stack:
#     macOS: torch==2.2.2 torchvision==0.17.2 (CPU/MPS)
#     Linux cpu: torch==2.2.2 torchvision==0.17.2 (CPU wheels)
#     Linux t4_gpu: torch==2.4.2 torchvision==0.19.1 (from ${TORCH_CHANNEL:-cu124})
# - Installs project from pyproject extras only (.[cpu] or .[t4_gpu])
#
set -e

# -------- Args --------
AUTO_YES=""
VARIANT=""      # cpu | t4_gpu (auto if not provided)
for arg in "$@"; do
  case "$arg" in
    --yes|-y) AUTO_YES="--yes" ;;
    cpu|t4_gpu) VARIANT="$arg" ;;
    *) ;;
  esac
done

ask_yes_no() {
  local prompt="$1"
  if [[ -n "$AUTO_YES" ]]; then
    echo "Auto-yes: $prompt -> yes"; return 0
  fi
  read -p "$prompt " -n 1 -r; echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

# -------- Colors & cleanup --------
if tput setaf 0 >/dev/null 2>&1; then
  COLOR_GRAY="$(tput setaf 8)"; COLOR_RESET="$(tput sgr0)"
else
  COLOR_GRAY=$'\033[90m'; COLOR_RESET=$'\033[0m'
fi
cleanup_render(){ printf '\r\033[K%s' "${COLOR_RESET}"; tput cnorm 2>/dev/null || true; }
trap cleanup_render EXIT INT TERM

# -------- Spinner --------
run_and_log() {
  local log_file; log_file=$(mktemp)
  local description="$1"; shift
  tput civis 2>/dev/null || true
  local prev_render=""; local cols; cols=$(tput cols 2>/dev/null || echo 120)
  (
    frames=( '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏' ); i=0
    while :; do
      local last_line=""
      [[ -s "$log_file" ]] && last_line=$(tail -n 1 "$log_file" | sed -E 's/\x1B\[[0-9;?]*[ -/]*[@-~]//g')
      local prefix="${frames[i]} ${description} : "; local plain="${prefix}${last_line}"
      (( ${#plain} > cols )) && plain="${plain:0:cols-1}"
      local head="${plain:0:${#prefix}}"; local tail="${plain:${#prefix}}"
      local render="${COLOR_RESET}${head}${COLOR_GRAY}${tail}${COLOR_RESET}"
      [[ "$render" != "$prev_render" ]] && printf '\r\033[K%s' "$render" && prev_render="$render"
      i=$(( (i+1) % ${#frames[@]} )); sleep 0.25
    done
  ) & local spinner_pid=$!

  if ! "$@" >"$log_file" 2>&1; then
    kill "$spinner_pid" &>/dev/null || true; wait "$spinner_pid" &>/dev/null || true
    printf '\r\033[K%s' "${COLOR_RESET}"; printf "❌ %s failed.\n" "$description"
    echo "ERROR LOG :"; cat "$log_file"; echo "END OF ERROR LOG"; rm -f "$log_file"; exit 1
  fi

  kill "$spinner_pid" &>/dev/null || true; wait "$spinner_pid" &>/dev/null || true
  printf '\r\033[K%s' "${COLOR_RESET}"; printf '✅ %s\n' "$description"; rm -f "$log_file"
}

# -------- Step 1: .env --------
if [ -f ".env" ]; then
  echo "Sourcing .env"; set -a; source .env; set +a
fi
# Bridge HF token envs (optional)
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"; fi

VENV_DIR=".venv"; PYTHON_BIN=""

# -------- Step 2: Python --------
echo "Searching for Python 3.11/3.12"
if command -v python3.11 &>/dev/null; then PYTHON_BIN="python3.11"
elif command -v python3.12 &>/dev/null; then PYTHON_BIN="python3.12"
elif command -v python3 &>/dev/null; then PYTHON_BIN="python3"; echo "⚠️  Falling back to 'python3'"
else echo "❌ Python 3.11/3.12 not found"; exit 1; fi
echo "✅ Using: $($PYTHON_BIN --version)"

# -------- Step 3: Build deps --------
uname_s="$(uname -s)"
if [[ "$uname_s" == "Linux" ]]; then
  if ! command -v g++ &>/dev/null || ! command -v cmake &>/dev/null; then
    run_and_log "apt-get update" sudo apt-get update
    run_and_log "Install build tools" sudo apt-get install -y build-essential g++ cmake
  fi
  if command -v nvidia-smi &>/dev/null && ! command -v nvcc &>/dev/null; then
    echo "GPU detected, CUDA Toolkit not found."
    if ask_yes_no "Install CUDA Toolkit 12.4 now? [y/N]"; then
      run_and_log "CUDA keyring" wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
      run_and_log "Install keyring" sudo dpkg -i cuda-keyring_1.1-1_all.deb
      run_and_log "apt update (CUDA)" sudo apt-get update
      rm -f cuda-keyring_1.1-1_all.deb
      run_and_log "Install CUDA 12.4" sudo apt-get -y install cuda-toolkit-12-4
    fi
  fi
elif [[ "$uname_s" == "Darwin" ]]; then
  if ! xcode-select -p &>/dev/null; then echo "Installing Xcode Command Line Tools…"; xcode-select --install || true; fi
fi

# -------- Step 4: venv --------
[ -d "$VENV_DIR" ] || run_and_log "Create venv at ${VENV_DIR}" "$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
run_and_log "Upgrade pip" pip install --upgrade pip wheel

# -------- Variant selection --------
if [[ -z "$VARIANT" ]]; then
  if [[ "$uname_s" == "Darwin" ]]; then
    VARIANT="cpu"
  elif command -v nvidia-smi &>/dev/null; then
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
    echo "Detected GPU: ${gpu_name:-unknown}"
    if [[ -n "$AUTO_YES" ]] || ask_yes_no "Use the GPU install ([t4_gpu])? [y/N]"; then VARIANT="t4_gpu"; else VARIANT="cpu"; fi
  else VARIANT="cpu"; fi
fi
echo "Chosen variant: ${VARIANT}"

# -------- ABI guards --------
run_and_log "Ensure NumPy/SciPy ABI compatibility" pip install "numpy<2" "scipy<1.13" --upgrade

# -------- Step 5: Torch pins --------
TORCH_CHANNEL="${TORCH_CHANNEL:-cu124}"
TORCH_VER_CPU="2.2.2"; VISION_VER_CPU="0.17.2"
TORCH_VER_GPU="2.4.2"; VISION_VER_GPU="0.19.1"

install_torch_mac(){   run_and_log "Torch (macOS)" pip install "torch==${TORCH_VER_CPU}" "torchvision==${VISION_VER_CPU}"; }
install_torch_cpu(){   run_and_log "Torch (CPU)"   pip install --extra-index-url https://download.pytorch.org/whl/cpu \
                                           "torch==${TORCH_VER_CPU}" "torchvision==${VISION_VER_CPU}"; }
install_torch_gpu(){   run_and_log "Torch (CUDA ${TORCH_CHANNEL})" pip install --extra-index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}" \
                                           "torch==${TORCH_VER_GPU}" "torchvision==${VISION_VER_GPU}"; }

if [[ "$uname_s" == "Darwin" ]]; then
  install_torch_mac
else
  case "$VARIANT" in
    cpu)    install_torch_cpu ;;
    t4_gpu) install_torch_gpu ;;
    *) echo "❌ Unknown variant '$VARIANT'"; exit 1 ;;
  esac
fi

# -------- Step 6: Project deps (pyproject extras only) --------
run_and_log "Install project (editable)" pip install -e .
if [[ "$uname_s" == "Darwin" || "$VARIANT" == "cpu" ]]; then
  run_and_log "Install extras [cpu]"    pip install -e ".[cpu]"
elif [[ "$VARIANT" == "t4_gpu" ]]; then
  run_and_log "Install extras [t4_gpu]" pip install -e ".[t4_gpu]"
fi

# -------- Step 7: Optional Hugging Face auth --------
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  run_and_log "Hugging Face login" huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN" --add-to-git-credential
fi

echo ""
echo "✅ Setup complete (variant: ${VARIANT}; OS: ${uname_s})."
echo "Activate:   source ${VENV_DIR}/bin/activate"
echo "Verify:     python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__)'"
