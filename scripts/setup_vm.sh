#!/usr/bin/env bash
# One-time setup ON the TPU VM. Follows vllm-torchtpu docs/getting_started/installation.md:
# Python 3.12 only, libibverbs for tpu-sync-torch, CPU torch index (avoid CUDA wheels).
# This is the "zero-to-serving" act of the video: run it under `script` or screen-record it.
set -euo pipefail
T0=$(date +%s)
sudo apt-get update -y && sudo apt-get install -y libibverbs1 git curl
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }

uv venv --python 3.12 ~/venv-vllm
source ~/venv-vllm/bin/activate
export UV_TORCH_BACKEND=cpu
uv pip install vllm-torchtpu openai

python -c "from vllm.platforms import current_platform; print('vllm platform:', current_platform.device_name)"
mkdir -p results
{ echo "date: $(date -Is)"; python -V; uv pip freeze | grep -Ei '^(vllm|torch|torch[-_]tpu|tpu|jax|libtpu|openai)'; } \
  | tee results/env_versions.txt
echo "setup_s: $(( $(date +%s) - T0 ))"
echo "Next: source ~/venv-vllm/bin/activate && huggingface-cli login (if needed) && bash scripts/run_experiment.sh"
