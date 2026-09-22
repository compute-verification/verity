# Pod-side environment for every W3 run (source me).  Profile flags per contract §2.
#   W3_VENV=/workspace/venv          vllm 0.28.0 (PyPI)            -> results in /workspace/results/w3
#   W3_VENV=/workspace/venv-nightly  vllm 0.28.1rc1.dev472+gd9105ea80 (POC pin) -> /workspace/results/w3n
export W3_VENV=${W3_VENV:-/workspace/venv}
source "$W3_VENV/bin/activate"
export HF_HOME=/workspace/hf
export HF_HUB_OFFLINE=1
export VLLM_BATCH_INVARIANT=1
export PYTHONPATH=/workspace/veritor/vllm-poc:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
if [ "$W3_VENV" = "/workspace/venv" ]; then
  export W3_RESULTS=${W3_RESULTS:-/workspace/results/w3}
else
  export W3_RESULTS=${W3_RESULTS:-/workspace/results/w3n}
fi
export W3_WORKLOAD=/workspace/results/w3/workload.json   # one workload for every version
mkdir -p "$W3_RESULTS/prod" "$W3_RESULTS/logs"
cd /workspace/veritor/vllm-poc
