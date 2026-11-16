# The filler

The filler is a chain of int8 tensor products (MMAs) on operands derived from a round's coins, with no DRAM traffic,
whose receipt a host recomputes bit for bit. This directory measures, on one RTX PRO 6000 (sm_120), how much of the
GPU's int8 tensor capacity is verified compute when a filler runs beside Llama-3.1-8B's decode and prefill matmuls, and
what that costs the step.

The served side is `dcap.cu`'s proxy of the step (the 32 layers' linears through cuBLASLt int8 and int8 decode
attention, each held to the host bit for bit before it is timed), not a served vLLM: norms, RoPE, softmax and
quantization are left out, so a step here is a lower bound on a served step. `comm/` is the same question for a
Llama-3.1-70B tensor-parallel decode step, one process a rank, with the filler inside its collectives.

## Try it

On any machine, run an arm's contract and record checks without a GPU:

~~~sh
uv run python3 benchmarks/pouw/filler/score.py check fill-v1
~~~

It prints `fill-v1: the contract and the record's limits hold`.

## Jobs

Each job is the command of a `research run` from a pushed commit, with the source tree as its working directory, on a
node with the GPU and CUDA 13.0; it takes its GPU with `gpu-lease`, runs its gates before anything is timed, and reads
the clocks without setting them. `run.sh JOB [ARGS...]` is the directory's runner: it starts the job's script (`score`
is `score.py`, `tpcomm` is `comm/tpcomm.sh`, every other job the script of its name) with the rest of its arguments and
the environment as they are. A job's knobs are environment variables, listed in its script's header.

~~~text
research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  -- bash benchmarks/pouw/filler/run.sh score ARM          # ARM: a name of arms.py, or path/to/module.py:Class
research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  -- python3 benchmarks/pouw/filler/score.py ARM           # the same job
~~~
