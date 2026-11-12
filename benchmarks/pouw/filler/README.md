# filler: verified tensor work in the capacity a served step leaves idle

The filler is a chain of int8 tensor products (MMAs) on operands derived from a round's coins, with no DRAM traffic,
whose receipt a host recomputes bit for bit. This directory measures, on one RTX PRO 6000 (sm_120), how much of the
GPU's int8 tensor capacity is verified compute when a filler runs beside Llama-3.1-8B's decode and prefill matmuls, and
what that costs the step. Per suite point it reads the step's slowdown, the filler's rate and the coverage, (the proved
fraction of the served MACs + the proved fraction of the filler's) ÷ (capacity × wall time); SCORE is the suite's
time-weighted mean coverage. `scoring.py`'s docstring has the suite, the arithmetic and each scorer version's reading.

The served side is `dcap.cu`'s proxy of the step (the 32 layers' linears through cuBLASLt int8 and int8 decode
attention, each held to the host bit for bit before it is timed), not a served vLLM: norms, RoPE, softmax and
quantization are left out, so a step here is a lower bound on a served step. `comm/` is the same question for a
Llama-3.1-70B tensor-parallel decode step, one process a rank, with the filler inside its collectives.

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

| Job | What it does | Writes under `$OUT` |
| --- | --- | --- |
| `score.py ARM` | the scorer: builds `dcap.cu`, `dscore.cu` and the arm's own driver, runs the receipt, scorer and serving gates, then every suite point in interleaved rounds of serving alone and serving under the arm's placement, and scores the run | `arm.json`, `build.json`, `dgate.jsonl`, `sgate.jsonl`, `dscore.jsonl`, `control.jsonl`, `ncu-filler.csv`, `ncu-control.csv`, `smi.csv`, `score.json`, `ledger/ARM/filler-suite.json` |
| `dcap.sh` | `dcap.cu` alone: its gate, DRAM read bandwidth, the linears over their row counts, decode attention and the decode step over a (batch, context) grid | `gate.jsonl`, `dcap.jsonl` |
| `fill.sh` | the decode step beside a chained int8 GEMM filler on a green-context split | `gate.jsonl`, `fill.jsonl` |
| `dfill.sh` | the DRAM-free filler: its receipts against the host, its rate alone, ncu's DRAM and tensor-pipe counters for one chunk, and the decode step beside it on a split and co-resident | `dgate.jsonl`, `gate.jsonl`, `dalone.jsonl`, `dfill.jsonl`, `ncu-*.csv`, `sass-dfill.txt` |
| `flin.sh` | the fused filler (`flin.cuh`): the decode linears as a GEMM whose filler warps run the chain, each linear against cuBLASLt, and the decode step through it | `fgate.jsonl`, `gate.jsonl`, `flin.jsonl`, `fstep.jsonl`, `ncu-*.csv`, `sass-flin.txt` |
| `psched.sh` | the phase filler inside decode attention on the step's own SMs, beside the partitioned filler | the gates' `.jsonl`, `pfill.jsonl`, `sass-dfill.txt` |
| `psched_probe.sh` | screens through the scorer's drivers: each `PROBES` entry is one `dscore` or `pscore` row; a screen and no score | `probe.jsonl`, `probes.txt`, `smi.csv` |
| `pfill_probe.sh` | the filler's rate alone for each (warps, chains) shape, and ncu's warp-state, scheduler and instruction sections for one chunk | `dgate.jsonl`, `dalone.jsonl`, `ncu-sections-*`, `sass-dfill.txt` |
| `tile_probe.sh` | issue slots an MMA: for each build, the pump's rate alone at each fold and ncu over one launch | `dgate-*.jsonl`, `dalone.jsonl`, `walone.jsonl`, `ncu-pump-*`, `ncu-warp-*`, `sass-*.txt` |
| `fold_probe.sh` | the cost a word of a fold of R rounds, for each R: SASS, the rate alone, the receipt gate, ncu | `dgate-R*.jsonl`, `dalone.jsonl`, `sass-R*.txt`, `sass-R*.counts`, `ncu-R*.txt` |
| `capfill_fold_probe.sh` | what each fold and the step's own SMs are worth beside the step: the pump on a split with each fold, the lane filler, and the pump co-resident at short chunks | `dgate-*.jsonl`, `dalone.jsonl`, `lfill.jsonl`, `co.jsonl` |
| `capfill_co_probe.sh` | a one-block pump on every SM beside the step, co-resident | `dgate-*.jsonl`, `dalone.jsonl`, `co.jsonl` |
| `capfill_fit_probe.sh` | why a one-block pump still holds the step off: ncu's launch statistics of the step's kernels, and the linears and attention each alone beside the pump | `dalone.jsonl`, `co.jsonl`, `ncu-step-*.csv` |
| `capfill_fstep_probe.sh` | the whole decode step co-resident with a one-block pump, its linears through `flin.cuh`'s fitted tiles | `dgate-*.jsonl`, `fgate-*.jsonl`, `flin-*.jsonl`, `fstep.jsonl`, `sass-*.txt` |
| `comm/tpcomm.sh` | on `TC_TP` GPUs of one NUMA node: the tensor-parallel decode step's arms, with the filler's gates on every rank first | `alone.jsonl`, `set*/`, `ncu-filler.csv`, `ncu-control.csv`, `sass.txt`, `topo.txt` |

A shell job also writes `commit.txt`, its build log and `rc.txt`, most write `gpu.txt` (the clocks and board power
read) and `errors.txt`, and a job that flips coins writes `coins.txt`.

## Arms

An arm is a filler and its placement beside the step; the scorer owns the workload, the timing, the gates and the
arithmetic (`arms.py`'s docstring is the contract). `arms.py` holds the baselines, `pfill_arm.py`, `fused_arm.py` and
`psched_arm.py` the others, and `fused_score.py`, `psched_score.py` and `pfill_fp_score.py` measure the arms whose
placement the contract can't name, through their own drivers (`fscore.cu`, `pscore.cu`), into rows the scorer reads.
The files an arm lists as its `sources` are its variant: their bytes' digest is the id its record carries, so an edit
to one of them, a comment included, is a new variant. `score.py check ARM` runs an arm's contract and record checks on
a machine with no GPU.

## Reading a run

These run on any machine, on a job's out directory.

~~~text
python3 benchmarks/pouw/filler/score.py report OUT [--json] [--as v4] [--coins X]   # score a measured run again
python3 benchmarks/pouw/filler/score.py tp-report OUT... --bin|--nvd X              # tpcomm.sh runs' score
python3 benchmarks/pouw/filler/score.py --leaderboard [--refresh] [--local DIR...]  # every scored attempt, best first
python3 benchmarks/pouw/filler/share.py OUT [--fill OUT] [--dfill OUT...] [--fstep OUT...] [--json F] # dcap.sh's tables
python3 benchmarks/pouw/filler/capacity_of.py [OUT] [--lfill|--dfill|--fstep F.jsonl] [--json]   # σ of a run or a probe
python3 benchmarks/pouw/filler/psum.py OUT... [--limit 1.05]                        # psched.sh's rows by point and arm
python3 benchmarks/pouw/filler/psched_probe.py OUT...                               # psched_probe.sh's rows
python3 benchmarks/pouw/filler/pfill_phase_screen.py NAME                           # psched_probe.sh's PROBES, by name
python3 benchmarks/pouw/filler/pfill_proved.py OUT [--json]                         # its SCORE at each proved fraction
python3 benchmarks/pouw/filler/comm/report.py OUT [--json]                          # tpcomm.sh's table and gates
~~~

## What the directory offers

`dfill_ref.py` and `pfill_ref.py` are the filler tiles' receipts in Python, held to the kernels' own host reference
(`pfill_host.cpp` builds `pfill.cuh`'s with g++ for the tests); `pfill_ref.py` and `pfill.cuh` take their folds, key
rules and lane starts from the receipts reference (`verity/protocols/pouw/receipts/`). One test outside the directory,
`../tests/test_receipts_adopters.py`, loads `pfill_ref.py`, `pfill.cuh` and `scoring.py`'s `FOLDS` and `RECEIPT_TAGS`
and holds them to that reference; nothing else outside the directory loads its Python or includes its kernels.
`arms.py`, `score.py`, `scoring.py`, `share.py`, `pfill_ref.py`, `dfill_ref.py`, `capacity_of.py` and `comm/report.py`
list in `__all__` the names other files read; the other files in an arm's `sources` list none, since a byte changed in
one is a new variant of every arm that lists it.
`scoring.py`, `share.py`, the two references and the readers above start no process and touch no GPU, and import
nothing beyond the standard library, numpy, the receipts reference and `verity_machine_model`; `score.py` and the
three arm runners build with `nvcc` and take a lease.

The tests are one suite, `uv run tools/verity/check/suites.py benchmarks/pouw/filler`, on a machine with no GPU.
`../tests/test_budget.py`, `test_eps_budget.py` and `test_lean_headline_dc.py` test `benchmarks/pouw`'s own modules.
