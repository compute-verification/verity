# The memory benchmark

This benchmark measures how much of a card's memory read rate a memory audit, PoUB, leaves uncertified. The audit has
the card read its store, here most of the card's memory, in timed passes, while vLLM serves Llama 3.2 1B from that
store. It runs on one GPU machine of your own. Lower is better.

The session is the node's own, on one card. A vLLM engine loads the checkpoint and copies its weights into one arena of
most of the card's memory, which is the store, and serves from it. The node prover reads the whole arena in passes, each
under a live coin from the verifier. The verifier times every pass and checks each one's answers against the store's
committed contents.

## Try it

~~~text
uv sync --all-packages --extra torch-cpu
VLLM_PY=/path/to/your/vllm/bin/python uv run verity bench memory --local
~~~

Both from the root of a clone of `main`. Nothing of ours is needed: no account, key, token, store or host. It needs the
card, vLLM, the CUDA toolkit and Lean below. The first line it prints is the share with its error, and the next says
what the session ran with. `verity bench memory --help` lists its other ways in.

## What you need

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, which runs every command here.
- **A Linux machine with one NVIDIA H200 or RTX PRO 6000 Blackwell.** These are the cards the tree has a data sheet
  (`hardware/nodes/`) for, and the share is of the card's own sheet. An H100 has none, so `--local` prints no number
  there, saying so, before it starts anything. Another card needs a data sheet (named with `LOCAL_NODE`), and a result
  against a sheet that is not `main`'s is not comparable.
- **A vLLM environment, `VLLM_PY`:** a Python with vLLM installed (`uv venv V && uv pip install --python V/bin/python
  vllm`). Without it `--local` runs nothing and says how to make one. It serves the checkpoint and fetches it:
  `unsloth/Llama-3.2-1B` at the revision core's pin names
  (`verity/protocols/resource_accounting/compute/pouw/verifier/lean/check/models.json`), about 2.5 GB from Hugging Face, not gated, so
  no token.
  Its vLLM and torch must be built for your card's CUDA. The recipe as written (no pin, index or extra) was run from
  nothing on an H200: `uv venv` fetched Python 3.12 itself, and the install gave vLLM 0.33.0 and torch
  `2.14.0+cu130`, which served the session; earlier runs used a vLLM development build and torch `2.13.0+cu129`.
- **The CUDA toolkit with `nvcc`** (under `/usr/local/cuda*` or `CUDA_HOME`). The driver alone is not enough: the
  session compiles its kernels for your card.
- **Lean:** `bash tools/verity/lean/setup.sh verity/protocols/resource_accounting/memory/poub/verifier/lean` installs `elan` and the
  verifier's dependencies. On your machine the Lean verifier builds from source: several minutes and about 4 GB of
  memory the first time, reused after.
  `setup.sh` printing `error: unknown executable cache` (this package has no Mathlib to fetch) and
  `lean-cache: no read credential (…); building from source` is expected: it still exits 0.
- **Disk:** the checkpoint, the caches and each run's records go into `LOCAL_WORK` (default
  `~/.cache/verity/bench-memory`), and the command writes nothing outside it but the Lean build in the clone and
  `elan` in `~/.elan`.
  `uv sync` fills `~/.cache/uv` before that: to keep both out of your home, set `UV_CACHE_DIR` (or `XDG_CACHE_HOME`)
  and `LOCAL_WORK` to directories elsewhere.
- **Time, disk and host memory:** from nothing on an H200, a run took about 33 minutes from the clone to the number
  (the clone about 5, the vLLM install about 4, `--local` about 24, the Lean verifier's first build included) and
  about 26 GB of disk (the vLLM environment 9, uv's cache 10, the clone with its `.venv` 4, the checkpoint 2.4), so
  start with about 30 GB free. Have about 76 GB of host memory, the larger of the peaks the runs reached.
- `GPU` picks the card (an index `nvidia-smi` gives; unset, the first). The session runs on this process's cores: a
  quarter to the engine, a quarter to the verifier and the rest to the prover (`taskset` narrows them;
  `ENGINE_CPUS`, `PROVER_CPUS` and `VERIFIER_CPUS` name others).

## What the number is

The first line is the share of the card's memory read rate, at its data sheet's read bandwidth, that one accepted
session leaves uncertified: the session's γ in its verifier's ledger at the specification's rate. 0 would mean the
whole read rate over the session was certified as spent reading the store, the arena the card serves from. **The
target is 0.14 at an error of at most 2^-100.**

- **Its error** is printed beside it with its power of two, as `δ = 2.8e-05 (2^-15.1)`. It reads
  `δ + Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed)`: δ is the chance, over the verifier's coins, that
  the share is wrong, and each window's adv_t is a prover's edge at telling SHA-512's output from random, which the
  number names but neither prices nor assumes away. The error is priced against one prover, one that makes at most
  2^80 SHA-512 compressions over the session on any hardware (`audit.QUERIES_LOG2`). At the 16 sketch rows a `--local`
  session registers, the sketch term is the largest, about 2^-15.1, far above the target's 2^-100; the deployed
  registration's 64 rows (`node.py --rows 64`), which `run.sh` doesn't pass yet, bring that term below it.
- **It varies from run to run** with the lateness allowance each session's calibration sets, through the timing term,
  and its error is a probability, the chance the share is wrong, not a band around it: two runs' shares can differ by
  more than the error.
- **The next line,** `ran with: coins local challenger; parameters development (…)`, says what the session ran with,
  from the verifier's record (the JSON's `ran_with`). Its coins were flipped on this machine. Its parameters are
  development ones because it registers 16 sketch rows where the deployed registration has 64. The detail names the
  arena against the card's memory, the paced form, the passes, the rows, the sample's term a pass group (the chance
  one pass group's openings miss what they test; it is not the error), the lateness allowed, the error and the prover
  it is priced against, and the sheet.
- **It is a measurement to improve on, not an audit.** The verifier and its coins run on the machine being measured,
  so you could re-roll them. The line under it says so, and the JSON's `audit` is false.
- **Who decided each number is in the document.** `gamma_decided_by` is `lean` where PoUB's Lean verifier decided the
  share. Where it is `python-ledger`, the share and its error are the Python ledger's reading of the accepted session,
  which the verifier records beside its own, and a line of the text says so. Where the document holds
  `numbers_decided_by`, it lists by key which printed numbers are the Lean verifier's and which are the Python
  ledger's: the error's sketch and sample terms are among the latter.
- **It certifies reads, not contents.** The audit certifies reads of the registered store whatever it holds; the
  verifier holds no checkpoint, so the store's bytes are the developer's choice, which the store's line says.
- **The store is the card's memory, not a request's.** A session whose store is below half the card's memory
  (`STORE_MIN`) prints no number: it would measure a request's reads, not the card's.
- **It is one session's.** The baseline to beat is your own run of this command at `main`'s commit on your machine,
  not a number from another.
- The share is of the one card's read rate and its host's: the verifier's node is one card of the card's node sheet,
  handed to it as `--node`. A line names the card and the sheet with the SHA-256 of the tree's data sheets
  (`hardware/`) and of the verifier (`verity/protocols/resource_accounting/memory/poub/verifier/`), Markdown left out, in the JSON's
  `machine` with the arena's GiB and vLLM's share of the card.

**Exit codes:**

| Code | Meaning |
| --- | --- |
| 0 | the session was accepted and its number printed |
| 1 | the verifier didn't accept the session, or its store was below half the card's memory: `memory: no number: WHY` |
| 2 | a command line or a record that can't be read, or another session form |
| 3 | no session reached a verdict (no NVIDIA driver, no data sheet for the card, no `VLLM_PY`, a checkpoint that can't be fetched, a process that failed): `memory: no number: WHY` |

**Where the result goes:** the text and one line of JSON on stdout. With `OUT=DIR`, the JSON goes to `DIR/memory.json`
and the benchmark result to `DIR/result.json`, its metric `memory-accounting/one-machine-read-rate-uncertified` (`metrics.toml`)
beside the bytes and the error. The session's records (the verifier's `verifier.json`, every process's log) go to a new
directory under `LOCAL_WORK/runs/`, which stderr names; `verity bench memory --from DIR` reads it again with nothing of
ours. The two reference numbers printed beside a node's number are runs in our evidence store, so `--local` prints
them as not read, with that reason.
Under `research run` the records go to `$RESEARCH_RUN_DIR/out` instead, beside the JSON's `memory.json`, and the
benchmark result to `$RESEARCH_RUN_DIR/result.json`, with the same metric.
`--from` takes the run's directory or its `out/`, the directory stderr names, where `--local` leaves its JSON as
`memory.json` beside the `verifier.json` it reads.

**Backs:** no Lean guarantee or
claim; the number is the ledger's γ of one session, which PoUB's own analysis bounds.

## What you may change

The prover's side, which is all of what the audit treats as untrusted. What lowers the share is the bytes the card
reads for the store in each stretch against what it could read.

On one card the share is mostly the timing term: the share of the session's span, from its first pass group's release to
its last one's deadline, that its passes don't cover at the sheet's bandwidth. Each pass is credited only the store's
bytes, which at that bandwidth take the time a bare read of the store would, so the share falls most with a faster pass:
a prover that reads and sketches the store nearer the card's bandwidth, with no gap between passes. The deadline A (the
slowest pass, gap and round trip its calibration measured, at `QUANTILE`, plus twice `JITTER_MS`) adds to the span only
at its end. The sample term is next; it falls with more openings a stretch, which the fixed form sets. The `ran with`
line prints the measured pass and the lateness allowed.

- the pass that reads the arena, and its pacing against the verifier's coins
  (`verity/protocols/resource_accounting/memory/poub/prover/remote_session/session.cu`, `benchmarks/remote_session/node.py`);
- how the arena is encoded and decoded on the GPU (`verity/protocols/resource_accounting/memory/poub/prover/word_codec/kv.cu`), and
  how the engine serves from it (`benchmarks/remote_session/engine.py`);
- the arena's size: `ARENA_GIB` (default 0.88 of the card's memory less 12 GiB, `arena_of`; 112 GiB on an H200, 72
  on an RTX PRO 6000) and vLLM's share `UTIL` (default 0.88). The document says what it was;
- the deadline's margin: `JITTER_MS` (default 50) and `QUANTILE` (default `max`). A smaller A shortens the span only
  at its end, and leaves a slow pass less room before it misses its deadline.

## What is fixed

- the verifier and its Lean (`verity/protocols/resource_accounting/memory/poub/verifier/`, its `lean/`);
- the ledger that turns a session into the share and its error (`verifier/session/verifier.py`), and the error rule
  (an error of 1 or more prints no number);
- the session's form: one paced session (`--paced 1:1000:2773:7`, a coin every pass group, seven credited stretches
  of a second at 2773 openings each, its groups counted at the pass the session's calibration measures on the card,
  `PACE_PASS_MS=0`). `--local` refuses any other `PACED` or `PACE_PASS_MS`, and any `UNPACED`, `SESSIONS`, `WINDOWS`,
  `WINDOW_MS` or `SAMPLES`;
- the data sheets the share is of, and this command.

A number from a changed verifier or sheet is not a result, and the digests in its `machine` line say which changed.

## How a cheating change is refused

Every pass answers challenges the verifier samples only after the pass's data is fixed, and the verifier checks each
answer against the store's committed contents and against time. A prover that skips reading part of the store, or
answers late, is refused: the session isn't accepted, and the command prints `memory: no number: WHY` and exits 1,
WHY read from the verifier's record: the session, the pass group it was refused at, and the check that failed at
which sample, as ``the verifier refused session 0 at pass group T: its check `receipt` failed at pass P, position Q``.

- **What you can watch refused:** `PLAN=skip:0.0625 uv run verity bench memory --loopback`, on the CPU. The prover
  skips a sixteenth of the store, the verifier refuses the session ("session 0 was not accepted (…)"), and the
  command prints no number and exits 1. Under `--local` the same `PLAN` reaches the node prover.
- A cheat on `--local` is a changed prover, which the verifier's checks above refuse when they catch it, or a changed
  verifier or data sheet, which no check of the session catches: the document's `machine` gives their SHA-256s, and a
  result whose digests are not `main`'s is not a result.

The team's own runs (the node run through both auditor devices, `--from` a run in the evidence store, and
`--loopback`) need our machines, our evidence store or `research run`.
