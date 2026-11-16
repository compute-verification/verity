# benchmarks/memory

`verity bench memory` is the front door for the memory audit's one number: the share of a node's memory read rate that
an accepted PoUB session leaves uncertified, at the data sheet's read bandwidth. It is the session's γ in its
verifier's ledger at the specification's rate, read from the verifier's record alone: nothing the prover recorded
reaches it. `run.py` is the whole command; the sessions are `benchmarks/remote_session`'s, which it starts and reads
and never copies.

## On a machine of your own

~~~text
uv sync --all-packages --extra torch-cpu
uv run verity bench memory --local
~~~

Both from the clone's root. One Linux machine with one NVIDIA H200 or RTX PRO 6000 (the cards the tree has a row and
a data sheet for; another needs both, named with `LOCAL_ROW` and `LOCAL_NODE`), the CUDA toolkit
(`nvcc`, under `/usr/local/cuda*` or `CUDA_HOME`), Lean's `elan` (`tools/verity/lean/setup.sh`; `lake` on the PATH or in
`~/.elan/bin`) and about 20 GB of disk. vLLM serves Llama 3.2 1B (in int8) on the GPU while a memory audit runs
against it, and the audit's verifier and the random challenges it sends (its coins) run as processes beside it. The
command prints the share of the GPU's memory read rate that the session leaves uncertified, with its error, as text
and as one line of JSON, and exits 0; a session the verifier refuses prints `memory: no number: WHY` and exits 1.
Lower is better: 0 would mean the whole of the GPU's memory read rate over the session was certified as spent reading
the audited store. The target is 0.14 at an error of at most 2^-100.

The text's first line is the number; the next, `ran with: coins local challenger; parameters development (…)`, says
what the session ran with, from the verifier's record (the JSON's `ran_with`): its coins, drawn on this machine, and
its parameters, which are development ones, since the session registers 16 sketch rows and the deployed registration
64; its detail names the rows, the openings and δ a window, the windows, the lateness allowed, the prover the error is
priced against and the sheet the share is of. **Because the verifier and its coins run on the machine being measured,
this is a measurement to improve on, not an audit:** the line under it says so, and the JSON's `audit` is false. Until
PoUB's Lean verifier gives its answer for the share's terms (`gamma_answer` in `verifier/session/verifier.py`), an
accepted session's share is undecided: this command and `--loopback` print the Python ledger's share and error, which
the verifier records beside it, and say so in a line of their own and in the JSON's `gamma_decided_by`
(`python-ledger`); a recorded run's (`--from`, `PERIOD`) prints no number.

The error is priced against one prover: one that makes at most 2^80 SHA-512 compressions over the session on any
hardware it reaches (`audit.QUERIES_LOG2`), never the node's own hash rate. At the 16 sketch rows a session registers
by default, the sketch term alone keeps the error far above 2^-100 against that prover; the registration's 64 rows
(`node.py --rows 64`), which `run.sh` doesn't yet pass, bring the term below it.

The share is of the one card's read rate and its host's, at the figures of the tree's data sheets: the verifier's node
is one card of the card's node sheet (`hardware/nodes/`), handed to it as `--node`. A line names the card and the sheet
with the SHA-256 of the tree's data sheets (`hardware/`) and of the verifier
(`verity/protocols/accounting/poub/verifier/`), Markdown left out, the JSON's `machine`; a result whose digests are not `main`'s was
measured against another verifier or other figures, and is not comparable.

- **What it needs from you:** nothing of ours. The first run downloads and builds into `LOCAL_WORK` only (default
  `~/.cache/verity/bench-memory`): a Python environment with the pinned vLLM, the checkpoint, the package caches and
  the attention taps, by `adapters/vllm/verity_vllm/ops/pod_bootstrap.sh`, and it removes nothing outside it. A
  bootstrap that would touch another file of your machine (your package caches, a fixed build path) is refused before
  it runs:
  the vLLM wheel from vLLM's index, torch from PyTorch's, and `unsloth/Llama-3.2-1B` at a pinned revision from
  Hugging Face, which is not gated, so no account or token is needed; the int8 checkpoint is made from it on your
  machine and checked against its pinned SHA-256s. A machine that can't reach Hugging Face stops there: the command
  prints "no number: the bootstrap failed" and the bootstrap's log names the download. `VLLM_PY` names an environment
  of your own instead, `LOCAL_ROW` another row (a card of another capability needs one: `verity-vllm workload`), `GPU`
  the card. With no `gpu-lease` installed, the tree's own lease program keeps its locks in `LOCAL_WORK`. The session
  runs on this process's cores, half to the verifier and half to the prover (`taskset` narrows them; `VERIFIER_CPUS`
  and `PROVER_CPUS` name others).
- **Where the result goes:** the text and the JSON on stdout; with `OUT=DIR`, the JSON in `DIR/memory.json` and the
  result in `DIR/result.json`, its row `memory-accounting/one-machine-read-rate-uncertified` (`metrics.toml`) beside
  the bytes and the error; the session's records (the verifier's `verifier.json`, every process's log) in a new
  directory under `LOCAL_WORK/runs/`, which stderr names. `verity bench memory --from DIR` reads such a record again
  with nothing of ours. The two reference numbers printed beside a node's number are runs in our evidence store, so
  `--local` prints them as not read, with that reason.

## What you may change

The prover's side, which is all of what the audit treats as untrusted: how the store is read in each pass and when
(`verity/protocols/accounting/poub/prover/node_store/serving.py`, the store's backend, its passes and their pacing
against the verifier's windows), how the KV cache and the store are encoded on the GPU
(`verity/protocols/accounting/poub/prover/word_codec/kv.cu`), and what of the engine's memory is registered as the
store and how it is laid out (`adapters/vllm/verity_vllm/protocols/poub/binder.py`, and `run.sh`'s `KV_PAGES`,
`WEIGHT_BLOCKS` and filler), the store's size among them: the audit certifies reads of the registered store whatever it
holds, and no PoUS runs behind the session to say the store is the served model's state. The baseline to beat is your
own run of this command at `main`'s commit on your machine, not a number from another.

## What is fixed

The verifier and its Lean (`verity/protocols/accounting/poub/verifier/`, its `lean/`), the ledger that turns a session
into the share and its error (`verifier/session/verifier.py`), the error rule (an error of 1 or more prints no number),
the session's form (one session of 7 windows of 1000 ms at 2773 openings a window; `--local` refuses any other
`SESSIONS`, `WINDOWS`, `WINDOW_MS` or `SAMPLES`), the data sheets the share is of, and this command. A number from a
changed verifier or sheet is not a result, and the digests in its `machine` line say which.

## How a cheating change is refused

Every pass answers challenges the verifier draws only after the pass's data is fixed, and the verifier checks each
answer against the store's committed contents and against time. A prover that skips reading part of the store, or
answers late, is refused: the session isn't accepted, and the command prints `memory: no number: WHY` and exits 1.
`PLAN=skip:0.0625 uv run verity bench memory --loopback` shows it on the CPU: the prover skips a sixteenth of the store,
and the verifier refuses the session. A served row runs the honest plan alone (`run.sh` refuses any `PLAN` but `serve`
before a session starts), so a cheat there is a change to the prover's side above, which the same checks refuse.

**Status: active.** The lane's number for a node serving under the audit comes from here.

**Run:**

~~~text
verity bench memory PERIOD [--dry-run]            the node run: one session on the whole GPU node, through both auditor devices
verity bench memory --from DIR|RUN [--session K]  the number of a recorded run: its record's directory, or a run id
verity bench memory --loopback                    both processes on this machine, on the CPU: a test of the command
verity bench memory --local                       vLLM serving Llama 3.2 1B on this machine's GPU, audited beside it
~~~

- **The node run** is `benchmarks/remote_session/gate_run.sh PERIOD` with one honest session (`PLAN=honest`, `PACED`
  one word, by default seven credited stretches of a second, each pass under a live coin of its own): the developer's
  node serves `ROLE` (default `LLAMA32_1B_BF16`, Llama 3.2 1B as upstream holds it, since the store's layout covers
  BF16 weights) with each engine's weights arena the store, the verifier runs on the auditor's verifier host, and the
  coins come from the challenger host under period `PERIOD`, a name its ledger hasn't used. The verifier holds no checkpoint and
  the period signs no model, so the store is the developer's choice, which the text and the JSON's `bound` say; the
  auditor's line of record credits the session on its other checks, conditional on `DeveloperStore`, the developer's
  word that the store holds the registered weights.
  `UNPACED=WINDOWS:WINDOW_MS:SAMPLES` asks for a session of one coin a window in place of `PACED`'s
  (`7:60000:33273` is seven windows of a minute with 33,273 openings each); the verifier's ledger holds its γ as it
  does a paced session's, and the auditor's line of record has no time-held line for it, which only a paced session
  gets. `gate_run.sh`'s variables pass through: `COMMIT` (the commit the verifier host is pinned at, on `main`'s
  first-parent line, whose tree must pin `ROLE`), `ARENA_GIB` and `UTIL` (the arena on each card and vLLM's share of
  it), `HOST_GIB` (whole GiB of the node's host memory the store holds beside the arenas, default none), `LEAF_CTAS`
  (the node's passes by the leaf pass on that many reading CTAs, default today's pass),
  `PACE_PASS_MS` (the pass the verifier host counts a stretch's groups at), `LEASE_BY`, `FETCH=1`, which stages
  the role's checkpoint on the node before checking it, and the three hosts' addresses, which it doesn't run
  without (`VY_CHALLENGER_1_HOST`, `VY_AUDITOR_1_HOST`, `VY_NEBIUS_2_HOST`). The session's form is `PACED` or
  `UNPACED` alone: the command sets `gate_run.sh`'s `SESSIONS`, `WINDOWS`, `WINDOW_MS` and `SAMPLES` from
  `UNPACED`. The command waits for the fire
  run to reach the evidence store and prints from it; interrupted, `--from` the fire run prints the same. `--dry-run`
  sends nothing to either gate and prints no number.
- **`--from RUN`** reads the verifier's record of a run in the store: a gate run's fire run, or a run of
  `remote_session/run.sh` that held the verifier. A run that holds several sessions needs `--session K`. A withdrawn
  run is refused before any of it is fetched. The text says what the number rests on: whether the verifier ran on the
  auditor's verifier host, the store's size beside the node's GPU memory and the checkpoint it was bound to (or that
  the verifier held none, so the store is the developer's choice), that a
  share of 1 or more certifies nothing, and, of a run `--loopback` left, that it was a loopback.
- **`--loopback`** runs `remote_session/run.sh` with `ROLE=both`, the prover the driver's CPU reference over a test
  store of 64 blocks. No node is audited, so it certifies nothing, and both its text and its JSON say so. It builds the
  Lean verifier in the tree, so under `research run` it takes `--cwd clone`, and `--store-read` for the references.
  `PLAN` is the prover's plan, `honest` unless set: a cheating one (`skip:0.0625`) is refused by the verifier.
- **`--local`** is the section above: `remote_session/run.sh` with `ROLE=both SERVE=row`, the Llama 3.2 1B row for this
  card (`LOCAL_ROWS` in `run.py`), after the bootstrap. Its row on the Metrics tab says it is no audit and that the
  share is the Python ledger's.

It prints the number as text, then one line of JSON whose first keys are `value`, `unit`, `commit` and `run`, and exits
0. A session that wasn't accepted, or a run that left no verifier's record, prints no number and exits 1; a command
line or a run that can't be read exits 2. The text's first line prints the number's error and says how it reads:
δ + Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed), δ the session's in its verifier's ledger, the declared
error the result's `error`, below, holds, with the same words in its unit; the JSON holds it under `error`. A session
whose error is 1 or more bounds nothing: the first line prints "no bound at this period" with the error in place of
the number, and the JSON's `value` is null with those words under `no_bound` (the rule and the words are
`verity/protocol.py`'s, one for every row's error).

Beside the number it prints two references, each read from its run in the store with the run's date and card: the
state bound of a PoUS session over its card's memory (its record's `s_bits`, S₀ = ⌊ρ·|C|⌋: a lower bound on the
device's memory, its on-chip state and a round of side-channel traffic together, not the device memory credited), and
the share a wipe left foreign at most (the bound its run's label states). Each prints the error its record states
(the PoUS certificate's `eps`, the `eps` in the wipe's label), and one whose error is 1 or more prints no number, by
the same rule. A reference that can't be read prints its reason in place of a number, and the session's number still
prints.

The command prints and writes through the harness kit's report (`kit.report`, `benchmarks/harness/README.md`), as
`verity bench compute` does: under `research run` it writes the printed document to
`$RESEARCH_RUN_DIR/out/memory.json`, outside one where `OUT` names a directory to `$OUT/memory.json`, and with
neither nothing. Where it writes, an accepted session of a node run or of `--from` also writes the run's result,
`$RESEARCH_RUN_DIR/result.json` (where `research` reads a run's result) or `$OUT/result.json`, through the kit's
writer, in the five fields every resource's benchmark reports. Each is read from
the verifier's record alone, and its `inputs` name the recorded run and the record's keys it came from: `metric` (the
printed number in its unit); `verified` (the bytes the verifier certified read over the session's span, its
`certified_bytes`); `capacity at spec` and `capacity at measured` (for each reading the ledger holds, the data sheet's
read bandwidth and the measured read rate, that rate times the span); `free at spec` and `free at measured` (γ of that
capacity, so free over capacity is the reading's γ, the printed number at the first, and verified and free are the
capacity but for the float's last place); and `error` (the probability the verified amount doesn't hold, read window
by window as the share is: the record's declared error, which adds the sketch's term and the sample's rejection and
binding terms to the ledger's own and holds no PRF advantage; no rate is in it, so it is one number for both
readings). A capacity's rate is the verifier's own figure for the node (`timing.py`), not the registered hardware's,
and its unit says so. The printed number is written once more under
`memory-accounting/node-read-rate-uncertified`, the id of its row on the console's Metrics tab (`benchmarks/memory/metrics.toml`).
That entry also carries the goal's target, 0.14, and says whether the number is within it or past it. A session whose
verifier held no checkpoint, every session now, still writes it, with `"bound": false` and its verdict naming that premise
(`within target; the verifier held no checkpoint`); a bound session's entry, from a record made while the verifier could bind
the store, holds neither. Where the number's error
is above 2^-100, the goal's bound, the verdict names that as well (`within target; the verifier held no checkpoint;
its error is above 2^-100`): the record's declared error where it holds one, else the ledger's own, and above it
where the record holds neither. At a share of 1 or
more, verified is 0 and free the whole capacity. At an error of 1 or more, `metric`, `verified` and each `free` are
written with no value and the reason as their `why`; the capacities and the error are written as ever. An amount
whose keys the
record doesn't hold as numbers is left out of the result. The two references are other runs' numbers and are not in
it; a loopback writes none, since it certifies nothing, and a session that wasn't accepted writes none, as it prints
no number. A result holds one measurement a name, so a run that prints a second session keeps its first one's result
and says so on stderr.

- `run.py`: the command.
- `references.json`: the two references' runs.
- `tests/`: the printer, the four ways in and the references, against records built in the tests, a stand-in for the
  store and one for the processes; no session runs in the suite (from this directory, `uv run --project ../.. pytest
  tests`; or `uv run tools/check/suites.py benchmarks/memory` from the clone's root).

**Reads and writes:** reads a run's `out/verifier.json` (a gate run's, under `out/host/`) and the references' runs
from the evidence store; under `research run` writes an accepted session's amounts to the run's result,
`$RESEARCH_RUN_DIR/result.json`, and the printed document to `$RESEARCH_RUN_DIR/out/memory.json` (outside one, with
`OUT` set, `$OUT/result.json` and `$OUT/memory.json`), and `--loopback`
leaves `run.sh`'s records beside the document. A run that ran `--loopback` keeps a loopback's document there whatever
it prints afterwards: it is how `--from` knows that run's session for a loopback. **Backs:** no Lean guarantee or
claim; the number is the ledger's γ of one session, which PoUB's own analysis bounds.
