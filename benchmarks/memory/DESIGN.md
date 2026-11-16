# The memory benchmark: design

## The team's own runs: the node run, the evidence store and the Metrics tab

Everything below needs our machines, our evidence store or `research run`; nothing above does.

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
- **`--local`** is the section above: `remote_session/run.sh` with `ROLE=both SERVE=node GPUS=1`, the node run's
  engine, prover and paced verifier on one card. Its metric on the Metrics tab says it is no audit, and, where the share
  is the Python ledger's, says that too. The served workload (`SERVE=row`), whose store is one request's reads and whose
  share was 1 or more, is not this benchmark: `run.sh` runs it directly.

It prints the number as text, then one line of JSON whose first keys are `value`, `unit`, `commit` and `run`, and exits
0. A session that wasn't accepted prints no number and exits 1; a command line or a run that can't be read exits 2;
and where no session ran to a verdict (a run that left no verifier's record, a gate run that didn't launch or never
reached the store, and under `--local` or `--loopback` a machine that lacks what it needs or a process that failed)
it prints no number and exits 3. The text's first line prints the number's error and says how it reads:
δ + Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed), δ the session's in its verifier's ledger, the declared
error the result's `error`, below, holds, with the same words in its unit; the JSON holds it under `error`. A session
whose error is 1 or more bounds nothing: the first line prints "no bound at this period" with the error in place of
the number, and the JSON's `value` is null with those words under `no_bound` (the rule and the words are
`verity/protocol.py`'s, one for every metric's error).

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
neither nothing, but `--local` outside a research run also leaves it as `out/memory.json` beside its records, where
`--from` reads it; the loopback, a node run and `--from` leave nothing with neither. Where the kit writes, an
accepted session of a node run or of `--from` also writes the run's result,
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
`memory-accounting/node-read-rate-uncertified`, the id of its metric on the console's Metrics tab (`benchmarks/memory/metrics.toml`).
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
it prints afterwards: it is how `--from` knows that run's session for a loopback.
