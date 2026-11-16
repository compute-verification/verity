# verifier_cost: design

What the two verifiers measured are, the measurements whose text is not in `README.md`, what the outputs hold and what
the suite checks. `README.md` says what the benchmark measures, how to run it, and describes four of its measurements.

## The workload and the two verifiers

A verifier's time against the size of the description it reads and against the size of the circuit that
description unrolls to. The rule it measures (AGENTS.md "Design rules"): a verifier's cost is
O(circuit description), never O(unrolled circuit).

The workload is Serve_v1 (`benchmarks/network_sim/workload/served.py`) at a ladder of configs up to B0 (`Config.b0()`,
30 layers), word-level and on bits. On bits it is `lift.on_bits`: every Definition Serve_v1 reaches, lifted to bits all
the way down (`ir="boolean"` can't derive Serve_v1; `lift.py`'s docstring says why). Two verifiers are measured:

* `measure`, the inference-only check: `inference-only-check --policy` (`InferenceOnlyCheck.Desc`, one summary per Definition) on an instance whose θ is
  the registered inputs and whose inference blocks are the root's Calls, one each. `--policy` leaves out θ's provenance
  (`InferenceOnlyCheck.Theta`), which reads the registration log and adds work in θ's length and the log's. Where the unrolled
  circuit is small enough (`--unrolled-max`), the instance expanded gate by gate also goes through
  `inference-only-check --policy --unrolled` (`InferenceOnlyCheck.check`, the gate-by-gate reference), and the two verdicts must agree.
* `partition`, the one-stage verifier's partition: `spot-difftest`'s `partition` (`Partition.units`, `strata`,
  the expanded sampling design and the digest, as the verifier computes them) on Serve_v1 under the template query of its
  Calls (`Q_template_instances`), under `Q_word` v1 and `Q_call` v1 at `X = 16` where its Calls' distinct Definitions are
  small enough (`--unrolled-max`), and on the served shape: the population program of Serve_v1's largest Definition at
  1 to 10⁶ instances under `Q_template_instance`, as a served window registers one. A template query must apply.


## The other measurements

`inference_only_cost.sh` times `InferenceOnlyCheck.Desc` step by step (`inference-only-phases`, `inference_only_phases/Main.lean`: decode, summaries, root reads,
the verdict, the slowest Definitions' summaries and their return pieces) beside `inference-only-check --policy`, on the `inference_only` suite's
vectors (`suite`), Serve_v1 word-level and on bits (`serve`), and GLM-5.3's rank 0 word-level and lifted to bits (`glm`);
`inference_only_cost.py`'s docstring has the cases.

~~~sh
research run --project verity --on vy-nebius-cpu-2 --pool network --source . --cwd clone --mem-gb 320 --timeout 12h \
  --send glm53-b1-rank0.descriptor.json.gz -- bash benchmarks/verifier_cost/inference_only_cost.sh [suite|serve|glm ...]
~~~

`flock_read.sh K...` measures Flock's Lean verifier (`verity-verify-proof verify --zk`) on the H1 tile proof block, step by step:
`flock_read.py` builds the 4×8 tile's statement circuit at each contraction K with one instance (`h1unit.Shape` in `verity/protocols/resource_accounting/compute/pouw/prover/lowering/h1unit.py`), and `flock_read/Main.lean`, built in a scratch Lake package against the checkout's verifier, times the
file's read, its SHA-512, `Zk.maskLogOf`, the decode and line split, each section's `Net.parse` (its tokenizer,
`Sparse.ofRows` and row checks), the class digest, `HmOut.parse`, `Stmt.setupHiddenOf` and one lincheck fold. Per K,
`$OUT` gets build-K.json (the proof block's ANDs, matrix entries, text bytes and its Program descriptor's size) and
phases-K.jsonl (one row per step: milliseconds, RSS and peak RSS in MB).

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-1 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/flock_read.sh 256 1024
~~~

`population.sh N...` times the one-stage run's population-order check (`Slots.popOk`) against a template population's
proof blocks: `population.py` writes, per N, a population of about N proof blocks over a layered call hierarchy (by default 234 root
batches of 234 templates over six levels of at most 1,800 composites, about 8,100 Definitions; `POPULATION_ARGS`
changes them), and `population/Main.lean`, built in a scratch Lake package against the checkout's verifier, times
`Partition.unitGates` (a range at a time, `TemplateQuery.unitGates`), `Slots.popOk` (from the ranges,
`popOk_eq_popOkRanges`: no replay block and no input gate is built), the one-stage check's call of it
(`OneStageZk.popHeld`, compiled in `Spot/OneStageZkStatement.lean`, which imports `Spot.PopOrder` so that it runs from
the ranges too; its row carries the root's input gates, counted) and the references: `TemplateQuery.gates` on every
proof block, each recomputing its offset and template through an unmemoized walk, and `popOk` as stated, off `Slots.layout`'s
population, at N and either side of it. It fails where a reference differs. The references run up to `REF_MAX` proof blocks
(default 50). `POPULATION_ARGS='--input-members M'` widens the root's input nodes to M members each, which only the
layout's input class reads: `popOk`'s row and `popHeld`'s must not grow with M.
`$OUT` gets population.jsonl, one row per N and function (proof blocks, milliseconds, peak RSS).

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-11 --source . --cwd clone --mem-gb 90 -- \
  bash benchmarks/verifier_cost/population.sh 20 50 100000 1000000 6963296
~~~

`members_parse.sh N...` times the per-member form's parse of its members (`OneStageZk.membersOf`) against their count:
`members_parse/Main.lean`, built in a scratch Lake package against the checkout's verifier, makes N members of one range
of proof blocks each, no proof block in two, and times the parse on them, then on the same members with one more whose range starts
inside the last member's. It fails where the parse gives back other members than those made, name by name and proof block by
proof block, or accepts the proof block in two. The parse reads each member once and sorts the members' ranges, so its time must grow
no faster than N log N: a time that quadruples where N doubles is work in the members squared. `$OUT` gets
members_parse.jsonl, one row per N and case (members, milliseconds, peak RSS).

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-11 --source . --cwd clone -- \
  bash benchmarks/verifier_cost/members_parse.sh 1000 25000 100000 1000000
~~~

`whole_run.sh run|desc SHAPE N...` times `Run.run`, the stream's whole run of one registration, against the population.
`whole_run.py` writes, per N, an accepted `run` input (`run`: a v1 audit on challenger coins with N/8 more ports) or the
program and query alone (`desc`), over a template population of about N proof blocks: `toy`, the difftest's three templates,
or `l5`, a GLM-5.3 MoE window's seven templates at the window's proportions (whose Python partition walks every gate,
so `l5` takes `desc`). The audit's sampling design is the stratified rate of two in the population, `stratified-rate:2/P` over its
P proof blocks, on `toy` the stratified sampling design at K = 3. Stated over P, it samples one proof block of each template and two
of `toy`'s largest at every N, so the sample stays four proof blocks while the population it is timed against grows.
`whole_run/Main.lean` times the `run` piece and each of `Run.run`'s steps,
`Slots.popOk`'s layout split into `Partition.evaluate`, `Partition.unitGates`, `Slots.inputGates` and `Slots.layout`; on
a program alone, `Program.parse`, one `Desc.gates` and one `TemplateQuery.gates` per range, the input gates' count and,
with `-- --unit-gates`, `Partition.unitGates`; `-- --unit-gates --ref` adds `Partition.unitGatesRef`, the walk proof block by
proof block `unitGates` replaced, and fails unless the two give the same gates; `--pop-ok` adds `Slots.popOk` last, whose
layout holds a proof block per input gate. `PERF_N` adds a `perf` profile at that N. `$OUT` gets whole_run.jsonl, one row per N and step (milliseconds, peak RSS).

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-1 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/whole_run.sh run toy 146 1000 10000 50000
~~~

`members_files.sh N|all...` times the per-member form's pass over every member's files against the members of one
circuit file. `members_files.py` writes one per-member case as the executable's `check` takes it: a driver's (`MF_CASE`,
an audit's `window/case-v2-members.json`), or a stand-in window's of `MEMBERS` members (default 300) of one proof block over
two templates (`benchmarks/one_stage/members_standin.py`'s: one member of each template sampled, the rest sharing their
template's circuit file unsampled), audited as a served window's audit step runs it, on Flock's prover, the Lean
verifier and the ZK server; v2's `check` must accept the case, and its seconds and peak RSS go to inputs.jsonl.
`members_files/Main.lean`, built in a scratch Lake package against the checkout's verifier, takes the case's first N
members of one circuit file (`all`: every member of the case) and times `OneStageZk.membersFiles` as the compiled
check runs it (`membersFilesKept`: the blobs' lookup tables read once, a circuit file read and parsed once,
`Spot/CircuitKept.lean`'s summary kept of it, and the strata of the partition's query evaluated on the program once
for the members that share the parameters' partition) and the pass as stated, `Member.files` a member, each parsing
its file and evaluating the query itself (`membersFiles_eq_kept` proves them equal). It exits 1 where the two disagree
on a check's result or a member's files are not held. The stated pass runs up to `REF_MAX` members (default 150). The
compiled pass's microseconds at N less those at 1, over N - 1, are its work on one more member of a file; the stated
pass's over N are a member's with its file's parse. `MEMBERS` may name several sizes, a window, a case and its rows
each: the rows at one N across the cases are the pass against the window's members at a fixed count of members handed.
A stand-in window's program does not grow with its members (its root holds the templates' nodes), so these rows do
not vary the program's size; each case's row says how large its program is and what one evaluation of its partition's
query costs on it, which is the part of the stated pass's work on a member that grows with the program. `$OUT` gets
members_files.jsonl: each case's row (its members, blobs and bytes, the circuit file's, its program's bytes and root
nodes, and the microseconds of one evaluation of its partition's query's strata) and one row per N and pass (members,
circuit files, microseconds, peak RSS, the result's digest).

~~~sh
research run --as proofs --project verity --on auto --source . --cwd clone --mem-gb 32 -- \
  bash benchmarks/verifier_cost/members_files.sh 1 2 16 150 all
research run --as proofs --project verity --on auto --source . --cwd clone --mem-gb 32 --env "MEMBERS=300 1200" -- \
  bash benchmarks/verifier_cost/members_files.sh 1 16 150
~~~

`flock_tmpl.sh K...` is the same proof block in `flock-ir-tmpl/v2` (`Flock/Tmpl.lean`): `flock_read.py --circuit-only`
builds the circuit, `python -m verity_flock.tmpl` writes its proof block's templated form (Definitions read off the flat proof block's
repeated blocks), and `flock_tmpl/Main.lean` times the templated read (`Tmpl.parse`) and diffs `Tmpl.unroll`'s lines
against the circuit's flat proof block byte for byte. Per K, `$OUT` gets emit-K.json (the levels, Definitions, rows and bytes),
unit-K.tmpl, read-K.jsonl and diff-K.jsonl (the PHASE lines) and each step's `/usr/bin/time -v`.

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-1 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/flock_tmpl.sh 256 4096
~~~

`flock_unroll.sh K...` times what the templated statement's read costs against the flat one's on the same proof block: `Tmpl.unrollAt` at
`MODE` (default `iv/hybrid/4`) whole and a stage at a time (`rows`, `posMap`, `Net.ofRows`, `invOf`, `posOk`, `placed`,
`Sparse.ofRows`), the logical `Tmpl.unroll`, and `Net.parse` on the circuit's flat proof block. Per K, `$OUT` gets unit-K.tmpl,
unroll-K.jsonl (every PHASE line, with its step) and each step's `/usr/bin/time -v`.

`flock_members.sh` measures what a member of a run waits before its first check when one process verifies the run's
members in turn (`verify-members`, which reads the run's program once and hands each member the circuit file the member
before it read) against a process a member (`verify`). Per spec, `one_stage/m1_unit.py` stages, proves and verifies one
statement bound to its program, and `flock_members.py` hands that member in `FM_MEMBERS` times each way, pinned to one
core, the two ways alternating over `FM_ROUNDS` rounds. Its numbers are a member's `before_check_ms` (the verifier's own
`inputs:` and `setup:` stage lines) alone, as the kept process's first member, which pays the one read, and as a later
member, with the memory the kept process holds between members (`rss_kb`: its resident memory at each `MEMBER` line).
Every process runs under the verifier's allocator setting, as its Python callers start it (`proofs.verifier_env`),
unless the job's environment sets a value of its own, and `members.json` records the value (`env`). Its guard: each
kept member's `VERDICT` lines are byte for byte `verify`'s alone, every member is accepted and the kept process exits
0; a run that fails it exits nonzero and none of its figures counts. The target is a later member's wait
with nothing of the program's in it, and nothing of a circuit file it shares with the member before it but the file's
bytes read and compared. `FM_ORDER` (the members by their spec's index, `0,0,1,0`) adds the specs' members in one kept
process: a member whose circuit file is another than the one before's pays for its own, and the process's peak RSS
says what the change of file holds at once.

~~~sh
research run --as proofs --on auto --cpus 16 --mem-gb 64 --max-min 150 --cwd source -- \
  env FM_MEM_GB=64 bash benchmarks/verifier_cost/flock_members.sh
~~~

`flock_sessions.sh census:NAME [N...]` measures the Lean verifier's seconds on a served block type's sessions, on one
core, against the statement's blocks. `flock_sessions.py stage` stages the type's flat statement at `full` as
`served_units/stage.py` stages a census type, and from the same lowering its first N blocks for each N. Per N, the CPU
prover proves one loopback `--zk` session sampling all N blocks, and `verify --zk` checks it as the proof server's
verdict pool calls it, pinned to one core; for the first N one more call takes two sessions of it. A proof server's
saved sessions of the full statement, sent as `sessions.tar` (one directory a session), are handed to one `verify`
together. `verify` reads the statement once a call
(its `inputs:` line) and then sets up and checks each session it was handed, so `summary.json`
(`verity/flock-sessions/v1`) splits a call into the read, paid once a circuit, and each session's seconds, with their
median at each block count and the line through them: the seconds a session pays whatever its blocks, and the seconds
a block. `phases.jsonl` is `flock_read/Main.lean`'s phases of the read, and `perf-symbols.txt` one call's profile. A
census statement binds no Program, so every session here is refused for want of `--program` once every check has
passed.

~~~sh
research run --as proofs --project verity --on auto --mem-gb 48 --source . --cwd source --declared-output 'out/*' \
  [--send sessions.tar] -- env FS_MEM_GB=48 bash benchmarks/verifier_cost/flock_sessions.sh \
  'census:ScaledMmFp8BlockSharedScale_v1{K=2048,N=4,G=128}' 1 16 256
~~~

`table_root/table_root.sh` measures the Lean verifier's SHA-512 rate and what a history table's root costs it at the
pinned capacity (`Spot.HistoryTable.CAPACITY`, `2^17` ranks): `table_root/Main.lean`, built in a scratch Lake package
against the checkout's verifier, times `Flock.Sha512.hash` at the message lengths the verifier hashes and at 1 MiB, the
frame-v3 node, leaf and tree leaf, `FV3.root` over `2^17` leaves, and whole table roots (`Spot.HistoryTable.rootAt`)
at the lengths given, each over `--tables` distinct tables. `$OUT` gets table_root.jsonl (one row per phase: calls, ns a
call, calls a second, milliseconds, peak RSS), cpu.txt and the build's log.

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-1 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/table_root/table_root.sh [--tables T] [--lengths 4096,40960,131072]
~~~

`ledger_replay/replays.sh W...` counts the replays one PoUW period check makes: `ledger_replay/replays.py` writes, per
W, an honest period of W windows as PoUW's period difftest generates one and counts the replays its reference
(`period_check.case`) makes, and the compiled check (`pouw-period serve`) answers the same case under
`ledger_replay/calls.py`, which counts its entries into `Spot.Challenger.checked` by a breakpoint it holds with ptrace
(x86-64 Linux; the nodes have no debugger). `$OUT` gets replays.jsonl, one row per W: each side's verdict and replays.

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-11 --source . --pool research --cwd source \
  --mem-wait 1800 -- bash benchmarks/verifier_cost/ledger_replay/replays.sh 1 2 3
~~~

`qcall.sh CONFIG X...` measures `Q_call` v1 (`Flock.Qcall`, an open walk in `tests/test_verifier_cost.py`) on one Serve_v1
config's word-level descriptor: `verity-verify-proof qcall-program` at each X, with the output's SHA-256 so two commits' outputs
compare, a `perf` profile at the first X, and the one-stage `partition` piece under `Q_call` v1 at X = 16. Its header has
the command.


## Outputs and the suite

`results.json` (`verity/verifier-cost/v1`) has one row per config and `ir` (word or Boolean): the description's bytes, Definitions and body
nodes, the unrolled circuit's gates, and per check its wall seconds (the best of `--repeat`), peak RSS and the verdict.
`partition.json` (`verity/verifier-cost-partition/v1`) has one row per config, `ir`, shape and query: the description's
bytes, the unrolled gates, the gates of the Calls' distinct Definitions (what `Q_word` and `Q_call` lay out), the wall
seconds, peak RSS and the population (null where the verifier refuses the query). `store` puts either in the store as
an `evidence/v1` record naming the run and its commit; it runs where the store's keys are (a VM), since a node's shell
has none.

Apart from `inference_only_glm`'s inputs and `flock_served`'s saved step, nothing here is a served Program: `on_bits`'s output and `inference_only_glm`'s derived Programs are
this benchmark's constructions, and nothing outside this directory reads them.

The suite (`tests/benchmarks/verifier_cost/test_mains.py`, which `check` runs after its Lean steps) holds every `<name>/Main.lean` here to the
verifier it times: each must still elaborate in the checkout's package, `whole_run.py DIR toy 146 --check` must still
accept the run it writes, and the ledger replay's generators must still write a ledger the challenger accepts and a
period PoUW's period check accepts. It builds no executable and runs no benchmark; a main in a new directory is held
without being listed.
