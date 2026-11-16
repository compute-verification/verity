# prove

`verity bench prove` is the one command for what a proof system's prover costs on a workload. Its code is
`verity-bench`'s (`benchmarks/harness`: `verity_bench/prove.py` holds the rules, `prove_flock.py` Flock's prover,
`prove_records.py` the store side and the table, its code in `prove_common.py` (the slots, the workloads, reading a
record), `prove_rows.py` (a steady-state row), `prove_store.py` (each input), `prove_table.py` (the table and what a
runner records) and `prove_render.py` (the text)); this directory holds its runner. It reads the store's labelled
records, with no GPU and no node, on a machine that has the store's client (`research`, in `uv sync`'s dev group) and a
read key for its remote (`R2_ENDPOINT`, `R2_BUCKET` and their key, or `RESEARCH_STORE_CONFIG`). Without them it refuses
in one line naming what it needs, exit 2 (`verity_bench/cli.py`'s `_store_missing`, before anything is read);
`--local-only` needs the client alone. `prove layers` needs neither when every layer is given as a directory.

```text
verity bench prove --system flock --workload glm53                  every block proved
verity bench prove --system spot --workload glm53 [--sampling ART]  Flock at the workload's sampling plan of record
    [--provers N]                                                   the prover count each type is priced at (default 1)
    [--cores N]                                                     the host's CPU cores a card each type is priced at (default 10)
    [--rows N]                                                      the N costliest block types, the rest in one row
    [--json]                                                        the table as JSON, every row whole
    [--steady-rows FILE]                                            also write the steady-state rows the planner reads
    [--local-only]                                                  this machine's catalog alone, not the store's remote
verity bench prove lean-verify --lean VERIFIER --proof TREE         the Lean verifier timed on one kept proof (below)
verity bench prove layers --flock R --zk R --sampling R --recursion R
                                                                    the proof benchmark by layer (below)
```

## What it prints

- **A row a block type** of the workload's census under its cut: the blocks of it proved per served token, the prover's
  ms per block, and where that figure is from: measured at steady state (a row of the proof server's rate), steady
  state, its family's (no row of its own, so the dearest steady-state row of its Definition and K at other statics, at
  2^Δk its seconds a block; priced, not measured, and counted in no steady-state share), measured in a single session
  (the `proof_overhead` card), modeled from k_log (the card's types' median seconds a witness bit), or
  missing. Each row names the run or art its figure is from, and a steady-state row its prover count.
- **A row priced from a steady-state row names the Definition that row's statements are of**, with its parameters,
  after the staged type: `` `TYPE` (`DEFINITION`) ``. A row may be staged for a census type in another Definition than
  the type's own, so the line says which form priced the type: `its Boolean version` where the tree registers that
  Definition as the type's (`verity.circuits.ops.boolean.versions.boolean_version`), `another Definition` otherwise, and nothing
  where it is the type's own. A family's row names the Definition of the row it is scaled from the same way, after that
  row's own type's. The JSON rows hold it as `definition`, null on a row no steady-state row prices. Which Definition
  was staged changes no figure: a row prices the census type it was staged for.
- **A type is priced at a stated prover count**, `--provers N`, one by default: only the steady-state rows at N provers
  measure it, so its figure doesn't depend on which counts were run or in what order. Among those rows it takes the
  fullest statement's, then the newest run's. A row at N provers that didn't stand is named on its type's row with
  its reason, and so are the other counts that measured the type.
- **A type is priced at a stated count of host CPU cores a card**, `--cores N`: the cores a card's provers hold on
  the host the table prices for, 10 by default, node A's share (the cores its provider gives a card). A type whose
  proving is bound by the host costs more GPU-seconds a statement at fewer cores, so among the rows at the stated
  prover count only those timed at N cores price a type, by the same order. A type those rows measured only at more
  cores is priced from the ones at the fewest cores above N, and its row says so ("timed at 24 cores, above the
  host's 10"). A row timed at fewer cores than N prices nothing, since the host's share is the least a prover holds:
  it is named on its type's row, like a row at another prover count. The family's rule takes the rows chosen this
  way.
- **The totals:** the prover GPU-seconds per served token, the tokens a prover GPU covers per second, and the prover
  GPUs a serving GPU needs, each with its range, whose other end prices every type not measured at steady state at the
  steady-state types' median seconds a witness bit. A total is called measured only when every block type is measured
  at steady state. When a block type has no figure the totals leave its proving out and are called partial, naming each
  such type with its share of the blocks, and of the MACs where the census holds them (a type with no time has no
  share of the proving time). Otherwise a total is modeled. The proof server's verifier's seconds are given where a
  steady-state row holds them, and "not measured" where none does. The served MACs the census holds in no proof block
  are in no figure, and the totals give their share.
- **The share of the proving time that rests on rows timed above the host's cores:** the types priced from a row
  timed at more than `--cores` cores, their own or their family's, with the cores those rows were timed at. On a
  host of the stated cores those types' figures are not measured.
- **How many priced rows ran with their card off their CPUs' NUMA node:** of the steady-state rows that price a type,
  those whose card was on another NUMA node than their provers' CPUs, those whose card was on their node, and those
  that don't record it, each with its share of the proving time. Then come the rows priced from the single-session
  card, whose table records no CPUs. This is in the table's notes and in the inputs of both proving totals. A row
  records its card's node itself (`card_numa_node`, against its `cores.numa_nodes`). For a row that doesn't, the
  harness's `verity_bench/data/off_node.json` holds a scan of the runs' own host records: the record whose runs it
  read, the runs it read by id (`read`), and those among them that were off-node (`off`). A run the scan read and
  doesn't list off-node was on its node, and a run it didn't read isn't recorded, whatever its id. The shares are of the whole proving time, so
  they need not add to 100%. Off-node rows still price their types: the line moves no value and no mark.
- **The Lean verifier's seconds and the proof's bytes per served token, modeled.** Both are "not measured" until a
  Lean verifier run of record exists (below). With one, the block type it ran on takes its figures, and every other
  block type is scaled from it; each line says `modeled, not measured`, then what was measured (the block type and its
  share of the proving time, its statement, the verifier's seconds on it with each run's, or its proof files' bytes,
  the run and its host) and what was scaled (how many block types, their share of the figure, and the basis). A total
  is called measured only when every block type has a run of its own.
- **Under a sampling plan that states its own prices, both totals.** A plan's sample is sized by its planner at the
  planner's price of each block, which the plan's record may state (`prove_s` a cell: its seconds for the blocks the
  cell takes). The table prices the same blocks from its own sources, so the two totals differ wherever those prices
  do. The command prints the prover GPUs per serving GPU at each, the plan's ms per block beside the table's on every
  row, each source's proving time at the table's prices over the plan's, and the block type that holds most of the
  difference. The plan's prices are its record's: they are no measurement the table reads, and they hold whatever its
  planner counts in a block's price. A plan that prices only some of its cells is refused.
- **Under a sampling plan, the prover GPUs a serving GPU at two designs,** a line each naming its source: at the
  sampling plan of record (its art), the table's total; and at the exfiltration run of record's design (the run that
  carries `run_of_record=exfiltration-glm53-compute-audit` and no `superseded_by`), read by `verity bench
  exfiltration`'s own reader of that run's rows: the planner's world year at 10% of total compute, whose proving is a
  share of total compute at an audit budget β, so a serving GPU holds that share / (1 − β) prover GPUs, modeled.
  Where the record can't be read, its line says why and the table stands.
- **The share of the proving time each source covers.**
- **The Lean verifier's seconds a block on every block type,** a row each in the main table's order, marked by what
  it rests on: measured on a statement of its own Definition, modeled on its Definition's measured statement at its
  AND gates, assumed at the rate fitted over every measured statement, none with no AND gates. A measured or modeled
  row names the run that measured its statement, that statement's blocks, and the circuit file the Lean verifier read
  it from (the unrolled file, or its templates), as the record's statement id says.
- **The steady-state rows it didn't read,** under "Unread": a labelled row whose staged Definition is no block type
  of the census under its cut, at any prover count. Each is named with what holds it, the type and Definition it
  staged, and its own ms per block where it is a steady-state measurement, else why it isn't one. It is on no row and
  in no figure, the range's other end included: a census that no longer holds a type leaves its measurements listed,
  not dropped.
- **The steady-state rows as the break-even planner reads them,** in `--steady-rows FILE`: a row for each block type
  its own steady-state row measures at the stated provers and cores, by its census name, with the statement's
  GPU-seconds and blocks that row timed, its provers, their cores and the run or art that holds it (`prove.steady_rows`;
  `benchmarks/spot_figures/breakeven`'s `--steady FILE`). A type priced from its family's row, a single session or the
  k_log model gets no row: it is no measurement of that type.
- **The totals per monitoring period over all compute:** with every block proved, the census's datacenter and the
  world's serving for a week, a month and a year; under a sampling plan, the compute and period the plan is sized for.
- **Under `--system spot`, SPOT's resampling holes in the code that ran**, over the runs among its inputs (not its
  arts): the label (`SPOT`, or `SPOT soundness` only when both holes are closed), each hole's status, and the theorem
  with its open premises as the lock lists them (`verity_bench.spot_status`; `spot_holes` in the JSON).

`--system flock` proves every block. `--system spot` takes the same costs at the blocks a sampling plan samples: a
system is a `prove.ProofSystem`, a census in and a table out, and SPOT wraps one with a plan.

## Inputs

Every input is a record of the store, named there by a label and never in code: what carries
`run_of_record=prove-WORKLOAD-SLOT` and no `superseded_by` (`prove_records`'s docstring has each slot's format).

| Slot | What carries the label |
| --- | --- |
| `census` | the workload's block census (`benchmarks/one_stage/block_census.py`'s art) |
| `cut` | the `partition` card that recuts it |
| `flock-steady` | each run of `benchmarks/one_stage/server_rate.py`, an art naming them in its `runs`, or an art holding one run's rows as its `rate.json` |
| `flock-single` | the `proof_overhead` card |
| `flock-warm` | each run of `benchmarks/served_units/stage_prove.sh` (optional): a type with no steady-state row takes its own statement proved on one card, after WARM warm sessions the median of RUNS timed ones, marked `measured from warm sessions` with its WARM and RUNS |
| `spot-sampling` | the sampling plan's art, for `--system spot` |
| `lean-verify` | each run of `verity bench prove lean-verify` (optional: without one the two figures are "not measured") |

```text
research data label TARGET run_of_record prove-glm53-census --by LANE     name TARGET the glm53 census of record
research data label OLD superseded_by NEW --by LANE                       retire OLD once NEW replaces it
```

An input that isn't labelled or is superseded is refused, by name and with the label that would fix it; so is a slot
two unsuperseded targets claim, and a sampling plan and a census that don't cover each other. A steady-state run or
art that is superseded, a run that holds neither `out/rate/rate.json` nor a proof server report and an art that names
no run are left out and listed under "Refused".

## A steady-state row

A steady-state run's format is `out/rate/rate.json` as `benchmarks/one_stage/server_rate.py` writes it: a list of
rows, one a staged type and prover count. The command reads, by the script's names:

| Field | What it is |
| --- | --- |
| `stage.word` | the census type the row was staged for, by its census name, matched to the census's block types (optional: without it the type is `stage.definition`'s) |
| `stage.definition`, `stage.k_log` | the staged Definition's id, the one its statements are of, and log2 of a block's witness bits |
| `stage.blocks`, `stage.full` | the blocks in the statement proved, and in the type's full statement |
| `provers`, `gpus` | the provers that proved it, and the GPUs they held |
| `cores` | the CPU cores its provers held for its card: a count, or a mapping whose `count` is one (optional: see below) |
| `statements_per_s` | statements a second over the steady window |
| `warm`, `sessions`, `accepted`, `refused_n`, `unfiled` | the warm sessions a prover and the timed sessions, those accepted, those refused and those their verifier never filed |
| `exits`, `void` | each process's exit code, and why the row has no rate (only on a row that has none) |
| `flat` | whether the statement cache's hit rate was flat |
| `throttled.nr_throttled` | the times its CPUs were throttled while it ran |
| `warm_per_prover`, `idle_provers` | each prover's warm sessions, and the provers that filed no session in the window |
| `verify_s` | the verifier's seconds a statement (optional: "not measured" without it) |

A statement's GPU-seconds are `gpus / statements_per_s`, and a block's ms are that over `stage.blocks`, times 1000.

A row is a steady-state measurement when it passes six checks of the proof server's steady-state rules, the
harness's `verity_bench.steady`, which the command calls and never restates:

- **`accepted`:** every session it ran was accepted (`accepted` is `provers` times `warm` plus `sessions`), none was
  refused and each was filed by its verifier.
- **`exits`:** every process exited 0.
- **`not_void`:** it has a rate: no `void`, `error` or `skipped`, and a finite `statements_per_s` above 0.
- **`flat`:** `flat` is true.
- **`no_throttling`:** its CPUs were never throttled (`throttled.nr_throttled` is 0).
- **`every_prover_warmed`:** each of its provers finished a warm session before the timed ones (`warm_per_prover`),
  and none filed no session in the window (`idle_provers` is 0).

Only a pass counts. A row that fails a check, or whose fields can't show it ("cannot tell": a field it lacks, a count
that isn't one), prices nothing, as does one that lacks a field its price needs. It is named on its block type's row
with each check it didn't pass, the check's answer and its reason, as `` `flat` (cannot tell): its row lacks `flat` ``.
The rules' four other checks are no condition of a row's price here: three read the run's other files (the lease and
the card, the card at its start and end, a level window), and `one_numa_node` reads `cores.numa_nodes`, which a row
may lack. A row with no `stage.definition`, one that failed or was skipped, is listed under "Refused".

A row's cores are the CPU cores its provers held for its card. A row states them as `cores`: the count, or a mapping
that holds it as `count` beside what else it says of them (their list, their NUMA nodes). A run's row that states
none takes its run's, the `cpus_used` of the run's `out/host.json`, where the run holds that file and it states a
count. Any other row that states none, an art's or a run's without the file, is read as timed at 24 cores, and so is
every staged type of a proof server report. A row whose `cores` is anything else is not a steady-state measurement.

A run's rows completed or corrected after the run go in an art that holds them as `rate.json`, the same list of rows,
with the run in its meta's `runs`. Labelled, the art gives those rows in place of the run's own: they are ordered as
that run's, each type's row names the art as its record and the run that measured it, and the run's files aren't
read. Such an art holds one run's rows: one that names several runs is listed under "Refused". An art that holds no
`rate.json` stands for the runs it names, each read from its own files.

A run from before `server_rate.py` holds no `rate.json`, only its proof server reports: each `*/server-kK/server.json`
with the `types.json` beside it. Such a run is read from those, each staged type of a report as its row at K provers,
a steady-state measurement when the report wasn't stopped or timed out, every prover exited 0, every session was
accepted and its cache was flat. A run that holds `rate.json` is read from it alone.

The serving rate is not a record. It is a stated, modeled input: the census's MACs a served token at the fleet's assumed
serving MFU (`benchmarks/slowdown/data/reference.toml`) of the card's data-sheet peak (`benchmarks/census/hardware.json`).


## The Lean verifier's seconds and the proof's bytes

`verity bench prove lean-verify` runs Flock's Lean verifier on one kept proof and writes the record the table reads:

```text
verity bench prove lean-verify --lean VERIFIER --proof TREE [--repeat 3] [--out DIR]
    [--word TYPE] [--definition ID] [--k-log K] [--blocks N] [--full N]   the block type, where TREE's stage.json doesn't say
    [--session DIR] [--circuit F] [--public F] [--program F] [--partition F] [--registered F] [--tables DIR]
```

VERIFIER is the Lean verifier's executable, as `lake build` leaves it under `verity/protocols/proofs/flock/verifier`. TREE
holds one session directory (`session.json` and its proof files, `<table>.rep0.bin` and `<table>.rep1.bin`) and the
statement it was proved on (`circuit.txt`, the public file, and the verifier's program (`program*.json`),
`partition.json` and `registered.json` where the statement has them), each found by its name. A run is one process of the verifier's `verify` on the session,
as `benchmarks/one_stage/a0.py` runs the verifier, so its seconds hold the verifier reading the statement, setting it
up and verifying the session. The command runs it `--repeat` times and records each run's seconds, the proof files'
bytes, the block type, the host (node, cores, CPU model) and the label of an accept. A run the verifier refuses ends the command: the record
says refused, with the verifier's reason, and nothing is run again.

The partition and the circuit are held before the verifier starts, and the program is not held to the circuit. The
circuit's second line binds a partition (a rule and a digest). The command hashes a partition file under the bound rule
and ends with the two digests when the file isn't the bound object or the circuit binds nothing for it. Under the
provisional rule (`flock-circuit/unit-cover/v0`) the verifier takes no partition object and refuses a session given
one: a partition file that is the bound cover is checked and not handed to it, and the record says so
(`statement.bound`). The circuit is held to `stage.json`'s `circuit_sha512` where it states one. The second line's
`program_sha512` is no digest of a program file on a unit statement: the stager writes there the SHA-512 of the unit's
name with a SHA-256 over the staged Definition's id and the row layout, which a program descriptor doesn't hold. So the
command compares no program with it. It records which program was handed (`statement.program`: its digest as the
verifier computes it and the Definition it names) and ends only when that Definition isn't `stage.json`'s `word`, a
comparison of two names.

**What an accept holds is the record's `label`,** and the table prints it word for word with the seconds. At `--zk`
on a circuit under the provisional rule the verifier takes the units as the circuit states them and compares no
Program with the circuit, so it accepts the same session with another computation's Program. An accept there is
labelled "accepted on the provisional unit cover, units as stated, the Program not held to the circuit": it is about
the circuit file, and doesn't say the verifier knows what the statement computes. Any other accept is labelled by its
partition rule alone.

As a research run, from a clone on a CPU node, with the store's read-only key (`--store-read`: the job fetches the
arts, and without the key its first fetch ends it). `lean_verify.sh` fetches the proof's art (kind `flock-proof/v1`: the
session directory and what of its statement the art holds) and, with `--with ART PATH`, each file the statement keeps
in another art (a circuit too large for the proof's art, a program made after it) beside it in one tree, then builds
the verifier:

```text
research run --project verity --on NODE --mem-gb GB --timeout SECONDS --source . --cwd clone --store-read -- \
    bash benchmarks/prove/lean_verify.sh ART [--with ART PATH]... [--repeat N] [--word TYPE] [--k-log K] [--blocks N]
research data label RUN run_of_record prove-glm53-lean-verify --by LANE --ref RUN
```

The node's scratch holds the circuit twice, the fetched file and the verifier's archive of it.

The table reads `out/lean-verify.json` from each labelled run. A block type's record is its newest run's. A run whose
record the verifier refused, that wrote no record, or whose block type the census doesn't hold under its cut measures
nothing and is listed under "Refused".

**What is scaled, and on what basis.** A block type with a run of its own takes the statement's seconds (the median
of its runs) and its proof files' bytes, each over the statement's blocks. Every other block type takes the measured
statement's seconds, and bytes, a witness bit times its own 2^k_log bits, so a full statement of any type is priced
as the measured one:

- **Bytes:** a proof file's length is fixed by its statement's shape (its size m, the block's k_log and the opening's
  claims: `Flock.Proof.decodeRead`, `Flock.Zk.reader`) and by nothing else of the block type. Every full statement has
  m = 35, so its proof is the measured one's but for the lincheck's rounds, two field elements a unit of k_log, and a
  ring switch for each opening claim past the first two.
- **Seconds:** the rounds and Ligerito queries of a session are fixed by m too, but the verifier also reads and sets up
  the statement from its circuit file and folds the lincheck over the circuit's matrix entries, work that grows with a
  block's description. Scaling prices every type's description at the measured type's, and no measurement of a second
  type backs that.

Both figures assume full statements, as the proving time's steady-state rows do. They are a model from one block type
until more types have runs of their own.

## The proof benchmark by layer

`verity bench prove layers` prints one table of four benchmark results, Flock, Flock with zero knowledge, with sampling
and with recursion, each read from what its own command wrote (`verity_bench/proof_layers.py`, with a reader a format;
the table it prints is `verity_bench/proof_layers_print.py`). It runs nothing. Each
flag takes a run's `out/` directory (or the run directory holding it), read with no store, or a run id, fetched through
the evidence store where `research` is installed and this machine holds the store's read key. A run id the store can't
give is refused with why and the fix: give the run's `out/` as a directory. A layer not given is printed as "not run".

```text
verity bench prove layers [--flock DIR|RUN] [--zk DIR|RUN] [--sampling DIR|RUN] [--recursion DIR|RUN]
```

| layer | reads | prover seconds | Lean verifier |
|---|---|---|---|
| `--flock` | `benchmarks/prove_path` with `PP_PLAIN=1`: an arm's `audit.log` (its `PROVE-PATH` and `A5` lines, or under `PP_RECORDS=1` its `RECORDS` line), its `summary.json` and its rank or window `summary.json`, and `ran-with.txt` beside the output; or `a5_llama1b_h200.sh`'s plain out | Flock's prover a sampled proof block, the mean over every rank's sampled proof blocks | every rank accepted, each rank's own audited `A5` line with `lean-verify/m0` true |
| `--zk` | `prove_path`'s zk arm, or `benchmarks/one_stage/a5_llama1b_h200.sh`'s out | as above | as above |
| `--sampling` | `a5_llama1b_h200.sh`'s out (`summary.txt`, `verdicts.txt`, `machine.txt`, a5.sh's `a5/a5.log` and `a5/a5/summary.json`), or a `prove_path` arm | as above, from the audit's `summary.json` timing | the audit accepted, with `lean-verify/m0` true |
| `--recursion` | `benchmarks/recursion_e2e`'s `recursion_e2e.json` | wall, one recursive run's path without Lean's step | the headline's `ok` and `composed` both true |
| `--flock`, `--zk`, `--sampling` | `benchmarks/one_stage/layers.sh`'s layer: its `out/layers.txt`, `out/job.log`, `out/a6/summary.json`, each session's `lean-verify.log` and `gpu-lease-usage.jsonl` beside `out/`; the run's directory gives each layer its own (`flock/`, `zk/`, `sampled/`) | Flock's `prove_total_s` a statement | the plain run's every statement accepted, or the audit accepted and complete with every Lean verdict, every check before the sample and every negative case true |

The directory's files decide its reader, and a flag refuses another command's output. A `prove_path` directory holding
more than one honest arm is refused: give the arm's directory. A `--flock` benchmark result must state the plain proof,
and a `--zk` or `--sampling` one the zk proof (the sampling layer is Flock with zero knowledge at the sampling plan):
one whose output states another proof, or none, prints no number and writes null with why, whichever command wrote it.
Each printed benchmark result states its proof (`plain`, `zk`, or "not stated"), so a plain run's numbers in the
sampling layer's say plain.

A `layers.sh` run (`benchmarks/one_stage/layers.sh all`) proves stand-in values, never served. Its benchmark result's
workload opens with `stand-in values, never served:` where `layers.txt` says so, else `values not stated:`, then its
statements, each with its proof blocks and template, and the source commit `job.log` states. Its figures are a
statement's, not a proof block's: each cell gives one figure a statement, `3.0 s a statement of 1,000 proof blocks`, and
none is divided into a figure a proof block, since a statement's seconds don't grow with its proof blocks (a statement
of 20 sampled proof blocks can take most of one of 1,000's). Under the three metric ids it writes null, with why. Its
proof is `zk` only where every session's Lean `verify` line has `--zk`, and `plain` where the summary says `plain` and
none has it. Its device is the cards gpu-lease's usage record names for each `flock-circuit prove`. Its ε at δ is the
audit's `wrong_units`: `at most 100 of 1,000 incorrect proof blocks except with probability 2^-20`. Its `ran with:` line
is `layers.txt`'s first line, where that names its coins and parameters in these words.

Each benchmark result keeps the commit its output states it ran at, and the line `- commits:` under the `ran with:`
lines names each: a prove_path or a5 result's from its `machine.txt` (`git rev-parse HEAD`, for example `0123456789a
(machine.txt's git HEAD)`), a `layers.sh` result's from `job.log`'s `source`, recursion's from its `params.commit`. No
output states a branch. The printer can't know `main` offline, so the line ends with how to check: a benchmark result
reruns from `main` only where `git merge-base --is-ancestor COMMIT origin/main` holds.

An a5 benchmark result's workload opens with `served values:` where its summary's `source` is `served` and its
`REGISTERED` line's `kind=served`, `stand-in values (summary.json source stand-in):` where its source is `stand-in` and
nothing says served, and `values not stated:` otherwise. Its `ran with:` line is its `COINS` line's `draw:` and sessions
clauses where that says `one local challenger session`: coins `local challenger`, and parameters "not stated", since the
line says the key is made at the deployed LMS parameters but the challenger is a development stand-in for the auditor's.
A run made before the `COINS` line has `ran with: not stated`.

A prove_path served-arm run (`flock/out` or `zk/out` under the run's directory) takes its `ran with:` line from
`ran-with.txt`, which names its coins and parameters in these words. Its workload opens `served per ran-with.txt (...)`
from that file's `served` line. Where the audit's own `source` field says otherwise (prove_path writes `stand-in` for a
sealed served window), the two statements disagree, and the workload says so and prints both: `the output disagrees on
its values: ran-with.txt says served (...), the audit's source field says stand-in:`. Under `PP_RECORDS=1` the arm has a
`RECORDS` line and no `PROVE-PATH` line: its benchmark result is accepted where `RECORDS` is ok and accepted with its
negatives refused, and the A5 line's Lean verdicts are true. A `*-forged` arm with no `audit.log` is no forged run: the
workload says `no forged run`, and the note under the table gives its `commit.log`'s error.

A benchmark result prints its numbers only when its Lean verdict is an accept. Any other prints none, and says why under
the table. When no benchmark result prints a number, the first line is `prove layers: no number: WHY` (each layer's why)
and the command exits 1; otherwise it exits 0.

Under the table's title, each benchmark result's coins and parameters are a line of their own, `- LAYER: ran with: coins
COINS; parameters PARAMETERS (DETAIL)`: COINS is `deployed challenger`, `local challenger` or `in-process` and
PARAMETERS `deployed` or `development`, each only where the output says so outright, "not stated" for a half it doesn't
say, and `ran with: not stated` where it says neither. DETAIL, the output's own words, is also the JSON's `detail`. No
output yet states deployed parameters; `layers.txt` and prove_path's `ran-with.txt` state development ones. Unless every
benchmark result ran with a deployed challenger at deployed parameters, the line under the title is `a measurement, not
an audit`. The JSON's `audit` is always false, even then: this command checks no challenger's signature, so nothing it
prints is an audit.

The table's columns, after the layer and its workload:

- **proof:** `plain`, `zk`, or "not stated", the proof the output says it holds.
- **device:** what the output states it ran on, else "not stated", never a guess. A `layers.sh` benchmark result's is
  gpu-lease's record (above). Recursion's is CPU where its params
  say `GPU=0` and `OPROVE_GPU=0` (job.sh's CPU prover); a prove_path or a5 result's is read from its timing's `gpus` and
  `device` (a5's `proving_of`): `CPU (timing gpus 0)` where every member's `gpus` is 0.
- **ε at δ:** what the output states. An a5 summary's `epsilon` table gives its first pair (the audit's δ) and its
  last (the smallest δ): `at most 3 of 500 incorrect proof blocks except with probability 0.5; at most 40 except with
  probability 1e-9 (summary.json epsilon, the first and last of its 6 pairs)`. Without one, the audit's `wrong_units` (a
  `layers.sh` summary, an a5 summary, or an arm's one `A5` line) gives one pair. A prove_path arm with neither prints
  `ε not stated at δ 0.5` from its `PROVE-PATH` line's `delta`.
- **prover, Lean, proof bytes:** each cell starts with its figure's unit and what it is per, so no figure is read as
  another's: `2.5 s a sampled proof block (Flock's prove_total_s, the mean over 8)`, `400,000 bytes a sampled proof
  block (...)`, and recursion's `1500 s a run, wall (one recursive run's path without Lean's step: ...)`. The four
  benchmark results stay one table: the three a sampled proof block compare cell for cell, and recursion's figures are
  a whole run's.

The table is followed by its JSON line, and under a research run, or with `$OUT` set, it writes `proof-layers.json`.
Its schema is `verity-bench/proof-layers/v2`: `{schema, audit, results}`, one object for each layer's benchmark result.
v1 named that list `rows`; nothing reads either by key.

Run as `benchmarks/prove/run.py layers` (`LAYER_METRICS`), it also writes three of `metrics.toml`'s metrics through the
kit's writer, each a figure a sampled proof block:

| id | figure |
|---|---|
| `benchmarks/proof-layer-prover-seconds` | Flock's `prove_total_s`, the mean over the sampled proof blocks |
| `benchmarks/proof-layer-lean-seconds` | Lean's verify seconds, the mean over the sampled proof blocks |
| `benchmarks/proof-layer-proof-bytes` | one proof's bytes over its sampled proof blocks: an a5, `layers.sh` or prove_path result's one rep's (its reps are timed repeats of one proof, and differing reps give none). prove_path's own `per_unit` sums the reps, so its benchmark result reads them from the arm's rank or window `summary.json`, and prints `every rep's` only where no such file states them |

Under each id it writes three measurements, in this order:

1. the Flock layer, member `flock`;
2. the zero-knowledge layer, member `zero knowledge`;
3. the sampling layer last, with no member. That is the metric's own reading, Flock with zero knowledge at the
   sampling plan, the system as served, and the one the metric's title, unit and axis are for.

prove_path and a5 both give these figures by dividing a member's session timing by its proof blocks sampled
(`draw_counts`), as prove_path's own `per_unit` does. A member whose timing has no count of sampled proof blocks (one
keyed by its chunk's name) gives no figure, and the benchmark result says so under the table. A layer that was not run,
was unreadable, has no Lean accept or gives no figure a proof block writes its value as null, with why, and never a
number. Recursion is no member, because its seconds are a whole recursive run's. Its numbers on the Metrics tab are
`benchmarks/recursion_e2e/metrics.toml`'s two metrics, which the recursion job writes itself.

## On a machine of your own

The install, from a clone of the tree, with [uv](https://docs.astral.sh/uv/getting-started/installation/):
`uv sync --all-packages --extra torch-cpu` (the printer and `research`, the store's client, which only a run id needs).
The runs below build Flock's circuit prover, `flock-circuit` (Rust, `cargo`), and Lean's `verity-verify-proof` (the
tree's Lean toolchain, through `elan`) from the tree themselves. Each writes its result under `$OUT`, default
`$RESEARCH_RUN_DIR/out` (a5's `./out` outside a run, recursion's `/tmp/rec-bench-PID/out`); give that directory to the
printer's flag. Each from the tree's root:

1. **The Flock, zero-knowledge and sampling layers**, on one H200 with about 450 GB of memory: the commands of
   `benchmarks/one_stage/README.md`'s section "The three benchmarks of the proof-layers table", as written there. They
   are the vLLM adapter's one bootstrap (`adapters/vllm/verity_vllm/ops/pod_bootstrap.sh`, which makes the vLLM
   environment, the int8 checkpoint and the sm_90 taps under `WORK`), three partition files, and three runs of
   `benchmarks/one_stage/a5_llama1b_h200.sh`. Give their three `OUT` directories to `--flock`, `--zk` and `--sampling`.
2. **The recursion layer**, on 32 CPU cores with 128 GB (`run.py`'s memory behind the ZK server), about an hour at
   K = 4096:
   `taskset -c 0-31 bash benchmarks/recursion_e2e/job.sh K=4096 N=2 GPU=0 FLOCK_WORK=DIR`. Give its `$OUT` to
   `--recursion`.

Then `verity bench prove layers --flock A --zk B --sampling C --recursion E`, which reads only those directories.

`benchmarks/prove_path/run.sh` is no step here: before any build it refuses, as `PP_LLAMA`, a workload directory
`a5_llama1b_h200.sh` made, as its header says. What needs a team node today: recursion's launcher, `verity bench
recursion_e2e run`, starts a `research run` on a team node; `job.sh` as above is the same job without it.

## What you may change

The prover's side of the four runs: its code, its build, the machine, its cores and its device. Each benchmark result's
figures are then that prover's, read from its own output, and it names its workload, proof, device and coins as the
output states them.

## What is fixed

The printer's rules, in `verity_bench/proof_layers.py`: a benchmark result prints numbers only on its Lean accept;
the proof the output states must be the layer's (`plain` for Flock, `zk` for zero knowledge and sampling);
a prove_path directory must hold one honest arm; a figure a sampled proof block is a member's session timing over its
proof blocks sampled (`draw_counts`). The Lean verifier is the tree's, built from it by each run, and its verdict is
what the benchmark result's Lean cell reads. The workload, its partition and its sample are printed in the benchmark
result, so a changed partition is a different benchmark result.

## How a cheating change is refused

- **prove_path's forged arm.** With `PP_FORGED_PARTITION`, `run.sh` runs a second arm with `--forged`: a proof block's
  committed value changed and committed consistently, the prover's self-check skipped (`FC_ZK_SELF_CHECK=skip`) so it
  proves the false witness. That arm passes (its `PROVE-PATH` line has `forged` true and `ok` true) only when every
  check before the sample passed, the forged proof block was sampled, the audit record rejects and the member holding it
  was stopped: behind the ZK server, no proof of it was released; with `PP_PLAIN=1`, Lean refused the proof it read
  (`forged_row`, `records_audit.forged_outcome`). The printer reads the honest arm only, so the forged arm's verdict is
  in its own `audit.log`, not in the table.
- **The recursion job's negative runs.** `job.sh`'s `negative` step runs the Lean step again with rep 0's inner coins
  flipped (`NEG`, default `challenge,query`), each verified under every V* statement. The result's `negative_refused`
  holds each, and `parts_ok`, which `ok` needs, needs every one refused; the printer's accept needs `ok` and
  `composed`. With `forge-after` in `STEPS`, coefficient, direction and message forgeries made on the honest releases
  are put to Lean beside their honest controls (`forged_refused`), which is reported and not part of `ok`.
- **The Lean verifier's refusal.** A proof Lean refuses gives its benchmark result no number: prove_path's needs every
  rank's A5 line with each `lean-verify` verdict true, a5's its checks accepted and each verdict true, recursion's
  Lean's composition accepted.

## The metrics

`run.py`'s `METRICS` are the first five metrics of `benchmarks/prove/metrics.toml`, and its `LAYER_METRICS` the last
three (above, "The proof benchmark by layer"). `METRICS` are the prover GPUs a serving GPU needs (the headline), the
prover GPU-seconds, the Lean verifier's seconds and the proof's bytes a served token, each SPOT's at the workload's
sampling plan of record; and the prover GPUs a serving GPU at the exfiltration run of record's design
(`benchmarks/prover-gpus-per-serving-gpu-at-record`, modeled, its input the run it is read from). Under a research run,
`--system spot` writes them into the run's `result.json` through the harness kit (`prove_records.record`) and `--system
flock` writes nothing; outside one neither writes, and what it prints is the same either way:

```text
research run --on auto --project verity --source . --cwd source --store-read -- \
    uv run --extra torch-cpu python benchmarks/prove/run.py --system spot --workload glm53
```

A measurement's mark (`measured`, `modeled` or `partial`) is its total's, and the table's words for it (for the Lean
verifier's two figures `modeled, not measured` with each source's share (own, family, fitted), the fitted rate, the
block type measured and what its accept holds) come first among its inputs, then the records the table read; the
proving totals' words also give the plan's own prices where its record states them. The prover GPUs a serving GPU are
the proving total over the stated serving rate, so their mark is the weaker of the two's (measured, then partial, then
modeled), first among their inputs, then proving's words and the serving rate's. A total the table gives as not
measured has no value, says why and is marked `modeled`. Both prover GPUs a serving GPU, at the plan of record and
at the exfiltration run of record's design, carry the goal's target, at most 0.0545 (`prove_records.GPUS_PER_SERVING_GPU_TARGET`:
proving's share of the 10% compute budget, until the proof system's own number is set), and, with a value, the verdict `meets` at or under it or `misses`
over it; no other measurement carries a target. The validation is
the six checks a steady-state row is held to: passed when some block type is priced from rows that passed them all
and no block type is left out of the totals. The run keeps the table as `out/prove.json`, and its fingerprint names
the sampling plan.
