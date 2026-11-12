# prove

`verity bench prove` is the one command for what a proof system's prover costs on a workload. Its code is
`verity-bench`'s (`benchmarks/harness`: `verity_bench/prove.py` holds the rules, `prove_records.py` the store side and
the table); this directory holds its runner. It reads the store's labelled records and runs on any machine that reaches
the store, with no GPU and no node.

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
  Definition as the type's (`verity.definitions.boolean.boolean_version`), `another Definition` otherwise, and nothing
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
- **The share of the proving time each source covers.**
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
serving MFU (`verity/protocols/inference_only/rho/data/reference.toml`) of the card's data-sheet peak (`benchmarks/census/hardware.json`).


## The Lean verifier's seconds and the proof's bytes

`verity bench prove lean-verify` runs Flock's Lean verifier on one kept proof and writes the record the table reads:

```text
verity bench prove lean-verify --lean VERIFIER --proof TREE [--repeat 3] [--out DIR]
    [--word TYPE] [--definition ID] [--k-log K] [--blocks N] [--full N]   the block type, where TREE's stage.json doesn't say
    [--session DIR] [--circuit F] [--public F] [--program F] [--partition F] [--registered F] [--tables DIR]
```

VERIFIER is the Lean verifier's executable, as `lake build` leaves it under `verity/protocols/flock/verifier`. TREE
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
