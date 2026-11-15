"""`verity bench prove`'s store side: it reads a workload's inputs from the store's labelled records, prices the
workload under a proof system (`prove`) and renders the table. It runs on any machine that reaches the store: no GPU, no
node.

A record is named in the store, never in code (`served_table`'s rule). Each input is what carries the label
`run_of_record=prove-WORKLOAD-SLOT` and no `superseded_by`:

| slot | what carries the label | what is read |
| --- | --- | --- |
| `census` | one `block-census/v1` art | its `census.json` (`prove.Census.of`) |
| `cut` | one `partition` card (`spot-card/v1`) | the block types it recuts (`prove.Census.recut`) |
| `flock-single` | one `proof_overhead` card | each block type's single-session seconds (`prove_flock.Session`) |
| `flock-steady` | any number of runs, or arts naming their `runs` | the rows of each run's `RATES`, the file `benchmarks/one_stage/server_rate.py` writes: a row a staged type and prover count (`prove_flock.Steady`, `steady_row`); an art that holds `ART_RATES` gives those rows in place of its one run's own (`steady_rows`); a run from before `RATES` gives its proof server reports (`REPORTS`), a staged type of a `server-kK` report as its row at K provers |
| `flock-warm` | any number of runs of `benchmarks/served_units/stage_prove.sh` | each group's `WARM_SUMMARY` (`prove_time.py`'s) with its `WARM_CONFIG`: a statement proved on one card, after WARM warm sessions the median of RUNS timed ones, every session accepted (`prove_flock.Warm`, `warm_rows`) |
| `spot-sampling` | one art holding `mix.json` | the sampling plan (`prove.SamplingPlan.of`), for `--system spot` |
| `lean-verify` | any number of runs of `verity bench prove lean-verify` | each run's `LEAN_RECORD`: one statement of a block type the Lean verifier accepted, its runs' seconds, its proof's bytes and its `label`, what the accept holds, which the Lean line prints word for word (`prove_flock.LeanVerify`, `lean_verify`) |

The command refuses an input that isn't labelled, one that is superseded and a slot two unsuperseded targets claim, and
says which and how to label it. A steady-state run or art that is superseded, a run that holds neither `RATES` nor a
report and an art that names no run are left out and listed under "Refused", as is a row that names no staged Definition (one that
failed or was skipped).

A row is a steady-state measurement when it passes six checks of the proof server's steady-state rules
(`verity_bench.steady`) and holds what prices it (`unpriced`). The six are `CHECKS`, the rules' own `accepted`,
`exits`, `not_void`, `flat`, `no_throttling` and `every_prover_warmed`: every session it ran was accepted, none refused
and each filed by its verifier, every process exited 0, it has a rate, its statement cache's hit rate was flat, its
CPUs were never throttled, and each of its provers finished a warm session and filed one in the window. This reader
states none of them itself, so it and the harness give one answer on each. Only a pass counts: a row that fails a
check, or whose fields can't show it ("cannot tell"), prices nothing, and is named on its block type's row with each
check it didn't pass, the check's answer and its reason, in the rules' words, then with what it lacks to be priced
(`row_why`). The rules' four other checks are no condition here: three read a run's other files (its lease and card,
the card at its start and end, its window's sessions), and `one_numa_node` reads `cores.numa_nodes`, which a row may
lack. They are read through a run's own verdict instead (`own_verdict`): a run of the harness's steady-state point
(`STEADY_POINT_TOOL`) judged its row by all ten as it ended, and its attempt carries that verdict as its result's
validation. A run of the point whose verdict is `failed` prices none of its rows, each refused with the run and the
checks it didn't pass, and one whose attempt carries no verdict can't be told and is refused too. Any other run, one
timed outside the harness's point or before it, has no verdict of its own and is judged by `CHECKS` alone, whatever
the other four say. `report_why` judges a report.

A row whose card was on another NUMA node than its CPUs still prices its type. Where a row records its card's node
(`CARD_NUMA`), it is off that node when its CPUs (`cores.numa_nodes`) aren't on that node alone, and on it otherwise.
Where it records none, its run decides by `OFF_NODE`'s scan of the runs' own host records: off where the scan lists
the run, on where the scan read the run and doesn't list it, and not recorded where the scan didn't read it
(`off_node`). The table's notes and the proving totals' inputs count the priced rows of each kind with the share of
the proving time they hold, and the rows priced from the single-session card, whose table records no CPUs
(`prove.report`'s `off_node`).

A type is priced at the stated prover count (`--provers`), from the rows at that count only, and at
the stated host CPU cores a card (`--cores`): from the rows whose provers held that many, else from those that held
the fewest above it, named on the type's row and counted in the totals; a row timed at fewer cores prices nothing. A
row's cores are its `cores`, a count or a mapping with one as its `count` (`row_cores`); where a run's row states
none, its run's (`HOST`'s `HOST_CORES_FIELD`); and `prove.UNSTATED_CORES` for any other row that states none and for
every proof server report. A row
whose staged Definition is no block type of the census under its cut is on no row and in no figure: it is listed
under "Unread", with its own figure where it is a steady-state measurement.

A row prices the census type it was staged for (`stage.word`), whatever Definition was staged (`stage.definition`).
A type's row names the Definition its figure's statements are of (a row's `definition`): alone where it is the type's
own, as "its Boolean version" where the tree registers it as the type's (`boolean_version`), and as "another
Definition" where it doesn't, so a reader sees which form priced the type. A type priced from its family's row names
that row's Definition the same way.

The Lean verifier's seconds and the proof's bytes a served token are `not measured` with no `lean-verify` run of
record. With one, a block type the Lean verifier was run on takes its record's figures and every other type is priced
from them (`prove`'s rules: its Definition's measured types at its size, else the rate fitted over every measured
statement), and both totals print as `modeled`, each line naming what was measured (the block type, its share of the
proving time, its statement, the run and its host) and the share of the figure its own records, its family's and the
fitted rate price, with the basis. A table of its own gives each block type's Lean verifier seconds a block with its
mark (`P.LEAN_MARKS`: measured, modeled or assumed) and the run, statement and circuit file it rests on. A run
whose record the Lean verifier refused, that holds no record or whose block type the census doesn't hold measures
nothing and is listed under "Refused".

The serving rate isn't a record: it is modeled from the census's MACs a served token, the fleet's assumed serving MFU
(`FLEETS`) and the card's data-sheet peak, and the output says so.

SPOT's table also prints the prover GPUs a serving GPU at two designs, a line each naming its source: at the sampling
plan of record (its art), the table's total; and at the exfiltration run of record's design (`at_record`, the run),
read by `verity bench exfiltration`'s own reader of that record's rows (`exfiltration.rows_of_record` and `at_share`):
the planner's world year at `exfiltration.SHARE` of total compute, whose proving is a share of total compute at an
audit budget β, so a serving GPU holds that share / (1 - β) prover GPUs, modeled. Where that record can't be read the
line says why and the table stands.

Under a research run, a runner that names metric ids for the table's totals (`benchmarks/prove/run.py`'s `METRICS`)
has `record` write them through the harness kit: a measurement a total of SPOT's table at the workload's sampling plan
of record, its mark (measured, modeled or partial, as the table says it) the first of its inputs, and `CHECKS` as the
run's validation; and the figure at the exfiltration run of record's design (`AT_RECORD`), modeled, its input the run
it is read from. A table with every block proved records nothing.

The code is in `prove_common` (the slots, the workloads, reading a labelled record), `prove_rows` (a steady-state row),
`prove_store` (each input), `prove_table` (the table and what a runner records) and `prove_render` (the text); this
module names what its importers use.
"""

from __future__ import annotations

from verity_bench import prove as P
from verity_bench.boolean import boolean_version
from verity_bench.prove_common import (
    ART_RATES,
    CARD_SCHEMA,
    CENSUS,
    CHECKS,
    CUT,
    LEAN,
    RATES,
    SAMPLING,
    SINGLE,
    STEADY,
    SYSTEMS,
    WARMED,
    WORKLOADS,
)
from verity_bench.prove_render import (
    HOST_CORES_ARE,
    LEAN_ROWS,
    _design,
    _n,
    off_node_line,
    render,
)
from verity_bench.prove_rows import (
    OFF_NODE,
    STEADY_POINT_TOOL,
    OffNodeScan,
    judged,
    off_node,
    off_node_scan,
    own_verdict,
    row_cores,
    row_why,
    steady_row,
    unpriced,
)
from verity_bench.prove_store import (
    CENSUS_FILE,
    CENSUS_KIND,
    FLEETS,
    HOST,
    HOST_CORES_FIELD,
    LEAN_RECORD,
    PLAN_FILE,
    REPORTS,
    WARM_CONFIG,
    WARM_SUMMARY,
    at_record,
    census,
    lean_verify,
    report_why,
    steady_rows,
    warm_rows,
)
from verity_bench.prove_table import (
    AT_RECORD,
    FIGURES,
    GPUS_PER_SERVING_GPU_TARGET,
    STRENGTH,
    TABLE,
    build,
    measurements,
    record,
    validation,
)
from verity_bench.served_table import RUN_OF_RECORD, SUPERSEDED_BY
from verity_bench.steady import CARD_NUMA

__all__ = [
    "ART_RATES",
    "AT_RECORD",
    "CARD_NUMA",
    "CARD_SCHEMA",
    "CENSUS",
    "CENSUS_FILE",
    "CENSUS_KIND",
    "CHECKS",
    "CUT",
    "FIGURES",
    "FLEETS",
    "GPUS_PER_SERVING_GPU_TARGET",
    "HOST",
    "HOST_CORES_ARE",
    "HOST_CORES_FIELD",
    "LEAN",
    "LEAN_RECORD",
    "LEAN_ROWS",
    "OFF_NODE",
    "OffNodeScan",
    "P",
    "PLAN_FILE",
    "RATES",
    "REPORTS",
    "RUN_OF_RECORD",
    "SAMPLING",
    "SINGLE",
    "STEADY",
    "STEADY_POINT_TOOL",
    "STRENGTH",
    "SUPERSEDED_BY",
    "SYSTEMS",
    "TABLE",
    "WARMED",
    "WARM_CONFIG",
    "WARM_SUMMARY",
    "WORKLOADS",
    "_design",
    "_n",
    "at_record",
    "boolean_version",
    "build",
    "census",
    "judged",
    "lean_verify",
    "measurements",
    "off_node",
    "off_node_line",
    "off_node_scan",
    "own_verdict",
    "record",
    "render",
    "report_why",
    "row_cores",
    "row_why",
    "steady_row",
    "steady_rows",
    "unpriced",
    "validation",
    "warm_rows",
]
