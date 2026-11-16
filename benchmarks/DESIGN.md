# Benchmarks: design

What a benchmark has to follow: what it may import and write, the harness it shares, the interface the core benchmarks
keep, and how its numbers are declared as metrics. `README.md` has the index and how to run one.

## What a benchmark may import and write

- **It imports like any other caller.** `verity` from the owning module, an integration through its public modules
  (`tests/test_vllm_public_api.py`; only the integration imports vLLM, `tests/test_vllm_boundary.py`), the harness,
  and other benchmarks.
- **Nothing outside imports it,** by package name, by a path put on `sys.path` or loaded by file. A number a protocol
  uses reaches it through a card file citing an `art:` id, and code that a protocol, a device or a verifier runs is not
  a benchmark: it moves out to the package that uses it. `BENCHMARKS_IMPORTERS` in `tests/test_boundaries.py` names the
  files that still import a benchmark, and only shrinks.
- **It writes measurements, under its run's output directory** (`$OUT` under `research run`, else its `--out`). The
  run publishes them to the evidence store, and tables are rendered from the store on demand. Results, tables and
  plots are not committed (`tests/test_repository.py`); the tree keeps code, the parameters code reads and small
  fixtures beside their generators.
- **It writes no verdict.** Whether an audit passed is the verifier's to say. A benchmark that runs a
  verifier records what the verifier returned, and its own gates say only whether its measurement stands.

Some files here still break these rules: a kernel or a store the integration loads, a probe whose rate a card file
holds, the exporter of Lean data a guarantee reads. Each moves out to the package that uses it, and none is a pattern
for a new benchmark.

## The harness

`benchmarks/harness/verity_bench` (the distribution `verity-bench`) is the shared harness. It measures what an arm, a
protocol option or an experimental install, adds to served inference against stock vLLM, through the vLLM
integration's public interfaces. A spec in `benchmarks/harness/specs/` names a preset, its arms and its sweep, and expands to
points, each with an identity. `verity-bench run SPEC` records one run per point without a passing run, `check` passes
only when every point has one, `compare` and `table` accept results only when their identities agree, and
`served-table` and `breakdown` render from the evidence store. It also holds the code behind `verity bench overhead`.
The harness's kit (`harness/verity_bench/kit.py`) holds what runners share, each written once: the GPU lease, the host
stamp, the paired ratio, the result's writer, which records measurements and refuses a verdict of record, and the
report, the one way `verity bench compute`, `verity bench memory` and `verity bench inference_only` print and write
their number. The report prints the text, then one line of JSON whose first keys are `value`, `unit`, `commit` and
`run`. Under a research run it writes that line to `$RESEARCH_RUN_DIR/out/NAME.json` and the measurements to
`$RESEARCH_RUN_DIR/result.json`; outside one, with `OUT` set, to `$OUT/NAME.json` and `$OUT/result.json`; with
neither set, or set empty, it writes nothing. A command's `--out DIR` names the place in place of those.
`benchmarks/harness/README.md` is its contract: how a point is timed and what a result says.

## The core benchmarks' interface

The core benchmarks take and print the same things in the same form. Each rule is a call of the harness's kit
(`harness/verity_bench/kit_report.py`, re-exported by `kit`), so a command follows it by calling it.

1. **What a command takes:** `--from DIR`, a run's record as a directory (the run's directory as `research run` leaves
   it: its `job.json` and `out/`). It is read with nothing of ours: no store, no host, no key and no import of
   `research`. A run id is taken only where `research` is installed, and is fetched to such a directory first, past
   the store's withdrawals: a run withdrawn whole, or in a part that holds a file the command reads, is refused with its
   reason before anything is fetched, and a withdrawal of only other files of it is printed beside the number and is
   the document's `withdrawn` (`kit.report(..., withdrawn=...)`). `kit.record_dir(arg, paths=...)` gives the directory,
   and `kit.record(arg, paths=...)` the directory and the withdrawals (`kit.Record`); each refuses anything else in one
   line naming the fix (`kit.NoRecord`).
2. **One line of coins and parameters,** right after the command's first line: `ran with: coins COINS; parameters
   PARAMETERS (DETAIL)`, and in the document `ran_with: {coins, parameters, detail}`. COINS is `deployed challenger`,
   `local challenger` (its own process on this machine) or `in-process` (the serve holds the challenger); PARAMETERS
   is `deployed` or `development`; DETAIL is the command's own words (its sample, δ, scheme, key). `kit.ran_with(coins,
   parameters, detail, verified=...)` refuses other words. `verified=True` is passed only where the command itself
   checked the challenger's signature on the coins against a pinned key. `deployed challenger` without it is the
   record's own word: the object holds `verified: false`, and the line `coins by the record's own word, not checked` is
   printed under the audit line. A record that predates the field prints `ran with: not stated` (`ran_with=None`), and
   `ran_with` null.
3. **A number only from a run the Lean verifier accepted:** a command's headline, its number, is its text's first line,
   opening with `NAME:` (`compute: 0.75 …`). `kit.report(..., accepted=False, why=WHY)` prints `NAME: no number: WHY`
   in its place: a first line that opens with `NAME:` is dropped, and the rest of the text follows as the command's
   detail, which states no number of the run. It writes `value` null beside `no_number`, writes each measurement with no
   value and its why, and returns exit code 1, which `main()` returns; an accepted run returns 0.
4. **Not an audit:** the document holds `audit: true` only for coins the deployed challenger flipped, verified by the
   command against a pinned key, at `deployed` parameters. Anywhere else the line `a measurement, not an audit` is
   printed under the `ran with` line, and the document holds `audit: false`. A record's own word about its coins is
   stated as such and is never an audit. The kit derives it from `ran_with`; the command does not choose it.
5. **Its README's three parts,** under these headings: `## What you may change`, `## What is fixed` and
   `## How a cheating change is refused`.
6. **Its numbers on the dashboard:** every number the command prints is also a measurement, written by the kit's writer
   (`kit.report`'s `measured`), under the `id` of a metric of its directory's `metrics.toml` (a variant of one
   number, such as a clock reading, is that metric's `member`), and the metric's `goal` is the benchmark's goal. Only
   a run of ours through `research run` reaches the Metrics tab ("Metrics" below); a person outside gets the print and
   `result.json`.

**A made-up record** (a fixture, a test's record, anything no run left) is reported with `kit.report(...,
made_up=True)`: the line `a made-up record, not a run's` is printed under the audit line, the document holds
`made_up: true` and is no audit, and every measurement written from it is marked `modeled` (figures no run measured)
and carries `made_up: true`, so its `result.json` passed on alone never reads as a run's measurement. A fixture record
passes `made_up=True` wherever it is reported, whatever its own files say of its coins, parameters or commit.

`kit.report(name, text, doc, measured, ran_with=..., accepted=..., why=..., made_up=..., withdrawn=...)` is the command's
last step, and the printed and written forms are the kit's (`harness/README.md`'s "The kit").

A command declares that it follows the interface in its own directory's `pyproject.toml`, in the change that makes it
follow:

```toml
[tool.verity.interface]
core = "verity bench NAME"
```

`tests/benchmarks/harness/test_interface.py` reads that key from every `benchmarks/*/pyproject.toml` and holds each
directory that declares it to its README's three headings, its call of `kit.report` with `ran_with` and `accepted`, and
each measurement name its source gives as a metric of its `metrics.toml` with a `goal` (rule 6, for a directory the
test's `DASHBOARD_NOT_YET` doesn't name, each entry saying what it writes that is no metric). The harness's suite lists
the directory and its README among its inputs, which the test checks too.

## Metrics

### Where a metric is declared

Each benchmark or loop declares the numbers a lead is held to in a `metrics.toml` beside the script its command runs
(the first `.py` or `.sh` under `benchmarks/`, `demos/`, `infra/` or `tools/` the command names:
`benchmarks/serve_tax/run.sh`'s in `benchmarks/serve_tax/metrics.toml`, and a loop's, such as
`infra/control/store_index.sh`, beside the loop), one `[[metric]]` table per number. Each metric's `owner` owns its
file: the longest `owns` entry in `.agents/skills/using-slack/registry.json` covering the file names that handle, so a
lead's numbers in another lead's directory take a file entry of their own there. The console's Metrics tab reads every
`metrics.toml` on `main` under `benchmarks/`, `demos/`, `infra/` or `tools/`, and `research metrics show` reads the
tab's report.

### Where a value comes from

A metric's value has one of two sources. A run's value is the measurement its `id` names in the run's `result.json`,
which the benchmark records it under (`serve_tax`'s `Model.metric`) and `research run` preserves to the evidence store;
it reaches the tab only from that preserved `result.json`, through the store's index, up to an hour after the run is
preserved. A loop that samples a number (the control pod's, `infra/control/loop_metrics.py`) sends it as a reading to
the site's `POST /api/metrics` (`research.metrics.send`) under a site key with `panels:write`; an agent's identity
can't send one. A reading names the `command` that computed it and never a run, and the site refuses a reading that
names a run. So every value on the tab is a preserved run's or a sent reading's. Of the readings sent for a metric, the
tab's values are those that name no `period`, `over` or `member`, from the last 62 days: each of the last day, and the
last of each three hours before it. Beside them it lists the newest reading of each line that names a `member`, which
stays listed while it is within that window, whether or not the loop still sends it.

Its owner keeps a metric within its `every`: the control pod's rerun loop (`infra/control/metrics_rerun.py`) launches
the `command` of a metric whose newest value is older than its `every` as a `research run`, within the bounds its
docstring lists, and a metric whose command's script is under `infra/control/` is a loop's, which the loop sends.

### Keys

A metric's keys:

- `id`: `OWNER/NAME`, under its owner's prefix.
- `title`: the number's headline on the tab, three to six plain words with no digit or backtick.
- `owner`: a handle in the registry.
- `unit`: the unit its value is in, at most three words (`hours`, `GB`), a rate's two sides, either side of " a ",
  counted apart (`prover GPUs a serving GPU`); a span of time or compute goes in the measurement's `period` or `over`,
  not the unit.
- `axis`: one sentence, on one line and ending with a period, saying what the chart's y axis is.
- `better`: `lower` or `higher`.
- `command`: the one command that measures it, whose script's directory records it under `id`.
- `every`: how often its owner measures it, in minutes, hours or days (`5m`, `12h`, `7d`); a value older than that is
  stale, so a number that reaches the tab through the index needs an `every` well over the index's hour.
- Optional: `floor`, the number it must not cross; `goal`, the narrowest goal on the Goals page it serves; `tags`, a
  list of words.

The site refuses a metric with any other key: it lists the metric with why and shows none of it, so a new key reaches
the site's reader before a metric uses it.

### Marks, periods and members

Each value carries its mark, how it was had, which the tab shows beside it: the measurement's `mark`, which the kit's
writer requires (`measured`, `modeled` or `partial`, `kit.MARKS`; a value of None takes the one it would have had)
and also writes as the first of its `inputs`, where the tab reads it until it reads `mark`. A sent reading carries
`mark` itself, which the site requires.

A value over a span of time or of compute says which, each as its own field of the measurement, held only where given:
`period`, one of `phase`, `week`, `month` or `year` (`kit.PERIODS`), and `over`, `world` or `datacenter` (`kit.OVER`). A
run's result holds one measurement a name, period and `over`, and the writer refuses a second, so one id may give a
value for each period and scope it is computed at, each in its own unit (`verity bench exfiltration`'s three, over a
week, a month and a year of a datacenter's and of the world's compute). A value for one member of a set (a deployed
guarantee, a lead) names it as `member`, a non-empty string held only where given, and the writer keys on it too, so one
id gives a value for each member (`lean/trust-gap`'s, one a deployed guarantee). Of a run's measurements under one id,
the tab's value is the one written last that names no member, or, where the run wrote none without a member, the one
written last: one point a run. Beside it the tab lists the measurements that name a member from the newest run that
wrote any. So a benchmark that writes one id for several periods, scopes or members writes the metric's own last: the
one its `title`, `unit` and `axis` are for (`exfiltration-gb-world-year`'s, the `year` over the `world`).

### Where the tab lists a number

The tab lists each number under every top goal its `goal` is or sits under, a top goal being a goal on the Goals page
that nothing waits on, other than `top/parked`. A number whose goal sits under `top/parked` and under no top goal reads
"parked".

### What the test refuses

`tests/test_metrics_registry.py` refuses a metric that lacks a key the list above doesn't mark optional or holds one it
doesn't list, an id declared twice across all the files, an unknown owner, an owner who doesn't own the metric's file,
an id outside its owner's prefix, a `title`, `unit` or `axis` that breaks its rule above, a `better` other than `lower`
or `higher`, an `every` that isn't a count of minutes, hours or days, a command whose script is missing or whose
directory never names the id, a metric whose command's first script is outside the directory of the file declaring it,
and a shared `metrics.toml` directly under `benchmarks/`.
