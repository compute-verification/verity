# Benchmarks

`benchmarks/` holds measurement only: each directory measures something, or holds the data measurements are computed
against. A benchmark is never the verifier, writes no verdict and computes no number a guarantee reads, and
nothing outside `benchmarks/` reads it. An application is one configuration file in `applications/`, which `verity run`
reads. This file is the directory's interface: its entry points, the shared harness, what a benchmark may import
and write, and the index of every directory.

## Entry points

`verity bench NAME [ARGS]` runs the directory NAME's own runner, its `run.py` or else its `run.sh`, and hands it ARGS (a
`-` in NAME is a `_`). A runner that needs a GPU or a whole node is the command of a `research run`, as its header
says. In a fresh clone, `uv sync --all-packages --extra torch-cpu` makes the environment every runner here runs in,
and `uv run verity bench NAME [ARGS]` runs one. What each measures is its row in the index below, and its runner's
header gives its machine, its arguments and what it writes. These are the directories that have one, with a runner's
subcommands or defaults in parentheses: `attester_load`, `bool_build_cost`, `certifier_term`, `codec_tensor` (the
tensor-core wide-word codec on one GPU), `compile_cost`, `compute`, `dsa_topk`, `exfiltration`, `flock_s8_block`, `inference_only`, `landing`, `lean_unread_gate`
(`--against REV`, its guard), `memory`, `mempool` (`--quick`), `network_traces`, `numerical` (one of
`verity_numerical`'s tools by name; `numerical judge` compares two results under the proof campaign's contract), `overhead` (`points`, `run`,
`table`), `pouw NAME` (`pouw/NAME/`'s runner; given no NAME, the names), `ppl` (`eval`, `submit`, `report`,
`baseline`), `prod_repro`, `proof_throughput` (`record`: one prover and two at once on one node's H200s), `prove`
(`--system flock|spot`; `lean-verify`, the Lean verifier timed on one kept proof; `layers`, the proof benchmark by
layer from its four runs, as `prove/README.md` says), `prove_path`, `poub_session`, `recursion_e2e` (`plan`, `run`,
`result`, `table`, `store`), `replay_ladder`, `route_capture`, `scan_cost`, `serve_tax`, `serving_cost` (`minify RUN`,
`exact`, `capture RUN`, `price`, `confirm`: "The serving cost in five stages"), `served_commit`, `served_errors`,
`served_units` (`fp8-coord:K=6144`), `unit_block`, `unit_lowering`, `unit_walk`, `verifier_cost` and `word_codec`.

`verity calibrate NAME [ARGS]` runs `calibration/NAME/`'s runner the same way. These measurements live there, not
here: `attester_timing` (`run`, `guard`, `serve`, `observe`, `score`, `table`, `target`), `bandwidth` (the device
ceilings PoUB's timing cites, one probe a run: `chip` and `write_gpu` on one GPU, `write_host` on the host's CPUs),
`circuit_properties` (`--models llama32-1b --out DIR`, Llama 3.2 1B's layer alone and its point, on a CPU),
`f32_exhaustive`, `native_peak` (bf16 at 8192^3), `rate_squaring` (`sq` on one GPU, `ifma` on the host's CPUs),
`tc_probe` (`sm120.mma.m16n8k32.e4m3`'s canary), `tc_probe_fp4` (`nvf4` with its SASS check), `tensor_duty` and
`training` (`CONFIG`, the training slowdown; `ENTRY --library --root DIR`, one library entry's step traced on a CPU).

A directory without a runner is run by the commands its README names, or its main module's docstring where it has no
README. `benchmarks/kernels/`'s directories have no `verity bench` name: each runs as its script's header says. `verity-bench` is the
harness's own command.

## The harness

`benchmarks/harness/verity_bench` (the distribution `verity-bench`) is the shared harness. It measures what an arm, a
protocol option or an experimental install, adds to served inference against stock vLLM, through the vLLM
integration's public interfaces. A spec in `benchmarks/harness/specs/` names a preset, its arms and its sweep, and expands to
points, each with an identity; its README's "Commands" lists what `verity-bench` does with them, and it also holds the
code behind `verity bench overhead`.
The harness's kit (`harness/verity_bench/kit.py`, whose docstring lists it) holds what runners share, each written
once, among them the result's writer, which records measurements and refuses a verdict of record, and `kit.report`,
the one way `verity bench compute`, `verity bench memory` and `verity bench inference_only` print and write their
number; its docstring says where it writes.
`benchmarks/harness/README.md` is its contract: how a point is timed and what a result says.

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

Some files here still break these rules (a kernel or a store the integration loads, a probe whose rate a card file
holds); none is a pattern for a new benchmark.

## The core benchmarks' interface

The core benchmarks take and print the same things in the same form. Each rule is a call of the harness's kit
(`harness/verity_bench/kit_report.py`, re-exported by `kit`), so a command follows it by calling it.

1. **What a command takes:** `--from DIR`, a run's record as a directory (the run's directory as `research run` leaves
   it: its `job.json` and `out/`). It is read with nothing of ours: no store, no host, no key and no import of
   `research`. A run id is taken only where `research` is installed, and is fetched to such a directory first, past
   the store's withdrawals: a run withdrawn whole, or in a part that holds a file the command reads, is refused with its
   reason before anything is fetched, and a withdrawal of only other files of it is printed beside the number and is
   the document's `withdrawn` (`kit.report(..., withdrawn=...)`), as `kit.record_dir` and `kit.record` give them.
2. **One line of coins and parameters,** right after the command's first line: `ran with: coins COINS; parameters
   PARAMETERS (DETAIL)`, and in the document `ran_with: {coins, parameters, detail}`. COINS is `deployed challenger`,
   `local challenger` (its own process on this machine) or `in-process` (the serve holds the challenger); PARAMETERS
   is `deployed` or `development`; DETAIL is the command's own words (its sample, δ, scheme, key). `kit.ran_with(coins,
   parameters, detail, verified=...)` refuses other words. `verified=True` is passed only where the command itself
   checked the challenger's signature on the coins against a pinned key. `deployed challenger` without it is the
   record's own word: the object holds `verified: false`, and the line `coins by the record's own word, not checked` is
   printed under the audit line. A record that predates the field prints `ran with: not stated` (`ran_with=None`), and
   `ran_with` null.
3. **A number only from a run the Lean verifier accepted:** a command's number is its text's first line,
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
   `result.json`. A number is measured once, by the run that measured it: a command that reprints another run's record
   passes that run's id (`kit.report(..., from_run=RUN)`), and the kit then prints and writes the document but writes no
   measurement. Under a research run each measurement also holds the `command` the run was launched as, from its
   `job.json`.

**A made-up record** (a fixture, a test's record, anything no run left) is reported with `kit.report(...,
made_up=True)`: the line `a made-up record, not a run's` is printed under the audit line, the document holds
`made_up: true` and is no audit, and every measurement written from it is marked `modeled` (figures no run measured)
and carries `made_up: true`, so its `result.json` passed on alone never reads as a run's measurement. A fixture record
passes `made_up=True` wherever it is reported, whatever its own files say of its coins, parameters or commit.

**A trial** is a run whose numbers are for its launcher alone: a probe from a branch, a study at other settings, a run
on a stand-in model, a comparison run. Start one with `research run --env VERITY_BENCH_TRIAL=1 ...` in front of the
registered command (the overhead, vLLM and steady doors forward the variable from the launcher's environment). Its
writer puts every measurement in `out/trial-measurements.json` and none in `result.json`, which says `trial: true`, so no
number of it reaches a metric's line. It is never a plan's passing run, and every record step refuses a labelled one
(`kit.trial_why`), naming the launch that makes it of record.

`kit.report(name, text, doc, measured, ran_with=..., accepted=..., why=..., made_up=..., withdrawn=..., from_run=...)` is the command's
last step, and the printed and written forms are the kit's (`harness/README.md`'s "The kit").

A command declares that it follows the interface in its own directory's `pyproject.toml`, in the change that makes it
follow:

```toml
[tool.verity.interface]
core = "verity bench NAME"
```

`harness/tests/test_interface.py` reads that key from every `benchmarks/*/pyproject.toml` and holds each directory that
declares it to its README's three headings, its call of `kit.report` with `ran_with` and `accepted`, and each
measurement name its source gives as a metric of its `metrics.toml` with a `goal` (rule 6, for a directory the test's
`DASHBOARD_NOT_YET` doesn't name, each entry saying what it writes that is no metric). The harness's suite lists the
directory and its README among its inputs, which the test checks too.

## Metrics

Each benchmark or loop declares the numbers a lead is held to in a `metrics.toml` beside the code that records them
(`benchmarks/serve_tax/run.sh`'s in `benchmarks/serve_tax/metrics.toml`, and a loop's, such as
`infra/control/store_index.sh`, beside the loop), one `[[metric]]` table per number. Each metric's `owner` owns its
file: the longest `owns` entry in `.agents/skills/using-slack/registry.json` covering the file names that handle, so a
lead's numbers in another lead's directory take a file entry of their own there. The console's Metrics tab reads every
`metrics.toml` on `main` under `benchmarks/`, `calibration/`, `applications/`, `infra/` or `tools/`, and `research
metrics show` reads the tab's report.

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
docstring lists, and a metric whose command's script is under `infra/control/` is a loop's, which the loop sends. A
metric's keys:

- `id`: `OWNER/NAME`, under its owner's prefix.
- `title`: the number's headline on the tab, three to six plain words with no digit or backtick.
- `owner`: a handle in the registry.
- `unit`: the unit its value is in, at most three words (`hours`, `GB`), a rate's two sides, either side of " a ",
  counted apart (`prover GPUs a serving GPU`); a span of time or compute goes in the measurement's `period` or `over`,
  not the unit.
- `axis`: one sentence, on one line and ending with a period, saying what the chart's y axis is.
- `better`: `lower` or `higher`.
- `command`: the one command that measures it, naming a script under `benchmarks/`, `calibration/`, `applications/`, `infra/` or
  `tools/`.
- `every`: how often its owner measures it, in minutes, hours or days (`5m`, `12h`, `7d`); a value older than that is
  stale, so a number that reaches the tab through the index needs an `every` well over the index's hour.
- Optional: `floor`, the number it must not cross; `goal`, the narrowest goal on the Goals page it serves; `tags`, a
  list of words; `version`, `changed`, `was`, `under`, `kind` and `needs` (below).

The site refuses a metric with any other key: it lists the metric with why and shows none of it, so a new key reaches
the site's reader before a metric uses it.

A metric has a version, a whole number on its row, 1 where the row gives none, written `ID@vN`. It versions what the
number means, not the code: it rises, in the change that makes it so, when what is divided by what, what the run
includes or which workload it is over changes, and then `changed` says in one sentence what changed from the version
before. A refactor, a new run, a new model or a new card does not raise it: a model or a card is a `member`. The kit's
writer stamps it on every measurement, 1 unless `kit_report.VERSIONS` names the id, which holds exactly the rows above
1, and no caller gives it. A line is an id, a version and a member: readings of an older version leave the line by
themselves and stay in the store under their version, so a changed meaning takes a raised version, not labels;
`superseded_by` stays for a run that was wrong. The test holds the unit alone: a unit changes only with a raised
version. Whether anything else, the axis among it, changed what the number means is its author's call. `was` lists a
row's former ids, so an id is renamed without breaking its line: the registry's reader (`research.metrics.registry`)
resolves each to the row.

A row `under = "OWNER/ID"` is a breakdown of that main number, one level deep; `research metrics show` indents it. `kind`
is `system` (it measures Verity; the default) or `platform` (the project). `needs` lists `parameter` where the value
depends on a security parameter and `calibration` where it uses a calibration number: each measurement then names
them, `parameters` as `{NAME: {value, from, basis}}`, `from` the application it is read from, and `calibration` as
`[{path, key}]`, `path` under `calibration/`. The kit refuses one naming none, by `kit_report.NEEDS`, held to the rows.

The site's reader (infra's) takes `version`, `changed` and `was` on a row and `version` on a measurement
(`SITE_TAKES_VERSIONS` in the registry's test), and `under`, `kind` and `needs` on a row (`SITE_TAKES_STRUCTURE`). One
part is not true yet: a reading sent from a loop through `research.metrics` carries no version, since the sender
(infra's) doesn't take one.

Each value carries its mark, how it was had, which the tab shows beside it: the measurement's `mark`, which the kit's
writer requires (`measured`, `modeled` or `partial`, `kit.MARKS`; a value of None takes the one it would have had)
and also writes as the first of its `inputs`, where the tab reads it until it reads `mark`. A sent reading carries
`mark` itself, which the site requires.

A value over a span of time or of compute names its `period` and `over`, and a value for one of a metric's lines its
`member`, each as `harness/verity_bench/kit.py`'s docstring defines it. The writer keys on all three, so one id gives a
value for each period, scope and member, each in its own unit (`verity bench exfiltration`'s three, over a week, a
month and a year of a datacenter's and of the world's compute; `lean/trust-gap`'s members, one a deployed guarantee).
Of a run's measurements under one id,
the tab's value is the one written last that names no member, or, where the run wrote none without a member, the one
written last: one point a run. Beside it the tab lists the measurements that name a member from the newest run that
wrote any. So a benchmark that writes one id for several periods, scopes or members writes the metric's own last: the
one its `title`, `unit` and `axis` are for (`exfiltration-total-gb`'s, the `year` over the `world`).

The tab lists each number under every top goal its `goal` is or sits under, a top goal being a goal on the Goals page
that nothing waits on, other than `top/parked`. A number whose goal sits under `top/parked` and under no top goal reads
"parked".

`tests/test_metrics_registry.py` refuses a metric that lacks a key the list above doesn't mark optional or holds one it
doesn't list, an id declared twice across all the files, an unknown owner, an owner who doesn't own the metric's file,
an id outside its owner's prefix, a `title`, `unit` or `axis` that breaks its rule above, a `better` other than `lower`
or `higher`, an `every` that isn't a count of minutes, hours or days, a command whose script is missing, a metric
the directory of the file declaring it never names, and a shared `metrics.toml` directly under `benchmarks/`. It also
refuses a `version` that isn't a whole number 1 or above, a `changed` missing above 1, given at 1 or more than one
sentence, a `was` that repeats an id or names a row's id, a `kit_report.VERSIONS` that isn't the rows above version 1,
a unit changed at its pinned version, and a row whose version and unit aren't pinned in `tests/metric_pins.json`; after
a metric is added, removed or raised, `uv run python tests/test_metrics_registry.py --pin` refreshes the pins, and
refuses a unit changed at its version.

## The serving cost in five stages

The serving cost of a model too large to optimize on (`serve_tax/README.md`'s "The serving cost") is had in five
stages, each `verity bench serving_cost STAGE` (`serving_cost/run.py`) run under `research run --store-read`. A stage
reads the record of the stage before it from the evidence store by the id of the run that stored it, never by a path,
and refuses a run whose result is not that record's kind or did not pass, naming what makes the missing record and
stating the record's contract (`serving_cost/run.py`'s `MAKERS`, `HOLDS` and `FINGERPRINT`). The records it reads
are stored by the runs that make them, and it launches none. `minify RUN` and `capture RUN` check that a run is a
passed record of their kind and print what it holds. `exact` checks the replay and the circuit-check report and stores
the verdict. Price and confirm compute their figures.

| Stage | What it runs | What it stores |
| --- | --- | --- |
| 1. Minify | a small copy of the model from random weights on one card, keeping the selectors that choose each code path (`serve_tax`'s `glm53-standin`) | the copy's identity, the selectors it keeps and its size class; and for each part and each code path the part can take, a unit cost in seconds a call or a step, `measured`, with the count of calls behind it |
| 2. Exact | `circuit-check` on the full model's bindings, and a replay on the copy | a verdict, not a number: `circuit-check`'s on the bindings, and the replay's on the copy (its tokens equal to the served tokens, its steps reproduced), saying whether the replay was the Python replay or a proof the Lean verifier accepted, so that exact is not read as proved |
| 3. Capture | a short run of the real model | for each part and code path its counts (calls a step, steps a request), the baseline step time, and the value-dependent inputs a price needs |
| 4. Price | minify's unit costs times capture's counts, over capture's baseline step time | the priced serving cost under `benchmarks/serving-cost-priced` (`serving_cost/metrics.toml`), never under the measured line, its `member` model, card, then arm (`glm-5.3-h200-prod-record`), `partial` while any (part, code path) is unpriced and `modeled` otherwise, with the capture, minify and units runs among its `inputs`, every (part, code path) it could not price, and the runs each priced one's unit costs came from |
| 5. Confirm | a stored run of the real model at scale (`--measured RUN`), and price's | the measured minus the priced as a measurement of its own, in percentage points, with both runs among its `inputs`: the measured figure under `vllm/prod-serving-cost-glm53` of the price's member over its stock unless `--member` names another; the measured serving cost stays the measuring run's |

Stage 1 is to write no serving-cost percentage and no point under the real model's metric: a `member` tells apart
configurations that read the same quantity, and a copy from random weights does not read the real model's serving
cost. Stage 4 names the runs it read in its `inputs` and not as `from_run`, since its number is its own and
`from_run` writes no measurement.

Exact holds a replay to the minify run its fingerprint names under `minify`, and where it names none records
`held_to_minify` false and says so. A replay whose fingerprint's `replay` says `lean` is a proof the Lean verifier
accepted, and any other the Python replay. Exact's result passes only where the replay passed and circuit-check's
report passed and holds every binding named. Price merges the minify run's unit costs with each units run's (`--units RUN`,
repeatable), refusing where two runs give one shape of one pass and path different costs, or where together they time
no rows × keys grid. A capture that names an exact run is refused unless that run is a passed exact record.

Two cases decide where a price comes from:

- **A cost that depends on the load,** such as the attested tick's holding a request back until its admission step,
  is stored as a count from stage 3, the steps an admission was held under the captured load, times the step time,
  and never as a unit cost from the copy.
- **A code path the copy never takes** is listed as unpriced, and the price is `partial`, not `modeled`, until a
  unit cost for that path comes from another source: the top-k at prefill steps is unpriced while the capture has
  prefill steps and no source's `shapes` is `prefill`.

## On VCF's Slurm cluster

The Verifiable Compute Foundation's cluster (`vcf.lucidcomputing.ai`: the 8-H100 node `h100-sxm-8-0` and the one-card
nodes `h100-nvl-1-0-cc` and `h100-nvl-1-1-cc`) runs a benchmark from a commit as one Slurm job under the account its credentials name.
Only the control pod holds the login key, so a lead sends the job from there with `infra/vcf/` (deployed under
`/data/vcf/`). From a checkout, build the payload, `git archive` of the commit and the command, and copy it to the pod:

~~~text
infra/vcf/payload.sh origin/main --tool native_peak -- python calibration/native_peak/measure.py --dtype bf16 --sizes 8192 \
  | research pods ssh vy-control-verity -- 'cat > /data/vcf/payloads/NAME.tar.gz'
research pods ssh vy-control-verity -- 'cd /data/vcf-jobs && SEND=/data/vcf/payloads/NAME.tar.gz setsid -f bash \
  /data/vcf/vcf-job-run.sh NAME main 30 /data/vcf/vcf-bench.sh --nodelist=h100-sxm-8-0 --gres=gpu:1 --cpus-per-task=16 \
  --mem=64G > NAME.launch.log 2>&1 < /dev/null'
~~~

The scripts' headers say what the job does (`vcf-bench.sh`), where `vcf-job-run.sh` copies the run directory and
how `vcf-publish.sh` publishes it; `NAME.launch.log` gives the Slurm job. Read it back as any run: `research data
show RUN`.

Such a record names its machine `vcf-NODE` (`execution.machine`) and its cost provider `vcf`. It has no lease and
no slot of ours (its memory limit is Slurm's `--mem`), its node is shared with other users, it has no rate, and
nothing preserves it while it runs (`research run-preserve` and `research status` from a VM don't reach it). The nodes
have network, Python 3.12 and no root. The DGX's driver is CUDA 13.0's, which `main`'s torch needs;
`h100-nvl-1-1-cc`'s is CUDA 12.8's, too old for it, and `h100-nvl-1-0-cc` refuses any GPU request. No node has `nvcc` or `ncu` on
its PATH. The H100 has a card file (`calibration/hardware/cards/h100-sxm-80gb.toml`), but a card-bound benchmark of the
harness still can't start there: its served run launches only on our machines, which an open change is curing. Use only idle
capacity: one job at a time, at most 30 minutes, one GPU and 16 CPUs. A job that pends on `Resources` while the
DGX's GPUs are idle may find them held by a reservation of the account (`scontrol show reservation`), which
`--reservation=NAME` runs it in.

## Index

Every directory has one row, in the group of what it measures; `tests/test_benchmarks_index.py` holds the rows to the
tree. A directory's status is one of:

- **maintained**: developed against the current tree, and its commands run from it;
- **frozen**: kept so that what it measured can be measured again, and changed only to keep it running;
- **data**: no measuring code, only numbers that other code takes as parameters.

### Protocol overhead

What the protocols cost: measured on served inference, and bounded on training.

| Directory | What it measures | Status |
| --- | --- | --- |
| `harness/` | `verity-bench`: its specs and sweeps (`specs/`, one TOML file each), recorded runs and results for an arm against stock vLLM, the served table of each protocol's overhead, and the kit runners share (`verity_bench/kit.py`) | maintained |
| `overhead/` | `verity bench overhead`, the one front door for overhead: a card's operating points C and L, a scheme's γ over stock at them in busy SM-time, and the table of γ from labelled runs of record | maintained |
| `served_commit/` | what the Commit, every value of every step committed, adds to served vLLM: throughput, time to first token and per-token latency | maintained |
| `application_runs/` | one application run's verdicts that accept, of how many, and its minutes, written by the run that measures (`run.py`, which runs `verity run`) | maintained |
| `demo_minutes/` | the onsite application's minutes from its first command to its verdict, read off its two `verity run` runs' records (`onsite_minutes.py`, on `run_minutes.py`) | maintained |
| `dsa_topk/` | the DSA indexer's top-k at the integration's vLLM pin made a function of its logits by a vLLM plugin: outputs run to run, its set against the registered relation's, and its cost against the pin's kernels | maintained |
| `replay_ladder/` | whether replaying a saturated window under its recorded schedule gives the served tokens across a ladder of dense and MoE models, run after run, against each request served alone; and the per-model status table | maintained |
| `serve_tax/` | what each change to serving (step trace, release order, admission on the network attester's tick, committing tokens and the KV cache) costs vLLM's own server against stock compiled vLLM at three loads, the serving cost (one number with a target: the served engine's best goodput against stock's, by vLLM's own benchmark client), and replaying a recorded window under its committed schedule | maintained |
| `prod_repro/` | one command over the serving cost benchmark's serve, load and replay: what the serve turned on, whether the replayed window's tokens are the served ones, what our changes cost over stock with its interval, and the advice a request | maintained |
| `serving_cost/` | `verity bench serving_cost`, the serving cost of a model too large to optimize on in five stages, each refusing without the stored record of the stage before it: the priced figure from a copy's unit costs and a capture's counts (`benchmarks/serving-cost-priced`), and the measured less the priced | maintained |
| `vllm_encoded/` | Llama-3.1-8B in vLLM with its weights and KV cache held as codec words and receipted by PoUB, as `verity-bench` arms beside the plain stack | maintained |
| `inference_only/` | the inference-only policy of Llama 3.2 1B as served, stated, and of the class this tree builds, with the Lean count's verdict on it, and each one's number: the least slowdown of training run inside the policy's inference blocks, under the min-cut bound | maintained |
| `slowdown/` | a lower bound on the slowdown a training run pays under the regime, and the time to its target, per scenario and monitoring period | maintained |
| `nci/` | the sources, parameters and policies the inference-only policy's bounds rest on, and a board of training methods with their slowdown brackets | maintained |
| `../calibration/training/` | the grader (`grader/`): a training step's matrix products, their multiply-adds and output wires, and the circuit's size, read off the Boolean circuit a lane hands over, beside training's slowdown code's trace of the same step | maintained |

### Kernels

GPU code held bit for bit to its reference or Definition before it is timed.

| Directory | What it measures | Status |
| --- | --- | --- |
| `hash/` | GPU kernels for PoUW's Dither prover (`dither/`: the no-fold and J10-R units, their gates and harnesses) and the served path's SHA-512 commitment library (`lib/`) | maintained |
| `kernels/` | the served path's exact kernels and what a kernel build ships with: the exact W8A8 linear (`s8_linear`), vLLM's int8-exact linear kernels' gates and benches (`exactlin`), the ship frame every build here sources (`ship.sh`, `ship_clock.c`, `ship_nvcc.sh`, `ship_probe.sh`), the nvcc a job's own library compiles with (`job_nvcc.sh`), and the Marlin MoE path's `moe_align_block_size` made a function of its inputs by the integration's `verity_vllm.engine.moe_align`, checked run to run and against the pin's op and timed against it (`moe_align`) | maintained |
| `pouw/` | Pearl-C's kernels on the RTX PRO 6000, their harness, and the served vLLM path and its audits; `contents.py` lists the rest | maintained |
| `../calibration/tc_probe/` | tensor-core instructions bit for bit against their silicon models, and the trust each instruction's evidence earns | maintained |
| `../calibration/tc_probe_fp4/` | the FP4 path on Blackwell: its conversions bit for bit, and each sm_120a multiplier priced against dense NVFP4 | maintained |
| `../calibration/native_peak/` | a card's measured dense-GEMM throughput for one datatype (`native-peak/v1`) | maintained |
| `../calibration/tensor_duty/` | the falsification test of the H200 card's tensor clock: dense tensor MACs a second over the host's wall clock by tensor duty cycle, operand class and fill, at maximum power, against 132 x 4096 x `clock.tensor` (timings only) | maintained |

### Proving and recursion

| Directory | What it measures | Status |
| --- | --- | --- |
| `one_stage/` | the one-stage audit end to end: committed values, registration, the sample, Flock's circuit prover and the Lean verifier, to the audit record and its integrity profile, with its negatives; and the two-stage audits, of a served window and of a toy replay block (its README's "The two-stage audits") | maintained |
| `prove_path/` | a served run's proof blocks proved after the fact: the compiler's Programs, a sample on live coins, Flock `--zk` behind the Lean ZK server, and the Lean verifier | maintained |
| `served_units/` | a served model's proof blocks proved one type at a time from synthetic values, the Lean verifier accepting the honest run and refusing forgeries | maintained |
| `unit_walk/` | the device witness's host walk of a served proof-block type's unit: CPU and wall seconds a statement at each width (lane groups a pass), distance asked ahead and pool size, with and without its row vectors kept, every walk's packed words held to the first's | maintained |
| `spot_proved/` | a replayed served step's values proved: one replayed step of a recorded replay block captured per TP rank under CUDA graphs, staged as each sampled type's proof blocks, and audited one stage end to end by the Lean verifier | maintained |
| `flock_s8_block/` | Flock `--zk`'s price for a proof block of adjacent cells of one int8 GEMM output row, on one GPU node and on the CPU | maintained |
| `unit_block/` | a word primitive's Boolean version built as one Flock proof block: the select's slot alone and with its rows' openings, against the 2^27 block | maintained |
| `prove/` | `verity bench prove`, the one command for a proof system's prover on a workload: per block type the blocks proved per token and the ms per block, and the prover GPUs per serving GPU, from the store's labelled records | maintained |
| `proof_throughput/` | the proof blocks a second Flock proves and the Lean verifier accepts on GLM-5.3's dot products, over the batch size and the number of provers at once | maintained |
| `commitments/` | the commitment schemes' cost: hashing leaves and building the tree, hiding leaves per row, and serving's per-step row tables | maintained |
| `../calibration/sampling_plan/` | the sampling plan for one served workload: a cost model fitted to proving measurements, and ε at δ over the verification overhead (a model, not a measurement) | maintained |
| `spot_figures/` | the SPOT post's Figures 1-3 for a model from its cards (`spot-card/v1`) and SPOT's planner: what an adversary on all of a scope's compute can exfiltrate, infiltrate or skip against total cost and against proof overhead, and the serve-time commitment's cost (a model, not a measurement) | maintained |
| `exfiltration/` | `verity bench exfiltration`: GLM-5.3's exfiltration capacity in GB over a monitoring period at 10% of total compute, the compute audit's part beside the network attester's (`certifier_term/`'s rows), every input labelled with its mark | maintained |
| `verifier_cost/` | a verifier's time against the size of the circuit description it reads and of the circuit that description unrolls to | maintained |
| `recursion_e2e/` | recursive witness zero knowledge, timed step by step: the inner proof, V* proved with `--zk`, and the Lean verifier | maintained |
| `recursion_figures/` | a recursion bench result's session scaled over the census's fleets and the monitoring periods: its cost, knowledge-soundness error and stop-leak bound per window and pooled | maintained |
| `remote_error/` | SPOT's cryptographic error in the remote application: the remote guarantee's cryptographic terms of SPOT at the application's parameters against a prover's SHA-512 calls, per window and pooled over the census's fleets and the monitoring periods (its README says what is proved and what is open) | maintained |

### Checks

| Directory | What it measures | Status |
| --- | --- | --- |
| `lean_records/` | a records run of `security/Proofs` under a base version of `tools/verity/lean/` and this tree's, and at a lower reservation: the build step's seconds, the tools' own CPU, peak memory and `lean` jobs at once, each arm on a lemma of its own | maintained |

### Circuits

| Directory | What it measures | Status |
| --- | --- | --- |
| `bool_build_cost/` | a workload's Boolean Build against its word Build: the descriptor's size and each Definition's proof-block rule | maintained |
| `compile_cost/` | what compiling a class-G window's Programs costs (`predict.steps.compile_steps`, GLM-5.3 at TP8 on one rank): wall time and peak RSS, by phase, its digests held to their pins | maintained |
| `../calibration/circuit_properties/` | structural properties of a Boolean circuit's counted operations, measured exactly: fan-out, fan-in, their count and s_× | maintained |
| `route_capture/` | what a model executes when served: every engine step's launch-shape class under CUDA-graph serving and the kernels of each class's first steps, and the (route, statics) pairs those steps need that class G's registered Definitions don't cover, with their owners | maintained |
| `unit_lowering/` | writing a PoUW H1 circuit file (a leaf's lane and combine, or the whole tile unit, at any tile R × C): peak memory and wall time by K, the file's AND rows, bytes and block (k_log), its SHA-512 held to a reference, and where the memory goes | maintained |

### Compute

| Directory | What it measures | Status |
| --- | --- | --- |
| `compute/` | `verity bench compute`, the front door for the compute audit's number: the share of a served model's MACs that Pearl+ credits over the tokens a served run served, from the verifier's side of a recorded run's record; and `h200_share.py`, the share of an H200's capacity an honest Pearl+ prover is credited, modeled from that share and a recorded tensor-rate run | maintained |
| `served_errors/` | error terms of the quoted guarantees that had no value, evaluated at the served sizes (PoUW's, Pearl+'s, Dither's, the integrity subprotocol's and the remote product's), each with its formula and premises (a model, not a measurement) | maintained |
| `../calibration/machine_model/` | the ACCEL model of each measured card: its rows, census, attack families and designs as data, and a design's ε from them (`machine-model`) | maintained |

### Network

| Directory | What it measures | Status |
| --- | --- | --- |
| `network_traces/` | the network attester's runs in front of a served endpoint, and their zero-knowledge proofs | maintained |
| `network_e2e/` | the network chain's steps after N2 (`chain.py`) held on fakes and recorded endpoint runs, the advice encodings (`encodings/`) and vLLM's scheduler traces (`traces/`) | maintained |
| `certifier_term/` | the network attester's term for GLM-5.3 at the load of a node, a datacenter and the world, in bits and GB, each row by its check's own counting rule: the scan as the monitoring-period check charges it, the scan under each change to its check, and the design with no run behind it, without its stop block and with it (`network_traces/leak_bounds.py`'s runner) | maintained |
| `network_sim/` | the workload's Programs (`workload/`): a Boolean MLP and vLLM's serving Program, which `verifier_cost` measures a verifier on | maintained |
| `../calibration/attester_timing/` | the bits an interval a server moves through when the network attester releases its bytes: one number from a server that times its writes by a bit, with a guard, a target and the candidate fixes as flags | maintained |
| `scan_cost/` | what the network attester's scan costs for each second of execution phase it covers: proving seconds (the round's sessions and their staging) and the Lean verifier's CPU seconds on one core, at a fixed identity, each with its stages, a guard on the round's answer, a baseline and two targets | maintained |
| `attester_load/` | what the network attester costs the requests it carries, at the application's schedule: the added time to a response's first and last byte through it against straight to the endpoint, the ticks' lateness at the capacity its schedule states, on a stub endpoint on a CPU and on a model on a card. The model arm's network attester runs in the bytes form, relaying vLLM's own framing, where the application's runs in the id form: the schedule and its cells' buckets are the same, and only what a cell may carry differs | maintained |
| `mempool/` | the network attester's prefix-cache pool as memory-checked RAM (`pool_ram`, held to vLLM's pool on the encodings' traces; `stream.py` holds `CacheStep_v3`'s stream to it after every call): a window's cost at its busiest stretch, and `MemCheck_v1`'s and `CacheStep_v3`'s vectors | maintained |
| `datacenter/` | a datacenter as Verity accounts for it, in a small Python API: the Program, hardware, binding, plan, run, claims and account | maintained |

### Memory

| Directory | What it measures | Status |
| --- | --- | --- |
| `pous/` | every PoUS implementation on one dataset, by its effective weight bandwidth | maintained |
| `../calibration/bandwidth/` | the device ceilings PoUB's timing cites: how fast a card and a host write labels, and the chip's on-chip stores; and a node's own report of its interfaces' line rates, its disks' interfaces and its cards' usable memory | maintained |
| `poub_session/` | PoUB's session harness: a verifier's and a prover's process over TCP, end to end. The remote protocol's application is `applications/remote.toml`, not this | maintained |
| `memory/` | `verity bench memory`, the front door for the memory audit's number: the share of a node's memory read rate one accepted PoUB session leaves uncertified, at the data sheet's read bandwidth, from a gate run on the whole node, a recorded run or a loopback | maintained |
| `kv_filler/` | the filler top-up beside vLLM serving, and what it costs inference | maintained |
| `word_codec/` | the store's Sloth codecs (the word codec, the KV codec that PoUB's `codec.py` mirrors, the wide codecs): each a byte-exact reference, a CUDA kernel and its benchmark | maintained |
| `codec_tensor/` | PoUB's receipts, leaves and receipt tree on the GPU, and the tensor-core wide-word codec | maintained |
| `property_redteam/` | attacks on the stacking property and on the gate model's counts behind the encode bound | maintained |
| `../calibration/rate_squaring/` | squaring rates modulo the codec's primes on the GPU and on the host's AVX-512 IFMA, the evidence for PoUB's stated squaring ceiling | maintained |

### Numerics

| Directory | What it measures | Status |
| --- | --- | --- |
| `ppl/` | the perplexity an intervention on a served model's matmuls or attention costs, against bf16 and plain int8; on one H200 (`h200.sh`), Llama 3.2 1B's as shipped and with its linears in Pearl-C's FP8 E4M3 form, as metrics | maintained |
| `../calibration/f32_exhaustive/` | a correctly rounded reference primitive (`F32LogRn_v1`, `F32ExpRn_v1`) against MPFR, or a Boolean version (`F32Sqrt_v2`) against its word, on all 2^32 binary32 words | maintained |
| `definitions/` | the GPU probes behind the numerics Definitions: the approximate instructions FlashAttention-2 and the RMSNorm kernel run (`MUFU` exp2, reciprocal, square root and tanh), measured on every 32-bit input | maintained |
| `numerical/` | the bench spine (`verity_numerical.bench`) that Flock and `circuit-check` import, and the per-unit-type cost inventory; the archived proof systems' tables are in the evidence store | frozen |

### Landing

How long the team's own changes take to reach `main`: measured on the verified queue's records, not on a protocol.

| Directory | What it measures | Status |
| --- | --- | --- |
| `landing/` | mark to `main` over the last 7 days, split as the queue batches (plain, Lean): each class's p90, median, count and share within 2 h | maintained |
| `lean_unread_gate/` | the unread gate a ready mark's preflight runs (`tools/verity/lean/unread_gate.py`) on one change: its time by stage and its git calls, from an empty gate cache, and its guard, the same verdict and `--json` as the gate at another revision | maintained |

### Communication

How fast the team's leads answer each other: measured on Linear's comments, not on a protocol.

| Directory | What it measures | Status |
| --- | --- | --- |
| `communication/` | over the last 24 hours, the minutes from a Linear comment ringing at most three lanes to each rung lane's answer on the issue, as a 90th percentile, and the asks open over 30 minutes (`response.py`, on `research msg linear response`) | maintained |

### Census data

| Directory | What it measures | Status |
| --- | --- | --- |
| `../calibration/census/` | the fixed quantities benchmarks are computed against: hardware, datatypes, networks, deployments, subcircuits and workloads | data |

### The repository

| Directory | What it measures | Status |
| --- | --- | --- |
| `tree/` | the lines of `tests/tree.toml`'s not-yet list, each a path `main` holds outside the layout of record, and the unowned among them (`not_yet.py`); the rows of `ARCHITECTURE.md`'s table of code to write (`not_built.py`); and the sites the ratchet lints find, their lists allow and the lints find past their entries, in total and by lead (`lint_sites.py`, over `lint_counts.py`); and the share of Python and Lean lines that are prose, the prose items over the length AGENTS.md allows and those naming a date, an id, a commit, a machine or a person, in total and by lead (`prose.py`) | maintained |
| `proof_status/` | the proofs' status as the Lean locks record it (`verity lean deployed`): the deployed guarantees, those at a trust gap of zero, each one's gap, and each lead's open premises and guarantees with no honest witness (`run.py`) | maintained |

### Security

| Directory | What it measures | Status |
| --- | --- | --- |
| `security_findings/` | the security review's serious findings still open (`findings.toml`): each closed only by the check `open_findings.py` holds for it, run on the checkout, the machine it runs on and the readings of what it can't reach (`readings.toml`) | maintained |
