# compute: the share of a served model's MACs that Pearl+ credits

`verity bench compute`, the compute benchmark, prints one metric for one workload: the share of the served model's
multiply-accumulates (MACs) that the Pearl+ verifier credits. PoUW, proof of useful work, is how this repository credits
a GPU's matrix multiplications: the prover commits to every product it computed, a random coin flipped after the
commitment picks a sample of them, and the verifier recomputes the sample and credits the whole only if the sample is
right. Pearl+ is PoUW's verifier for Pearl, a proof-of-work scheme whose products are FP8 (E4M3) matrix multiplications
on an H100 or H200. The number is the MACs Pearl+ credits over all the MACs the model computes on a fixed load, and its
target is 97%.

It is a measurement, never an audit: your own run writes every file of the record it reads, coins included, so nothing
in a record can show that the auditor flipped the coins rather than you, re-rolling them until a run passed. The output
says so on its third line, always, and refuses a record that claims a deployed challenger or deployed parameters
(exit 2). Reading a deployed run needs a check of the challenger's signature against a pinned key, which this command
doesn't have yet.

## What you may change

How the work is done, and nothing the verifier reads:

- **The kernel** that computes the served model's matrix multiplications
  (`verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm90/`: `pearl_c_sm90.cu`, the inert-work counter
  `debit_sm90.cu`, `build.sh`, built by `benchmarks/harness/setup/pearl_c_sm90.sh`). Any code may replace it that
  commits the same operand rows, checked words and served outputs bit for bit: the verifier recomputes a sampled tile
  from its opened rows and never sees how they were computed (`checkTile`). `build.sh` refuses a binary that fails
  `sass_gate.py`, which keeps an honest kernel replayable and is no check of the verifier's. A kernel wrong on any tile
  risks the whole run: one sampled wrong tile rejects it.
- **The schedule**: how the load's requests become engine steps and passes (one pass is the matrix multiplications of
  one engine step), and the streams, graphs and overlap on the card. Which rows each step computes is the load's, so
  what is left free is inside a pass.
- **The give-up rule**: which rows of each matrix multiplication the prover gives up, declaring them uncredited, before
  the coin is flipped (`served_passes.py volunteer`), at most ⌈m/64⌉ rows of the activations A a call and ⌈n/64⌉ rows of
  a fixed weight B (`voluntaryCap`, `checkRecords`); and which passes to drop (`served_passes.py retain`). Both credit
  nothing. Giving up rows that would fail the verifier's per-row rules costs less than having the run rejected.
- **The sample size, locally.** On your machine you are the verifier too, so you choose t, the number of tiles sampled
  (`served_passes.py verify --draws`, 459 by default). The verifier takes a share ε_s off the credit for what the
  sample could miss. Where no sampled tile is over its inert-work bound, ε_s is about ln(1/δ_s)/t: 1/100 at t = 459
  and δ_s = 1/100. Each sampled tile over its bound enlarges ε_s (its rule is below), so the number moves
  with the sample from one run to the next, on the same kernel and load. Doubling t halves ε_s, which raises the
  number by about half of ε_s's deduction (in the fixture below, from 95.66% to about 96.15%), and doubles the replay.
  In an audit t is the auditor's.

A faster kernel or schedule makes the run quicker but does not move the share; a give-up rule that loses fewer rows, or
a larger sample, does.

**Why 97% is out of reach today.** Pearl+ credits the model's linear layers, and the LM head (the last projection onto
the vocabulary) only where the passes hold its calls. Attention and the LM head are the rest of the MACs. At the fixed
load the linears are 96.89% of the model's MACs, so a run whose passes don't hold the LM head can't reach 97% however
little it gives up. A run whose passes hold the LM head's calls can credit up to 99.88%.

## What is fixed

What the number is of, and the verifier that decides it. Change any of these and the number is another benchmark's:

- **The model and the load**: `load.toml`. The model is `unsloth/Llama-3.2-1B` at its pinned revision, and the load is
  4 requests of 32 prompt tokens and 4 output tokens each. The run registers its load in its inference-only policy
  (`out/inference-only-policy.json`), as its class: the requests (slots), their prompt tokens and their output tokens.
  A record whose policy registers another model, revision or class is refused, naming both. The MACs the number divides
  by come from the model's shape and the class (`MODELS`, `model_macs` in `run.py`); attention is counted and never
  credited.
- **The verifier**: the Lean program `pouw-pearlplus` (package `pouw_pearlplus` in
  `verity/protocols/resource_accounting/compute/pouw/verifier/lean/pearlplus/`, `PouwCheck.PearlPlus`'s `check`) and its Python twin
  `benchmarks/pouw/pearl_c_vllm/verify_run.py`, which must agree. The number needs the Lean verdict.
- **The scheme**: `pearl-c-h100-v1-h3` alone (`schemeOf` refuses the RTX PRO 6000's schemes by name), with 64 × 64
  tiles and a matrix depth k (the length of the products' dot products) of at least 1,024 (`TILE`, `K_MIN`) or
  exactly 256, the one smaller depth the scheme also admits (`censusKs`), and at most 2^16 (`K_MAX`), and m and n at
  most 2^24 (`domainReason`). Its words follow the H100's FP8
  tensor core bit for bit (`hopperE4m3K32`), so the card is an H100 or H200.
- **The record's format**: the manifest schema (`manifestSchema`), the records root (`recordsRoot`), the coin record
  (`CoinRecord`) and the version of W (`wVersion`). W is the window record: the run's registration of each matrix
  multiplication call, its activation rows, its weight rows and its outputs.
- **The rules and the credit**: the per-row rules (`rowPasses`), the inert-work bound ρ = 1/400 (`INERT_BOUND_DEN`) and
  its prices (`inertWorkUnits`), the rule for tiles over it, δ = 1/100 (`DELTA`) and the credit formula
  (`creditAtAssumption`). The formula for the number is `run.py`'s docstring.

## How a cheating change is refused

Each paragraph names a cheat and the check that meets it, in the Lean verifier and in `verify_run.py`. The checks'
properties are stated in `security/Properties/Pouw/PearlPlus.lean` (`AcceptMeans`, `PlanMeans`, `TileAcceptMeans`,
`LoadedGivesUp`, `YIsTheUnscale`, `WorkIsTheCredit`, `OpeningBinds`) and proved in
`security/Proofs/Properties/Pouw/PearlPlus/`.

**What the credit rests on.** That a credited MAC was computed rests on SHA-512's collision resistance and on Pearl's
conjecture TT_OUT rev1 (`TTOutPearlCDevRev1`, per tile `TTOutTilePearlCH100Rev1`), which proofs' red-team rating
refutes (`security/Definitions/Pouw/Guarantees/PearlCS8.lean`), and no proved guarantee covers a Pearl+ run's credit:
the number counts what the verifier credits, and that this is work done is an open assumption. A bridge from acceptance
to the audit game is proved (`acceptGivesTheAudit` in `security/Proofs/Pouw/PearlPlus/Sample.lean`).

**Computing fewer products, or wrong ones.** `checkTile` recomputes a sampled tile's checked words from its opened rows
under the salt's noise and compares them with the committed leaf (`checked values`; non-finite inputs are
`inadmissible inputs`). A wrong tile the sample misses is neither refused nor found: every run has ε_s taken off its
credit, and each wrong tile is a chance that the sample lands on it and rejects the run.

**Opening other values than the committed ones.** Each opened row and leaf must open against its root:
`activation opening`, `weight opening`, `transcript opening`, `y opening`. This rests on SHA-512's collision resistance.

**Working ahead of the salt, or choosing it.** The salt is a coin the prover's noise derives from, with the operands'
roots (`fixedNoise`). Each pass's salt coin is held to the epoch's coin commitment: `the pass's coin commitment is not
the epoch's`, `the pass's salt is not its coin's and the weights'`, `the passes are not all of one salt`. A re-rolled
salt is refused only in an audit, by a verdict of record: one the Lean verifier gives with `--of-record`, which refuses
coins that didn't come live from the auditor's challenger (`ofRecordReason`). Without it the verdict is a development
one.

**Choosing the sample.** Tiles are sampled by a key of the coin's source and the run's commitment (`drawKey`). A coin
record of another commitment is `after-coin`; a verdict of record refuses a sample the serving machine took itself
(`of-record`). One
sampling session a root, ever, is the challenger's own rule, in Python on the auditor's device (`open_refusal` in
`verity/protocols/compose/verifier/live_challenger/auditor.py`), not in Lean. On your own machine nothing refuses a
retried sample.

**Dropping passes after the sample.** The run's commitment covers the retained passes (`runCommitment`) and must be
the coin record's (`after-coin`); given a class, the retained rounds must be exactly its rounds (`heldToLoad`).

**Skipping inert work.** `inertBoundCheck` replays a sampled tile's inert work exactly and requires at most ρ of the
credited cells' credit. A tile over it is credited nothing and enlarges ε_s (`overInertBoundDraws`): `exceeds the cap`.
Inert work below the bound, and any other skip, is what TT_OUT rev1 bounds.

**Crediting cheap or padded rows.** `rowsFailure`: a credited row must pass the noise floor and liveness (`floorOK`,
`liveRow`), and an excluded row must fail them; a filler row is zero and no row holds a word past the kept width
(`not zero past the kept width`). `checkRecords` limits voluntary rows.

**Crediting one call twice.** `passesDistinct` refuses `repeated-call` (an index repeated under one salt) and
`call-range`; `workloadReason` refuses repeated matmul indices.

**A smaller instance, or another scheme.** The verifier refuses a matrix multiplication outside the scheme's domain
(`domainReason`: k under 1,024 other than 256, k past 2^16, m or n past 2^24) and passes at a k_min or
tile other than the served instance's, except as a test instance, which credits nothing (`testCredit`) and which
`verity bench compute` refuses; `schemeOf` refuses other schemes.

**Serving an output other than the checked product.** On a call that commits its output y, `yReason` requires each
opened y word to be the unscale of the tile's U (`yOf`): `is not the unscale of the`. `tieOf`, the check that ties the
passes to W, refuses a call that commits no y.

**Crediting work nobody served.** `tieOf` requires W to register each call's activation rows and y and each fixed
operand's weight rows, and `heldToLoad` the class's fixed load. These are Lean code with no proved property. `run.py`
refuses an answer whose passes or credit hold more MACs than one load of the class can (its linears, and its LM head
where the passes hold the head's calls), in Python on the measuring side.

**Serving with weights other than the model's: not refused today.** The registration log can carry a checkpoint's
SHA-512 (`checkpoint` in `verity/protocols/registration/verifier/lean/RegistrationLog.lean`), but no check
recomputes the committed weight rows' root from the published checkpoint, so other weights of the same shapes can be
credited.

**A faster exact method.** No verifier refuses a program that gives the same words bit for bit (a Strassen-like schedule,
a low-rank split, a preprocessing of the fixed operand). One that does materially less work than the credit counts
breaks no check: it falsifies TT_OUT rev1, and with it what the credit means.

## Running it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone this repository, and run every command
below from the clone's root. The first `uv run` installs the workspace's Python environment (`uv sync` does it alone).

~~~text
uv run verity bench compute --from tests/benchmarks/compute/fixtures/record
uv run verity bench compute --from DIR --per-call --top 5
~~~

`--from DIR` reads a run's record from the directory DIR, through no network and with no key. `verity bench` starts
the command at the clone's root, so a relative DIR is taken from the root. Run it from the root too: `uv run` in a
directory with a `pyproject.toml` of its own, as `tests/benchmarks/compute/` has, makes that directory's environment
instead of the workspace's.
`tests/benchmarks/compute/fixtures/record/` is a record directory written by hand: its shapes are the served model's
and its totals add up, but no verifier wrote it and no card served it, and the output's fourth line says so, with what the record says
of itself under it; `result.json` marks its numbers modeled and made up, below. `--per-call` also prints where the MACs the credit leaves out went, by cause and by matrix
multiplication call (the top N, 10 by default).

The first lines of the output, on the fixture:

~~~text
compute: 95.66% of the served model's MACs (134,502,518,410 of 140,598,837,248), below the 97% target. ...
ran with: coins in-process; parameters development (sample size 459, delta 1/100, scheme pearl-c-h100-v1-h3, ...)
a measurement, not an audit
~~~

The second line says who flipped the coins (`local challenger`, a challenger you run as its own process; or
`in-process`, the run's own process), and that the sample size and δ are development ones. The fixture's sample was
taken by the serving machine itself: it is a development run. The lines after name the run and say that the Pearl+
verifier's answer accepts at ε_s and δ_s and the Lean verifier agrees: ε_s and δ_s are sampling's, the credit misses
more than the share ε_s of the work with probability at most δ_s. Then one line says what the credit rests on, and one
gives PoUW's verdict in the run's record of verdicts (`out/record/verdicts.json`, every protocol's verdict on the run):
on your own card it is not an accept, so PoUW credits nothing of the run, and the number is the Pearl+ verifier's answer
alone.

To keep what it prints as files too (`compute.json` and `result.json`, below), set `OUT` to a
directory: `OUT=/tmp/compute uv run verity bench compute --from tests/benchmarks/compute/fixtures/record`.

The tests: `cd benchmarks/compute && uv run --project ../.. pytest tests`.

## Making a record on your own card

A record is the directory a workload writes: the Pearl+ verifier's answer and the Lean verifier's verdict on it,
the run's policy, and the rest listed below. One command makes one: `serve.sh` serves the load on one card under
Pearl+, takes Pearl+'s verdict in Python and in Lean, writes the record directory and ends with `run.py --from` on it,
which prints the number:

~~~text
bash benchmarks/compute/serve.sh
research run --on NODE --project verity --source . --cwd clone --timeout 4h -- env WORK=/workspace bash benchmarks/compute/serve.sh
~~~

The first line, from the clone's root, is the one command a fresh clone needs on your own machine. The second is how we
run it on Verity's machines, whose environments are under `/workspace`: `research run` is our job runner, which runs
the command on one of our machines (a node; `NODE` is its name) and records the run under a run id. You need neither.

Its first step, `bootstrap`, installs what the serve needs where it isn't there yet: it runs
`adapters/vllm/verity_vllm/ops/pod_bootstrap.sh --cases LLAMA32_1B --gpu`, which puts under the directory `WORK` names
(`~/verity-work` unless set) a Python 3.12 environment with vLLM at commit d9105ea80 and torch 2.13.0+cu129
(`venv312`, from the vLLM wheel it keeps in `wheels/`), the checkpoint `unsloth/Llama-3.2-1B` at the revision
`load.toml` pins (about 2.5 GB from Hugging Face, in `hf/`) and flash-attention's taps (`cp/`, which serve.sh's serve
doesn't load); its install caches, under `WORK/.cache`, it removes where it made them. Its log is
`out/logs/bootstrap.log`, its own logs and its readiness record are under `out/logs/bootstrap/`, and its seconds are
a line of `out/steps.txt` like every step's. Where `benchmarks/harness/env.sh` finds the environment already (under
`WORK`, or at `VERITY_BENCH_VENV`), serve.sh says so in one line and skips the step, so keep `WORK` the same from one
run to the next.

The record directory is `RUN_DIR`, else the run directory under `research run`, else `./compute-run`. The command
exits 0 only where both verdicts accept and the number prints. Each step's log is `out/logs/STEP.log`, its status a
line of `out/steps.txt`, and its seconds are in `out/compute-run.json`'s `steps`. A share it prints is always this
run's: it refuses a record directory that already holds a record (remove it, or give `RUN_DIR` a new directory), a try
its checks of the machine refuse (exit 3) writes nothing, and a failed step ends it with exit 1, a sentence naming the
step and no number. `COINS=local`, the default, runs a local challenger:
infra's gate code on a scratch auditor host with a test key, each session's challenger its own process on 127.0.0.1,
so the Lean verdict is taken of record. Of record here means only that the Lean verifier took its of-record path
(`--of-record`) against the local challenger's key, the test key this run made, not the auditor's: the `ran with`
line says `Lean verdict of record against the local challenger's key`. Under it the serve runs inside a Pearl-C period
of the local gate's lane, opened before the serve on the run's circuit record (built first, on the CPU) and closed once
the serve's salt session has ended, so the verifier's sample is the lane's first reserve after the close, as PoUW's verdict
of record requires (`out/pieces/pouw/period.json` holds the period). `COINS=in-process` holds the challenger in the
serving process. `out/compute-run.json` says which (`challenger`), and the salt coin as the piece states it
(`salt_coin`). The serving machine's passes stay outside the record and are removed on exit, so the run leaves nothing
in `/tmp`.

It needs:

- one idle H100 or H200 (the scheme follows the H100's FP8 tensor core): `GPU` is its index, default 0 (`nvidia-smi
  -L` lists them), and serve.sh checks before any step that it is a compute capability 9.0 card with no process on it.
  Run under a `gpu-lease` already held (`research run --gpus 1`, or your own `gpu-lease 1 --on 0 --cores CPUS -- bash
  benchmarks/compute/serve.sh` to pin the run to some CPUs), it serves on that lease's card and takes no lease of its
  own;
- nvcc from CUDA 13.0 (12.8 or newer) at /usr/local/cuda-13.0, or wherever `CUDA_HOME` names: the kernel is built by
  `benchmarks/harness/setup/pearl_c_sm90.sh`, which hands `build.sh` that toolkit;
- network access, and `curl`, for what the `bootstrap` step downloads;
- this tree's `adapters/torch` importable (`verity_torch`, which the served model's PoUW hooks import):
  `benchmarks/harness/env.sh`, which serve.sh sources, puts every workspace member's packages on `PYTHONPATH`, this
  one's among them, and serve.sh checks it after the bootstrap, before the cubin;
- Lean v4.34.0, through [elan](https://github.com/leanprover/elan), to build the Lean verifier: `lake` on `PATH` or
  in `~/.elan/bin`, which serve.sh also checks first;
- disk under `WORK` for what the bootstrap installs, and under `SCRATCH` or `TMPDIR` (the run's scratch under
  `research run`) for the run's passes, which are removed on exit;
- memory for the Python verdict's workers, one a core unless `JOBS` says otherwise. It grows with `JOBS`: a smaller
  `JOBS` holds less and makes `python-verify` slower.

The two verdicts run on every core serve.sh may run on (`nproc`), so a run takes longer on fewer cores. The Lean
verdict has `LEAN_TIMEOUT` seconds: 3,600 on 96 or more cores, and 3,600 × 96 over `nproc` on fewer, 17,280 on 20.
Give `research run` a `--timeout` past the bootstrap, `LEAN_TIMEOUT` and the other steps together. For each step's
seconds on one H200, read the published run that `metrics.toml`'s rows read: its record's `out/compute-run.json`
holds them under `steps`.

## A record directory

The paths below are relative to the record directory.

- `out/pieces/pouw/pearlplus-answer.json` (required): the Pearl+ verifier's answer, one of its pieces (the files it
  writes under `out/pieces/pouw/`). It holds the verdict, the credit in MACs, and the sample's ε_s at δ_s.
- `out/inference-only-policy.json` (required): the run's registered policy. It names the served model, its revision
  and the class. The model's MACs are counted from it.
- `out/pieces/pouw/pearlplus-lean.json` and `out/pieces/pouw/pearlplus-lean-check.json`: the Lean verifier's verdict,
  and the record's comparison of it with the answer. Without either there is no number (exit 1, naming the file).
- `out/record/verdicts.json` (optional): the run's composed record, every protocol's verdict on the run, of which PoUW's
  verdict is printed beside the number. serve.sh writes PoUW's row as the window's compose reads it
  (`benchmarks/one_stage/served_verdicts.py`'s `piece`): `missing` with the first condition of record the run fails,
  as no window registration of SPOT's is made on your own card. The credit is then the Pearl+ verifier's alone.
- `out/pieces/pouw/verify.json` (optional): the Pearl+ verifier's document. Its per-call split says whether the
  passes hold the LM head's calls, and `--per-call` prints it.
- `out/pieces/pouw/development.json` (optional): the Pearl+ verifier's development document. Only its `coin_source`,
  where the passes' salt coins came from, is read.
- `out/compute-run.json` (optional): what the serving command ran with. Only its `challenger` and `development` are
  read, for the second line: `challenger` is `local` or `in-process`, and `development` true. A file saying the
  challenger is `deployed`, or `development` false, is refused (exit 2).
- `job.json` (optional): the run's own record of itself. The command prints its `run.id` as the run, and the commit
  its `store.conditions` name (`source.commit`, with `-dirty` when `source.dirty`). Without it the run is the
  directory's name and the commit is unknown. A `made_up` statement in it makes the record a made-up one: the output says so, and `result.json` marks each
  number modeled and made up.

A directory missing a required file is refused with exit 2, and the message names the file.

## What it refuses

It prints `compute: no number: WHY` and exits 1 unless the Pearl+ answer is an accept with a credit, and not a test
instance's (the answer's `profile.test_instance`, whose passes are not the served instance's). It prints no number
either when the record holds no Lean verdict, or when the Lean verdict differs from the answer in verdict, run
commitment or credit. It exits 2 when a file can't be read, when the policy's model or load is not the benchmark's or
can't give the model's MACs, when the answer's passes or credit hold more MACs than one load of the class can, or when
`out/compute-run.json` claims a deployed challenger or deployed parameters. It
reads only the verifier's side of the record, so the tokens and seconds the serving machine reports are never part of
the number.

## Files

- `run.py`: the command. Its docstring states the number's formula, each file of a record it reads, what it prints
  beside the number, what it writes and when it refuses.
- `load.toml`: the benchmark's load.
- `tests/benchmarks/compute/`: the printer, the model's MAC count and the written output, held to the record under its
  `fixtures/`.
- `h200_share.py`: a modeled share of an H200's capacity (below).

**What it prints:** the number as text and as one line of JSON (`value`, `unit`, `commit`, `run`), beside its target.
The first line says the number is what the Pearl+ verifier's own answer credits, not PoUW's verdict in the run's
record of verdicts (`out/record/verdicts.json`, every protocol's verdict on the run), which a line further down
prints. Under the answer's verdict a line says the credit rests on Pearl's conjecture TT_OUT rev1, which this
repository's red-team rates refuted.
The second and third are the harness kit's (`benchmarks/harness/README.md`): the coins and parameters the record states
(`ran_with` in the JSON), and `a measurement, not an audit`, always (`audit` is false, in `compute.json` and in each
entry of `result.json`). Where the record doesn't say whose challenger flipped a challenger's coins, the second line is
`ran with: not stated` and a line under it says what the record does state.

**What it writes:** two files, through the harness kit's report (`kit.report`): under `research run` in its run
directory, outside one where `OUT` names a directory, and otherwise nothing.

- `compute.json`: the printed document. With no number its `value` is null and `no_number` says why.
- `result.json`: the same result in the fields every resource's benchmark reports, through the kit's writer: the
  number under its id (below), `verified` (the credited MACs), `capacity` and `free` (each with no value and why: a
  record holds no registration of its hardware and no span the verifier holds its passes to; `capacity()` in `run.py`
  gives them for a registered site and a span) and `error` (the answer's δ_s, sampling's alone). With no number, only
  the number's entry, with no value and why. A made-up record's entries, such as the fixture's, are marked `modeled`
  and `made_up` true: no run measured them.

**Exit code:** 0 with the number printed; 1 with none; 2 when the command line, the record or its policy can't be read,
when the policy's load is not the benchmark's (`load.toml`), or when the split `--per-call` reads is not the answer's.

## Ours: the evidence store and the metrics

On our machines `--from` also takes a run id of our evidence store, read through research's store client with a key
that reads the store; a withdrawal of part of a stored record is honored (a withdrawn required file refuses the record,
a withdrawn optional one is read as absent). `verity bench compute --from RUN` reads a run id that names no directory.

`metrics.toml` holds the metrics the console's Metrics tab reads: `compute-accounting/verified-compute-share`
(the number) and the others `run.py` writes, each written by the workload that measured them, at `serve.sh`'s last
step, as a `--from` of another run writes none; `h200_share.py` prints `compute-accounting/h200-capacity-credited-share`
for the goal `accounting/h200-97-percent`: an honest H200's e4m3 rate (a `benchmarks/tensor_duty --arms fp32-beside`
run's arm A on uniform operands) over the card's ceiling at each reading of its tensor clock, times `run.py`'s share of
a workload. Both need the store:
`uv run python benchmarks/compute/h200_share.py --from RUN --duty RUN`. Its docstring states the model and what it leaves
out, which makes its number an upper bound on the served share.
