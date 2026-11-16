# compute: the share of a served model's MACs that Pearl+ credits

What you may change is how the work is done: the kernels that compute the served model's matmuls, how the model is
served under Pearl+ (how the load's requests become engine steps and passes, and the streams, graphs and overlap on the
card), and the give-up rule (which rows each matmul gives up before the coin, within the verifier's cap). A better
kernel, schedule or give-up raises the number.

What is fixed is what the number is of: the model and its revision, and the load (`load.toml`: Llama 3.2 1B at its
pinned revision, 4 requests of 32 prompt tokens and 4 output tokens each), the verifier and its Lean (`pouw-pearlplus`,
whose verdict the number needs), and the formula (`run.py`'s docstring). A record whose policy registers another model,
revision or load is refused. Change any of these and the number is another benchmark's.

`verity bench compute` prints the compute audit's number for one served run. The number is a share: the
multiply-accumulates (MACs) that the Pearl+ verifier credits, over the MACs the served model computes on the tokens of
the run's registered class. The command measures nothing itself. It reads the verifier's side of the run's record, and
nothing the developer's node reports of what it served. The number is the Pearl+ piece's own answer, not PoUW's row in
the run's record of verdicts, and the line under it says which coins and parameters the record states the run used.

You give it a run's record in one of two ways:

- `--record DIR`: a directory on disk holding the record. This needs nothing of ours: no evidence store, no host and
  no key. The command imports no store client, looks up no credential and opens no network connection.
- `--from RUN`: a run id of our evidence store. This is ours: it needs research's store client and a key that reads
  the store.

~~~text
verity bench compute --record DIR
verity bench compute --from RUN
verity bench compute --record DIR --per-call [--uncredited F] [--top N]
uv run python benchmarks/compute/h200_share.py --from RUN --duty RUN
~~~

The two options are separate, so the command never guesses: a path that names no directory is refused as a path, and
a run id never reads a local file. `tests/fixtures/record/` is a record directory, so
`verity bench compute --record benchmarks/compute/tests/fixtures/record` runs from a clone. Its record is made up: it
shows the format, not a served run's number. Run the command from the clone's root as `verity bench compute`, which
starts it there, so a relative DIR is taken from the root.

The tests run from this directory, which is not a workspace member, in the workspace's environment:
`uv run --project ../.. pytest tests`.

Outside `research run` it only prints. To keep what it prints as files too (`compute.json` and `result.json`, below),
set `OUT` to a directory: `OUT=/tmp/compute verity bench compute --record DIR`.

<!-- the serving command: one line here once it is on main -->
**Producing a record** (serving the model on a card and auditing the run) is not yet one command on `main`. A branch
adds it as serve.sh in this directory, which serves the load above on one H100 or H200 and writes a record directory
for `--record`. Until it lands, a record comes from a run on our machines.

Tonight's record uses a local challenger and development parameters: the coins come from a challenger process the
developer runs, not the auditor's device, and the sample size and δ are the development defaults (459 draws, 1/100).
The coins line says so. The card is an H100 or H200: the Lean verifier refuses the sm_120 schemes today.

## PoUW under Pearl+: what you may change, what is fixed, how a cheat is refused

This section was checked against the code by reading: its paths, names, constants and quoted reasons are the code's. No
run exercised it.

Pearl+ is PoUW's run verdict over `pearl-c-h100-v1-h3`, Pearl's FP8 (E4M3) proof of useful work. The prover commits
every pass's operand rows and every tile's checked values. The verifier samples tiles by their credit, with a coin
drawn after the commitment, replays each sampled tile from its opened rows, and credits the run at the scheme's named
assumption. The verifier is Lean, `PouwCheck.PearlPlus`'s `check`
(`verity/protocols/accounting/pouw/verifier/lean/pearlplus/PouwPearlPlus/Check.lean`, executable `pouw-pearlplus`).
The Python `benchmarks/pouw/pearl_c_vllm/verify_run.py` computes the same answer beside it, and `verity bench compute`
prints no number when the two differ or the record holds no Lean verdict.

**A number from your own machine is a measurement, not an audit.** With in-process coins the salt comes from an
in-process challenger and the sample from the node's own draw, and you control both. Run of record, the Lean verifier
refuses such a run as a development run (`ofRecordReason` in
`verity/protocols/accounting/pouw/verifier/lean/pearlplus/PouwPearlPlus/Coins.lean`): of record needs the coins live
from the auditor's challenger. Locally nothing stops you re-rolling either until a run passes. A run is one execution
phase; an audit pools execution phases over a monitoring period, which a local run doesn't.

### What you may change

Anything the verifier doesn't read, as long as what you commit is what it replays:

- **The kernel.** The served kernel is `verity/protocols/accounting/pouw/prover/kernels/pearl_c_sm90/`
  (`pearl_c_sm90.cu`, the inert-work counter `debit_sm90.cu`, `build.sh`), built by
  `benchmarks/harness/setup/pearl_c_sm90.sh`. Any code may replace it that commits the same operand rows, checked
  words, leaves and served y bit for bit: `checkTile` replays them from the opened rows and never sees how they were
  computed. `build.sh` refuses a cubin that fails `sass_gate.py`; that gate keeps an honest kernel replayable and is no
  verifier check. A kernel wrong on any tile risks the whole run: one sampled wrong tile rejects it.
- **The schedule.** How the load's requests become engine steps and passes (the tokens a step), and the streams,
  graphs and overlap on the card. The requests, their prompt tokens and their output tokens are the load's
  (`load.toml`): given the class (`served_passes.py verify --class SLOTS,PROMPT,OUTPUTS`), each round's rows are the
  class's (`heldToLoad`), so what is left free is inside a pass.
- **The give-up before the coin.** Which rows to give up as voluntary (`served_passes.py volunteer`), at most ⌈m/64⌉ A
  rows a call and ⌈n/64⌉ B rows a fixed operand (`voluntaryCap`, `checkRecords`), and which passes to drop
  (`served_passes.py retain`). Both credit nothing and are fixed before the sample.
- **The sample size, locally.** Locally you are the verifier too, so you choose t, the work-weighted tiles sampled
  (`served_passes.py verify --draws`, default 459). More samples take a smaller share ε off the credit and cost more
  replay. In an audit t is the verifier's, and the coin that picks the tiles is never yours.

### What is fixed

The verifier and everything it pins. Changing any of these makes another verifier, whose number is not this
benchmark's:

- **The verifier**: the Lean package `pouw_pearlplus` (`PouwCheck.PearlPlus`) and its Python twin `verify_run.py`.
- **The scheme and instance**: `pearl-c-h100-v1-h3` alone (`schemeOf` refuses the sm_120 records' schemes by name),
  64 × 64 tiles at `k_min` 1,024 (`TILE`, `K_MIN`), k ≤ 2^16 (`K_MAX`). Its words follow the sm_90 FP8 tensor core
  (`hopperE4m3K32`): the card is an H100 or H200, and on any other card your kernel must reproduce those words bit for
  bit.
- **The record's format**: the manifest schema (`manifestSchema`), the records root (`recordsRoot`), the coin record
  (`CoinRecord`) and W's version (`wVersion`).
- **The rules and the credit**: the per-row rules (`rowPasses`), the inert-work bound ρ = 1/400 (`INERT_BOUND_DEN`)
  and its prices (`inertWorkUnits`), the over-bound rule `uncredited`, δ = 1/100 (`DELTA`) and the credit formula
  (`creditAtAssumption`).
- **The model and the load**: `load.toml`. The MACs the number divides by come from the model's shape and the class
  (`MODELS`, `model_macs` in `run.py`), attention counted and never credited.

### How a cheat is refused

Each paragraph names a cheat and the check that meets it, in the Lean verifier and in `verify_run.py`. The checks'
properties are stated in `security/Properties/Pouw/PearlPlus.lean` (`AcceptMeans`, `PlanMeans`, `TileAcceptMeans`,
`LoadedGivesUp`, `YIsTheUnscale`, `WorkIsTheCredit`, `OpeningBinds`) and proved in
`security/Proofs/Properties/Pouw/PearlPlus/`. That a passed check means the work was done rests on SHA-512's collision
resistance (`cr/sha-512` in the lock) and on Pearl's assumption at the H100 record, TT_OUT rev1 (`TTOutPearlCDevRev1`;
per tile `TTOutTilePearlCH100Rev1`), which is conjectured. A bridge from acceptance to the game's audit is proved
(`acceptGivesTheAudit` in `security/Proofs/Pouw/PearlPlus/Sample.lean`), but no guarantee in the lock states a Pearl+
run's credit yet, and `verity/protocols/accounting/pouw/verifier/README.md` records TT_OUT rev1 as refuted by a
red-team rating.

**Computing fewer products, or wrong ones.** `checkTile` recomputes a sampled tile's checked words from its opened rows
under the salt's noise and compares them with the committed leaf (`checked values`; non-finite inputs are
`inadmissible inputs`). A wrong tile the sample misses is neither refused nor found: every run has ε taken off its
credit, and each wrong tile is a chance that the sample lands on it and rejects the run.

**Opening other values than the committed ones.** Each opened row and leaf must open against its root:
`activation opening`, `weight opening`, `transcript opening`, `y opening`. This rests on `cr/sha-512`.

**Working ahead of the salt, or choosing it.** The noise derives from the salt and the operands' roots
(`fixedNoise`). Each pass's salt coin is held to the epoch's coin commitment: `the pass's coin commitment is not the
epoch's`, `the pass's salt is not its coin's and the weights'`, `the passes are not all of one salt`. A re-rolled salt
is refused only in an audit, where a verdict of record refuses in-process coins (`ofRecordReason`).

**Choosing the sample.** Tiles are sampled by a key of the coin's source and the run's commitment (`drawKey`). A coin
record of another commitment is `after-coin`; a verdict of record refuses the node's sample (`of-record`). One sampling
session a root, ever, is the challenger's own rule, in Python on the auditor's device (`open_refusal` in
`verity/protocols/compose/device/live_challenger/auditor.py`), not in Lean. Locally nothing refuses a retried sample.

**Dropping passes after the sample.** The run's commitment covers the retained passes (`runCommitment`) and must be
the coin record's (`after-coin`); given a class, the retained rounds must be exactly its rounds (`heldToLoad`).

**Skipping inert work.** `inertBoundCheck` replays a sampled tile's inert work exactly and requires at most ρ of the
credited cells' credit. A tile over it is credited nothing and enlarges ε (`overInertBoundDraws`): `exceeds the cap`.
Inert work below the bound, and any other skip, is what TT_OUT rev1 bounds.

**Crediting cheap or padded rows.** `rowsFailure`: a credited row must pass the noise floor and liveness (`floorOK`,
`liveRow`), and an excluded row must fail them; a filler row is zero and no row holds a word past the kept width
(`not zero past the kept width`). `checkRecords` caps voluntary rows.

**Crediting one call twice.** `passesDistinct` refuses `repeated-call` (an index repeated under one salt) and
`call-range`; `workloadReason` refuses repeated matmul indices.

**A smaller instance, or another scheme.** The verifier refuses passes off `k_min` 1,024 and 64 × 64 tiles, except as a
test instance, which credits nothing (`testCredit`) and which `verity bench compute` refuses; `schemeOf` refuses other
schemes; `domainReason` refuses k past 2^16.

**Serving an output other than the checked product.** On a call that commits y, `yReason` requires each opened y word
to be the unscale of the tile's U (`yOf`): `is not the unscale of the`. Tied to W, `tieOf` refuses a call that commits
no y.

**Crediting work nobody served.** `tieOf` requires W to register each call's A rows and y and each fixed operand's B
rows, and `heldToLoad` the class's fixed load. These are Lean code with no proved property. `run.py` refuses an answer
crediting more MACs than the class's linears (and LM head), Python on the measuring side.

**Serving with weights other than the model's: not refused today.** The registration log can carry a checkpoint's
SHA-512 (`checkpoint` in `verity/protocols/accounting/registration/verifier/lean/RegistrationLog.lean`), but no check
recomputes the committed B rows' root from the published checkpoint, so other weights of the same shapes can be
credited.

**A faster exact method.** No check refuses a program that gives the same words bit for bit (a Strassen-like schedule,
a low-rank split, a preprocessing of the fixed operand). One that does materially less work than the credit counts
breaks no check: it falsifies TT_OUT rev1, and with it what the credit means.

## A record directory

A record directory is a run's directory as `research run` leaves it and our store keeps it. The paths below are
relative to it.

- `out/pieces/pouw/pearlplus-answer.json` (required): the Pearl+ verifier's answer. It holds the verdict, the credit
  in MACs, and the sample's ε_s at δ_s.
- `out/inference-only-policy.json` (required): the registered policy. It names the served model and the class: its
  slots, prompt tokens and output tokens. The model's MACs are counted from it.
- `out/pieces/pouw/pearlplus-lean.json` and `out/pieces/pouw/pearlplus-lean-check.json`: the Lean verifier's verdict,
  and the record's comparison of it with the answer. Without either there is no number (exit 1, naming the file).
- `out/record/verdicts.json` (optional): the run's composed record. PoUW's row is printed beside the number.
- `out/pieces/pouw/verify.json` (optional): the Pearl+ verifier's document. Its per-call split says whether the
  passes hold the LM head's calls, and `--per-call` prints it.
- `out/pieces/pouw/development.json` (optional): the Pearl+ piece's own document. Only its `coin_source`, where the
  passes' salt coins came from, is read, for the coins line below. In the evidence store, a withdrawal of it leaves
  the record readable: `--from` then neither fetches nor reads it, as if the record held none, where a withdrawn
  required file refuses the record.
- `job.json` (optional): the run's own record of itself. The command prints its `run.id` as the run, and the commit
  its `store.conditions` name (`source.commit`, with `-dirty` when `source.dirty`). Without it the run is the
  directory's name and the commit is unknown.

A directory missing a required file is refused with exit 2, and the message names the file.

## What it refuses

It prints no number, and exits 1, unless the Pearl+ answer is an accept with a credit, and not a test instance's (the
answer's `profile.test_instance`, whose passes are not the served instance's). It also prints none when the
record holds no Lean verdict, or when the Lean verifier's verdict differs from the answer in verdict, run commitment or
credit. It
exits 2 when a file can't be read, when the policy's model or class can't give the model's MACs, or when the credit is
more than one load of the class can hold. It reads only the verifier's side of the record, so the tokens and seconds
the developer's node reports are never part of the number.

## Files

- `run.py`: the command. Its docstring states the number's formula, each file of a record it reads, what it prints
  beside the number, the result it writes and when it refuses.
- `h200_share.py`: the number of the goal `accounting/h200-97-percent`, modeled: an honest H200's e4m3 rate (a
  `benchmarks/tensor_duty --arms fp32-beside` run's arm A on uniform operands) over the card's ceiling at each reading
  of its tensor clock, times `run.py`'s share of a served run of record. Its docstring states the model and what it
  leaves out, which makes the number an upper bound on the served share.
- `tests/`: the printer, the model's MAC count and the written result, held to the small record under
  `tests/fixtures/`, read as a directory and through a stand-in for the evidence store, so the suite fetches nothing.
  Capacity is held to one card file's two clocks.

**What a caller may import:** nothing outside this directory imports `run.py`. It is a command, which
`verity bench compute` starts as a process of its own, and its `__all__` lists the names its own tests read. **What it
never does:** it starts no job and measures nothing, it reads nothing the developer's node reports of what it served,
and it prints no number for an answer that is not an accept. With `--from`, fetching a record is the evidence store's
client's work; the command itself writes only the two files named below.

- **Hardware:** none. It runs on any machine, in seconds: with `--record` on any clone, with `--from` on one that can
  read the evidence store. The GPU work is the recorded run's.
- **What it prints:** the number as text and as one line of JSON (`value`, `unit`, `commit`, `run`), beside its
  target. The first line says the number is what the Pearl+ piece's own answer credits, not PoUW's row in the run's
  record of verdicts. The line under it (`coins` in the JSON) says which coins and parameters the record states the
  run used: the answer's sample source (`profile.source_from`: `challenger` or `node`), the passes' salt coins
  (`development.json`'s `coin_source`: `challenger` or `in-process`), whether the Lean verifier ran as of record
  (`--of-record` in `pearlplus-lean-check.json`'s `argv`), and the answer's draws and δ. What the record doesn't state
  it prints as not stated. It ends the line with `development` where the record holds PoUW's development piece
  (`development.json`) or any of the three is not live, which today is every record, since only that piece states the
  salt coins. `challenger` names a challenger session, the auditor's or a local one: the record doesn't say whose. It
  never refuses on them. With `--per-call`, also where the MACs the credit leaves out went, by cause and by matmul call (the top N
  calls, 10 by default), from the Pearl+ verifier's split of its credit (the record's `verify.json`, or the file
  `--uncredited` names).
- **What it writes:** two files, through the harness kit's report (`kit.report`, `benchmarks/harness/README.md`), as
  `verity bench memory` does: under `research run` in its run directory, outside one where `OUT` names a directory,
  and otherwise nothing. With no number printed it writes neither.
  - `compute.json`: the printed document, `$RESEARCH_RUN_DIR/out/compute.json` under `research run`, else
    `$OUT/compute.json`.
  - `result.json`: the same result in the fields every resource's benchmark reports, through the kit's writer. Under
    `research run` it is the run directory's (`$RESEARCH_RUN_DIR/result.json`), where `research` reads a run's
    result; outside one it is `$OUT/result.json`. Its measurements are
    `compute-accounting/verified-compute-share` (the printed number, its unit and its target, under its id in
    `metrics.toml`, where the console's Metrics tab reads it), `verified` (the credited MACs), `capacity`, `free`
    (capacity less verified) and `error` (the answer's δ_s, sampling's alone). A recorded run's record holds no
    registration of its hardware and no span the verifier holds its passes to, so `capacity` and `free` are one entry
    each, with no value and why.
    `capacity()` in `run.py` gives them for a registered site and a span, at each data-sheet reading: a clock of a
    registered device's card file (`clock.tensor`, `clock.sm`).
- **Exit code:** 0 with the number printed; 1 with none (the answer is not an accept, the record holds no Lean
  verdict or one that is not the answer's, or the run left no answer); 2 when the command line, the run, its record
  directory or its policy can't be read, when the policy's load is not the benchmark's (`load.toml`), or when the
  split `--per-call` reads is not the answer's.
