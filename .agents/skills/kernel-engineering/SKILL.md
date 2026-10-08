---
name: kernel-engineering
description: >
  How to design, evaluate, record and land PoUW GPU kernels in Verity: Pearl-C (FP8 E4M3) and Pearl-C4 (NVFP4) on the
  RTX PRO 6000 (sm_120), or any kernel whose output a verifier recomputes bit for bit. Covers checking what's been tried
  before starting, the evaluation ladder (CPU conformance, a one-GPU screen, the timed whole-node panel row, served vLLM),
  what a timed run must pass (the SASS gate, 0xA5 poisoning and a no-write control, the reference verifier on the run's
  own transcript, interleaved cuBLASLt and CUTLASS baselines, γ at both FP32 prices), recording every attempt with its
  hypothesis and reason, hooking a variant into vLLM (KernelVariant, GateRecord), sharing node 2 (gpu-lease, the fill
  queue) with other kernel lanes, and leaving the tooling better. Use when writing or tuning a CUDA or PTX kernel, a
  mainloop, forming or an epilogue, running benchmarks/pouw/harness (bench.py, verify.py, sass_gate.py), adding a panel
  row (panel.py), timing against a baseline, installing a kernel in vLLM, or starting or taking over a kernel lane.
---

# Kernel engineering

Kernel authoring has been fast; what cost the PoUW kernel lanes time on Sep 30 was around it. Measured rows were voided
because the timed code committed something the verifier doesn't check; stale outputs passed the verifier; results stayed
on a node; and the same lesson was learned in three lanes. This skill is the workflow around the kernel: what to check
before you start, how to climb from a CPU test to a served row, what to record, and how to leave the tools better than you
found them.

It points at the tools rather than repeating them: the harness's `README.md` (`benchmarks/pouw/harness/`, #491), the
panel's `add-a-row.md`, and the vLLM integration API (#567). The pages lanes keep writing are in the notes
(`~/.research/notes`): `kb/sm120/` holds the facts, the technique catalog and the panel's pages, and `campaigns/pouw/`
the RTX PRO brief, the lanes' checkpoints, the price twins and the assessor's folder. The panel's tool (`panel.py` and its
`attempts.jsonl`) is still in the POUS Project store's `internal/pouw/panel/`, until its rows move to the evidence store. Parts
marked **(lands with …)** change when that work merges; [When the tooling lands](#when-the-tooling-lands) lists them.

## The objective

- **The score** is the timed whole-node slowdown: total time, hashing included, over the fastest plain GEMM of the same
  precision, timed in the same run on the same die. The headline shapes are prefill `m8192-n8192-k8192` and decode
  `m32-n8192-k8192`; tier 3 adds the served model's (n, k).
- **The constraints:** bit-exact against the scheme's reference; the reference verifier accepts the run's own
  transcript; γ ≤ 1% at both FP32 prices. A row with γ over 1% is never plotted.
- **The scorer is locked.** Never change the harness, a gate, the verifier or a statement parameter to make an arm pass.
  The statement parameters are the promotion period G, the domain's constants, ρ, the W1 prices Lean uses and the credit
  formula. Harness changes go to its owner; a statement change is a proposal to the coordinator.
- **Honest arms only.** An arm does the same work on every call and never detects the poisoned or dumped one. A result
  that looks too good (a hash-free slowdown under 1.00×, a jump far past its parent) goes to the assessor before anyone
  cites it. The harness's flag for this is planned, not built.

## Before you start: what's been tried

Ten minutes here saves a GPU window. For your line, version and mechanism:

1. **Approaches:** `campaigns/pouw/APPROACHES.md` in the notes, generated from the evidence store. Claim a new approach
   family (a new schedule, forming or mainloop, not a knob in your own kernel) with
   `research notes claim pouw/<slug> --lane <you> --hypothesis "..."`. `claim` refuses an approach that is live under
   another lane, or that was killed, unless you pass `--reopen WHY`. The CLI is on #240 until it merges.
2. **Attempts:** search the technique catalog, `kb/sm120/tried-techniques.md` in the notes, for your mechanism: one row
   per technique tried on sm_120, with its kernel and head, its measured effect and runs, and kept, dropped, open or
   superseded with the reason. Then the attempts since it was built: the harness's records
   (`research data select --kind kernel-attempt/v1`, or `python benchmarks/pouw/harness/ledger.py rows --refresh`),
   the panel's rows for the line (`kb/sm120/panel.md`), and the owning lane's latest checkpoints
   (`campaigns/pouw/rtx-pro/workers/*.md`).
3. **Gotchas:** `kb/sm120/gotchas.md` in the notes, one line per fact with its source. Read your toolchain's and
   the node's sections before your first build.
4. **Owner:** every kernel directory, shared module and tool has one owner lane (the table in
   `campaigns/pouw/rtx-pro/brief.md`). A change to someone else's code goes to them, as a PR against their
   branch or a message to them, never as an edit in their tree.

If it was tried and dropped, your hypothesis says what's different this time, or you pick something else.

## Where the code goes

- **Exact semantics:** `verity.ml.circuits.silicon`. A newly captured step lands as a model with a sampled fixture and an
  `instructions` entry (`benchmarks/tc_probe`, `benchmarks/tc_probe_fp4`). A kernel directory never defines a second one.
- **The scheme and its verifier:** `verity/core/protocols/pouw` (`verity.core.protocols.pouw`, stdlib only; its `PROTOCOL.md`).
- **The kernel:** a directory under `benchmarks/pouw/`, like GPU 1's `pearl_c_sm120/` on #540: `run.py` (`Dev`,
  `Pipeline`), `fixture.py`, `build.sh`, its sources, and its harness arm against `arm.py`'s contract. Where serving
  kernels finally live is compute-accounting's call; the loader keys on digests, so a later move is cheap.
- **The vLLM executor:** `integrations/vllm` (`verity_vllm/linear/impl/` once #567's migration reaches step 4).
- **A new circuit** (a Definition, a subcircuit template) passes `circuit-check` (`AGENTS.md` "Circuits").

## One attempt

An attempt is one change to a variant, with a parent, taken up the ladder only as far as it earns.

### 1. Name it before you run it

- **Parent:** the variant you start from: today its commit and kernel directory; once #567 lands, its `KernelVariant.id`
  (`<family>-<device>/<source digest>`).
- **Mechanism:** what changes, in short tags (`pipeline:tma-ws`, `unroll:2`, `release:fence-proxy`,
  `epilogue:wide-stores`); reuse the catalog's tag where one fits, so a search finds every attempt at it.
- **Hypothesis:** the expected effect and why, in one line: "WIDE stores cut STG L1 tag requests from 16 to 4; expect the
  tail to shrink".
- **The tier** that will decide it, usually 2a.

In a harness arm, these go in its build record, and the harness copies them into the attempt's record:
- `variant`: `ledger.variant_record(family=..., device=..., schemes=..., sources=..., binaries=..., prices=...)`. `prices`
  is the priced path the kernel runs, such as `cast-8.72`, `cast-32.06` or `lut256`.
- `lineage`: the parent, the mechanism tags and the panel line and version.
- The hypothesis: `verify.py --hypothesis "..." --parent ART`.

Outside the harness, they go in your status file's checkpoint and the panel row's `--description`.

A tuned or swept choice is part of the variant's sources: commit the winning knob, so its digest changes with it. A knob
that picks code at run time (`PEARLC_A_FORMS`) goes in `variant_record(..., config={...})`, or two forms share one id.
Serving never tunes.

### 2. Climb the ladder

Stop at the lowest tier that decides the attempt. Only a candidate that beats its parent at 2a gets a 2b window.

| Tier | Runs on | Establishes | How |
|---|---|---|---|
| 1 conformance | CPU | the build; the SASS gate on each binary; no spills; the twin against `verity.ml.circuits.silicon` on small tiles; the executor's step order on `RecordingDev` | the package's suite (`uv run tools/verity/check/suites.py benchmarks/pouw`); `python benchmarks/pouw/harness/sass_gate.py K.cubin --json gate.json` |
| 2a screen | one leased GPU, untimed | the device gates at the served shapes; a poisoned transcript that verifies and a no-write control that's rejected; a diagnostic interleaved time; the time split; ncu, compute-sanitizer, knob sweeps | `research run ... -- gpu-lease 1 --wait -- ...`, or fill chunks of at most 8 minutes |
| 2b timed | the whole node, locked-2100 | the measured panel row | `bench.py` under `research run --no-sampler` and `gpu-lease 8 --wait --timed`, then `verify.py --tier 2b` after the lease, in the same run |
| 3 served | vLLM on the card | the served arms, the time split, the verify job, the served row | [Hooking a kernel into vLLM](#hooking-a-kernel-into-vllm) |

The harness's `README.md` has the build, the untimed run and the timed run with its verify step: copy them. They pass
`--timed`, and #590 adds `--no-sampler`.

Every GPU tier holds to these. Each came from a voided row, a lost window or GPUs held idle:

- **Time the verifier's bytes.** The timed code emits the transcript the reference verifier recomputes: the version's
  own commitment of A and B, seeds derived from the roots on the device, the version's format. A stand-in commitment,
  hash or format doesn't count, even one that costs the same. That voided GPU 1's 2.42× and Pearl-C4's attempt 5. A
  new commitment path is verified at 2a, on a transcript your timed code wrote, before it is timed at all.
- **Poison, then control.** Fill every buffer the transcript reads with 0xA5 just before the call you dump, including
  buffers a setup step wrote (`root_a`, `seed_a`, scales). A control arm writes nothing into them, and the verifier must
  reject it. Give each arm its own inputs.
  - **In the harness:** implement `dump(dev, call, shape, dir)` and `call.transcript_buffers`. `bench.poison_dump` then
    poisons, dumps the transcript and writes the no-write control under `negative/`, and `verify.py` judges both.
  - **What to leave out:** tensor maps and a one-launch tree's `atomicInc` counters. A counter filled with 0xA5 never
    reaches the last CTA.
  - **What it refuses:** a transcript buffer that overlaps the call's operands.
  - **Outside the harness:** your code does it, as #540's `e2e.py` does.
- **Gates before timing, in the same run.** A shape and flag set is timed only after its gates passed on the device in
  that run, and a failed gate writes no measurement. The harness's order does this; a bench of your own must too.
- **The baseline is the fastest plain GEMM, timed beside you:** same card, shape and die, interleaved rep by rep after a
  warm-up, in items of equal device time, with the SM clock of every rep. cuBLASLt is autotuned over its top algorithms,
  CUTLASS's sm_120 kernels are tried, and the winner is recorded. A faster divisor is adopted only after a timed
  whole-node run confirms it; until then, show both.
- **Decode is a dependent chain** (`--decode-method dependent-chain`, the harness's default): call i+1's A is derived on
  the device from call i's y. Independent calls don't count.
- **A lease holds a GPU for GPU work only.** Every GPU command runs under `gpu-lease` with `GPU_LEASE_WHO=<your id>`, for
  at most 30 minutes. Take the GPU's identity from `GPU_LEASE_UUID`; `nvidia-smi -i` ignores `CUDA_VISIBLE_DEVICES`. Build,
  prepare operands and verify outside the lease, as the harness runs `verify.py` after it. A timed run takes `--timed`;
  otherwise the fill queue keeps running beside it.
- **Every run goes through `research run`**, so its outputs reach the evidence store even if your VM dies:
  `--source <clean checkout> --cwd source`, never hand-rolled `nohup`. A fill chunk's outputs under
  `/workspace/pouw/fill-out/` are not in the store until you run
  `research data put --kind evidence/v1 --tree DIR --preserve` on them; do it when the chunk ends, and cite the `art:` id.
- **A tree never changes between its gate and its window.** Run from your own checkout at a pushed commit, never edit a
  file a running job reads, and put the next change on a new commit.

**γ at both prices.** Compute γ at the cheapest bit-exact FADD (8.00 W1 per word) and at the measured in-loop price
(1047/125 = 8.376), publish the larger, and say which won.
- The prices come from the Lean guarantees (`campaigns/pouw/price-twins-lean/` in the notes), not from a device record
  or a diagnostic.
- In the harness you don't compute or type γ. The attempt record looks it up in `price_twins.json`, the Lean
  guarantees' twins, on the variant's `prices` path, at both prices. A result that types γ in gets no record.
- The path you price is the path you run. A kernel that casts with 64-bit stores at a 160-byte stride is priced at that
  cast's 8.72 per code; if the block scale is priced on `lut256` (8.55), forming runs `lut256`. Name the path from the
  kernel your forming step launches at the run's commit, not the one you plan to reach.
- A kernel change that moves the priced path (another cast, store width or scale path) moves γ: re-derive it, and say so
  in the row. Changing a price is a statement change, for the coordinator.

### 3. Record it, whatever happened

Every attempt that ran is recorded, screens, failures and drops included: a negative result nobody wrote down gets tried
again.

- **The run** is in the store because it went through `research run`; a fill chunk's through `research data put`.
- **A harness run records itself.**
  - `verify.py` writes one `kernel-attempt/v1` record per (arm, shape) under the run's `ledger/`, verified or not. It holds
    the variant and parent, the tier, the gates' version, the metrics, the time split, γ at both prices and the
    verifier's lines.
  - It also declares the records in the run's `outputs.json`. Under `research run --on`, the runner publishes them with
    the Attempt (#590); the workload has no store key, so `verify.py --publish` can't.
  - Elsewhere, `verify.py --publish --push` publishes them, or `ledger.py publish RUN_DIR --by <you>` does it later.
  - `panel.py render` shows them beside `attempts.jsonl`. A 2a screen is logged there and never plotted.
- **Your judgment is a label on the record:**
  `ledger.py outcome ART <outcome> --reason "..." --ref <run> --by <you> --push`. Numbers go in records and judgments
  in labels; `research data label` refuses a measurement.
- **Anything priced or timed outside the harness is a panel row:** `panel.py append`, with every field described in
  `kb/sm120/panel-add-a-row.md` in the notes.
  - A measured row needs the verifier's commit, its accept line, the transcript naming the run, the control's reject
    line and both sides' per-rep clocks; `panel.py` refuses the row without them.
  - Add `--noisy` when anything shared the GPU or the node.
  - After a store `EAGAIN`, run `panel.py render`; never append again.
  - Hand appends stop once 90% of tier-2 runs record themselves.
- **The keep-or-drop line,** with its reason and run id, goes in your status file.

### 4. Decide, and write down why

Each attempt ends with one outcome and a one-line reason that cites its evidence:

| Outcome | Means |
|---|---|
| screened | ran at 2a, not worth a 2b window; the reason is optional |
| frontier | a distinct approach family worth a 2b window |
| kept | in the variant from now on |
| dropped | not kept; the reason says why ("fold: −2.2%, the stage release's MEMBAR waits on the step's stores") |
| voided | it measured the wrong thing: `panel.py append --attempt N --revision "..." --voids` |
| superseded | replaced by a named later attempt |

- Put each "why" where the next reader will look: an attempt's reason; an approach's status and reason
  (`research notes approach pouw/<slug> killed --by <you> --reason "..."`); a coordinator's ruling, cited by its time.
- A variant chosen to serve is pinned, citing its ruling (a `pin` label once the ledger lands). Nothing serves "the
  fastest at serving time".
- A change to agreed semantics or architecture goes through the coordinator; its rationale belongs in Notion.

## Hooking a kernel into vLLM

**Today** the path is #540's `benchmarks/pouw/pearl_c_vllm/`:
1. `setup.sh`, off the timed path: the venv, the model, FlashInfer's JIT.
2. `window.sh` under `gpu-lease 8 --wait --timed`: its gates, then interleaved stock-FP8, Pearl-C, no-hash and BF16
   reps, keeping poisoned passes for the verifier.
3. `verify.sh <window run> 48`: accept prefill, accept decode, reject the control.
4. `panel_rows.py e2e.json <verify dir> --append`.

Each migrated head is re-verified and re-timed through the coordinator, from its own checkout, one window at a time. Don't
add another install path: three already patch vLLM's linears. Until the migration's step 3, wrap through `engine.hooks`
in `protocol_options.pouw`, never by assigning `quant_method.apply`.

**(Lands with #567 and bc-f4e8ae34's migration onto it:**
- A kernel directory exports `VARIANT` (family, device, schemes, schedules, shapes) and `STEP_IO` (each step's group and
  per-call reads and writes) beside `Dev` and `Pipeline`, and names its steps in `GROUPS` (`a_tree`, `forming`, `gemm`,
  `cleanup`, `tile_hash`, `screen`).
- Its identity is `KernelVariant.id`, the digest of its sources. The loader keys modules by it, and every served pass
  records it in `PassCommitment.kernel`, so a row names exactly what ran.
- It serves only through `register`, with a passing `GateRecord` of its own, and `select` by pin. The gate covers every
  served shape: exact, poisoned passes accepted, controls rejected, schedules equal, the SASS gate passed under the pinned
  toolkit, all on the priced path it declares.
- The harness times it through the same `KernelPackage`, `Schedule` and `lanes.py` the executor serves with, so what was
  timed is what serves.**)**

## Working beside other lanes

Between kernel lanes there is ownership, one copy of each shared type, and GPU etiquette. An ordinary change waits on
nobody.

- **One owner per thing.** A kernel directory, the hashing module, the harness, the serving API and the panel each have
  one owner lane. Change theirs by a PR against their branch or a message to them, and review their PRs when they ask.
- **Interfaces are data, in one place.** Scheme names, hashing formats, schedules, step groups and kernel variants are
  defined once (`verity.core.protocols.pouw.serving`, #567). Never keep a local copy of such a table (tree hashes, keys, arm tables); if
  you find one, replace yours with the shared one, or tell its owner.
- **Messages:** a one-line checkpoint per result in your status file. Message your coordinator (never through a note)
  only to unblock, to ask for a decision, or when your result changes someone's plan (lane contract §3a).
- **Serialize your own edits:** write, then commit, then run, never a write and a launch at once. Commit and push often:
  a VM reset has wiped uncommitted work twice.
- **The assessor's folder and id are its own** (`campaigns/pouw/red-team/` in the notes). Send it code or a job through the
  coordinator.

## Leave it better than you found it

Kernel lanes run fast, in parallel, on the same tools, so small fixes compound. Whenever something bites, and before you
finish a session:

- **Tripped over a tool, a command or a doc?** Fix it at the source if it's yours or plainly broken, in a small PR of its
  own (`.agents/skills/friction/SKILL.md`, §1). Otherwise write a friction note, or two lines naming the earlier one under
  `recurs:`.
- **Learned a fact** about the toolchain, the card, a library or the node? Add one line to
  `kb/sm120/gotchas.md` in the notes with its source (a run id, a commit, a status file and time), in its section.
  When a line is wrong, correct it; don't add a contradiction.
- **Checked a rule by hand that code could check?** It belongs in the tool, with the fix in its failure message: propose it
  to the tool's owner, pricing it with `.agents/skills/process-design/SKILL.md` if it gates other lanes.
  [What enforces what](#what-enforces-what) lists the known gaps.
- **Found dead machinery?** Delete it in your next PR, or name it to its owner. Examples: a stand-in commitment; a lane
  bench the harness has replaced; an opt-in mode (`PEARLC_*`) that lost and is no longer timed; a second SASS gate or
  `verify.py`; a copied scheme table.
- **Changed how something is done?** Update this skill in the same PR.

## What enforces what

A rule that code checks costs nothing until it fires. Where a rule is still prose, this table names the planned check;
until it lands, you are the check.

| Rule | Checked today by | To be checked by |
|---|---|---|
| The timed code emits the transcript the verifier checks | `verify.py` (#491) accepts only an `ACCEPT` line naming the transcript's sha256; `panel.py` refuses a measured row whose transcript doesn't name the run | the harness timing through the served `KernelPackage` (#567 with #491) |
| Poisoned buffers, and a no-write control rejected | the harness for an arm with `dump` (#491 `80bff34cc`), the arm otherwise; `verify.py` rejects a row whose control is accepted; `panel.py` refuses a measured row without the control's reject line naming the run | every arm on `dump` (GPU 1, GPU 5 and #588 still to take it, server.md 18:18Z); `GateRecord.poisoned_accepted` and `controls_rejected` (#567) |
| No `.FTZ`, and no FP32 `MUFU` outside the pinned IEEE sequences; a pinned toolkit and flags | `sass_gate.py` (#491), exit 7; an unpinned toolkit or ptxas flag, `-lineinfo` included, fails closed | also `GateRecord.sass_ok` (#567) |
| Gates before timing, in the same run | `bench.py`'s order (#491): exit 5 when a known-bad variant passes, 6 on the consistency gate | — |
| An interleaved same-die baseline, with per-rep clocks | `bench.py`'s warm-up, equal-time items and per-item NVML; `panel.py` requires both sides' clocks and tags a gap over 0.5% | — |
| Decode as a dependent chain | `bench.py`'s default; `panel.py` keeps other measured decode rows off the plots | — |
| A timed run holds a `--timed` lease | `gpu-lease --timed` pauses the fill queue; nothing checks that a row's run had one | the attempt record naming its lease (to propose with #491's self-recording) |
| γ at both prices, the larger published, from the Lean guarantees, on the path that ran | the harness's attempt records (#491): γ from the price twins on the variant's declared `prices`, and a typed-in γ refused; outside the harness, nothing (`panel.py --gamma` is typed by hand) | a check that the declared path is the one the binary runs; `variant_refusals` on a priced-path mismatch (#567) |
| Every attempt recorded | `research run` custody for runs; the harness's `kernel-attempt/v1` records (#491), published by the runner under `research run --on` (#590); other panel rows by hand | — |
| A variant serves only with its own passing gate, and nothing tunes at serving | nothing (#540 caches kernel modules under fixed names) | `register`, `select` and `variant_refusals` (#567) |
| No CPU work inside a GPU lease | nothing | none planned; a candidate for `gpu-lease`, which could report a lease's idle GPU time |
| Measurements aren't labels | `research data label` refuses measurement keys | — |

## When the tooling lands

Whoever lands one of these updates this skill in the same PR, and deletes what it replaces.

| Lands | Who | What changes here | What goes |
|---|---|---|---|
| #491, the harness, on `main` | harness owner (bc-0de2d624), through a train | commands point at `main`, not a tree assembled from branches | per-kernel copies of the SASS gate and `verify.py` (`pearl_c_sm120/`) |
| #491's self-recording: on #491 (#583); the runner's publishing on #590 | bc-1a23b70c, with the harness owner and the panel's renderer | done: "What's been tried" step 2, "Name it", "Record it", the γ rows above | hand `panel.py append`, once 90% of tier-2 runs record themselves |
| the technique catalog: first version `kb/sm120/tried-techniques.md` (notes); then generated from the records | bc-1a23b70c | done: "What's been tried" reads it | — |
| the facts page: first version `kb/sm120/gotchas.md` (notes), hand-kept; then `fact` records | bc-1a23b70c | done: "Gotchas" and "Learned a fact" point at it; later, it is generated from `fact` records | research-notes `kb/sm120-kernels.md` (now a pointer) |
| lane benches as tier-2a harness arms | harness owner | tier 2a runs in the harness, gated and recorded | `mainloop_bench`, `nvf4_bench` and `run.py check` as separate benches |
| #240, the approach registry's CLI, on `main` | the research coordinator | `research notes claim`, `approach`, `approaches` run from `main` | — |
| #567, then the migration's steps 2 to 4 | bc-f4e8ae34 | "Hooking a kernel into vLLM" | #540's `e2e.install`, `vllm_bench.install`, the fixed-name module cache |

## Before you finish

- Every run you started is in the evidence store (`research data preserved <run|art>` exits 0), fill outputs included.
- Every attempt has a row or a record, an outcome and a reason, so the next agent finds what you tried and dropped.
- Every fact you learned is a line in `kb/sm120/gotchas.md` in the notes; every friction you hit is fixed or noted.
- Your PR description says what you fixed and what you deleted along the way.
- You hold no GPU lease, and no job of yours reads a tree you are about to change.
