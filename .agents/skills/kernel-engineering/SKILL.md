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

The workflow around a PoUW kernel. Commands are in the harness's `README.md` (`benchmarks/pouw/harness/`). Paths
`kb/...` and `campaigns/...` are in the notes (`~/.research/notes`). The panel's tool (`panel.py`, `attempts.jsonl`) is
in an agent store, at the path `kb/sm120/panel-add-a-row.md` gives.

## The objective

- **The score** is the timed whole-node slowdown: total time, hashing included, over the fastest plain GEMM of the same
  precision, timed in the same run on the same die. The headline shapes are prefill `m8192-n8192-k8192` and decode
  `m32-n8192-k8192`; tier 3 adds the served model's (n, k).
- **The constraints:** bit-exact against the scheme's reference; the reference verifier accepts the run's own
  transcript; γ ≤ 1% at both FP32 prices. A row with γ over 1% is never plotted.
- **The scorer is locked.** Never change the harness, a gate, the verifier or a statement parameter (the promotion
  period G, the domain's constants, ρ, Lean's W1 prices, the credit formula) to make an arm pass. Harness changes go to
  its owner; a statement change is a proposal to the coordinator.
- **Honest arms only.** An arm does the same work on every call and never detects the poisoned or dumped one. A result
  that looks too good (a hash-free slowdown under 1.00×, a jump far past its parent) goes to the assessor before anyone
  cites it.

## Before you start: what's been tried

For your line, version and mechanism:

1. **Approaches:** `campaigns/pouw/APPROACHES.md`. Claim a new approach family (a new schedule, forming or mainloop, not
   a knob in your own kernel) with `research notes claim pouw/<slug> --lane <you> --hypothesis "..."`; it refuses one
   live under another lane or killed, unless you pass `--reopen WHY`.
2. **Attempts:** the technique catalog, `kb/sm120/tried-techniques.md`, for your mechanism. Then the attempts since:
   the harness's records (`research data select --kind kernel-attempt/v1`, or `ledger.py rows --refresh`), the line's
   panel rows (`kb/sm120/panel.md`), and the owning lane's checkpoints (`campaigns/pouw/rtx-pro/workers/*.md`).
3. **Gotchas:** read your toolchain's and the node's sections of `kb/sm120/gotchas.md` before your first build.
4. **Owner:** every kernel directory, shared module and tool has one owner lane (the table in
   `campaigns/pouw/rtx-pro/brief.md`). Change theirs by a PR against their branch or a message, never an edit in their
   tree.

If it was tried and dropped, your hypothesis says what's different this time, or you pick something else.

## Where the code goes

- **Exact semantics:** `verity.ml.circuits.silicon`. A newly captured step lands there as a model with a sampled fixture
  and an `instructions` entry (`benchmarks/tc_probe`, `tc_probe_fp4`), never as a kernel directory's own copy.
- **The scheme and its verifier:** `verity/core/protocols/pouw` (stdlib only; its `PROTOCOL.md`).
- **The kernel:** a directory under `benchmarks/pouw/`, like `pearl_c_sm120/`: `run.py` (`Dev`, `Pipeline`),
  `fixture.py`, `build.sh`, its sources, and its harness arm against `arm.py`'s contract. Where serving kernels finally
  live is compute-accounting's call; the loader keys on digests, so a later move is cheap.
- **The vLLM executor:** `integrations/vllm`.
- **Shared tables:** scheme names, hashing formats, schedules, step groups and kernel variants are defined once, in
  `verity.core.protocols.pouw.serving`. Never keep a local copy (tree hashes, keys, arm tables); replace one you find.

## One attempt

An attempt is one change to a variant, with a parent, taken up the ladder only as far as it earns.

### 1. Name it before you run it

- **Parent:** the variant you start from, by its `KernelVariant.id` (`<family>-<device>/<source digest>`).
- **Mechanism:** short tags (`pipeline:tma-ws`, `unroll:2`, `epilogue:wide-stores`); reuse the catalog's tag where one
  fits, so a search finds every attempt at it.
- **Hypothesis:** the expected effect and why, in one line ("WIDE stores cut STG L1 tag requests from 16 to 4; expect
  the tail to shrink").
- **The tier** that will decide it, usually 2a.

In a harness arm these go in its build record, which the attempt's record copies: `variant` is
`ledger.variant_record(..., prices=...)`, `prices` naming the priced path the kernel runs (`cast-8.72`, `cast-32.06`,
`lut256`); `lineage` holds the parent, the tags and the panel line and version; the hypothesis is
`verify.py --hypothesis "..." --parent ART`. Outside the harness, they go in your status file and the panel row's
`--description`.

A tuned or swept choice is part of the variant's sources: commit the winning knob, so its digest changes with it. A knob
that picks code at run time (`PEARLC_A_FORMS`) goes in `variant_record(..., config={...})`, or two forms share one id.
Serving never tunes.

### 2. Climb the ladder

Stop at the lowest tier that decides the attempt. Only a candidate that beats its parent at 2a gets a 2b window.

| Tier | Runs on | Establishes | How |
|---|---|---|---|
| 1 conformance | CPU | the build; the SASS gate on each binary; no spills; the twin against `verity.ml.circuits.silicon` on small tiles; the executor's step order on `RecordingDev` | `uv run tools/verity/check/suites.py benchmarks/pouw`; `sass_gate.py K.cubin --json gate.json` |
| 2a screen | one leased GPU, untimed | the device gates at the served shapes; a poisoned transcript accepted and a no-write control rejected; a diagnostic interleaved time; the time split; ncu, compute-sanitizer, sweeps | `research run ... -- gpu-lease 1 --wait -- ...`, or fill chunks of at most 8 minutes |
| 2b timed | the whole node, locked-2100 | the measured panel row | `bench.py` under `research run --no-sampler` and `gpu-lease 8 --wait --timed`, then `verify.py --tier 2b` after the lease, in the same run |
| 3 served | vLLM on the card | the served arms, the time split, the verify job, the served row | [Hooking a kernel into vLLM](#hooking-a-kernel-into-vllm) |

Copy the build, untimed and timed commands from the harness's `README.md`. Every GPU tier holds to these:

- **Time the verifier's bytes.** The timed code emits the transcript the reference verifier recomputes: the version's
  own commitment of A and B, seeds derived from the roots on the device, the version's format. A stand-in commitment,
  hash or format doesn't count, even one that costs the same. A new commitment path is verified at 2a, on a transcript
  your timed code wrote, before it is timed at all.
- **Poison, then control.** Fill every buffer the transcript reads with 0xA5 just before the call you dump, including
  buffers a setup step wrote (`root_a`, `seed_a`, scales). A control arm writes nothing into them, and the verifier must
  reject it. Give each arm its own inputs.
  In the harness, implement `dump(dev, call, shape, dir)` and `call.transcript_buffers`; `bench.poison_dump` poisons,
  dumps and writes the control under `negative/`, and `verify.py` judges both. Outside it, your code does it (as
  `pearl_c_vllm/e2e.py` does). Leave out tensor maps and a one-launch tree's `atomicInc` counters (one filled with 0xA5
  never reaches the last CTA).
- **Gates before timing, in the same run.** A shape and flag set is timed only after its gates passed on the device in
  that run, and a failed gate writes no measurement; a bench of your own keeps the harness's order.
- **The baseline is the fastest plain GEMM, timed beside you:** same card, shape and die, interleaved rep by rep after a
  warm-up, in items of equal device time, with every rep's SM clock. cuBLASLt is autotuned over its top algorithms,
  CUTLASS's sm_120 kernels are tried, and the winner is recorded. A faster divisor is adopted only once a timed
  whole-node run confirms it; until then, show both.
- **Decode is a dependent chain** (`--decode-method dependent-chain`, the harness's default): call i+1's A is derived on
  the device from call i's y. Independent calls don't count.
- **A tree never changes between its gate and its window.** Run from your own checkout at a pushed commit, never edit a
  file a running job reads, and put the next change on a new commit.

**Sharing node 2.**

- Every GPU command runs under `gpu-lease` with `GPU_LEASE_WHO=<your id>`, for at most 30 minutes. Take the GPU's
  identity from `GPU_LEASE_UUID`; `nvidia-smi -i` ignores `CUDA_VISIBLE_DEVICES`.
- A lease holds a GPU for GPU work only: build, prepare operands and verify outside it (you are the check).
- A timed run takes `--timed`, which pauses the fill queue; otherwise the fill queue keeps running beside it (you are
  the check that a row's run held one).
- Every run goes through `research run` (`.agents/skills/writing-runs/SKILL.md`). A fill chunk's outputs under
  `/workspace/pouw/fill-out/` reach the store only by `research data put --kind evidence/v1 --tree DIR --preserve`:
  run it when the chunk ends, and cite the `art:` id.

**γ at both prices.** Compute γ at the cheapest bit-exact FADD (8.00 W1 per word) and at the measured in-loop price
(1047/125 = 8.376), publish the larger, and say which won.

- The prices come from the Lean guarantees (`campaigns/pouw/price-twins-lean/`), never a device record or a diagnostic.
  The harness looks γ up in `price_twins.json` on the variant's `prices` path; never type it. Outside the harness
  `panel.py --gamma` is typed by hand (you are the check).
- The path you price is the path you run. A kernel that casts with 64-bit stores at a 160-byte stride is priced at that
  cast's 8.72 per code; if the block scale is priced on `lut256` (8.55), forming runs `lut256`. Name the path from the
  kernel your forming step launches at the run's commit, not the one you plan to reach.
- A change that moves the priced path (another cast, store width or scale path) moves γ: re-derive it and say so in the
  row. Changing a price is a statement change.

### 3. Record it, whatever happened

Every attempt that ran is recorded, screens, failures and drops included: a negative result nobody wrote down gets tried
again.

- **A harness run records itself:** `verify.py` writes one `kernel-attempt/v1` record per (arm, shape), verified or
  not, which the runner publishes under `research run --on` (elsewhere `verify.py --publish --push`, or
  `ledger.py publish RUN_DIR --by <you>` later). A 2a screen is logged and never plotted.
- **Your judgment is a label on the record:**
  `ledger.py outcome ART <outcome> --reason "..." --ref <run> --by <you> --push`. Numbers go in records and judgments in
  labels.
- **Anything priced or timed outside the harness is a panel row:** `panel.py append`, with every field
  `kb/sm120/panel-add-a-row.md` describes. Add `--noisy` when anything shared the GPU or the node. After a store
  `EAGAIN`, run `panel.py render`; never append again. Hand appends stop once 90% of tier-2 runs record themselves.
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
- A variant chosen to serve is pinned, citing its ruling. Nothing serves "the fastest at serving time".
- A change to agreed semantics or architecture goes through the coordinator.

## Hooking a kernel into vLLM

The path today is `benchmarks/pouw/pearl_c_vllm/`:

1. `setup.sh`, off the timed path: the venv, the model, FlashInfer's JIT.
2. `window.sh` under `gpu-lease 8 --wait --timed`: its gates, then interleaved stock-FP8, Pearl-C, no-hash and BF16
   reps, keeping poisoned passes for the verifier.
3. `verify.sh <window run> 48`: accept prefill, accept decode, reject the control.
4. `panel_rows.py e2e.json <verify dir> --append`.

A migrated head is re-verified and re-timed through the coordinator, from its own checkout, one window at a time.
Add no other install path. Wrap through `engine.hooks` in `protocol_options.pouw`, never by assigning
`quant_method.apply`.

A variant serves only with a passing `GateRecord` of its own (`serving.variant_refusals` says why not): exact at every
served shape, poisoned passes accepted, controls rejected, schedules equal, the SASS gate passed under the pinned
toolkit, all on the priced path its `KernelVariant` declares. `pearl_c_vllm/` caches modules under fixed names and checks
none of this (you are the check) until the executor serves through `serving`'s `register` and `select` by pin: then a
kernel directory exports `VARIANT` and `STEP_IO` and names its steps in `GROUPS`, each served pass records
`KernelVariant.id` in `PassCommitment.kernel`, and that PR rewrites this section.

## Leave it better than you found it

- **Tripped over a tool, a command or a doc?** `.agents/skills/friction/SKILL.md`.
- **Learned a fact** about the toolchain, the card, a library or the node? One line in its section of
  `kb/sm120/gotchas.md`, with its source (a run id, a commit). Correct a wrong line rather than add a contradiction.
- **Checked by hand a rule code could check?** Propose the check to the tool's owner, with the fix in its failure
  message. The rules above marked "you are the check" are the known gaps.
- **Found dead machinery** (a stand-in commitment, a lane bench the harness replaced, a `PEARLC_*` mode that lost, a
  second SASS gate or `verify.py`, a copied scheme table)? Delete it in your next PR, or name it to its owner.
- **Changed how something is done?** Update this skill in the same PR.

The assessor's folder and id are its own (`campaigns/pouw/red-team/`): send it code or a job through the coordinator.

## Before you finish

- Every run you started is in the evidence store (`research data preserved <run|art>` exits 0), fill outputs included.
- Every attempt has a record or a row, an outcome and a reason; every fact a gotchas line; every friction fixed or noted.
- Your PR description says what you fixed and what you deleted along the way.
- You hold no GPU lease, and no job of yours reads a tree you are about to change.
