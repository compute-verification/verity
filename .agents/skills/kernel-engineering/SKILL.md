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

The workflow around a PoUW kernel. Commands and the harness's mechanics are in its `README.md`
(`benchmarks/pouw/harness/`). `kb/...` and `campaigns/...` are in the notes (`~/.research/notes`); the panel's tool
(`panel.py`, `attempts.jsonl`) is in an agent store, at the path `kb/sm120/panel-add-a-row.md` gives. "You are the
check" marks a rule no code enforces yet.

## The objective

- **The score** is the timed whole-node slowdown: total time, hashing included, over the fastest plain GEMM of the same
  precision, timed in the same run on the same die, at prefill `m8192-n8192-k8192` and decode `m32-n8192-k8192` (tier 3
  adds the served model's (n, k)).
- **The constraints:** bit-exact against the scheme's reference; the reference verifier accepts the run's own
  transcript; γ ≤ 1% at both FP32 prices (a row over 1% is never plotted).
- **The scorer is locked:** never change the harness, a gate, the verifier or a statement parameter (the promotion
  period G, the domain's constants, ρ, Lean's W1 prices, the credit formula) to make an arm pass. Harness changes go to
  its owner; a statement change is a proposal to the coordinator.
- **Honest arms only:** the same work on every call, never detecting the poisoned or dumped one. A result that looks too
  good (a hash-free slowdown under 1.00×, a jump far past its parent) goes to the assessor before anyone cites it. The
  assessor's folder and id are its own (`campaigns/pouw/red-team/`): send it code or a job through the coordinator.

## Before you start: what's been tried

For your line, version and mechanism, read `campaigns/pouw/APPROACHES.md`, the technique catalog
(`kb/sm120/tried-techniques.md`), then the attempts since: `research data select --kind kernel-attempt/v1` (or
`ledger.py rows --refresh`), `kb/sm120/panel.md` and the owning lane's checkpoints (`campaigns/pouw/rtx-pro/workers/*.md`).
An idea tried and dropped comes back only with a hypothesis saying what's different. Claim a new approach family (a
schedule, forming or mainloop, not a knob in your own kernel) with `research notes claim pouw/<slug> --lane <you>
--hypothesis "..."`, which refuses one live under another lane or killed unless you pass `--reopen WHY`. Read your
toolchain's and the node's sections of `kb/sm120/gotchas.md` before your first build. Each kernel directory, shared
module and tool has one owner lane (`campaigns/pouw/rtx-pro/brief.md`): change theirs by a PR against their branch or a
message, never an edit in their tree.

## Where the code goes

- **Exact semantics:** `verity.ml.circuits.silicon`. A newly captured step lands there as a model with a sampled fixture
  and an `instructions` entry (`benchmarks/tc_probe`, `tc_probe_fp4`), never as a kernel directory's own copy.
- **The scheme and its verifier:** `verity/core/protocols/pouw` (stdlib only; its `PROTOCOL.md`).
- **The kernel:** a directory under `benchmarks/pouw/` like `pearl_c_sm120/` (`run.py` with `Dev` and `Pipeline`,
  `fixture.py`, `build.sh`, its sources, a harness arm against `arm.py`'s contract). Where serving kernels finally live is
  compute-accounting's call; the loader keys on digests, so a later move is cheap.
- **The vLLM executor:** `integrations/vllm`.
- **Shared tables** (scheme names, hashing formats, schedules, step groups, kernel variants) are defined once, in
  `verity.core.protocols.pouw.serving`. Never keep a local copy (tree hashes, keys, arm tables); replace one you find.

## One attempt

An attempt is one change to a variant, with a parent, taken up the ladder only as far as it earns.

### 1. Name it before you run it

Its parent (`KernelVariant.id`, `<family>-<device>/<source digest>`), mechanism tags (`pipeline:tma-ws`,
`epilogue:wide-stores`; the catalog's where one fits, so a search finds every attempt at it), a one-line hypothesis (the
expected effect and why) and the tier that will decide it, usually 2a. In a harness arm they go in its build record
(`variant`, whose `prices` names the priced path the kernel runs: `cast-8.72`, `cast-32.06`, `lut256`; `lineage`) and
`verify.py --hypothesis "..." --parent ART`; outside it, in your status file and the panel row's `--description`.
Commit a tuned knob's winner, so the digest changes with it; a knob that picks code at run time (`PEARLC_A_FORMS`) goes in
the variant's `config`, or two forms share one id. Serving never tunes.

### 2. Climb the ladder

Stop at the lowest tier that decides the attempt. Only a candidate that beats its parent at 2a gets a 2b window.

| Tier | Runs on | Establishes | How |
|---|---|---|---|
| 1 conformance | CPU | the build; the SASS gate on each binary; no spills; the twin against `verity.ml.circuits.silicon` on small tiles; the executor's step order on `RecordingDev` | `uv run tools/verity/check/suites.py benchmarks/pouw`; `sass_gate.py K.cubin --json gate.json` |
| 2a screen | one leased GPU, untimed | the device gates at the served shapes; a poisoned transcript accepted and a no-write control rejected; a diagnostic interleaved time; the time split; ncu, compute-sanitizer, sweeps | `research run ... -- gpu-lease 1 --wait -- ...`, or fill chunks of at most 8 minutes |
| 2b timed | the whole node, locked-2100 | the measured panel row | `bench.py` under `research run --no-sampler` and `gpu-lease 8 --wait --timed`, then `verify.py --tier 2b` after the lease, in the same run |
| 3 served | vLLM on the card | the served arms, the time split, the verify job, the served row | [Hooking a kernel into vLLM](#hooking-a-kernel-into-vllm) |

Every GPU tier holds to these:

- **Time the verifier's bytes:** the timed code emits the transcript the reference verifier recomputes (the version's own
  commitment of A and B, seeds derived from the roots on the device, the version's format), never a stand-in, even one
  that costs the same. A new commitment path is verified at 2a, on a transcript your timed code wrote, before it is timed.
- **Poison, then control:** fill every buffer the transcript reads with 0xA5 just before the dumped call, setup-written
  ones included (`root_a`, `seed_a`, scales), but not tensor maps or a one-launch tree's `atomicInc` counters (one never
  reaches the last CTA). A control arm that writes nothing into them must be rejected; each arm has its own inputs. The
  harness does this for an arm with `dump` and `call.transcript_buffers`; outside it your code does, as
  `pearl_c_vllm/e2e.py` does.
- **Gates before timing, in the same run:** a shape and flag set is timed only after its gates passed on the device in
  that run, and a failed gate writes no measurement; a bench of your own keeps the harness's order.
- **The baseline is the fastest plain GEMM (cuBLASLt autotuned, CUTLASS's sm_120 kernels; the winner recorded), timed
  beside you** as the harness times it: same card, shape and die, interleaved rep by rep after a warm-up, every rep's SM
  clock. A faster divisor is adopted only once a timed whole-node run confirms it; until then, show both.
- **Decode is a dependent chain** (`--decode-method dependent-chain`): call i+1's A is derived on the device from call
  i's y. Independent calls don't count.
- **A tree never changes between its gate and its window:** run from your own checkout at a pushed commit, never edit a
  file a running job reads, and put the next change on a new commit.
- **Sharing node 2:** every GPU command runs under `gpu-lease` with `GPU_LEASE_WHO=<your id>`, for at most 30 minutes,
  and holds the GPU for GPU work only: build, prepare operands and verify outside it (you are the check). Take the GPU's
  identity from `GPU_LEASE_UUID`; `nvidia-smi -i` ignores `CUDA_VISIBLE_DEVICES`. `--timed` pauses the fill queue, which
  otherwise runs beside you (you are the check that a row's run held one). Every run goes through `research run` (the
  writing-runs skill); a fill chunk's outputs under `/workspace/pouw/fill-out/` reach the store only by `research data
  put --kind evidence/v1 --tree DIR --preserve` when the chunk ends, cited by its `art:` id.
- **γ at both prices:** the cheapest bit-exact FADD (8.00 W1 per word) and the measured in-loop price (1047/125 =
  8.376); publish the larger and say which won. Prices are the Lean guarantees' (`campaigns/pouw/price-twins-lean/`),
  never a device record or a diagnostic: the harness looks them up (`price_twins.json`); outside it `panel.py --gamma`
  is typed by hand (you are the check). Price the path you run, named from the kernel your forming step launches at the
  run's commit, not the one you plan to reach (64-bit stores at a 160-byte stride cost that cast's 8.72 per code; a
  block scale priced on `lut256`, 8.55, means forming runs `lut256`). Moving the priced path (cast, store width, scale
  path) moves γ: re-derive it and say so in the row. Changing a price is a statement change.

### 3. Record it, whatever happened

Every attempt that ran is recorded, screens, failures and drops included. A harness run records itself (`verify.py`'s
`kernel-attempt/v1` records, published as its README says); a 2a screen is logged, never plotted. Your judgment is a
label, `ledger.py outcome ART <outcome> --reason "..." --ref <run> --by <you> --push`: numbers in records, judgments in
labels. Anything priced or timed outside the harness is a panel row: `panel.py append` with every
field `kb/sm120/panel-add-a-row.md` describes, `--noisy` when anything shared the GPU or the node; after a store
`EAGAIN`, run `panel.py render`, never append again. Hand appends stop once 90% of tier-2 runs record themselves. The
keep-or-drop line, with its reason and run id, goes in your status file.

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

An approach's status and reason go in `research notes approach pouw/<slug> killed --by <you> --reason "..."`; a
coordinator's ruling is cited by its time. A variant chosen to serve is pinned, citing its ruling: nothing serves "the
fastest at serving time". A change to agreed semantics or architecture goes through the coordinator.

## Hooking a kernel into vLLM

The path today is `benchmarks/pouw/pearl_c_vllm/`, each script as its header says: `setup.sh` off the timed path,
`window.sh` under `gpu-lease 8 --wait --timed`, `verify.sh <window run> 48`, then `panel_rows.py e2e.json <verify dir>
--append`. A migrated head is re-verified and re-timed through the coordinator, from its own checkout, one window at a
time. Add no other install path: wrap through `engine.hooks` in `protocol_options.pouw`, never by assigning
`quant_method.apply`.

A variant serves only with a passing `GateRecord` of its own, at every served shape and on the priced path its
`KernelVariant` declares (`serving.variant_refusals` says why not). `pearl_c_vllm/` caches modules under fixed names and checks
none of this (you are the check) until the executor serves through `serving`'s `register` and `select` by pin; then a
kernel directory exports `VARIANT` and `STEP_IO` and names its steps in `GROUPS`, each served pass records
`KernelVariant.id` in `PassCommitment.kernel`, and that PR rewrites this section.

## Leave it better than you found it

Friction goes by the friction skill, and a PR that changes how something is done updates this skill (AGENTS.md "Working
principles"). A fact about the toolchain, the card, a library or the node is one line, with its source (a run id, a
commit), in its section of `kb/sm120/gotchas.md`, correcting a wrong line rather than contradicting it. A rule you checked
by hand goes to the tool's owner as a proposed check, with the fix in its failure message. Dead machinery (a stand-in
commitment, a lane bench the harness replaced, a losing `PEARLC_*` mode, a second SASS gate or `verify.py`, a copied
scheme table) is deleted in your next PR or named to its owner.

## Before you finish

Every run you started, fill outputs included, is preserved (`research data preserved <run|art>` exits 0); every attempt
has a record or a row, an outcome and a reason; every fact has a gotchas line and every friction is fixed or noted; your
PR description says what you fixed and deleted along the way; you hold no GPU lease, and no job of yours reads a tree you
are about to change.
