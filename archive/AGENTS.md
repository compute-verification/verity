# Backends: rules for the proof campaign

Read the root `AGENTS.md` first. This file is the policy shared by every proof
backend and by the campaign-side numerical work under `benchmarks/numerical/`.
C-Flock is the family in active development; A-GKR, B-Ligero, D-SP1 and the
VOLE research backend are frozen: they stay in the tables and under these
rules, but are not developed.

## What is fixed

- The subcircuit is `verity.ml.circuits.targets.target.FIRST`. Lanes do not change it.
  The backend transition unit is one 16-product update with arbitrary finite
  `c`; the proof unit is `GemmCoordinate<K>` from `c_0 = 0` plus the epilogue.
- Backend families must agree on profile, input/output bitstrings, supported domain,
  the exact function and proof-unit endpoints. Auxiliary witness layouts are theirs.
- Semantic disagreements go into `verity/ml/circuits/silicon/discrepancy_log.json` (an entry
  with an owner), then to the coordinator. Changing a reference
  implementation resolves nothing by itself.
- Proof gadgets (selectors, quotient/remainder, leading-bit, lookup layouts,
  auxiliary columns) live with the backend or in `benchmarks/numerical/`, not in
  `verity_sp1.proofs`, unless one proves to be a durable backend-independent
  obligation representation.

## Labels every number carries

~~~text
proof_class     ARITHMETIC_DIAGNOSTIC | NON_ZK_PROOF_DIAGNOSTIC | NON_ZK_PROOF
                | COMPLETE_HVZK_BACKEND | COMPLETE_ZK_BACKEND | NO_PROOF
authentication  included | excluded
~~~

Never compare across `proof_class`, and never compare two numbers that were
not measured **at the same batch size, on the same hardware, at the same
soundness target** (`contract.IDENTITY_FIELDS`; `benchmarks/judge.py` refuses
otherwise). The decision workload is B=4096 K=1536 proof units of one input set on
one machine at 2^-128; the canonical comparison is the three frozen tables
below. Transition-unit and small-B numbers are diagnostics of a
backend's own scaling, not a comparison. A non-ZK kernel is a diagnostic. An
implementation whose privacy is honest-verifier zero knowledge is
`COMPLETE_HVZK_BACKEND`: a valid milestone, not the target.
`COMPLETE_ZK_BACKEND` is reserved for a construction whose malicious-verifier
privacy (the verifier-coin-commitment step or another justified mechanism) is
implemented, with its transcript order, what the verifier commits to, the
binding requirement, message timing, abort/restart rules, privacy argument and
extra round/communication/prover cost written down and red-team reviewed. The
expected path is: arithmetic diagnostic -> complete HVZK -> malicious-verifier
ZK. Guest-cycle shares are not proving-time shares: measure wall time, memory,
shards and fixed costs directly.

## Canonical tables (frozen 2026-09-22)

Rendered from the evidence store by `python -m verity_numerical.bench.tables
--root <store> [--format md|json] [--snapshot]` (`verity_numerical/bench/tables.py`).
These are the only canonical tables; everything else (memory, committed
elements, hash bytes, verifier seconds, transcript bytes, the RTT sweep,
coupling diagnostics) is drill-down behind a Table 2 or Table 3 cell, never
another top-level table. Caption constants: K = 1536; B = 4096; soundness
2^-128, target and achieved (union-bounded over sub-batches); dense execution;
FP32 accumulation; FMA = 2 FLOP; the frozen input set of the row's subcircuit;
native denominator = datasheet dense peak of the exact target device
(`Target.native_peak`); authentication = excluded (until `included` runs exist).

~~~text
Table 1  Candidate proof systems (definitional: no performance, no status)
         Candidate | Description | ZK | Verifier model | Soundness | Setup | Interaction | Security assumptions
         rows = tables.CANDIDATES: A-GKR (NON_ZK_PROOF), B-Ligero (COMPLETE_ZK_BACKEND; COMPLETE_HVZK_BACKEND
         as drill-down), SP1 (any sound class); ZK is No | HVZK | Malicious-verifier ZK (two variants allowed)
Table 2  Headline proving overhead by target (one row per verity.ml.circuits.targets.target.TARGETS)
         Device | Datatype / target | Native dense peak, spec | Native dense peak, measured | A-GKR | B-Ligero | SP1
Table 3  Prover phase decomposition (one row per populated Table 2 cell, the same artifact)
         Device / target | Candidate | Total prover time | Witness | Encoding + commitment | Arithmetic | Lookup | ZK additional | Serialization | Other
~~~

Cell = the best valid implementation of the backend family in its declared proof
class: min `t.total` over valid results, printed as R_native_spec / R_proved
(`2.6e7×`), R_proved = 2KB / `t.total`; R_proved and the artifact id stay in
the JSON and in a numbered markdown footnote; `—` = no valid result. The
measured column is `292 T (0.94)` from a `native-peak/v1` artifact (meta:
device, dtype, accumulate fp32, method "cublas dense gemm 8192^3, median of 5",
flop_per_second, spec_flop_per_second, fraction_of_spec, sass, driver, cuda)
matched by normalised device + dtype; `…` when absent. `--snapshot` freezes the
cell artifacts as a `dataset-snapshot/v1` named `tables-<UTC date>`.

Same-SKU invariant: the prover ran on the row's native device, `hardware.gpu.name`
≡ `native_peak.device` under `tables.normalise_sku` ("NVIDIA A100-SXM4-80GB" ≡
"NVIDIA A100 SXM4 80GB", "NVIDIA H100 80GB HBM3" ≡ "NVIDIA H100 SXM5 80GB"); a
run on any other device never populates a cell.
Validity (`tables.reject_reasons`; empty = valid): `contract.validate()`
empty; `profile` is the row's subcircuit; K = 1536 and B = 4096 (B_real = B when
present); `security.target` ≤ −128 and, when present, `security.achieved_log2`
≤ −128; `instances` equals the subcircuit's frozen ref (BF16: `bench-instances/v1`
`vu-k1536` [0, 4096) manifest 059103cf…; FP8 Ada: `bench-instances-fp8-ada/v1`
`vu-k1536-fp8-ada`, digest `contract.FP8_ADA_INSTANCES_DIGEST`; Hopper has no
frozen set yet), or is a re-packing of it that an `instance-equiv/v1` artifact
(campaign morning-tables BRIEF §5) shows byte-identical, labelled
`verified=accepted` by a non-producer of it and of the result (user decision
2026-09-24; the cell's footnote names it); `proof_class` is the backend family's
declared class; authentication excluded (column B-Ligero + in-proof hash:
`included-hash`, Poseidon2-BabyBear per row, no sharing; Ajtai, BLAKE3 and
shared tiles are drill-down); same SKU; **independently verified** -- a label
`independently_verified=true` (the store's back-fill key; `verified=accepted`
is also accepted) whose `by` is no producer: not the producing attempt's
campaign, an asserter of the result's own `candidate`/`label` labels, the lane
a `lane` label or its meta (`lane`, `by`, `registered_by`, `constructed_by`)
names; an attempt-less result naming none is never verified; **not
downgraded** -- the latest `red-team-*` `proof_class` verdict on the artifact
is not below the fingerprint's class; **not pulled** -- the latest `finding`
label on the artifact does not start `PULLED` (one starting `UNDER
RE-VERIFICATION` keeps the cell, marked `(under re-verification)` with the
finding in its footnote, until a later `CLEARED` label). A result's backend family is recognised from
`software.backend.name` (`verity-gkr`, `b-ligero chain…`, `sp1`); component
benchmarks (`verity-direct`, `archive/gkr/...`, `b-encode`) are none. The
renderer lists every rejected result with its first reason.
Table 3 identity: T_total = witness + encoding_commitment + arithmetic +
lookup + zk_additional + serialization + other, each cell `0.41 s (37%)`,
Other = t.total − Σ buckets − Σ joints (≥ −1 %·t.total); a fused stage
`t.joint.a+b` prints once as `a+b: 0.41 s (37%)` in the first bucket column it
covers and `↞` in the others.

Drill-downs D1-D3 (`python -m verity_numerical.bench.drilldown --root <store>
[--format md|json]`, `verity_numerical/bench/drilldown.py`): D1 backends and
statements with their security properties, one row per `drilldown.VARIANTS` entry (its recogniser and
its ZK / operands hidden / binding / soundness / Table 2 column, with sources);
D2 fastest measured `t.total` per `VARIANTS` entry × subcircuit, marked ✓ or with the first
reason letter of `tables.reject_codes`; D3 communication and verifier cost per
Table 2 cell, read only from the cell's result envelope (`…` where absent).
Non-canonical; layout fixed 2026-09-24 (campaign morning-tables BRIEF §1).

The approved spec's tables (`kb/TABLES.md`: Table 1 backends and their
security, Table 2 end-to-end N ÷ P per subcircuit, hardware class and
commitment scheme over the full committed relation on a core-defined scheme,
results on an algebraic hash flagged and marked `(alg.)`, Table 3 with
serving and proving overhead) render as a labelled preview with
`python -m verity_numerical.bench.views --root <store> [--rules spec|unfiltered|legacy]`
(`--format entities`: the docs site's JSON, with ids from `verity.census` and `verity.core.claims`);
`--parity` checks that the legacy-rules views select exactly today's cells
(the switch's gate; `--take-snapshot --record` stores the check as a
`tables-parity/v1` artifact). The steward renders it as a preview for a
`[[render]]` entry with `preview = true`. It replaces the canonical tables only
at the labelled switch: an entry with `published = "views"` publishes
`views --published` as `<stamp>-tables.md` and the frozen tables as the
drill-down `<stamp>-tables-frozen.md`; `python -m verity_numerical.bench.switch`
renders the switch digest (`benchmarks/numerical/CHANGELOG.md`). Since
2026-09-26 the docs site replaces the markdown tables for reading (Daniel):
the steward's entry is `published = "entities"`, which renders only the
entities JSON `<stamp>-tables.json` (the site's input, mirrored to the
Project store's `internal/tables-render/latest.json`) and runs `--parity` as an
internal correctness gate that publishes nothing. Renders land outside the
notes (the entry's `out`, absolute or under `$RESEARCH_RENDERS`): they are
regenerated from the store on demand, never committed.

## Proof model: interactive is fine, transferability is not required

Decided 2026-09-21 (user). The Verity verification is an interactive session
with one verifier; a proof need not convince a third party afterwards. So
**designated-verifier** protocols (VOLE/IT-MAC ZK: QuickSilver, Antman-style
SIMD compression) are in scope and are labelled with
`zk_mode = "designated-verifier"` in the fingerprint, never mixed with
publicly verifiable results in one ratio. Their assumption set is: LPN with
regular noise over a large field (the sVOLE expansion, the only assumption
beyond symmetric-key), one batch of base oblivious transfers (any public-key
assumption, post-quantum options exist), and a correlation-robust hash for OT
extension; no trusted setup, no pairings, no knowledge assumptions; soundness
of the online phase is statistical given the correlation, and the
correlation is input-independent (preprocessable). **LPN awaits the user's
ratification as an acceptable assumption**; the earlier "hash-based only"
requirement is what excluded it. The VOLE backend built meanwhile
(`archive/vole`) is frozen.

## Soundness is a parameter

No error target is baked into a protocol. The per-batch error is derived from
the application budget (sum over proofs plus the other protocol errors), and
`verity_numerical.security.accounting` union-bounds mechanically over every
proof and sub-batch. Reported operating points:

~~~text
2^-80        diagnostic / sensitivity point (never the default claim)
2^-96/-112   optional intermediate sweep
2^-128       primary matched-security comparison point
~~~

Keep explicit in every table which configuration first meets the operating
point: for the frozen subcircuit the quadratic Goldilocks challenge extension
cannot reach 2^-128 at any batch size (the lookup identity term floors near
2^-109 from the 65,536-row tables); the cubic extension is the first that does.

## Latency is a first-class axis

For each backend report, beside prover compute at zero-latency transport:
verifier compute; **sequential challenge depth** (network round trips on the
critical path after all available parallelism and batching -- distinct from
the count of protocol rounds, since independent messages issued before the
next challenge cost no extra trip); total communication; and predicted or
measured end-to-end wall time at RTT in {0, 0.1, 1, 10, 50} ms. The network
floor is depth x RTT. No hard round budget yet: the measurements decide.

## Tensor cores are the target machine

Decided 2026-09-21 (coordinator, on the user's direction). The prover runs
on the same class of hardware as the computation it proves, and its
first-class consumer is the **tensor core**, not the SIMT core. Consequences:

- Cost models and the census are stated in tensor-core terms: a field
  multiplication is priced as UINT8/INT8 limb products with INT32 accumulation
  and deferred reduction (TensorZKP, USENIX Security 2026, is the calibration:
  sumcheck and linear-time encoding as matrix multiplication), not as 64-bit
  SIMT multiplies. A 64-bit element is 8 limbs (64 limb products per multiply),
  a 31/32-bit element is 4 (16). The field question was therefore reopened and
  re-priced (`note:r20-proof/tensor-cost/20260922T0450Z-report-tensor`, same day): 31-bit fields
  win every cell (A 1.1-1.2x, B 1.6x at 2^-128) because a 2^-128 challenge is
  24 limbs in either family while base and trace-times-challenge products are
  4x and 2x cheaper. **Tensor-core kernel work uses BabyBear (degree-6
  extension at 2^-128; 2^27 two-adicity; TensorZKP's layout).** The Goldilocks
  CPU implementations stay as algorithmic diagnostics. Realistic utilisation
  is the calibration to replace first: 0.172 of INT8 peak (TensorZKP's
  degree-2 sumcheck) is assumed until a kernel of ours is measured.
- A prover design must say which of its inner loops are matrix
  multiplications (sumcheck rounds, encoding, LPN expansion, batch checks)
  and which are not (hashing, hint generation), with the non-matmul part's
  share of the wall time. Anything on the critical path that cannot be a
  matmul is a cost to justify.
- Nothing is hashed inside the circuit unless a design shows it must be; the
  commitment layer is chosen so the Merkle hash runs natively (Blake3/SHA on
  SIMT at tens of GB/s is fine when there is no recursion).
- CPU measurements remain valid **diagnostics** of algorithmic scaling
  (committed elements per proof unit, depth, constraint counts); the decision
  hardware is a tensor-core GPU, and `overhead.vs_native_peak` is always
  against the proved subcircuit's anchor tensor-core peak (`Target.native_peak`),
  never against the prover's own card.

## Trust: when we trust a silicon circuit

Decided 2026-09-22 (lane `trust`). For every numerical target (a `Target`: one
subcircuit's exact tensor-core relation on one architecture) "do we trust this circuit, and why"
is **computed from evidence, never asserted**: `assess` in `benchmarks/tc_probe/trust.py`
evaluates predicates over store artifacts (ids + labels) and repo fixtures and
returns a level with the exact evidence that satisfies or fails each criterion;
`Instruction.status` is reported next to it so the two can disagree visibly.

- **UNPROBED** -- no hardware evidence for the instruction on the anchor SKU.
- **HYPOTHESIS** -- a model agrees somewhere (any SKU, or one family of a
  sweep), or is inherited from a sibling instruction that does.
- **PINNED** -- on the anchor SKU: `tc-evidence/v1` with 0 mismatches over
  >= N_min = 10^7 elements (one default `tc_probe --sweep` is 2.5e7; the
  1.4e5-element FP8-family run alone cannot pin) covering the canonical
  families (random sweep, adder-width, zero-products, accumulator LSB walk,
  specials) and the regression tile of every resolved D-entry naming the
  target; SASS shows one MMA-class instruction or the lowering is documented.
- **TRUSTED** -- PINNED and (a) an independent reference agrees (Hawkeye
  source-close, autoproof model5, a second compiled path or harness) or every
  disagreement with it is resolved with hardware as arbiter; (b) every
  discrepancy-log entry naming the target is resolved/excluded; (c) relation
  == model: the checker accepts >= 2,000 honest proof units incl. edge families and
  rejects >= 100 mutations (`bench-result/v1` validation); (d) hardware-in-the-
  loop: the frozen input set re-executed on the anchor with 0 mismatches
  (`tc-evidence/v1` labelled `family=frozen-instances instances_dataset=<ds>
  instances_tier=<tier> mismatches=0`); (e) an independent verifier accepted a
  full-contract proof; (f) no open `trust_downgrade` label.
- **TRUSTED_STABLE** -- TRUSTED on >= 2 physical devices of the SKU or >= 2
  driver/ptxas environments with identical words (the F1 Triton-cache guard).

Trust is re-opened only by an **invalidation trigger**: a canary/probe
mismatch on the anchor, a driver/ptxas/toolkit change on the anchor (re-run
`benchmarks/tc_probe/tc_probe.py --instruction <id> --canary`, < 1 min, exit 1 on
any mismatch, labelled `family=canary`), an independent verifier rejecting an
honest run, a new D-entry naming the target, a red-team finding. The
**procedure** is frozen in code with the criteria (`RELITIGATION_PROCEDURE`):
reproduce on the anchor -> bisect model vs relation vs prover with the stored
tiles -> open a D-entry with the hardware evidence -> the level downgrades
automatically (an open D-entry blocks TRUSTED, a mismatch on the anchor blocks
PINNED; nobody edits a level) -> resolve with hardware as arbiter, the tile
joins `REGRESSION_TILES`, close with `trust_downgrade=resolved:<D-id>`.

Dossier: `python benchmarks/tc_probe/trust.py <target|instruction|--all>
[--root STORE] [--format md|json] [--publish]` renders the criteria table with
evidence ids, D-entries, references, SASS/lowering, the missing items to the
next level, the triggers and the procedure; `--publish` writes a
`trust-dossier/v1` artifact (refs = every cited artifact; labels `target`,
`instruction`, `level`, `anchor`), pushes `--preserve`, and `Instruction.dossier`
/ `Target.dossier` cite it. Label contract: the docstring of `trust.py`.

## Census: every backend family, the same columns

~~~text
materialized witness elements   committed elements     bytes encoded
hash/commit work                field operations       memory passes / bytes moved
sequential challenge depth      communication          peak memory
~~~

A difference in one column (A-GKR commits ~4x fewer elements than B-Ligero) is an
explanatory variable, not the verdict.

## Workloads and the decision

~~~text
one group               debugging
16-product update       first complete integrated checker
K=1536 GemmCoordinate   the architecture decision (handoff section 9.5)
~~~

The headline statistic is throughput, `vu.per_second`, and better still
**overhead**: `overhead.vs_native_peak` = R_native(target) / R_proved, where
R_proved = 2·K·B / `t.total` is the prover's own FLOP/s over the proof units it
established (`rate.proved_flop_per_second`, K MACs = 2K FLOP per unit) and
R_native is the datasheet dense tensor-core peak of the **proved subcircuit's**
conformance anchor at FP32 accumulate (`rate.native_peak_flop_per_second`,
`Target.native_peak`, `contract.native_reference(profile)`): A100 BF16
312 TFLOPS for FIRST (9.85 ps per K=1536 proof unit, ~1e11 units/s), RTX 4090 FP8
330.3 TFLOPS for FP8_ADA_MMA, H100 FP8 1978.9 TFLOPS for FP8_HOPPER_WGMMA.
Emit all three with `contract.rate_measurements(...)` on every proof-unit-level
number under the fingerprint's `profile` (the validator recomputes them);
the question every run answers is "how many native seconds does one proved
second buy". Peak is the conservative reference (a real decode step runs
below peak), so the true overhead is lower than the reported one.

Nothing is selected as the winner on the small cases. The decision is the
lowest complete numerical-proving cost at matched soundness, privacy and
domain on the K=1536 proof unit. A batch may be split into independently committed
sub-batches; the result then reports both `B` (proof units per proof) and
`N_subbatches`, the soundness error is the union bound over them, and the
per-unit cost includes every sub-batch -- performance is never improved by
quoting a per-proof error for a chopped batch. Security parameters are stated
per run (`security.target` in the result fingerprint); the protocol documents
are `archive/gkr/PROTOCOL.md` and `archive/direct/PROTOCOL.md`, and the red
team reviews their transcripts before any implementation claims a proof class
above diagnostic. Report the six timing buckets (witness,
encoding/commitment, arithmetic, lookup, ZK-additional, serialization), peak
memory, bytes moved, transcript bytes, rounds, verifier work.

Every A-GKR/B-Ligero number at K=1536 is measured on the frozen input set
`bench-instances/v1` (`fixtures/bench-instances/v1/`, loader and recipe
`verity_numerical.bench.instances`): the `vu-k1536` tier, 4,096 valid proof units
(3,456 real Qwen2.5-1.5B coordinates including the 24 hardware-confirmed
captures, 640 adversarial-but-valid corners, interleaved so every prefix is
representative). B=1, 64, 4096 are the id prefixes `[0:1]`, `[0:64]`,
`[0:4096]`; `B_proof` and `N_subbatches` are contiguous id ranges of it, and
the result fingerprint's `instances` field records which (`contract.instances_ref`).
The judge refuses two results whose `instances` disagree and warns when either
lacks it. Negatives (`vu-k1536-neg`, each tagged with the implementation
mistake it equals) and the transition-unit tier (`tu-k16`) are for the
checker, not for timing.

The public pole-redraw rule of the lookup argument (redraw `z_T` only when it
lies in the public table image) is provisionally accepted; the red team checks
that the condition depends only on public data, that the transcript records
redraws deterministically, that no prover-controlled restart exists, that no
other zero denominator can induce a witness-dependent abort, and the soundness
bound is stated for the conditioned challenge space (uniform over the
complement after rejection), not reused from a whole-field theorem.

## Budget

The campaign's pods (`vy-*`) are under the `verity-campaign` cap: **$120
total, $5/h ceiling** (2026-09-21). Concurrency is not expanded because budget
is available; the $5/h ceiling rises only when several validated lanes are
blocked on independent hardware work. **Hard review point before WP5 GPU
optimisation**: the evidence then decides whether further GPU spend is
justified; nothing is pre-authorised. Foreign `r18-*` pods are outside both
lifecycle control and campaign accounting.

## D-SP1 is the system-level reference

`3.58 s` core and `4.46 s` compressed for K=1536 (`archive/sp1/benchmarks/dot_product/baselines.json`)
are the end-to-end path a numerical backend is eventually compared against, in
addition to the isolated relation. Do not describe D-SP1's cost as
authentication-dominated: the measured guest is 82% statement parsing and
opening-table construction, 4.2% relation, 1.3% authentication hashing
(`r20260921-224405-22dc`). A true relation-only D-SP1 diagnostic (statement and
table construction removed too) is wanted for a fair arithmetic baseline; it
blocks nothing.

## How work is done

- All measured work is a recorded `research` run on a pod; proving and GPU
  probes never run on a laptop or the control pod. `--exclusive` on shared
  pods. Agents do not create or terminate pods; the coordinator does, and the
  campaign's pods (`vy-*`) are under their own budget cap.
- Commit before `--source .`; ignored files are not shipped.
- A lane's exit gate is tests + recorded run ids + the concrete unresolved
  items. `main` is the integration point; another lane's branch is not
  canonical.
- Rust for anything whose speed is the result; Python for references,
  checkers and adversarial search. CPU baselines first, GPU after parity.
- New subcircuits go through the benchmark spine. Input sets are
  `input-set/v1` / `vllm-vu-set/v1` directories read by id
  (`verity_numerical.bench.generate` makes synthetic ones from the IR). A
  family lowers template T in its own `templates/<t>.py`
  (`verity_numerical.bench.lowerings`), and
  `python -m verity_numerical.bench.cell plan|run|check|register` runs a
  (backend, statement, input set) cell through the backend's `bench/drivers/`
  module.
