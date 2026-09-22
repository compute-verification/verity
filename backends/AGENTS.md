# Backends: rules for the proof campaign

Read the root `AGENTS.md` first. This file is the policy shared by every proof
backend and by the campaign-side numerical work under `backends/numerical/`.

## What is fixed

- The target is `verity.verification.target.FIRST`. Lanes do not change it.
  The backend transition unit is one 16-product update with arbitrary finite
  `c`; the Verity VU is `GemmCoordinate<K>` from `c_0 = 0` plus the epilogue.
- Candidates must agree on profile, input/output bitstrings, supported domain,
  the exact function and VU endpoints. Auxiliary witness layouts are theirs.
- Semantic disagreements go into `fixtures/discrepancy_log.json` (an entry
  with an owner), then to the coordinator. Changing a reference
  implementation resolves nothing by itself.
- Proof gadgets (selectors, quotient/remainder, leading-bit, lookup layouts,
  auxiliary columns) live with the backend or in `backends/numerical/`, not in
  `verity.verification`, unless one proves to be a durable backend-independent
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
otherwise). The decision workload is B=4096 K=1536 VUs of one instance set on
one machine at 2^-128; transition-unit and small-B numbers are diagnostics of a
candidate's own scaling, not a comparison. A non-ZK kernel is a diagnostic. An
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
correlation is input-independent (preprocessable). **Pending the user's
ratification of LPN as an acceptable assumption** before a VOLE build lane
starts; the earlier "hash-based only" requirement is what excluded it.

## Soundness is a parameter

No error target is baked into a protocol. The per-batch error is derived from
the application budget (sum over proofs plus the other protocol errors), and
`verity_numerical.security.accounting` union-bounds mechanically over every
proof and sub-batch. Reported operating points:

~~~text
2^-80        diagnostic / sensitivity point (never the default production claim)
2^-96/-112   optional intermediate sweep
2^-128       primary matched-security comparison point
~~~

Keep explicit in every table which configuration first meets the operating
point: for the frozen target the quadratic Goldilocks challenge extension
cannot reach 2^-128 at any batch size (the lookup identity term floors near
2^-109 from the 65,536-row tables); the cubic extension is the first that does.

## Latency is a first-class axis

For each backend report, beside prover compute at zero-latency transport:
verifier compute; **sequential challenge depth** (network round trips on the
critical path after all available parallelism and batching -- distinct from
the count of protocol rounds, since independent messages issued before the
next challenge cost no extra trip); total communication; and predicted or
measured end-to-end wall time at RTT in {0, 0.1, 1, 10, 50} ms. The network
floor is depth x RTT. No hard round budget yet: Candidate A may reduce its
depth and the measurements decide.

## Tensor cores are the target machine

Decided 2026-09-21 (coordinator, on the user's direction). The prover runs
on the same class of hardware as the computation it proves, and its
first-class consumer is the **tensor core**, not the SIMT core. Consequences:

- Cost models and the census are stated in tensor-core terms: a field
  multiplication is priced as UINT8/INT8 limb products with INT32 accumulation
  and deferred reduction (TensorZKP, USENIX Security 2026, is the calibration:
  sumcheck and linear-time encoding as matrix multiplication), not as 64-bit
  SIMT multiplies. A 64-bit element is 8 limbs (64 limb products per multiply),
  a 31/32-bit element is 4 (16). The field question is therefore **reopened**:
  the 2026-09-21 "Goldilocks stays" conclusion was priced on SIMT cores.
- A prover design must say which of its inner loops are matrix
  multiplications (sumcheck rounds, encoding, LPN expansion, batch checks)
  and which are not (hashing, hint generation), with the non-matmul part's
  share of the wall time. Anything on the critical path that cannot be a
  matmul is a cost to justify.
- Nothing is hashed inside the circuit unless a design shows it must be; the
  commitment layer is chosen so the Merkle hash runs natively (Blake3/SHA on
  SIMT at tens of GB/s is fine when there is no recursion).
- CPU measurements remain valid **diagnostics** of algorithmic scaling
  (committed elements per VU, depth, constraint counts); the decision
  hardware is a tensor-core GPU, and `overhead.vs_native_peak` is always
  against the anchor's tensor-core peak.

## Census, both candidates, same columns

~~~text
materialized witness elements   committed elements     bytes encoded
hash/commit work                field operations       memory passes / bytes moved
sequential challenge depth      communication          peak memory
~~~

A difference in one column (A commits ~4x fewer elements than B) is an
explanatory variable, not the verdict.

## Workloads and the decision

~~~text
one group               debugging
16-product update       first complete integrated checker
K=1536 GemmCoordinate   the architecture decision (handoff section 9.5)
~~~

The headline statistic is throughput, `vu.per_second`, and better still
**overhead**: prover seconds per VU divided by the native seconds the same
VU takes on the machine being proved (`overhead.vs_native_peak`, against
the A100's dense BF16 tensor-core peak: 9.85 ps per K=1536 VU, ~1e11 VU/s;
`contract.NATIVE_REFERENCE`). Report it on every VU-level number; the
question every run answers is "how many native seconds does one proved
second buy". Peak is the conservative reference (a real decode step runs
below peak), so the true overhead is lower than the reported one.

Nothing is selected as the winner on the small cases. The decision is the
lowest complete numerical-proving cost at matched soundness, privacy and
domain on the K=1536 VU. A batch may be split into independently committed
sub-batches; the result then reports both `B` (instances per proof) and
`N_subbatches`, the soundness error is the union bound over them, and the
per-VU cost includes every sub-batch -- performance is never improved by
quoting a per-proof error for a chopped batch. Security parameters are stated
per run (`security.target` in the result fingerprint); the protocol documents
are `backends/gkr/PROTOCOL.md` and `backends/direct/PROTOCOL.md`, and the red
team reviews their transcripts before any implementation claims a proof class
above diagnostic. Report the six timing buckets (witness,
encoding/commitment, arithmetic, lookup, ZK-additional, serialization), peak
memory, bytes moved, transcript bytes, rounds, verifier work.

Every Candidate A/B number at K=1536 is measured on the frozen instance set
`bench-instances/v1` (`fixtures/bench-instances/v1/`, loader and recipe
`verity_numerical.bench.instances`): the `vu-k1536` tier, 4,096 valid VUs
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

## SP1 is the system-level reference

`3.58 s` core and `4.46 s` compressed for K=1536 (`benchmarks/dot_product/baselines.json`)
are the end-to-end path a numerical backend is eventually compared against, in
addition to the isolated relation. Do not describe SP1's cost as
authentication-dominated: the measured guest is 82% statement parsing and
opening-table construction, 4.2% relation, 1.3% authentication hashing
(`r20260921-224405-22dc`). A true relation-only SP1 diagnostic (statement and
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
