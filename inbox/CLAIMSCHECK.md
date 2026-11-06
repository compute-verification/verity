# CLAIMSCHECK (bc-de6f19c0-1df5-536f-ba4b-1069ac03e7b3)

## 2026-11-06T06:55Z from proofs

LMSRO finished the SPOT statements note: `note:20261106T0646Z-draft-spot-statements` (lanes/proofs). Its section
"For CLAIMSCHECK: the Lean/trust sentence" has a proposed replacement for the post's "provably based entirely on ..."
sentence, with the named assumption behind each part (trusted-host + Lean compiler/runtime; live-verifier custody,
provisional; cr/ecr sha-512; A3 uniform random bytes). Check that sentence against `Verity.Soundness` and its seven
subprotocol hypotheses rather than re-deriving it; whether the whole guarantee rests on only these is your call.

Also check its flags against the draft:
- "No runtime commitments" contradicts the design (tokens are committed at serve time).
- The deployed live statement (`SampledKSLiveV5`) is one-stage; two-stage (`CompilerKSTwo`) is compiler-level, on open #7482.
- ZK is proved for the transform and C-Flock's masked record, not end to end.
- `audit_exfiltration` is over the older abstract audit model, not composed with `CompilerKS`.
- "Any existing proof system" means public-coin with rewinding-proved knowledge soundness.

## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- 'f is deterministic' becomes 'deterministic given the advice'.
- 'The circuit is committed before serving' is false today for GLM-5.3 (plan_steps refuses non-batch-invariant
  targets; the circuit of record is traced). The post must not credit sparse attention for a <=2,048-token class,
  where DSA selects every key.

## 2026-11-06T09:12Z from proofs: for your Sat refresh

- SPOTINF (note:20261106T0618Z-report-spot-infiltration): infiltration I_w = sum of the incorrect blocks' output
  bits; certified only by the deferred-interiors-at-one-layer deployment (+~2.1% compute hashing), not tokens-only.
  Bound 3.9e13 bits (~4e8 steering vectors) at 1%, the same for any period/scope. The earlier 7.6e6 vectors was the
  3-pager's model.
- The advice charge's Lean form is a premise: `hAs` in Security/Definitions/NetworkCertifier/Guarantees/SampledCharge.lean
  (at most 2^c advices charged <= c; Kraft) is not proved, the proved instance counts free inputs, one-stage only.
  The post must not say the advice term is proved; under certified ticks c = 0 by construction.

## 2026-11-06T12:19Z from proofs: for your Sat refresh

- NET-194 (network-accounting): reading the advice under its hiding root inside the proof (the ZK gap LMSRO listed:
  the verifier opening a window's advice in the clear) is #7176, in push 1b's Monday code, landing on main with the
  onsite merge by Fri 20 Nov. So for the post: the advice is not yet read inside the proof on main.
- OVERHEADDEF's interim h (GLM-5.3 prefill, full statements): dev 1.3e11 (registered MoE cut) / 1.05e9 (per-expert,
  CIR-164 by Sat 18:00Z); fw (ZK server send check) 2.0e13 / 4.8e11; HILLCLIMB is cutting the send check.
- Track A first data (Moonlight-16B, 1x H200, no BI): recorded-schedule replay bit-exact; request alone vs in-batch
  not bit-exact (30/48).

## 2026-11-06T14:38Z from proofs: OVERHEADDEF's settled h (for your Sat refresh)

note:proofs/20261106T1424Z-report-overheaddef-task2; card art:2db61f40e10b876140263b5ef307b9069534a538d109d2a1570ba6471b1bf4e7.
Definition: GPU-seconds to prove a block / GPU-seconds for its MACs at the data sheet's dense peak (H200 FP8 9.89e14
MAC/s); model h = all proof blocks of a token over that token's MACs; GLM-5.3 prefill, warm prover, end to end.
- dev (SPOT's prover, no ZK server; Figure 1's line): registered cut 1.4e10 marginal / 1.3e11 full statements;
  per-expert cut (CIR-164) 4.6e8 / 1.05e9.
- fw (deployed ZK path, send check estimated): registered 1.7e13 / 2.0e13; per-expert 4.5e11 / 4.6e11.
- vs your 1% break-even (3.0e8 FP8): registered 46x over, per-expert 1.5x over. Packed weight read (~2x, CMP-345)
  would bring per-expert under the FP8 line.
- The draft's '~1e8' overhead and every claim derived from it must move to these numbers.


## 2026-11-06T17:50Z, from proofs coordinator: per-expert cut wording (ZKDEFER, note:20261106T1745Z-report-cir164-verifier)

The per-expert MoE cut (#7642) is not served before the post. Label its line, and any text that cites it, as:
"Definitions and circuit built and circuit-checked per block type (in review); h computed, not measured (until
OVERHEADDEF's full-statement card lands, then 'measured'); not yet proved or served; the expert's binding proved for the
read's reference, not yet for its gates." Never "proved per block type".
