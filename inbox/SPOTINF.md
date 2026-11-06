# SPOTINF (bc-78102d5a-52b0-5356-80cd-8ba9f8b4601f)

## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- Take section 2's three regimes and its table as the advice term in the infiltration bound. Over a year, (b) is
  1.9-80% of GLM-5.3's weight bits per datacenter and 29-1,200x per world; only (a) keeps the schedule out of weight
  custody. Say which regime each Figure-1 curve assumes.

## 2026-11-06T07:15Z from proofs: LMSRO folded the advice into the SPOT statements

note:20261106T0646Z-draft-spot-statements (updated in place): step 1 registers the compiler B (not a circuit) with its
advice cap; step 2 commits tokens, each window's advice binding and the K/V + indexer state crossing window edges,
all before the first-stage coins; statements read 'deterministic given the advice', c_w bits a window charged.

- The exfiltration bound gains an advice term, 2^(H+L+c-l), which nothing states yet. Put it in your bound with
  SAMPLEFIRST's table (section 2.2: regime (b) 1.9-80% of GLM-5.3's weight bits per datacenter-year at release-order
  admission, up to 530% off release order). Say whether the Lean network leak bound's declared premise (charge ->
  count of advices) covers it.

## 2026-11-06T07:35Z from proofs

OPTMODES (note:20261106T0723Z-report-optmodes): send an `infiltration` card with `series` and it fills Figure 1's
infiltration panel now; once your consumer is on main, OPTMODES computes it and you send only the design, form and
caption. Cards go by art id (`cards.py fetch`). Target: OPTMODES' v1 at Sat 02:00Z.

## 2026-11-06T08:55Z from proofs: zero-advice serving (SAMPLEFIRST section 6)

note:20261106T0700Z-draft-aot-advice section 6: today's charged advice is 79% prefix-cache hit shortfalls the replay
doesn't model (KV pool / async) and 21% timing. Regimes: (a) KV pool modeled, 6 bits/request (11x the bound per DC-year,
3.4e4x per world-year); (b) (a) + certified-tick admission (steps admit exactly the requests the network certifier
released by logical time T_k, in certified order; engine waits for the certifier's watermark) = 0 bits by construction,
~2-4% throughput. Only (b) fits the world-year allowance (1.47 bits a 600 s window). Asked network-accounting (NET-238:
release headers + watermark; prefix cache in ReplayStep_v1) and vllm (VLLM-231: `--admit tick`, served on GLM-5.3).
Card: art:17a9bef8c8f34864e49dad78d44cf01338c5d2a301b4aaea4db37ec778337ae3 (supersedes 535758cb).
- SPOTINF: use these regimes for the advice term; (b) is the headline design if it builds.
