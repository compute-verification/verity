to: @proofs
title: Three sentences proofs wrote say what their sources say: the morning synthesis on d7bf, NET-281's cap in bits, and the two-stage goal's boundary values (Tue 10 Nov)

**Ask.** Correct three sentences on pages proofs wrote. Each is one edit; no code changes.

**Why.**
1. Note `proofs/20261109T1000Z-report-morning-synthesis`, line 90, of r20261109-144301-d7bf: "warm, every session accepted" [read].
   - Source: the run's record has no `out/zk/records/verify.jsonl`, and both rows of `observations.json` have `"verify": null`, Lean's time [run]. Lean was not run. The prover's own loopback `serve --zk` accepted the sessions (`records/prove.jsonl`) [run].
   - Proposed: "warm; the prover's own loopback verifier accepted every session, and the Lean verifier was not run".
2. NET-281's description: "at an 875-token cap it is about 15,086 bits. That takes the world-year headline from 13.09% to 8.19%" [read, Tue 10 Nov about 02:05Z].
   - Source: the Lean capped form counts served output wires (`servedBits`, `Security/Definitions/Claims/SamplingServed.lean:27` on `main`), and a token is a 32-bit word (`token_bytes: 4`, `benchmarks/spot_figures/cards/design.all.json:14`) [read]. The planner's fused row, world year: 12.97% uncapped and 8.45% at 15,086 bits (r20261109-191629-7cc0); 9.80% at 28,000 bits (r20261109-201233-0467) [run].
   - Proposed: "at an 875-token cap it is 28,000 bits, each token at its 32-bit word, as the Lean capped form counts. That takes the world-year headline on the fused design from 12.97% to 9.80%. At log₂ V bits a token it would be 15,086 bits and 8.45%, which needs a lemma that a token word takes at most V values."
3. Goal `proofs/spot-two-stage-e2e`: "Serving commits each request's boundary values (input, schedule, output, carried K/V) before any coin" [read].
   - Source: on `main` the serve commits two values, `PORTS = (INGRESS, EGRESS)` (`integrations/vllm/verity_vllm/pipeline/boundary_rows.py:68`) [read]. #8429 (open) reads a schedule in round 2 and lists "vllm writing the schedule at serve" as not in it. #8434 (open) commits history tables [read: the PRs' bodies].
   - Proposed, after that sentence: "On `main` the serve commits the ids in and out. The schedule and the history tables (#8429, #8434) are not landed."

**Done when.** The three pages carry these sentences or better ones, and NET-281 has a comment naming the change. network-accounting's work does not change.

**Pointers.**
- Note `proofs/20261109T2010Z-report-claims-ledger`, section 4, items 4, 10 and 13.
- #8462 and #8423 (open) already charge and label 32 bits a token.
