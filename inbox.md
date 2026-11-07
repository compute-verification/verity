[2026-11-07T21:01Z] a05480b86837
Proofs coordinator 21:00Z, from ci's reviewer: #7823 and #7833 grow tests/test_retired_terms.py's count of the retired term 'law' in Lean names from 133 to 142. The Glossary term is "sampling design". Rename these on your branches before they land:
- Security/Proofs/Flock/Bridge/DrawnLaw.lean: lawOf (:285), lawOf_cases (:291), drawnFrom_lawOf (:302), lawSized_lawOf (:327) (#7823); coveredLaw_bernoulli (:52), coveredLaw_ok (:58), notBernoulli_drawnLaw (:123), notBernoulli_lawOf (:318) (#7833).
- verity/core/service/sampled_proofs/SampledProofs/OneStage.lean: coveredLaw (:360) (#7833).
For example: lawOf -> designOf, coveredLaw -> coveredDesign, notBernoulli_lawOf -> notBernoulli_designOf. Run check.py --update --moved where a lock lists a renamed name (V5's statement reads lawOf). Regenerate the records, rerun the quick suites, and re-mark both PRs. #7335's restack follows the new names.

[2026-11-07T21:01Z] e42e5195b769
Proofs coordinator 21:05Z. NEXT TASK, after your current one (the retired-term renames on #7823 and #7833, re-marked, and #7335's restack opened). Daniel ruled at 12:51 PM PST:

RULING. SPOT's architecture, as the blog post states it, is three roles: an inference server (vLLM serving, commits its values), a proof server (replays sampled requests, proves them with flock-circuit prove --zk, Figure 1's dev arm) and a verifier (the challenger's live coins and signed log, plus the Lean verifier). SPOT has NO firewall and NO ZK server; those belong to the recursion post and system. The proof server talks only to the verifier, and bits to the verifier are not exfiltration. SPOT's ZK becomes Flock.SecurityProofs.MaskedZK (ZKSTACK takes the Lean).

TASK SPOT3ROLE, yours. New branch cursor/proofs-spot3role-<hex> from origin/main.
1. verity/core/protocols/sampled_proofs/parties.py and its README: SPOT's parties are the inference server, proof server and verifier, with the challenger on the verifier's side. Channels: the inference server sends the verifier registration and commitments, and sends the proof server replay inputs (one way). The proof server sends the verifier proofs and gets coins and draws back, nothing else. audit() needs no firewall pin. Map every consumer first (PoUW and the network certifier import sampled proofs). Keep what recursion needs under recursion's name, and report another protocol's dependence rather than deciding it. No new core-to-ml edges.
2. benchmarks/one_stage: sampled sessions run under prove --zk directly, no Lean ZK server, so the end-to-end path is what the dev prices measure. Keep the ZK-server path behind a flag for recursion.
3. Docs that state SPOT's parties or trust list with a firewall or ZK server (DESIGN.md and its Glossary, the sampled_proofs README, the one_stage README, spot_figures captions and cards, skills). SPOT's ZK-only premises become commitment hiding plus the honest proof server's randomness; Claims.ZKServer.Loop.Src leaves SPOT's list. Don't edit AGENTS.md's One-proof-service rule: its scope is waiting on Daniel.
Run the changed suites remotely (--quick), open with research queue open --issue PRF-454 --by proofs, and mark it.
Deliver: the consumer map in inbox/proofs-coordinator.md by 22:30Z, and the PR marked by Sun 06:00Z.

[2026-11-07T21:06Z] 3400151ceec7
Proofs coordinator 21:20Z: updates to your SPOT3ROLE task (item 2). Daniel's 1:02 PM PST ruling:
(1) Verity's core protocols (PoUW, PoUS, network certification; the remote and onsite deployments) KEEP the ZK server and the recursion layer: they wrap SPOT. Only SPOT on its own, the research release, has the three roles and no ZK server (#7843 records this in AGENTS.md). So don't delete the firewall party. Factor it: SPOT's own parties are the inference server, proof server and verifier, and the core deployment wraps SPOT with the ZK server as now. Keep the core path's behavior and tests unchanged.
(2) The SPOT post won't claim zero knowledge. SPOT's docs state soundness and its trust list only, with no ZK premises.
(3) Heads-up: Daniel wants "sampled proofs" renamed to "SPOT". A separate worker plans that rename as one certified move after Sunday's critical PRs land. Use the current names; the move renames them.

