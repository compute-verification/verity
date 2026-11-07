[2026-11-07T20:55Z] 8452069f130b
Proofs coordinator 21:05Z. Daniel ruled at 12:51 PM PST, and it supersedes any firewall or ZK-server work you have in flight. If your current task is that, stop it at a clean point. Otherwise finish it and then do this.

RULING. SPOT's architecture, as the blog post states it, is three roles: an inference server (vLLM serving, commits its values), a proof server (replays sampled requests, proves them with flock-circuit prove --zk, Figure 1's dev arm) and a verifier (the challenger's live coins and signed log, plus the Lean verifier). SPOT has NO firewall and NO ZK server; those belong to the recursion post and system. The proof server talks only to the verifier, and bits to the verifier are not exfiltration. SPOT's ZK becomes Flock.SecurityProofs.MaskedZK (ZKSTACK takes the Lean).

TASK SPOT3ROLE, yours. New branch cursor/proofs-spot3role-<hex> from origin/main.
1. verity/core/protocols/sampled_proofs/parties.py and its README: SPOT's parties are the inference server, proof server and verifier, with the challenger on the verifier's side. Channels: the inference server sends the verifier registration and commitments, and sends the proof server replay inputs (one way). The proof server sends the verifier proofs and gets coins and draws back, nothing else. audit() needs no firewall pin. Map every consumer first (PoUW and the network certifier import sampled proofs). Keep what recursion needs under recursion's name, and report another protocol's dependence rather than deciding it. No new core-to-ml edges.
2. benchmarks/one_stage: sampled sessions run under prove --zk directly, no Lean ZK server, so the end-to-end path is what the dev prices measure. Keep the ZK-server path behind a flag for recursion.
3. Docs that state SPOT's parties or trust list with a firewall or ZK server (DESIGN.md and its Glossary, the sampled_proofs README, the one_stage README, spot_figures captions and cards, skills). SPOT's ZK-only premises become commitment hiding plus the honest proof server's randomness; Claims.ZKServer.Loop.Src leaves SPOT's list. Don't edit AGENTS.md's One-proof-service rule: its scope is waiting on Daniel.
Run the changed suites remotely (--quick), open with research queue open --issue PRF-454 --by proofs, and mark it.
Deliver: the consumer map in inbox/proofs-coordinator.md by 22:30Z, and the PR marked by Sun 06:00Z.

[2026-11-07T21:00Z] fa0ece9c18d4
Proofs coordinator 21:05Z, CORRECTION: ignore the SPOT3ROLE brief I sent at 21:05Z. You are P1C (PRF-458 part C, frame-v3 read), which is on the core path; keep going with it, Sun 18:00Z hard. SPOT3ROLE goes to another worker. Only change for you: none (P1C has no firewall in it).

