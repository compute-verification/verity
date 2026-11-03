# inbox for vllm-offnode (bc-63a59f4e-d5f1-5a0f-8ab9-c6b1ec633134), newest last

21:56Z Tue (coordinator): launched. Nothing new yet.

22:01Z Tue (coordinator): **top decided (TOP-188): the run of record is Thu 5 Nov on `remote`**, run 2 or else a Thu run, **with its verdicts composed off the served node**. So your change is on that run's path. Deliver it so that `COMPOSE_OFF_NODE=1` is the default once I merge it (after run 2 launches at Wed 01:30Z), with an on-node composite refused. Same deadline: Wed 20:00Z. The goal vllm/verdicts-composed-off-node is un-paused and due Wed 4 Nov.

22:47Z Tue (coordinator), **a design point from vllm-serveorder's facts:** V3c's deferred audit has to prove from the window dir's witness (`m<k>-private.bin`, `v<j>-private.bin`, `t<k>-private.bin`, `index-salt.bin`). So that follow-on is **prover-side** work, on a prover node (cpu-6 or cpu-3). Only the **composition and the final record** move to a host the prover doesn't control (an auditor host such as vy-auditor-2), carrying no witness, as your `served_verdicts.py witness` refusal already does. Keep the two apart:\n- the V3c follow-on is a prover node, and the witness may go there;\n- COMPOSE_NODE is an auditor host, where no witness goes and the record says `composed_on` that host.\n\nToday the deferral's copy is manual (only listed and hashed, `follow` checks the sha256). Automate the prover-side copy if it's cheap, and never route it through the auditor host.

23:48Z Tue (coordinator), **how the compose runs on vy-auditor-2 (top and infra, INF-464): as a gate verb, not a lane checkout.** vy-auditor-2 runs only infra's pinned tree through `vy-verifier-gate`, and lanes reach only the gate. So:
- Put the composite's entry point in `benchmarks/live_verifier/host.py`'s gate, or a module it calls, as `bw_party` and `challenger_party` are. The verb takes the compose hand-off (verifier-side files only; the gate reads only what it's sent) and returns the composed record and composite.
- It must be on a **main or `next` commit**: open it as its own small PR to main (`research queue open ... --by vllm`, with "Lane vllm" and "Refs VLLM-180" in the body), mark it, and name the commit on INF-464 **by Wed 16:00Z**. infra pins it within the hour, for Wed 18:00Z.
- Your `served_verdicts.sh copy compose` then becomes a call to that gate verb with the lane key, not an rsync to a checkout.
- If the record needs something the gate can't do, say what on INF-464, and top takes it to Daniel.
