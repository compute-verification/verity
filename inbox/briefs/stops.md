# STOPS: run the two-stage audit's whole chain at the smallest configuration and list every stop before an accept

**Why.** Nothing has run the two-stage sampling design end to end. Today the second stage has run only on a CPU at small sizes and stops at a refusal. Daniel left it to me whether to bring the end-to-end run forward, and I did: the vLLM lane is asked (Linear VLLM-339) for one run of the whole chain to an accept, at a small configuration, by Thu 12 Nov 6 PM PST. You find the stops on our side now, so each has an owner today.

**What is known to stop it** [from this morning's readings; check each by a run]:
1. **The form's switch.** The Lean verifier refuses the two-stage form last (`FORM := false`, `verity/protocols/spot/verifier/lean/Spot/SecondStage.lean`, about `:135-138`), by the rule that a verifier gains no form ahead of its proof. Workers WIRED and FRAME are on the proof.
2. **`second-wires` at the served block's shape** (worker GATHER).
3. **The coin limit** (pull request #8914, being landed).
4. **The vLLM lane's second-stage command is not on `main`** (pull request #8875, head `5dd58e0400d7`; its runs at small sizes: `d417`, `e5c6`; `08a4` refused at a table of 2^41 bits).

**Your job:** on a branch of your own that is never merged and never opened as a pull request, build the Lean verifier with the form's switch on, and run the whole chain at the smallest configuration #8875's command takes (serve or its small stand-in, the boundary commitment, the first stage, the replay, the interior commitment, the second stage, the proofs, the Lean verifier), through `research run`. List every refusal and failure in order, each with its file and line, its owner (a named worker of ours, the vLLM lane, kernels), and whether a fix exists on a branch. Then apply the fixes that exist (WIRES' two checks, GATHER's wire, #8914, #8940) on your branch and run again, until the run reaches an accept or a stop nobody owns.

**Label every output** "the form's switch on, ahead of its proof; not `main`'s verifier". An accept from this branch is evidence about the stops, never a result.

**Deadlines (UTC):** 19:30Z first entry: the first run's id and the stops so far; 22:00Z (2 PM PST) the list, which I give the vLLM lane; then an entry at each rerun; Thu 12 Nov 20:00Z the state for Thursday's run.
