# GATHER: the second stage's design check passes on the served replay block (verifier code and its proof)

**The stop.** On today's served record's layout the two-stage verifier's second stage answers `second-wires` before it reads anything (`verity/protocols/spot/verifier/.../second_stage.py:1158-1160`; Lean `Spot/SecondStage.lean`, `nodeWires` at about `:562-575`). The served token select (`TokenSelect_v4{V=154880}`, inlined) has a `Combine` whose `best` and `idx` arguments each read one leaf of each of several `Chunk` nodes, and `nodeWires` makes no wire of an argument that gathers several nodes. So no served record can pass the second stage, whatever else is fixed. Nobody holds this.

**Sources:** in `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`: `## TWOPROVE, Wed 11 Nov 08:27Z` (points 1 and 4: the work is in `nodeWires`, `wireRuns`, `gather`, `Rows`, `readOk`'s wire case and their lemmas `readOk_wired`, `check_ok_*`, plus the mirror and the difftest); `## ATTACK, Wed 11 Nov 15:07Z` (the run on the mirror at the served layout: links 3, wires none; and: served as one template, not inlined, the wires are empty and the design check passes). The vLLM lane's command is pull request #8875 (head `5dd58e0400d7`), read-only for you.

**First entry, 19:15Z:** the shape exactly (the Definition's nodes, from the served record's program); then two ways with what each costs, and your pick:
- **(1) the verifier derives a wire for a gathering argument:** the Lean of record, its lemmas, the mirror, the difftest. "The verifier gains no form ahead of its proof": say which lemmas must land with it.
- **(2) the served Definition changes so that no argument gathers** (the select as its own template, or the gather as its own proof block): what that does to the proof blocks' count and size, and so to the proving cost and to the largest table.
Start building (1) at once unless your reading says (2) is clearly smaller; I answer your entry by mailbox within the half hour if I am awake to it, so do not wait.

**Then:** the pull request, with a difftest case built from the served record's shape that fails on `main`. It widens what the verifier accepts, so end your turn with a ready line; I put a second reader on it.

**Deadlines (UTC):** 19:15Z first entry; Thu 12 Nov 03:00Z the pull request's ready line, or the exact lemma that stops it.
