[2026-11-13T14:07Z] 1c0c0d177abc
From proofs (coordinator), Fri 13 Nov 14:07Z. Your leg is taken: `79b8b4ec6e22` is in the batch ASSEMBLY merges now. A new order, a small code change with a date: **the profile no longer checks a stratum's `k` against the draw's rule (soundness calls it L1). A pull request by Sat 14 Nov 6 PM PST (Sun 02:00Z). Daniel has that hour from me.**

**What it is** [my sizing of this morning, read by one worker and not by you; check each line]
- The stratum's size, `k`, work and floor are already data in the profile. `verity/protocols/interface/CoreProtocol/Profile.lean` (about lines 70 to 76 on `main` `053b8e50fe8`) adds a second check of `k` against the draw's rule, and `SamplingDesign.ofFlock`. The same comparison is made where the draw is checked (`Commitments/Draw.lean`, about line 326, in the base package `core_commitments`).
- The change: drop that comparison and its Python twin (`profile.py` in SPOT's verifier, about line 689), and move `SamplingDesign.ofFlock` to its one caller (`Spot/Pieces.lean:373`). The profile then names no draw.
- What it changes: the profile's reader accepts a work design whose `k` is not the rule's, for its two callers (the certifier's period check and PoUW's inputs). `Core.WorkStratum.problem`'s record moves in two locks (`security/lean-audit.json`, `security/Properties/lean-audit.json`).

**Step 1, now, by reading on `origin/main`; an entry by 15:00Z**
1. The definition's text before and after, and the Python twin's.
2. Every caller of the profile's reader. For each: does anything it concludes rest on the refusal of a `k` that is not the rule's? PoUW's credit checks the rule itself; **the certifier's period check is not read.** This is the one question that can stop the change.
3. Every listed guarantee that reads `Core.WorkStratum.problem` or `SamplingDesign.ofFlock`, by name, owner and lock, and whether its text changes.
4. The tests and difftest rows that hold the Lean and the Python together on that refusal.

**Step 2, the code, after chain 2 is on `main`** (#9734; I send the word "carry" with FRAME's steps, not before 15:00Z). The change crosses the files chain 2 moves, so no code before it.
- A new branch `cursor/proofs-addrwc-profile-<hex>` off `origin/main`. Push only to it.
- The Lean, the Python twin, the tests, `check.py --update` for the two locks, one records run and the changed suites on a node. No build on your VM.
- The pull request's body names the two callers' owners (the certifier: network-accounting; PoUW's inputs: compute-accounting) and says in one sentence each what they must now check themselves. `Lane proofs`. You do not mark it; name the head to me.

**STOP RULE.** If a caller's conclusion rests on the refusal, or a listed guarantee's text would change, stop and bring me the two texts. That goes to soundness before any code.

Timers of 30 minutes or less, as before. Entry 15:00Z.

