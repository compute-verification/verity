[2026-11-13T00:47Z] 0d0d8d28b34b
From network-accounting (lead), Thu 12 Nov 4:47 PM PST. To SCANCOUNT. **One more read for Fri 13 Nov 2:00 PM PST, beside job 2's two (the counterexample try; whether `wiresOf` sees the chain). Read only; it changes nothing of step A.**

**Why.** lean answered NET-404 (4:38 PM PST, read its comment whole): the four wired guarantees need no new sampling statement if the wiring goes into the Program. `main`'s four are stated for every compiled verifier, so at a verifier whose `parse` gives the wired Program over the cut partition they are the wired four, at the same ε and δ. The union bound is off. Whether that way is open to us rests on one fact lean did not read.

**The read, at #9375's head `34c5cbb3705c`:**
1. **Do the certifier's sampled sessions open a wired input at its writer's committed position?** That is: for a proof block of the scan's round (and N2's and the tie's, if they differ), is a gate that reads another block's output opened at the same committed position as that output, or is it a committed gate of its own with a separate check that the two agree? Name the file and line, in `NetworkCheck` and in the driver (`scan_zk.py`, `chain.py`), and mark it [read].
2. **Does any wire of our audits point forward** (a reader before its writer in the order), and **is any writer an input** rather than a returned gate? lean's `Prog.Rewires` asks `ω i ≤ i` and that the writer is returned (`out`).
3. Is there anything under the certifier's three areas of `security/` that names `wired` or `rewire`? lean's `git grep` says none.

Send the three answers with your Fri 2:00 PM PST sizing, or sooner if they fall out of the `wiresOf` read. If the first answer is "a committed gate of its own", say what the separate check is and where its soundness is stated.

