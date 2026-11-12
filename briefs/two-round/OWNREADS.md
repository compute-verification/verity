# OWNREADS: a session's registered reads are its template's reference reads

Read `COMMON.md` beside this file first, and its last section before anything in this brief: where they differ, that section wins.

**Yours, in one sentence.** Prove the lemma between two statements FRAME has written on the base: from `refReadsOwn h` (decided on what the verifier holds) to `OwnReads h` (the fact about runs the address proof uses).

**Why it is needed.** FRAME's entry `## FRAME, Thu 12 Nov 23:23Z` in `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, section 5 ("The open case, and the lemma it needs"), is part of this brief: read it whole. In short: at an output port that no link or wire names, round 2 of the two-round audit passes a session that reads a value the boundary registration registers in place of its own port of the interior record. Round 2 cannot tell an input port from an output port, so it passes it, and the address proof is false there. What closes it with no new premise on the run: round 2's `second-values` check (`valuesOk`) holds each session's registered reads to its template's reference reads but for the run's fields, so a read's value at a port is the reference's. Then the condition is decided on what the verifier holds: for each template, each reference read at a port no link or wire names has a value that is one of the interior registration's ports. FRAME states that as `refReadsOwn h` and the fact it gives about runs as `OwnReads h`, both in `WiringOk.lean` on the base. The lemma between them is JSON work on `jeq`, `blankReads` and `setKey`, and no lemma on `valuesOk` exists under `security/Proofs` today.

**Your file.** A new file beside `WiringOk.lean` (under `security/Proofs/Flock/SpotFlock/`), named `OwnReads.lean`, a root on a line of its own in `security/Proofs/lakefile.toml`. Its main theorem: `refReadsOwn h = true →` round 2's check passes on a run `→ OwnReads h` at that run, in whatever exact form FRAME's two statements make it. If a statement of FRAME's cannot be met as written, write the smallest change under `NEEDS` for FRAME, with the reason, and go on with the changed text as a hypothesis.

**Reuse, do not restate.** RUNSAS has lemmas about round 2's check and JSON equality on `origin/cursor/proofs-runsas-7e3a` (`RunsAs/Statements.lean`, `RunsAs/Round2.lean`, among them `jeq_self_of`); read them first and import what fits. One vocabulary: a definition that exists under another worker's name is imported, never copied.

**The same lemma later.** FRAME notes it is also what a later widening needs to cover a unit that reads a registered value (weights, a table) at an input port. State your lemma so that case is not shut out, but prove only what `OwnReads h` needs.

**Base.** `origin/cursor/proofs-two-round-base-84b5` at its newest head (it was `241c9bf91ac2` at 23:23Z); merge `origin/cursor/proofs-runsas-7e3a` if you import from it. Merge when they move; never rebase. **Your branch:** `cursor/proofs-ownreads-9a1c` (add a letter if it exists).

**Whose files.** Yours: `SpotFlock/OwnReads.lean`. Not yours: `WiringOk.lean`, `Code.lean`, `AddressOf.lean` and the model (FRAME's), `RunsAs/` (RUNSAS's and RUNCOINS's), the verifier's Lean under `verity/`.

**Builds.** vy-nebius-2, pool `proofs`: `research run --on vy-nebius-2 --pool proofs --as proofs --mem-gb 12 --timeout 3600 --cwd source -- bash -lc 'python3 tools/verity/lean/lean_changed.py --try FILE'`. A slot there has been taking 20 to 40 minutes tonight: launch a try as soon as the file is written and keep writing. Send nothing to cpu-8, cpu-10, cpu-12 or cpu-13.

**Owed.** 20 minutes: first entry (your agent id, your branch, your plan in five lines, and the exact statement you will prove). By 02:30Z Fri: the statement in a try. By 10:00Z Fri: the proof compiled, or the smallest named hypothesis that stands in for the part you could not prove. By 15:00Z Fri: your word on what compiles by Sat 02:00Z.
