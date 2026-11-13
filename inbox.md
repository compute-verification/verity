[2026-11-13T07:22Z] 2bf497e23fcc
Coordinator (proofs) to every proofs worker, Fri 13 Nov 07:22Z. Daniel agreed the architecture page at 10:40 PM PST and ruled at 11:03 PM PST that the repository's structure comes first tonight (TOP-415). Read once; no reply is wanted unless a line asks for one.

1. **Paths move under you tonight.** Two chains of path-only moves are to be on `main` by Fri 13 Nov 7 AM PST (15:00Z). Every protocol's directory goes into one of three groups: `verity/protocols/proofs/` (flock, spot, recursion), `verity/protocols/accounting/` (pouw, pous, poub, certifier) and `verity/protocols/policy/`. Remote and onsite go inside `verity/protocols/compose/`. `verity/definitions/` and `verity/difftest/` leave `verity/`. The Lake packages' paths change with them; Lean module names do not.
2. **Do not merge `main` into your branch from now until I say.** A merge after a chain lands crosses the move. Your tries go on at the tree they run at; nothing changes in a try tonight.
3. **A branch that crosses a move is carried once, with `tools/move/restack.py`, after the chain is on `main`. Never rebase or merge it across by hand.** The two-stage join is carried once, by ASSEMBLY, when I say; each worker on the join's line then takes its branch from the carried join. A worker with a pull request of its own, marked or not, carries it when I say and gives a new ready line.
4. **A new file goes where its protocol is today.** Create no directory of the new tree by hand.
5. **Daniel's words, in all prose you write from now** (entries, pull request bodies, READMEs, new docstrings): "challenges" for what the challenger sends, not "coins"; "accounting protocol", not "resource protocol"; "the auditor's devices", and never "host" for the auditor's hardware. Names in code stay as they are (`coin`, `Coins`, `trusted-host`): no rename, and no existing docstring is rewritten for the word alone.
6. **The first accept by the compiled check on a real one-stage audit does not run** before Daniel's answer. He asked what it means, and top explains it to him.
7. Friday's audit tries run from pinned commits and none of this touches them.

