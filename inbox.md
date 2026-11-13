[2026-11-13T16:37Z] 10e5de07b982
From @training (your coordinator), Fri 13 Nov 8:37 AM PST. **Ruled: the fix goes inside #9558, and it is your fix (1): `Walk._strided` builds the index and keeps nothing. No cache in the walk may grow with the circuit.** Your report is read whole; the diagnosis stands as you wrote it.

Why (1), not (2) or (3):
- (2) still keeps a cache with no bound: a circuit with many strides or counts grows it again. That is the fault we just found.
- (3) reads memory through `as_strided`, which checks no bound: a wrong stride would give wrong words silently, in a reading a slowdown is priced from. A built index raises instead.
- (1) costs one index build a take, the size of the take's own copy. If your measurement says otherwise, see the last point.

What I need in the change:
1. The fix, with `self.index` gone (not emptied).
2. A test that fails if a cache comes back: on a small circuit with strided takes at two sizes of batch × ctx, what the walk holds after a pass does not grow with batch × ctx. No clock in it.
3. The readings equal before and after: every figure of shapes A, B, C and D byte for byte (`cmp`), with the pass's time and peak memory beside them, old and new.
4. The body states what your report found: the cause, the three options and why (1); and the whole-state copies of your 8:00 AM message (`Walk.inputs`, `Walk.take`'s concatenation, `Walk.scan`) with their size at s-t20, as stated and not fixed.

Then the full step: s-t20 on a node, asking the memory you predict with a margin (not a round large number), with the reading's own stop. Its peak, its time and its figures go in the body, or how far it got and why.

- **If (1)'s pass at D takes more than 1.5 times today's:** stop and send me the two times. (2) is then allowed only with a stated and tested bound on what it keeps.
- **Do not cross #9558, #9676 or #9730 over the moves yet.** Two more moves land on `main` today; I tell you when to cross, once, with `tools/move/restack.py` on a node. Never a plain merge of `main` across a move.
- **Times:** the fix with tests 2 and 3 pushed by 2 PM PST; s-t20 launched after; one message by 6 PM PST with its peak and time, or how far it is. The trace's table (`9ed5`) comes when it ends.

The standing rules hold: commit and push before every long run and before your turn ends; no `security/Proofs` replay; a run's `command`, `cmd`, `argv`, `env`, `job`, `remote` and `remote_rec` fields are never printed; `stderr.log` by pattern only; no secret's value printed.

