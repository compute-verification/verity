[2026-11-13T02:00Z] cceeac294ab9
From FRAME's 01:38Z and 01:46Z entries, ruled:
1. You own `BitsInUnits h` (stated in FRAME's `InsideOf.lean`: a source bit's gate and a written bit's gate, at leaf `u` of member `t`, are gates of the proof block of member `t`'s unit `u`), after `MemberRowed`, `WrittenOnce h` and `RowsArePorts h`.
2. You also own, last, the lowering fact `wiresMove` needs: every bit of an output row port of a unit of a member is a written bit. FRAME states where it is used.
3. `RowsArePorts` from `two h s = some d` is fine as you state it.
4. Your branch adds a root to `security/Proofs/lakefile.toml`. Lean's lead wants a new root as one line in `security/Proofs/Flock.lean` and the lakefile unchanged. FRAME makes that change on its L1 branch; merge it back and drop your lakefile line.

