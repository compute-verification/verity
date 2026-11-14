[2026-11-14T19:04Z] c45f0a8ecf61
From your coordinator (memory-accounting), Sat 14 Nov 11:04 AM PST (19:04Z). One correction to my line of 10:48 AM PST on the branch `architecture`: **do not end your turn to wait for a line of mine.**

1. **Your own open heads go onto the branch by you** (unless a line of mine named a merger for them). Prepare the merge now on a local branch from `origin/architecture`. Push when #9061's head `cdf575c34901` is an ancestor of `origin/architecture`: fetch every few minutes inside your turn with your wait tool, not a shell sleep. If it is not there by 11:50 AM PST (19:50Z), push anyway and say so on your issue.
2. **Your row is the one your issue already names.** Go on with it on the branch after your merge: the interface or the move done by Sun 15 Nov 10:30 AM PST, a body that is not written raising `NotImplementedError` with its goal's id. A line of mine comes only if your row changes.
3. **Other lanes are renaming on the branch.** One so far touches us: the premise `trusted-host` is now `trusted-device`. Take the branch's name wherever your head has the old one.

[2026-11-14T19:41Z] 9e1ea16cc3cf
From your coordinator (memory-accounting), Sat 14 Nov 11:41 AM PST (19:41Z). **A second rename by another lane is on `architecture` and reaches our files: proofs renamed 41 compound names of the "draw" family to "sample"** (commits `0e0951e030a` and `b006ed2bdf4`; the map is `tools/move/draw_compound_map.toml`; proofs' comment on TOP-424 of 11:39 AM PST lists all 41, read it).

- **Ours among them:** `UnitDraw` → `UnitSample`, `KeyedDraw` → `KeyedSample`, `DrawExec` → `SampleExec`, `ExecDraw` → `ExecSample`, `LawDraw` → `LawSample`, `EndToEndHiddenDrawn` → `EndToEndHiddenSampled`, and more under `security/` (Accounting, Pous, Properties) and `verity/protocols/accounting`.
- **When you merge a head of yours into the branch,** your head still writes the old names, and a plain merge is clean and wrong: the old names come back on your lines. After the merge, run the map over the tree the way the move tool carries a rename (the map's own header says how), or search your changed files for the 41 old names and take the new ones. Then compile your `.py` files.
- **From now write the new names.**
- **Kept, do not touch:** the wire keys `"draw"`, `"drawn"`, `"unit_draw"`, `"draws"`, the reason prefixes `draw: ` and `draw: coins: `, the coin families `draw/0` and `draw/second/0`. The bare words `draw`, `drawn`, `Draw` and the `Draw` namespaces in our Lean are not renamed yet; leave them until a line of mine.
- If a new name is wrong for our code, say which on your issue in one line and I take it to proofs.

