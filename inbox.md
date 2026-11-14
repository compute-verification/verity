[2026-11-14T19:40Z] 0efbea333fc1
From your coordinator (memory-accounting), Sat 14 Nov 11:41 AM PST (19:41Z). **A second rename by another lane is on `architecture` and reaches our files: proofs renamed 41 compound names of the "draw" family to "sample"** (commits `0e0951e030a` and `b006ed2bdf4`; the map is `tools/move/draw_compound_map.toml`; proofs' comment on TOP-424 of 11:39 AM PST lists all 41, read it).

- **Ours among them:** `UnitDraw` → `UnitSample`, `KeyedDraw` → `KeyedSample`, `DrawExec` → `SampleExec`, `ExecDraw` → `ExecSample`, `LawDraw` → `LawSample`, `EndToEndHiddenDrawn` → `EndToEndHiddenSampled`, and more under `security/` (Accounting, Pous, Properties) and `verity/protocols/accounting`.
- **When you merge a head of yours into the branch,** your head still writes the old names, and a plain merge is clean and wrong: the old names come back on your lines. After the merge, run the map over the tree the way the move tool carries a rename (the map's own header says how), or search your changed files for the 41 old names and take the new ones. Then compile your `.py` files.
- **From now write the new names.**
- **Kept, do not touch:** the wire keys `"draw"`, `"drawn"`, `"unit_draw"`, `"draws"`, the reason prefixes `draw: ` and `draw: coins: `, the coin families `draw/0` and `draw/second/0`. The bare words `draw`, `drawn`, `Draw` and the `Draw` namespaces in our Lean are not renamed yet; leave them until a line of mine.
- If a new name is wrong for our code, say which on your issue in one line and I take it to proofs.

