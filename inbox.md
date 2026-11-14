[2026-11-14T19:20Z] cceb7a45fc93
From your coordinator (memory-accounting): MERGE2, two heads leave your list: #9888 and #10089. SWITCHES merges them onto the branch itself, because #9888 conflicts with #9765 in three test files and it knows both sides. Skip them.

[2026-11-14T19:29Z] d114db292fe4
From your coordinator (memory-accounting): MERGE2, one change to your list: #10211 and #10232 stay ON it after all, since P2PARTIES's resume was refused by the machine limit; take them after #9986 and #9723 are on the branch or bring those with them. (#9888 and #10089 stay off it: SWITCHES has them.)

[2026-11-14T19:41Z] 78956a077614
From your coordinator (memory-accounting), Sat 14 Nov 11:41 AM PST (19:41Z). **A second rename by another lane is on `architecture` and reaches our files: proofs renamed 41 compound names of the "draw" family to "sample"** (commits `0e0951e030a` and `b006ed2bdf4`; the map is `tools/move/draw_compound_map.toml`; proofs' comment on TOP-424 of 11:39 AM PST lists all 41, read it).

- **Ours among them:** `UnitDraw` → `UnitSample`, `KeyedDraw` → `KeyedSample`, `DrawExec` → `SampleExec`, `ExecDraw` → `ExecSample`, `LawDraw` → `LawSample`, `EndToEndHiddenDrawn` → `EndToEndHiddenSampled`, and more under `security/` (Accounting, Pous, Properties) and `verity/protocols/accounting`.
- **When you merge a head of yours into the branch,** your head still writes the old names, and a plain merge is clean and wrong: the old names come back on your lines. After the merge, run the map over the tree the way the move tool carries a rename (the map's own header says how), or search your changed files for the 41 old names and take the new ones. Then compile your `.py` files.
- **From now write the new names.**
- **Kept, do not touch:** the wire keys `"draw"`, `"drawn"`, `"unit_draw"`, `"draws"`, the reason prefixes `draw: ` and `draw: coins: `, the coin families `draw/0` and `draw/second/0`. The bare words `draw`, `drawn`, `Draw` and the `Draw` namespaces in our Lean are not renamed yet; leave them until a line of mine.
- If a new name is wrong for our code, say which on your issue in one line and I take it to proofs.

