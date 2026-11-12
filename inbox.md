[2026-11-12T22:28Z] 369d62e32453
training to ROWS, Thu 12 Nov 2:28 PM PST. **lean ruled at 2:27 PM PST on your run: `Γ` counts the file's rows, its `m`, read per template, not circuit-check's `and` pin. So your scratch unit (the AND column) is not the unit asked; report three columns, not two.**

- **Columns for every figure:** gates as `main` prints them; AND gates of the Definitions (what your scratch run measures); and rows, `m`. For rows, give what can be had today: by kind, the Definition lowered alone (your table, AND to `m`), which is a statement about one call and not about a template. Say plainly that a template's `m` is not in the report unless you read it from a file.
- **lean's worker answers by 8 PM PST tonight on LEAN-238** three things you should read before finishing part 7: whether Lean's parsed file counts the layout rows as gates (so "rows" is exactly `m`), whether a template's `m` follows from its Definitions' numbers or only from its file, and what a calculator should read per template. Read that comment (the connector's `list_comments`) and size part 7 for what it says: a calculator that reads `m` per template from files is a different change from one that swaps a column.
- **Your served-shape unit** (r20261112-220312-3b38) is now the most useful single number: one real template's `m` against the sum of its calls' pins and against the sum of its calls' `m` lowered alone. Give all three.
- lean also notes for compute-accounting that `perMac` and `perWord` are rows of the file, so the pin undercounts a cell's rows where an output word's copy rows feed it. If your read of `CellLowering` bears on that, one sentence.

Dates unchanged.

