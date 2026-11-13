[2026-11-13T07:03Z] 5fbc9b92ac8e
GROWN, Thu 12 Nov, 11:03 PM PST.

**714f is held to `m-t20-2x-hybrid-grow-depth-18`'s config: "equal, 1 expected difference", with no `record_only` line, where 18b2 needs the one it has. The `-b` copy is gone from the branch, and the held entry and `-18` are equal as before.**

RAN, r20261113-064950-da37 (vy-nebius-cpu-11, CPU, exit 0; "rc held=0 714f=0"). It ran the branch at 4da2178230; the next commit only deletes the `-b` copy. The command is `of_record` itself:
- once on the tree's own `of_record.toml`, for the held entry and `-18`;
- once with `--held`, a temporary data file whose one entry is `[held."m-t20-2x-hybrid-grow-depth-18"]`, naming r20261103-063831-714f (art `24056ffd…`, run.json sha256 `88d66c68…`), with the same `partial` and `step` as the tree's entry and no `record_only`.

No change to `of_record.py`. Outputs: `out/of_record.txt` and `out/of_record-714f.txt` of the run.

| entry, run | verdict | updated | log | named |
|---|---|---|---|---|
| `m-t20-2x-hybrid-grow-depth`, d64c (held, unchanged) | equal, 1 expected difference, 1 named line | 444 of 444 | 47 of 49 | `fwd/theta mul [[]] x4, 4 MACs` |
| `-18`, 18b2 | equal, 1 expected difference, 1 named line | 660 of 660 | 47 of 49 | `fwd/theta mul [[]] x4, 4 MACs` |
| `-18`, 714f (`--held`) | equal, 1 expected difference | 660 of 660 | 48 of 49 | none |

**The lines that differ between 714f's verdict and 18b2's:**
- `log`: 48 of the record's 49 products for 714f, 47 for 18b2.
- 18b2's `named` line: "in the record's step and not the config's: fwd/theta mul [[]] x4, 4 MACs; of_record.toml names it: the program's running total of the loss…".
- Everything else is the same: `start` m-t20 (3669435ebb3d), `steps` 10217, step 3 against the modal structure 10190 of 10217 steps have, `updated` 660 of 660, `E` "608,038,092 B after 3 steps; the record's model, 608,040,849 B after 10217", and the one expected difference, `norm` (148 multiplies by an updated gain, 148 normalisations, 1332 products of the backward's).

**Why** (READ, the two records' `structures`): 18b2's program adds each forward's loss times its share into a running tensor, so its modal step has `fwd/theta mul [[]]` x8 where the config's has x4. 714f's older program reads the loss into Python, so its record has the config's x4 and nothing to name.

The body will list 714f's point as the same job as `-18`.

