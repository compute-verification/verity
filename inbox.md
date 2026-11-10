[2026-11-10T08:09Z] bb5b31a89749
From DIGESTMOVE (NET-301), Tue 10 Nov 12:25 AM PST. **The move does send Daniel a DM: one line, "definition `NetworkCheck.Period.auditOf` (`NetworkCheck.Period`) changed", under Security and again under Security/Properties. The brief says `--update --moved` sends nothing.** No guarantee's statement, premises or record changed.

- [run] Fast path with `--update --moved` on the main-based head (r20261110-061749-6028, outcome `regenerated`, every AUDIT PASS): in both Security locks `auditOf` goes 194e926b… → 8c970cee… under its own name; the five leave `reads["NetworkCheck.Period"]` and appear under `reads["NciCheck.DefinitionDigest"]`. The update's review counts it the same: "1 lines say the same statements, printed differently, rehashed, moved or lifted".
- [run] `spec_alert.message(origin/main, that head with its records)`, read on my VM without sending: exactly those two lines, tally "definitions: 1 changed" each.
- [read] Why: a recorded hash covers the names of the constants a definition's term calls (`Facts.lean`, `canonText`), and `auditOf` now calls `NciCheck.DefinitionDigest.closureDigest`. `spec_alert` lists a hash that changed under the same name when a `.lean` file changed, and reads no moves map.
- [argued] An alias `slotEntryDigest` in Period.lean does not avoid it: the alias would rehash under its old name instead. Only declaring the five in the new module under their old `NetworkCheck.Period.*` names would send nothing, and that is against the names posted on NET-287.

I am not stopping: you asked me to build on `checked`, so the head is now 3cc3f48641 on 00d07520b3 (same diff; `merge-tree` against `next` lists nothing) and its runs are in line. The PR opens unmarked as briefed, with this at its top. Say if you want another shape.

