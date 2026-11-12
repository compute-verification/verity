[2026-11-12T22:23Z] d32733c6d306
COUNTS to training, Thu 12 Nov 2:23 PM PST. TRN-86 item 2: **the lock lists `HECharged` and the records run passes** (RAN r20261112-210712-8c6c, vy-nebius-cpu-1, rc 0, tree `7500b9a2b6ce…`, outcome "regenerated").

- `security/Proofs`: "PASS  198509 declarations in 3967 modules; axioms propext, Classical.choice, Quot.sound". `security`: every failure a record difference, then `--update`: "AUDIT: PASS"; `security/Properties`: "AUDIT: PASS". No form failure: the premise rule and the witnesses rule both take `ReadsOn`.
- The record, by the run: `HECharged`'s premises are assumptions `Nci.Cut.Assumptions.Exports`, `…Hardness`; domain `Nci.Cut.NoSharedPair`, `…ReadsOn`, `…ValuesInjective`; open none. RAN (a key-by-key diff of the record against my lock): only the three theorems' records, `ReadsOn`'s digest, and the `reads` of `Definitions.Nci.Cut` (new: `ReadsOn`, `Work.inside`, `workHEOn`) and `Definitions.Nci.Guarantees.Cut` (new: the three statements) differ; no other record. Committed as `d7a3766726f8`, pushed.
- No new `open` or `domain` entry was asked beyond `ReadsOn`.
- Out now, both on vy-nebius-cpu-1, at `d7a3766726f8`: `check.py --build security/Proofs` r20261112-222153-ecb9 and `check.py --build security` r20261112-222153-d70a (the kernel replay). The changed suites go out when one of them ends. Then the draft, unmarked.
- This run is a fast result with no kernel replay; nothing above is a Lean verdict yet.

