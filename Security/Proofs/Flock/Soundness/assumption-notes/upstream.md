# Upstream status

From `../ASSUMPTIONS.md`.

Whether a dependency proves what these theorems assume, or what they prove by hand, is computed, not written here. The
watch list is the `upstream` section of `lean-audit.json`, one entry per statement:

- `A1-johnson-mca` and `udr-mca`: A1 (proved here from DKT26) and its unique-decoding fallback;
- `collision-resistance`, `keyed-cr` and `rom-cr`: `SHA512CRStrict` and `SHA512CRExpected`, against upstream's generic,
  keyed and random-oracle statements of collision resistance;
- `list-size` and `list-size-code`: the Johnson list bound (`ListSize.closeMsgs_card_le`);
- `tensor-fold`: the fold union in `LigeritoLevels`;
- `novel-basis`: the novel basis (`Arith.Correct.xhat_poly`, `L ≤ 40`).

`python tools/verity/lean/upstream.py Security/Proofs` prints each entry's status at the pinned
revisions: the upstream theorems about it, which are proved, and which nobody has acknowledged. The audit runs that pass
over ArkLib, with ArkLib's own `sorry` baseline, and fails on a proved theorem no entry acknowledges. `--lean`, the
exact pass over everything imported (VCVio and CompPoly too), is run at a dependency bump.
