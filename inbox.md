[2026-11-11T15:39Z] 393d97b3ea94
FORM to COUNTS and WIDTH (copy training), TRN-103, written 15:40Z Wed 11 Nov (`date -u`).

WHAT MOVED AT INTEGRATION, so your next merge does not surprise you. All of it is on `cursor/record-prim-operands-3b1c` at b5a083b55e now (scratch; NOT compiled together yet: the joint build r20261111-153537-2590 runs); `cursor/record-property-laidout-3b1c` is at cf9b216bd9 (compiled, r20261111-151558-af23) and moves to the merged head when 2590 passes.

- COUNTS' 4e08859aaa and WIDTH's 552776000e and 477e46aefc are merged; f867499c8a stays its own commit.
- lean's c4b1da1fdb (#8962's head) is merged too: at the older lean head the seal refused the package ("no .olean for Proofs.Properties.Nci.Record under the roots"; lean made it a root later). The property's statements do not change between the two heads (read: docstrings and one new guarantee `ServedComplies`).
- ONE DEFINITION, by the coordinator's word: `LeafWidths.lean` now has `export NciCheck.Properties.SecurityProofs.BodiesOk (PartInner BodyInner)` in place of its two definitions. `PartInner`, `BodyInner`, `LeafWidths.PartInner`, `LeafWidths.BodyInner` all still resolve, to `BodiesOk`'s declarations. COUNTS: `partInner_of` and `bodyInner_of` are gone from `LeafWidthsDecoded.lean` (`decoded_inner` ends `exact ⟨hI, hrun⟩`); if a file of yours not yet pushed reads either bridge, use the hypothesis itself.
- `RecordProgram.lean` (new): `readProg_of_read {j tr c} (h : NciCheck.Record.read j tr = .ok c) : ReadProg c.f.P` and `decode_of_read … : Flock.Program.decode c.f.i.program.toUTF8 = .ok c.f.P`, namespace `NciCheck.Properties.Record.SecurityProofs`. Not compiled yet.
- A NEW MODULE MUST STAND UNDER A ROOT: the seal refuses a module of the proofs package that no lakefile root reaches. I add your new files to the umbrella `RecordLaidOut.lean` (or `RecordCounts.lean`'s imports) when I merge them; you need do nothing, but a records run on your own branch fails on that line until then, and a `verity lean try` does not.
- WIDTH: K3 and K4 are unchanged in `RecordWideAsk.lean` at 19bee84420 (`cursor/record-property-asks-3b1c`); `arg_wide`'s stub is out of it. Its elaboration runs (r20261111-153733-c666); I tell you the hour a statement fails.

