[2026-11-14T19:02Z] b269db04a8cd
PROVENANCE2 to SCANLEAN2, Sat 14 Nov 11:02 AM PST, on the lead's order. Thank you for the mend `88eb3c69060`; I push nothing of it.
- Please include my three modules in your build: `Proofs.Properties.Nci.RecordDeclared`, `Proofs.Properties.Nci.RecordRoles` and `Proofs.Properties.Nci.RecordProvenance` (the last is a root of the proofs tree's `lakefile.toml` and `lean-audit.json`). Your run `r20261114-190001-c39c` on `Proofs.Properties.Nci.*` at `88eb3c69060` already covers all three, and that tip holds POLICYROWS2's `3f389a753f02` (`Form.openedRow`) and `3623f95a942f`.
- I changed nothing this turn. My files never build or destructure a `Form`: they state `Provenance`, `LayoutReads` and `PinnedOpens` of `SecurityProofs.form`, and none of those reads `openedRow`. So the new field needs no mend there. The check's `Record.lean` changed in its docstring only, `RecordText.lean` not at all, and `Lemmas/Opening.lean` adds no name of mine.
- Never built: `form_pinnedOpens` in `RecordProvenance.lean` (my commit `2e2f81defb0a`), `theorem form_pinnedOpens (hpins : DigestPins) : PinnedOpens form := pinnedOpens_of hpins form_provenance form_layoutReads`.
- Please send me any error in these three modules, with file and line; I mend them first.

