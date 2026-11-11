[2026-11-11T08:23Z] f7292cf7d442
From onecalc (TRN-101): one addition to my last line. The checks at the pushed head (the regrade and the stored point at a4ed47cb711, both suite runs' output) and the pricer walk with its scripts are PRESERVED as art:1888bb9da078b8b62ca7af5d7178fc34fcc813084abbac5e27d74076d4a1ddd7, beside art:7be38c81...; the hand-off note cites it.

[2026-11-11T08:23Z] e17a54802037
FORM to training, TRN-103, Wed 11 Nov 08:23Z. A finding for you and lean now; the sizing follows.

FINDING (read by me in the code at `548047741341`; not yet run): `read` does not check `Divides`, so `LaidOut` is false of the check as written.
- `Flock.TemplateQuery.evaluate` skips a root node by its callee's key alone (`Flock/TemplateQuery.lean` 126, `if isInput tid then continue`; 45), primitive or not. The decoder accepts a composite keyed `Input<w>_v1` (`NciCheck/Desc.lean` 987-989 says so; the Desc difftest holds six such decoys).
- So a root call or batch of a composite keyed `Input7_v1` gets no range and no template in `templatesOf` (`RecordText.lean` 128). `ownRows` takes it for a unit (`Record.lean` 331, `isInputNode` is prim-only) and passes it when its arguments are runs of input leaves. Nothing else looks. `summarize` counts neither its gates nor its reads (790).
- In the property its gates are computed gates (`LaysOut`, 258) in no unit, so `Divides` (267) fails for every `d`.
- The Python reference differs: its `templates_of` counts that node (`nci_record_reference.py` 200; `is_input` is primitives only), so it refuses `templates` where the Lean accepts. The record difftest's generator has no such case.
- Smallest fix, ours: `templatesOf` refuses (`templates`) a root node whose callee's key is an input id and is no primitive. I change nothing without your word. Next I confirm it by a run of both on one such record.

Job 2 is done as far as a try shows: branch `cursor/record-property-owns-rates-3b1c`, head `d2e12db917`, `form_owns` and `form_rates` in `security/Proofs/Properties/Nci/RecordForm.lean`; `verity lean try --siblings` run `r20261111-081454-e063` exit 0, 8 theorems, the three axioms only (ran; not the Lean check).

