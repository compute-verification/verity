[2026-11-11T16:14Z] f31090504fcc
COUNTS to training and FORM, copy WIDTH, TRN-115, Wed 11 Nov 8:13 AM PST. K1 `leafWide` COMPILES AND IS PUSHED. K1 and K2 are both done.

RAN: `verity lean try security/Proofs/Properties/Nci/RecordLeafWide.lean`, run r20261111-160932-b995 on vy-nebius-cpu-1: "lean exit 0 in 36 s"; "axiom probe: none of its 12 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". Branch `cursor/training-record-counts2-a8e2`, head 4c0c3bda04 (it has WIRED's a619b35892 merged, two new files, clean).

  theorem leafWide (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
      (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) : LeafWide P rs W

FEWER hypotheses than the ask: no `hT`, no `hS`, and `ts` is any array, `W` any wiring. New file only, `RecordLeafWide.lean`, namespace `NciCheck.Properties.Record.SecurityProofs`; imports `LeafWidthsDecoded`, `LeafWidthsResolve`, `RecordFields`, `RecordPrim`, `RecordRowBits`, `RecordTakes`. FORM: add it to the umbrella. Public in it besides: `param_wide`, `rowLeaf_wide`, `outLeaf_wide`, `read_wide`, `unit_member`, `root_leaf_lt`, `unit_gate_lt`, `extract_get`, and two COPIES you may fold at integration: `Resolve.argIdx_lt` (the bound inside FORM's `Resolve.unit_leaf_ok`) and `LeafWidths.srcOf_ty` (WIDTH's private `src_ty`).

NOT RUN: on FORM's 8f25d696c2 (its `LeafWidths.lean` exports `PartInner` and `BodyInner`; K1 reads `decoded_wide`, `resolve_wide`, `arg_wide`, `BodyWide` only). I merge 8f25d696c2 next and rerun my five files with their importers; one line when it passes.

Names compiled, mine: `outFits` (K2), `leafWide` (K1), `wire_shape` (M4), `wire_of_writer`, `field_leaf`. Names left, mine: M3 (a row of kind theta is held), P2 (where a block's reads land), P3 (a slot's fresh rows are no more bits than the summary's fresh), `form_counts`. P2 is being written now.

