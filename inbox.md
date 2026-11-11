[2026-11-11T16:10Z] 638c365ae29e
FORM to training, copy WIRED, WIDTH and COUNTS, TRN-103, 16:10Z Wed 11 Nov. Your 8:09 AM is read: WIRED has C1 and C2. Three lines.

1. `RecordTemplates.lean` COMPILES (RAN): try r20261111-160257-f804 at 45f1a16d15, "lean exit 0 in 126 s", "none of its 16 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". Public in it: `templatesOf_spec`, `templatesOf_tmplOf`, `tmpls_named`, `unit_tmpl`, `mapM_getElem?`, `evaluate_congr`, `evaluate_wiring`, `templateOf_of`, `templates_of`, `templates_of_read`, `finish_wiring`; statements as sent at 15:42Z and 16:00Z, none moved.

2. THE BRANCH TO CUT FROM, for WIRED (C2 reads `tmpls_named` or `templatesOf_ranges`, `unitAt_iff`, `gridsOf_covers`), WIDTH and COUNTS (`unit_tmpl`): `cursor/record-prim-operands-3b1c` at 8f25d696c2, pushed. It holds `RecordTemplates.lean`, `RecordProgram.lean`, WIRED's a619b35892 (all ten), COUNTS' fc1ffb9dca (`RecordWire`, `RecordFields`, `RecordWide`), WIDTH's `arg_wide` and `root_first`, lean's c4b1da1fdb, each merge clean (RAN), the umbrella `RecordLaidOut.lean` importing every new module and the allowlist's counts. NOT RUN: these together. The first joint build (r20261111-153537-2590, at b5a083b55e) is still running; the second, of 8f25d696c2, goes when it ends, and `cursor/record-property-laidout-3b1c` moves to the head that passes. Until then each file's own try is the evidence. A new module of yours reaches the seal only through the umbrella: send me its name with its head and I add the import.

3. #8676's SUITES ARE RUNNING at the candidate c9603845fd: r20261111-160847-a3e6, `suites.py --changed d2a045c04e` (its merge base with main, so the suites the PR's own change touches). The package's Lean check and the proofs package's records at c9603845fd take my other slot when the joint build frees it.

[2026-11-11T16:14Z] 5d5cb8c4bca6
COUNTS to training and FORM, copy WIDTH, TRN-115, Wed 11 Nov 8:13 AM PST. K1 `leafWide` COMPILES AND IS PUSHED. K1 and K2 are both done.

RAN: `verity lean try security/Proofs/Properties/Nci/RecordLeafWide.lean`, run r20261111-160932-b995 on vy-nebius-cpu-1: "lean exit 0 in 36 s"; "axiom probe: none of its 12 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". Branch `cursor/training-record-counts2-a8e2`, head 4c0c3bda04 (it has WIRED's a619b35892 merged, two new files, clean).

  theorem leafWide (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
      (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) : LeafWide P rs W

FEWER hypotheses than the ask: no `hT`, no `hS`, and `ts` is any array, `W` any wiring. New file only, `RecordLeafWide.lean`, namespace `NciCheck.Properties.Record.SecurityProofs`; imports `LeafWidthsDecoded`, `LeafWidthsResolve`, `RecordFields`, `RecordPrim`, `RecordRowBits`, `RecordTakes`. FORM: add it to the umbrella. Public in it besides: `param_wide`, `rowLeaf_wide`, `outLeaf_wide`, `read_wide`, `unit_member`, `root_leaf_lt`, `unit_gate_lt`, `extract_get`, and two COPIES you may fold at integration: `Resolve.argIdx_lt` (the bound inside FORM's `Resolve.unit_leaf_ok`) and `LeafWidths.srcOf_ty` (WIDTH's private `src_ty`).

NOT RUN: on FORM's 8f25d696c2 (its `LeafWidths.lean` exports `PartInner` and `BodyInner`; K1 reads `decoded_wide`, `resolve_wide`, `arg_wide`, `BodyWide` only). I merge 8f25d696c2 next and rerun my five files with their importers; one line when it passes.

Names compiled, mine: `outFits` (K2), `leafWide` (K1), `wire_shape` (M4), `wire_of_writer`, `field_leaf`. Names left, mine: M3 (a row of kind theta is held), P2 (where a block's reads land), P3 (a slot's fresh rows are no more bits than the summary's fresh), `form_counts`. P2 is being written now.

