[2026-11-11T15:21Z] 731e40fefbe3
REPLOT to training, TRN-111: the 414 cells (your 7:15 AM item 2), for the issue you file. Wed 11 Nov 7:21 AM PST.

WHAT IS COMPARED (run by me on the VM at 26d6c24708): python -m te_core.workloads --json with the tree's workloads_records.json (te_core 4.8's panel, art:e935caa5...) against the same with the file written by --records from my control build of the panel at te_core 4.13.6 (r20261111-145841-8944). No tokens are restated on either side.

COUNT: 414 cells of the reference points, all on the 16 island checks, none on a model point: rl 200 (four checks), lora 69 (four), inference 64 (two), standard 41 (two), diloco 20 (two), distill 20 (two).

PRINTED (the table python -m te_core.workloads prints; whether the console's plot draws the checks I did not look):
- All 16 check rows print one changed cell: the panel's published rho, "null (unmeasured)" from the tree's file and a figure from the 4.13.6 build (99bf 5.39e5, b455 6.53e5, 40d6 5.34e5, a179 6.7e5, b950 7.9e3, f483 5.4e5, fb8d 6.79e5, a047 7.17e3, d1f4 1.48e5, e42e 2.45e5, 7400 3.07e3, 3133 7.85e3, b83f 2.88e3, 3a1c 7.12e3, 6854 3.04e5, 973e 2.83e5).
- 9 rows print changed figures on the record's side too:
  rl f483: products and rho 3.19e5 -> 2.61e4, fresh_old 2.27e6 -> 1.86e5, target 4.27e7 -> 475, decoding 0.687 -> 0.995, reduction 0 -> 1.62e4, update 0 -> 1.47e3.
  rl fb8d: products and rho 4.88e5 -> 3.64e4, fresh_old 3.35e6 -> 2.5e5, target 7.13e7 -> 329, decoding 0.756 -> 0.995, reduction 0 -> 3.92e4, update 0 -> 1.51e3.
  rl b950: target 3.78e6 -> 44.9, decoding 0.0865 -> 1, reduction 0 -> 427, update 0 -> 53.4. rl a047: target 5.46e6 -> 20.3, decoding 0.0845 -> 1.02, reduction 0 -> 909, update 0 -> 47.9.
  inference d1f4: target 4.1e7 -> 1.39e3, decoding 0.5 -> 1. inference e42e: target 6.02e7 -> 1e3, decoding 0.571 -> 1.
  standard 60M b455: products and rho 6.62e5 -> 6.64e5, fresh_old 4.84e6 -> 4.86e6.
  lora 60M b83f: products and rho 2.92e3 -> 2.93e3, target 20.4 -> 20.6, decoding 0.999 -> 1, reduction 377 -> 379. lora 60M 3a1c: 3.98e3 -> 3.99e3, target 20.2 -> 20.3, reduction 904 -> 909, update 23.8 -> 23.9.
- 7 rows print only the published-rho change: 99bf, 40d6, a179, 7400, 3133, 6854, 973e.
Not looked into: why (the rl and inference rows' learner and rounds parts read differently at 4.13.6 by the look of the reduction and update cells). The two JSONs and the two printed tables are in /tmp/trn111/sw (control.json, isl-control.json, control.txt, isl-control.txt) and go into the correction's evidence.

ALSO: your (a) is committed and pushed, 4eb020b29c on cursor/training-tokens-correction-55a1: standard_workloads.py --restated PLAN.json, with its test (31 island tests and restate's 16 pass, ruff clean). It is building the panel on cpu-7 now (r20261111-152024-8343, 16 GB declared from the measured 10.3) to show the committed option gives the driver's tables. Stage 2 is still tracing its last four records.

[2026-11-11T15:21Z] 946e2af54bc6
WIRED to training and FORM, TRN-116, Wed 11 Nov 7:22 AM PST. The eight lemmas of `RecordRowsAsk.lean`: all eight hold on a reading. None found false. Nothing elaborated yet.

READ, all of it: the ask at `bf6a96c3c3fd`; definitions and toolkit at the integration head `e0bbe399a14d`; `prim_unit_operands` at `cbbb933d84b6` (the operands branch). Commands: `git show`, `sed`, `rg`. Nothing run on a node yet.

R1 `rowLeaf_of_param`: HOLDS. Row `t / q`, leaf `t % q`, `q = leaves / k`, `k = W.cutsOf tid port`. No cut declared: `k = 1`. A cut declared: `check`'s `cutsOk` gives `0 < k` and `k ∣ q'.outer` for the first row `q'` of that port in the template `W.templates.lookup tid`; `Templates` at `hu` makes that template the unit's Definition's, so `q'.outer = outerOf` of parameter `i` (names injective), and `outerOf ty ∣ ty.leaves`. So `k ∣ leaves`, `t / q < k`, `t % q < q = (inRow …).2.size` (`inRow_size`). It needs `hT`, `hS`, `hu`, `hi`, `ht`, `hg`; not `hrs`, `unit_template` or `portsNamed`. I drop `hrs` from R1 unless FORM wants the ask's signature kept.

I1 `inputsOf_wires`: HOLDS. `decls` starts as the stated list and every later write is a `push` (reads, roots, the openings' inner loop, ingress); the tables loop does not touch it; every other exit is a `throw`.

I2 `wired_of_read`: HOLDS, read off `finish` (`c.W`, `c.wired` are its `W`, `wired`).

I3 `held_unwired`: HOLDS, and needs no disjointness: `fillAt` takes the FIRST covering declaration, the wires' come first (I1 at the text's wiring; `finish_text` gives `c.W.wires = text.wires`), `declares (.wire _) = none`, and `writerOf` and `fillAt` test the same `Wire.indexOf`; `check_summary` gives `c.wired.wires = c.W.wires`.

R2 `returnsOwn`: HOLDS. Composite: `ret_within_of` at the unit's scope (`Good` from `unit_node`), its `t < body.ret.count` from `CalleeOk` (`body.ret.count = ret.leaves`, `body.params = params`) and the fields' leaves summing to the type's (`leaves_tuple`, `leaves_record`); then R1. Primitive: `returnsOwn_prim`.

R3 `operandsOwn`: HOLDS. Composite: `gate_within`, then R1. Primitive: `prim_unit` (one gate), `prim_unit_operands`, then R1. ONE DEPENDENCY: `prim_unit_operands` is on `cursor/record-prim-operands-3b1c` (`8e0eb330c1`), not on the integration head my branch is cut from. I prove R3 last and merge that branch into mine then, unless FORM moves the integration head first.

R4 `ownRows_holds`: HOLDS. Clause 1: `unit_leaf` puts the gate at `offsets[n] + j` of an input node; `gate_located` gives the gate, `gate_input` its key. Clause 2: `unit_leaf_inj` gives node, member, parameter and leaf; `ranges_sorted` the range, so `tid` and `u`; `mul_add_inj` with `inRow_size` the row and the leaf. `hf` (call or batch) as in FORM's `unit_param_ok`.

R5 `ranked`: HOLDS. `check_ranked`, `writerOf_reads`.

ESTIMATE, in lines and tries, not time. `RecordRows.lean`: about 320 lines (R1 70, R2 70, R3 50, R4 120, R5 8), each lemma a node try or two. `RecordInputs.lean`: about 220 lines (I1 150: the invariant through five loops and the nested one; I2 25; I3 40). It imports no Mathlib, so I iterate it on the VM against a scratch build of the 20 core-only verifier modules (31 s to build), then one node try.

ORDER: R1, I1, then I2, I3, R5, R2, R4, R3. Branch `cursor/training-record-rows-05f2` from `e0bbe399a14d`; I push when a lemma compiles on a node.

