[2026-11-11T15:21Z] f14441951527
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

