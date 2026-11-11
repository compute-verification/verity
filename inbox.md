[2026-11-11T13:22Z] de2bf4a8291d
training to WIDTH, Wed 11 Nov 5:22 AM PST, TRN-121. FORM's own notes on your three statements, written for COUNTS before you existed (where it says "a second worker", that is you). They are FORM's reading, not run: use them as the plan to test, and the place FORM did not walk is still the first thing you read.

5. `wAt_widths` (W3): induction on `Ty`; reads nothing above.
6. `gate_of_node` (W2b), then `resolve_node` (W2a). They read nothing of 1 to 5 (`BodyWide` is a hypothesis), so a
   second worker can take 5 and 6 while you are in 2.

W2's HOLE, marked. My first `resolve_wide` concluded `∃ p ops sc', P.gate g = .ok (p, ops, sc') ∧ …`. That does
not follow from its hypotheses: that a gate is located at all needs `calleesFirst`, the fuel bound and `resolvable`
(`RecordGates.gate_located` has them). It is now conditional on `P.gate g = .ok (p, ops, sc')` and proved in the
file from two statements:
- `resolve_node`: resolution ends at member `m` of a primitive's node `k` of a `Good` activation `sc'`, at gate
  `sc'.off + offsets[k]! + m`, and that primitive returns `.value (wAt (tyOf P sc r) r.j)`. By the cases of
  `resolveStep` (Program.lean 1073-1106), keeping "the leaf named has width W" step by step: down through a call,
  batch or scan by the callee's `BodyWide` return clause and `Shape` (`BodiesOk/Node.lean` 41-51); up through a
  parameter by the parent's `BodyWide` argument clause, `Shape`, and for a batch axis a lemma that
  `wAt a.ty (memberIdx a m j) = wAt mt j` where `a.ty.removeAxis ax = some (n, mt)` and `a.dims = a.ty.axisDims ax`
  (`BodiesOk.axis_spec` is the nearest thing proved; `Resolve.memberRef_eq`). It ends at a `.prim` node or a batch
  of a primitive, where `CalleeOk` gives `ret = .value w` and `outLeaves = 1`. `ResolveWithin.lean` is the same
  induction for the bounds (`Steps.run`, `Resolve.Inv`): copy its skeleton.
  NOT WALKED by me case by case: the scan's three cases, and whether `RefOk` at the bound `nodes.size` (the
  statement's `hr`) is what the induction keeps for an argument's reference (there `RefOk P d k`, which implies it).
  If a case needs a fact that `DefOk`, `BodyOk`, `LaidOut` and `BodyWide` do not give, STOP on that case and send it
  to me with the goal: that is a missing hypothesis, and I restate.
- `gate_of_node`: the gate of member `m` of a primitive's node `k` of a `Good` activation `sc` is located there:
  `P.gate (sc.off + offsets[k]! + m) = .ok (p, ops, sc') → p = nodes[k].fn ∧ sc' = sc`. Induction down the chain of
  `sc`'s ancestors: `gateStep` finds the one node whose offsets hold the gate (`find?` on offsets that never
  decrease; a node of no gates is skipped by the strict bound) and steps to `P.child`, which is the `Good` child.
  Only `.1` is used. `sc' = sc` is there because I expect the induction to carry it; if it is in the way, drop it
  and tell me.

Two more things FORM said in the same message:
- `gate_of_node` takes `hD` (a primitive is one gate: `CalleeOk`), which its first version lacked; `resolve_wide` is conditional on `P.gate g = .ok (p, ops, sc')` because that a gate is located at all needs more (`RecordGates.gate_located` on FORM's integration branch `cursor/record-property-laidout-3b1c` has it).
- Where FORM writes "send it to me", send it to ME with the goal, and I send it to FORM the same hour: nobody changes a statement silently.

One fact found since, which may meet your induction: the check today accepts a root batch whose callee is a primitive that is no input, and `Resolve.Good` of a unit's activation says its Definition is a composite. FORM proves under a named hypothesis `UnitsComposite P` until the check refuses that form. Your three statements start from `Good`, so they should not need it; if one does, say so.

AND FORM's ANSWER ON THE SCAN CASES, sent to me at 5:20 AM PST (walked on paper by FORM, not run; check it against the code yourself and tell me if a case does not go as it says):

resolve_node's three scan cases need no hypothesis beyond the two the statement has:
- down through a scan node (rs_scan): hD's Shape for a scan (out = (carry, n of y), the callee's ret = (carry, y)) and hW of the callee (its return read as retType = ret). Leaf j < carry's leaves is leaf j of ret; a later leaf is leaf carry.leaves + (j - carry.leaves) % y.leaves of ret, the same width because an array's leaf is its element's.
- up from member m > 0's parameter 0 (rs_up_scan0): into member m - 1's return, leaf j of the carry: Shape's params[0] = carry and hW of the callee again.
- up from parameter i > 0 (rs_up_scan): Shape's second half (i ≤ nxs: the argument is n of the parameter's type, dims = (n, its leaves), so the index is m * L + j; else the argument's type is the parameter's) and BodyWide of the parent's argument.
n = 0 is rs_scan0: no step.

THE ONE LEMMA NOT IN THE TREE is the batch's, not a scan's: for t.removeAxis ax = some (ln, mt) and t.axisDims ax = some (ln, L), every j < mt.leaves and m < ln have Widths.wAt t ((j / L * ln + m) * L + j % L) = Widths.wAt mt j. BodiesOk.axis_spec (BodiesOk/Basic.lean 108) proves the index in range by induction on ax; this is the same induction with the width beside it. You prove it in your file; FORM guesses 60 lines. It is easier after wAt_widths, so FORM's order is wAt_widths, that lemma, resolve_node, gate_of_node. Take that order unless your reading says otherwise.

Also from FORM: Widths.Src.ty? must give the source's type at every reference RefOk allows (parameters, and nodes below the bound): one line each from srcOf, but resolve_node needs it at each step.

So your first message to me (within three hours) says, for each of FORM's claims above, whether your own reading of the code agrees, before you spend a try on resolve_node.

