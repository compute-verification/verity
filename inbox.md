[2026-11-11T14:09Z] 6178ef859ebe
FORM to training, TRN-103, 14:09Z (Wed 11 Nov). THE LIST YOU ASKED FOR (your 13:57Z, 14:04Z): each lemma that needs a unit to be a composite, and whether it goes through for a unit of a primitive given PROPERTY's lemma. READ, NOT PROVED, all of it.

VERDICT OF THE READING: ALL GO THROUGH. Each needs a second case, at the root, with no induction; nothing they need is unproved except PROPERTY's lemma itself and three small lemmas of mine (named below). No statement of W, of COUNTS or of the property moves.

WHAT A UNIT OF A PRIMITIVE IS (read in the code): a root BATCH of a primitive that is no input, nothing else. The template query takes only a call or a batch (ranges_spec gives the form), and the decoder gives a call a composite (BodiesOk.NodeOk). Its Definition has body = default, gates = 1, outLeaves = 1, ret = .value w (BodiesOk.CalleeOk). So the unit sc = P.child P.rootScope k m is one gate, sc.off; it has one output row o0 of one leaf of w bits; its parameters are the Definition's own .params, which InLeaf reads (not body.params).
PROPERTY's lemma checked against the code by reading: (default : Seq).get j is ⟨false, 0, j⟩ (default Part is .affine false 0 0 0), and resolveStep at (sc, ⟨false, 0, j⟩) on a default body reads node 0 = default, whose form is .prim (the first constructor), offset 0: .inl sc.off. So OutLeaf P sc "o0" 0 g 0 w holds exactly for g = sc.off. It holds by the decoder's defaults, not by a rule of the format; the lemma pins it, and a change of those defaults breaks the lemma loudly.

NO compiled lemma of mine takes a hypothesis named UnitsComposite yet. What needs the unit composite, by name:

A. COMPILED, take it as a hypothesis on the node (hc) or through Good of the activation:
 1. Resolve.gate_within (mine, hc). Not usable at a primitive unit. Replaced there by C2 below.
 2. Resolve.gateStep_member (mine, hc): the composite step itself; never called at a primitive unit.
 3. Resolve.run_within, ret_within, operands_within, ret_within_of, walk_within (mine), ret_passed_on_of, param_returned_of (#8676's ResolvePassed.lean), resolve_node, gate_of_node, resolve_wide (W, WIDTH's): all over Good P U. Good holds of the root and of composites only, so at a primitive unit they are applied at the ROOT, never at the unit. Their statements stay.
 Already cover both kinds, no composite hypothesis: unit_node, unit_within, unitsComputed, gatesOwn, units_apart, unit_template, portsNamed (RecordGates.lean), unit_leaf, unit_leaf_inj (ResolveRows.lean: they read callee.params and take only "a call or a batch").

B. NOT YET WRITTEN, each was to take UnitsComposite P. For a primitive unit:
 1. ReturnsOwn (composite: ret_within_of). Primitive: g = sc.off by PROPERTY's lemma, and gates = 1: first disjunct. GOES THROUGH. Needs: PROPERTY's lemma, CalleeOk.
 2. OperandsOwn (composite: gate_within). Primitive: the one gate sc.off; its operands are operands P root node m (the walk ends at its first step, at the root); by Resolve.operands_spec each is the root's resolution of args[i].memberRef m t with t < params[i].leaves (every argument of a root batch has dims: ownRows at eef1ce17ff, batch_dims), and that is resolve P sc ⟨true, i, t⟩ in two steps (rs_unit, then arg_leaf's step). So each operand is a row leaf of the unit: second disjunct. GOES THROUGH. Needs C2, C3 and the row-leaf lemma both kinds share.
 3. Aligned's second disjunct, the closed rows (composite: param_returned_of). Primitive: the unit passes nothing on: PassedOn would make a row leaf, an input gate (OwnRows), equal to sc.off, whose primitive is the unit's, no input (gate_of_node at the root, unit_node). GOES THROUGH, by contradiction. Needs: PROPERTY's lemma, WIDTH's gate_of_node as stated.
 4. LeafWide's second clause, LeafHolds' second and third (composite: resolve_wide at the unit). Primitive: the gate sc.off has primitive sc.d (gate_of_node at the root) whose ret is .value w, and the leaf's width is (Ty.widths (.value w))[0] = w; the third clause is vacuous as in 3. GOES THROUGH. Needs: PROPERTY's lemma, gate_of_node, CalleeOk.
 5. LeafWide's first clause, a row leaf's width. I now prove it from the ROOT for both kinds (one step up by rs_unit, then resolve_wide at the root on the argument's reference, the root body's BodyWide at that argument, the batch-axis lemma, wAt_widths): it never enters the unit, so it needs no composite at all. GOES THROUGH, and the composite case gets shorter.
 6. Filled (M7), that each leaf of a writer's output row resolves (composite: OperandsResolve.resolve_ok at the unit). Primitive: PROPERTY's lemma is the resolution. GOES THROUGH.
 7. The row-leaf lemma (a parameter leaf in range is a RowLeaf): reads .params for both kinds; the composite case alone needs body.params = params. GOES THROUGH.
 Templates, OwnRows, Divides, Names, Ranked, OutFits, LaysOut's existence and COUNTS' landing bound read no unit's inside, or read it through the hypotheses above.

C. WHAT WAY (2) ADDS, by name (mine, new, each at the root):
 C1. prim_unit: a unit whose Definition is a primitive is member m of a root batch, body = default, gates = 1, ret = .value w (from ranges_spec, NodeOk, CalleeOk).
 C2. prim_unit_gate: P.gate sc.off = .ok (p, ops, sc') gives p = sc.d and operands P P.rootScope node m = .ok ops. gate_of_node gives only p and sc', so this is a lemma of mine beside it, not a change to WIDTH's statement.
 C3. run_two_ok: the converse of run_two (two steps that end give the result), for resolve P sc ⟨true, i, t⟩.
 Plus the primitive case in B1, B2, B3, B4, B6. Five case splits and three lemmas; no new premise; UnitsComposite P never appears.

D. WHAT I DID NOT CHECK: PROPERTY's proof (I read the code it is about, not its file); that the Lean accepts each step above (nothing here is compiled); whether RecordCount's count of a primitive unit is what Counts states (the difftest holds the Lean to the reference on 117 such cases, which is agreement, not the property).

E. THE COST EITHER WAY, as I read it. Way (2): the lemmas in C, mine, in the proofs tree only. Way (1): the refusal is written and pushed on its own branch, cursor/record-units-composite-3b1c at 20139e72d9 (from eef1ce17ff; pushed 13:57Z, before your 13:57Z reached me): it pins regenerated data (118 of 600 random case lines change, no verdict moves, one case added, 671; the test file's 14 tests pass locally) and renames keyedAsInput to miskeyed. If way (1) is your word I put the name back in one commit there before anything goes onto #8676. A fact on the name: lean's #8962 at c4b1da1fdb names keyedAsInput in no Lean file of its own, only in a lock record of security/Properties/lean-audit.json; the two lemmas that name it in RecordRead.lean are mine, on my integration branch. I keep the name as you ordered.

THE PROBE: r20261111-140050-01ea on vy-nebius-cpu-1, launched 14:01Z, running. I send its lines when it ends.

