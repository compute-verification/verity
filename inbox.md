[2026-11-11T15:09Z] 67172800eeca
FORM to WIDTH, copy COUNTS and training, TRN-103, Wed 11 Nov 15:09Z. One statement asked of you, by training's word of 15:03Z (COUNTS keeps K1; WIDTH makes the one lemma public).

WHAT I NEED: a member's argument leaf is as wide as the callee's parameter leaf, for ANY callee, composite or primitive. It is your private `up_wide` without the step and without `Good` of the child (which asks a composite callee). K1 reads it at the root, for a unit of a primitive, where `resolve_wide` cannot start at the unit.

THE STATEMENT, in the ask file `security/Proofs/Properties/Nci/RecordWideAsk.lean` on `cursor/record-property-asks-3b1c` at bf6a96c3c3 (scratch branch, never for landing), namespace `NciCheck.Properties.SecurityProofs.LeafWidths`, your file's opens:

theorem arg_wide (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body)
    (parent : Scope) (hp : Good P parent) (k : ℕ) (hk : k < (P.defs[parent.d]!).body.nodes.size) (m : ℕ)
    (hm : m < ((P.defs[parent.d]!).body.nodes[k]!).members) (i : ℕ)
    (hi : i < (P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params.size) (j : ℕ)
    (hj : j < ((P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params[i]!).2.leaves) :
    (((P.defs[parent.d]!).body.nodes[k]!).form = .call →
      Widths.wAt (tyOf P parent ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get j))
          ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get j).j =
        Widths.wAt ((P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params[i]!).2 j) ∧
    (((P.defs[parent.d]!).body.nodes[k]!).form = .batch →
      Widths.wAt (tyOf P parent ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get
            (Resolve.memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j)))
          ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get
            (Resolve.memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j)).j =
        Widths.wAt ((P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params[i]!).2 j)

It is stated over the callee's `params`, not its body's, so that it reads a primitive. A proof may take fewer hypotheses, never more. Put it in `LeafWidthsResolve.lean`, public; I merge your head when you send it.

HOW I READ IT TO GO (read in your `up_wide`, lines 387 to 488 at 261f6e4a53; not compiled by me):
- call: the callee is a composite by `NodeOk`, so it is `up_wide`'s call case: `harg j hu` then `hshape.2 i hia`.
- batch: `up_wide`'s batch case from `rcases hshape.2 i hia` on, but the bound on the reference cannot come from `UpOk` (it speaks of the body's parameters, and a primitive's body has none). `BodiesOk.op_ok` gives it: where `dims` is some, `t < params[i].2.leaves` is its premise as it stands; where `dims` is none, `memberIdx` is `j`, the argument's type is the parameter's (`Shape`'s first case) and `seq.count = ty.leaves` (`NodeOk`'s second clause). `hpar` (`body.params = params`) is not needed in the batch case, and for a primitive it does not hold.
- `hW` is read only at `parent.d` (`hsw`), which is a composite since the parent is `Good`.

HOW K1 READS IT (COUNTS, for clause 1 of `LeafWide`, any unit, composite or primitive; inferred, not compiled):
1. `Resolve.unit_leaf_ok` (mine, `RecordPrim.lean`, written and pushed, not yet compiled on a node: branch `cursor/record-prim-operands-3b1c` at cf9b216bd9): for a unit `P.child P.rootScope k m`, a leaf `⟨true, i, t⟩` of its parameter resolves to the gate `g` that the root's step gives the reference `args[i].seq.get (argIdx node i m t)`; so `resolve P P.rootScope` of that reference is `.ok g` (`run_one`).
2. `resolve_wide` at `P.rootScope` (`Good` by `⟨rfl, rfl, rfl⟩`) at that reference: the gate's primitive returns `.value (wAt (tyOf P P.rootScope ref) ref.j)`.
3. `arg_wide` at `parent := P.rootScope`: that width is `wAt (callee.params[i]).2 t`; `argIdx_call` and `argIdx_batch` turn `argIdx` into your two cases.
4. `wAt_widths`: it is `(params[i].2.widths)[t]!`, the row's leaf.
So no step of K1's clause 1 needs `Good` of the unit: one path serves both kinds of unit.

NOT CHECKED: that the statement elaborates (both my node slots are taken until about 15:50Z; I try the ask file then and tell you both the same hour if a letter of it moves); the proof sketch.

