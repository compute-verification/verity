import Proofs.Accounting.TS.Session.Lanes.Layer

/-!
# The named premises a leaf's 33 linked sessions take

`Accounting.Guarantees.TS.LeafLanesSound` takes two premises beyond `CheckedSessionSound`'s, each the reading of a
session's registered rows as the words its Definition reads, through a decode per statement (`decL` for a lane's
ports, `decC` for the combine's):

* `LeafChecks` (the 33 circuits; owner @compute-accounting): each lane's and the combine's circuit computes its
  Definition, so on every wire assignment its evaluation agrees with at every unit, the decoded `ok` word reads `1`
  only where `LaneOk` (or `CombineOk`) holds of the decoded inputs;
* `LeafShared` (the proof service's sharing and the verifier's registrations; owner @proofs): the 33 sessions'
  committed rows, read through the decodes, are one leaf, one partials record and one receipt, with lane `l`'s index
  `l` and every `ok` word `1`.

Like `Accounting.Assumptions.TS`'s other session premises they read `RowsSession`, in the proofs package, so they sit
here until C-Flock's spec is extracted.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Accounting.TS.Checked Accounting.TS.Lanes Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)

/-- `accounting-ts-leaf-checks` (a fact of `h1zy.PouwH1LaneZy_v1` and `PouwH1Combine_v1`'s circuits, to be computed by
a check; owner @compute-accounting). **Each of a leaf's 33 statements computes its Definition at its decoded ports**: at
lane `l`'s session `sess (some l)`, on every wire assignment `vals` the key program's evaluation agrees with on every
unit's wires, if the `ok` word that `decL` reads from `vals`' registered reads is `1`, then `decL`'s view of them meets
`LaneOk` at `dq` and `part`: its partials row is the honest one at its rows, its public inputs and its index, and each
of its 32 cells has `z = Σ_k a_k b_k` and `y = dq [z, s_a, s_w]`. At the combine's session `sess none` the same holds
of `decC`'s view with `CombineOk`: the rows' XOR in lane order is the receipt.

The decodes read only the registered reads (`rowsReg`), so the premise holds of a lane's circuit only if every input
its `ok` depends on is registered: its A and B rows, admit bits, record, partials row, salt, position, roots and index.
A port read only from the public file is free in `vals`, and a lane whose index is free can check any lane's cells.
`dq` is the Program's dequant on words (`PouwZyDequant_v2`'s word view) and `part` the honest lane partial
(`h1zy.lane_partial`'s u32 words, #3095's order: sets `T_0..T_{KS-1}`, `O′`, `D_1..D_{m-1}`, words `2s` the set's
y-XOR and `2s + 1` its z-XOR). -/
def LeafChecks {K nw : ℕ} {Pub : Type} (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (sess : Option (Fin 32) → RowsSession)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw) : Prop :=
  (∀ l (vals : Fin (sess (some l)).Circ.N → Bool),
      (∀ u, ∀ g ∈ (sess (some l)).P.io u, (sess (some l)).Circ.eval vals g = vals g) →
      (decL (sess (some l)) (rowsReg (sess (some l)) vals)).ok = true →
      LaneOk dq part (decL (sess (some l)) (rowsReg (sess (some l)) vals))) ∧
    ∀ vals : Fin (sess none).Circ.N → Bool,
      (∀ u, ∀ g ∈ (sess none).P.io u, (sess none).Circ.eval vals g = vals g) →
      (decC (sess none) (rowsReg (sess none) vals)).ok = true →
      CombineOk (decC (sess none) (rowsReg (sess none) vals))

/-- `accounting-ts-leaf-shared` (the proof service's one opening per committed row, and the verifier's registrations;
owner @proofs). **A leaf's 33 sessions open one leaf, one partials record and one receipt**: on every path, at proof
`i` of round `w`, lane `l`'s committed rows at its registered reads (`openedReg`, the registrant's commit-time
openings `RowsSession.wo`), read through `decL`, are `laneView`: the leaf's A rows at its row slots and B rows at its
column slots, the leaf's record, row `l` of the partials, the path's public inputs (admit, salt, position, roots), the
index `l` and `ok = 1`; and the combine's, read through `decC`, are the 32 partials rows, the receipt and `ok = 1`.

It holds by construction when the service registers each committed row once and the verifier registers its own words:
* every session that reads a row of the leaf's A, B and z/y tables registers it at the same root with the same
  commit-time opening, the prover's at commit (`ServedRegistered` registers the roots; this asks that the opening
  `rdU`'s values and salts are the same one in all 33);
* the partials are committed once, after the draw, and every one of the 33 sessions registers each partials row at the
  record's root with the one opening the service keeps with the record, fixed before any of their audit coins.
  `SampledProofs.Produced.opened` (`P-opened`, cursor/proofs-combine1-3a85) checks that each session opened the
  produced value against the record's root, which fixes the root, not the opening: under a hiding commitment two
  openings of one root are a SHA-512 collision, which this premise doesn't take as a term, so the record keeps each
  row's opening (or its digest) and every session registers that one;
* the verifier registers each lane's salt, position, roots, admit bits and index `l` (distinct `l = 0..31`), and every
  statement's `ok` row, at their known content, as ports of `own`. -/
def LeafShared {Ω : Type} {S nπ K nw : ℕ} {Pub : Type}
    (sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw)
    (pub : Paths Ω S → Fin S → Fin nπ → Pub) (parts : Paths Ω S → Fin S → Fin nπ → Fin 32 → Fin nw → BitVec 32)
    (receipt : Paths Ω S → Fin S → Fin nπ → Fin nw → BitVec 32) (leaf : Paths Ω S → Fin S → Fin nπ → Leaf K) :
    Prop :=
  ∀ x w i,
    (∀ l, decL (sess x w i (some l)) (openedReg (sess x w i (some l))) =
      laneView (leaf x w i) (pub x w i) (parts x w i l) l) ∧
    decC (sess x w i none) (openedReg (sess x w i none)) = ⟨parts x w i, receipt x w i, true⟩

end Accounting.Assumptions.TS
