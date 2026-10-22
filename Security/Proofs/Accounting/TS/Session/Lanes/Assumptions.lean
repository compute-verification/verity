import Proofs.Accounting.TS.Session.Lanes.Layer

/-!
# The named premises a leaf's 33 linked sessions take

`Accounting.Guarantees.TS.LeafLanesSound` takes two premises beyond `CheckedSessionSound`'s, each the reading of a
session's registered rows as the words its Definition reads, through a decode per statement (`decL` for a lane's
ports, `decC` for the combine's):

* `LeafChecks` (the 33 circuits; owner @compute-accounting): each lane's and the combine's circuit computes its
  Definition, so on every wire assignment its evaluation agrees with at every unit, the decoded `ok` word reads `1`
  only where `LaneOk` (or `CombineOk`) holds of the decoded inputs;
* `LeafReads` (the decodes and the verifier's registrations; owner @proofs): where the 33 sessions' registrants'
  rows agree at every root and position two of them read (`Accounting.TS.Lanes.RowsAgree`), their committed rows,
  read through the decodes, are one leaf, one partials record and one receipt, with lane `l`'s index `l` and every
  `ok` word `1`.

That the registrants' rows agree is not a premise: each registrant's opening opens its root at its position, so two
of them at one root and position are one row's bytes or an explicit SHA-512 collision
(`Accounting.TS.Lanes.rowsAgree_or_leafColl`), which the leaf layer's relation admits (`LeafColl`).

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

/-- `accounting-ts-leaf-reads` (the decodes and the verifier's registrations; owner @proofs). **Where a leaf's 33
sessions' registrants' rows agree, the decodes read one leaf, one partials record and one receipt**: on every path, at
proof `i` of round `w`, if the registrants' rows of the 33 sessions agree wherever two of them read one root at one
position (`RowsAgree`), and at each session the registrants' rows at the ports its known content `known x w i k`
names are the known rows (`KnownHeld`), then lane `l`'s committed rows at its registered reads (`openedReg`, the registrant's
commit-time openings `RowsSession.wo`), read through `decL`, are `laneView`: the leaf's A rows at its row slots and B
rows at its column slots, the leaf's record, row `l` of the partials, the path's public inputs (admit, salt, position,
roots), the index `l` and `ok = 1`; and the combine's, read through `decC`, are the 32 partials rows, the receipt and
`ok = 1`.

It asks nothing of the openings the sessions register: one row read by several sessions at one root and position is
one row's bytes in all of them by `RowsAgree`, which holds up to an explicit SHA-512 collision
(`Accounting.TS.Lanes.rowsAgree_or_leafColl`). (#3131's `SampledProofs.Produced.shared`, `P-shared`, checks the same of
the openings the verifiers of record report, with no collision escape; the proof doesn't read it.) What it does ask is of
the decodes and of the registered files the verifier hands the 33 sessions:
* every session registers each shared value at one port, the same in all 33 (one root, domain, leaf count and schema):
  the leaf's A, B and z/y tables at the registration's roots (`ServedRegistered`), and the partials at the post-draw
  record's root, the one each session opened them against (#3131's `SampledProofs.Produced.opened`, `P-opened`), and
  every port of each circuit is registered (#3131's `Flock.Registered.checkComplete`);
* `decL` and `decC` read each value's rows by that port and the row's position, lane `l`'s A row at row slot `s` at
  row `rowOf l s`, its B row at column slot `t` at row `colOf l t` and its partials row at row `l`, so rows that agree
  decode to one leaf and one partials record;
* the known content the verifier hands each session (`known x w i k`) is each lane's salt, position, roots, admit bits
  and index `l` (distinct `l = 0..31`), and every statement's `ok` row, at the ports the decodes read them from
  (#3131's `SampledProofs.Produced.known`). That the registrants' rows at those ports are that content is no longer
  asked: `KnownHeld` is a hypothesis here, and `LeafLanesSound` proves it from flock-verify's known-content check
  (`Flock.Registered.checkKnownAll`) unless the known-content tie collides (`LeafKnownColl`). -/
def LeafReads {Ω : Type} {S nπ K nw : ℕ} {Pub : Type}
    (sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession)
    (known : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → Known)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw)
    (pub : Paths Ω S → Fin S → Fin nπ → Pub) (parts : Paths Ω S → Fin S → Fin nπ → Fin 32 → Fin nw → BitVec 32)
    (receipt : Paths Ω S → Fin S → Fin nπ → Fin nw → BitVec 32) (leaf : Paths Ω S → Fin S → Fin nπ → Leaf K) :
    Prop :=
  ∀ x w i, RowsAgree (sess x w i) → (∀ k, KnownHeld (sess x w i k) (known x w i k)) →
    (∀ l, decL (sess x w i (some l)) (openedReg (sess x w i (some l))) =
      laneView (leaf x w i) (pub x w i) (parts x w i l) l) ∧
    decC (sess x w i none) (openedReg (sess x w i none)) = ⟨parts x w i, receipt x w i, true⟩

end Accounting.Assumptions.TS
