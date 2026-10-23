import Proofs.Accounting.TS.Session.Lanes.Holds

/-!
# The named premises a leaf's 33 linked sessions take

`Accounting.Guarantees.TS.LeafLanesSound` takes two premises beyond `CheckedSessionSound`'s, each the reading of a
session's rows as the words its Definition reads, through a decode per statement (`decL` for a lane's ports, `decC` for
the combine's):

* `LeafChecks` (the 33 circuits; owner @compute-accounting): each lane's and the combine's circuit computes its
  Definition at its port rows, so on every wire assignment its evaluation agrees with at every unit, the decoded `ok`
  word reads `1` only where `LaneOk` (or `CombineOk`) holds of the decoded inputs. It reads every port row
  (`portRead`), registered or not, so it is a fact of the circuits alone; that every port they read is registered and
  opens at its registered root is proved from flock-verify's complete check, part of `leafLayer`'s acceptance
  (`RowsSession.rowsReg_eq_portRead`, `RowsSession.portRow_opens`, `Flock.Registered.complete_opens`), not assumed;
* `LeafReads` (the verifier's registrations; owner @proofs): where the 33 sessions' registrants' rows agree at every
  tree and position two of them read (`Accounting.TS.Lanes.RowsAgree`), their committed rows at the leaf's data ports
  hold one in-range leaf and the receipt, each word's encoding at its place in the layout. The known ports' rows (the
  admit bits, the public inputs, lane `l`'s index and every `ok` word `1`) are not asked: flock-verify's known-content
  and known-read checks, part of `leafLayer`'s acceptance, give them (`Accounting.TS.Lanes.laneKnownHolds_of_check`,
  `combKnownOk_of_check`). Nor are the partials rows: they are proof-layer values, and lane `l`'s `partials` and the
  combine's `p<l>` read one row of one registered tree (`PouwWindow.Lanes.ownOf`, which the guarantee takes of each
  session's registered file) in full (the known-read check at `PouwWindow.Lanes.readFull`), so they are one row
  (`Accounting.TS.Lanes.partials_bits`), whatever it holds.

That the registrants' rows agree is not a premise: each registrant's opening opens its root at its position, so two
of them at one tree and position are one row's bytes or an explicit SHA-512 collision
(`Accounting.TS.Lanes.rowsAgree_or_leafColl`), which the leaf layer's relation admits (`LeafColl`).

Like `Accounting.Assumptions.TS`'s other session premises they read `RowsSession`, in the proofs package, so they sit
here until C-Flock's spec is extracted.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Accounting.TS.Checked Accounting.TS.Lanes Flock.SoundAtBridge PouwWindow.Lanes
open Pouw.RelativeSecurity (Paths)

/-- `accounting-ts-leaf-checks` (a fact of `h1zy.PouwH1LaneZy_v1` and `PouwH1Combine_v1`'s circuits at their tile's
`K`, backed by circuit-check and `h1zy`'s tests, not proved; owner @compute-accounting). **Each of a leaf's 33 circuits
computes its Definition at its port rows**: at lane `l`'s session `sess (some l)`, whose circuit is
`PouwH1LaneZy_v1{TILE}`'s lowering at its tile's `K` (any positive `K`), on every wire assignment `vals` the key
program's evaluation agrees with on every unit's wires, if the `ok` word that `decL` reads from `vals`' port rows
(`portRead`) is `1`, then `decL`'s view of them meets `LaneOk` at `dq` and `part`: its partials row is the honest one at
its rows, its public inputs and its index, and each of its 32 cells has `z = Σ_k a_k b_k` and `y = dq [z, s_a, s_w]`.
At the combine's session `sess none`, whose circuit is `PouwH1Combine_v1{TILE}`'s at `K ≥ 384` (`h1zy.LEAST_K`, the
least `K` at which C-Flock proves the combine), the same holds of `decC`'s view with `CombineOk`: the rows' XOR in lane
order is the receipt.

It reads every port row, registered or not, and asks nothing of `own`: that every port the decodes read (`ok`'s cone,
the A and B rows, admit bits, record, partials row, salt, position, roots and lane index) is a port of `own` that
opens at its registered root is flock-verify's complete check (`Flock.Registered.checkComplete`, part of `leafLayer`'s
acceptance), from which `RowsSession.rowsReg_eq_portRead` proves the registered reads are the port rows. What stays
assumed is that the circuit file a session's flock-verify reads computes its Definition, read through the decodes: the
Lean has no lowering of a Definition to a circuit, and the session's relation (`CheckedRel`) is that circuit's own
evaluation. The circuit's input wires follow `PouwWindow.Lanes.wireOf?`'s map (#3180,
`class_statement.unit_lowering`'s order: the input rows one after another, wire `w` the port, row bit and field
`wireOf?` gives), which makes the port rows `decL` and `decC` read the Definition's inputs; that the circuit file lays
its inputs out so is part of this premise until TMPL1's templated read. `dq` is the Program's dequant on words
(`PouwZyDequant_v2`'s word view) and `part` the honest lane partial (`h1zy.lane_partial`'s u32 words, #3095's order:
sets `T_0..T_{KS-1}`, `O′`, `D_1..D_{m-1}`, words `2s` the set's y-XOR and `2s + 1` its z-XOR). -/
def LeafChecks {K nw : ℕ} {Pub : Type} (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (sess : Option (Fin 32) → RowsSession)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw) : Prop :=
  (∀ l (vals : Fin (sess (some l)).Circ.N → Bool),
      (∀ u, ∀ g ∈ (sess (some l)).P.io u, (sess (some l)).Circ.eval vals g = vals g) →
      (decL (sess (some l)) (portRead (sess (some l)) vals)).ok = true →
      LaneOk dq part (decL (sess (some l)) (portRead (sess (some l)) vals))) ∧
    ∀ vals : Fin (sess none).Circ.N → Bool,
      (∀ u, ∀ g ∈ (sess none).P.io u, (sess none).Circ.eval vals g = vals g) →
      (decC (sess none) (portRead (sess none) vals)).ok = true →
      CombineOk (decC (sess none) (portRead (sess none) vals))

/-- `accounting-ts-leaf-reads` (the verifier's registrations; owner @proofs). **Where a leaf's 33 sessions' registrants'
rows agree, their data ports hold one leaf and one receipt**: on every path, at proof `i` of round `w`, if the
registrants' rows of the 33 sessions agree wherever two of them read one tree at one position (`RowsAgree`), and at
each session the registrants' rows at the ports its known content `known x w i k` names are the known rows
(`KnownHeld`), then the leaf is in range (`LeafInRange`: s8 codes, u32 scale encodings, s32 z, u16 y encodings), lane
`l`'s committed rows at its registered reads (`openedReg`, the registrant's commit-time openings `RowsSession.wo`), read
bit by bit (`portBit`), hold at the layout's places at tile `T` (`PouwWindow.Lanes.lanePorts`) the leaf's A rows at its
row slots and B rows at its column slots and the leaf's record (`LaneDataHolds`), and the combine's hold the receipt
(`CombDataHolds`). It asks nothing of the partials rows, which the decodes read as they are.

It asks nothing of the openings the sessions register: one row read by several sessions at one tree and position is
one row's bytes in all of them by `RowsAgree`, which holds up to an explicit SHA-512 collision
(`Accounting.TS.Lanes.rowsAgree_or_leafColl`). (#3131's `SampledProofs.Produced.shared`, `P-shared`, checks the same of
the openings the verifiers of record report, with no collision escape; the proof doesn't read it.) Nor does it ask of
the known ports (admit, pub, `ok`): their rows' bits are their known content by flock-verify's checks. What it does ask
is of the registered files the verifier hands the 33 sessions, beyond their form (`PouwWindow.Lanes.ownOf`: one tree
per value in all 33, each port at its layout row, which the guarantee takes): the leaf's A, B and z/y tables and the
receipt are registered at the registration's roots (`ServedRegistered`), so the committed rows at those trees are the
leaf's and the receipt's encodings. That every port of each circuit is registered is not asked: `leafLayer`'s
acceptance includes flock-verify's complete check (#3131's `Flock.Registered.checkComplete`). -/
def LeafReads {Ω : Type} {S nπ K nw : ℕ} (T : Tile)
    (sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession)
    (known : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → Known)
    (receipt : Paths Ω S → Fin S → Fin nπ → Fin nw → BitVec 32) (leaf : Paths Ω S → Fin S → Fin nπ → Leaf K) :
    Prop :=
  ∀ x w i, RowsAgree (sess x w i) → (∀ k, KnownHeld (sess x w i k) (known x w i k)) →
    LeafInRange (leaf x w i) ∧
    (∀ l, LaneDataHolds K T (portBit (sess x w i (some l)) (openedReg (sess x w i (some l)))) (leaf x w i) l) ∧
    CombDataHolds nw T (portBit (sess x w i none) (openedReg (sess x w i none))) (receipt x w i)

end Accounting.Assumptions.TS
