import Proofs.Accounting.TS.Session.Lanes.Holds

/-!
# The named premise a leaf's 33 linked sessions take

`Accounting.TS.Lanes.leafLanes_soundAt` takes one premise beyond `CheckedSessionSound`'s, the reading of a session's
rows as the words its Definition reads, through a decode per statement (`decL` for a lane's ports, `decC` for the
combine's). `Accounting.Guarantees.TS.LeafLanesSound` takes what stays of it once each Definition's `ok = 1` is proved
to mean `LaneOk` or `CombineOk`, that the lanes' and the combine's circuits compute their pieces' compositions on bits
(`Accounting.TS.Lanes.LaneLowers`, `CombineLowers`, which give it by `leafChecks_of_computes`, `laneComputes_of_lowers`
and `combineComputes_of_lowers`):

* `LeafChecks` (the 33 circuits): each lane's and the combine's circuit computes its Definition at its port rows, so on
  every wire assignment its evaluation agrees with at every unit, the decoded `ok` word reads `1` only where `LaneOk`
  (or `CombineOk`) holds of the decoded inputs. It reads every port row (`portRead`), registered or not, so it is a fact
  of the circuits alone; that every port they read is registered and opens at its registered root is proved from
  flock-verify's complete check, part of `leafLayer`'s acceptance (`RowsSession.rowsReg_eq_portRead`,
  `RowsSession.portRow_opens`, `Flock.Registered.complete_opens`), not assumed.

Nothing is asked of the rows the sessions open. The leaf the layer opens and the receipt it checks are the ones those
rows decode to (`Accounting.TS.Lanes.leafOf`, `receiptOf`); the known ports' rows (the admit bits, the public inputs,
lane `l`'s index and every `ok` word `1`) are flock-verify's known-content and known-read checks', part of `leafLayer`'s
acceptance (`Accounting.TS.Lanes.laneKnownHolds_of_check`, `combKnownOk_of_check`); the rows several statements read
(each A row four lanes, each B row four lanes, each partials row lane `l` and the combine) are read in full at one
registered tree (`PouwWindow.Lanes.ownOf`, `readFull`, which acceptance checks), so they decode alike
(`Accounting.TS.Lanes.read_bits`). That the registrants' rows agree is not a premise: each registrant's opening opens
its root at its position, so two of them at one tree and position are one row's bytes or an explicit SHA-512 collision
(`Accounting.TS.Lanes.rowsAgree_or_leafColl`), which the leaf layer's relation admits (`LeafColl`).

Like `Accounting.Assumptions.TS`'s other session premises it reads `RowsSession`, in the proofs package, so it sits
here until C-Flock's spec is extracted.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Accounting.TS.Checked Accounting.TS.Lanes Flock.SoundAtBridge PouwWindow.Lanes
open Pouw.RelativeSecurity (Paths)

/-- `accounting-ts-leaf-checks` (a fact of `h1zy.PouwH1LaneZy_v1` and `PouwH1Combine_v1`'s circuits at their tile's `K`,
backed by circuit-check and `h1zy`'s tests, not proved). **Each of a leaf's 33 circuits computes its Definition at its
port rows**: at lane `l`'s session `sess (some l)`, whose circuit is `PouwH1LaneZy_v1{TILE}`'s lowering at its tile's
`K`, at a `K` that tile binds at (`32 ∣ K` and `K < 2^17`, `GemmAccumulatorS8`'s bind; from `2^17` a step of its s8
accumulator can clamp, so `z` need not be the exact dot, and the verifier refuses such a tile,
`PouwWindow.Lanes.Tile.check`), on every wire assignment `vals` the key program's evaluation agrees with on every unit's
wires, if the `ok` word that `decL` reads from `vals`' port rows (`portRead`) is `1`, then `decL`'s view of them meets
`LaneOk` at `dq` and `part`: its partials row is the honest one at its rows, its public inputs and its index, and each
of its 32 cells has `z = Σ_k a_k b_k` and `y = dq [z, s_a, s_w]`. At the combine's session `sess none`, whose circuit is
`PouwH1Combine_v1{TILE}`'s at `K ≥ 384` (`h1zy.LEAST_K`, the least `K` at which C-Flock proves the combine), the same
holds of `decC`'s view with `CombineOk`: the rows' XOR in lane order is the receipt.

It reads every port row, registered or not, and asks nothing of `own`: that every port the decodes read (`ok`'s cone,
the A and B rows, admit bits, record, partials row, salt, position, roots and lane index) is a port of `own` that opens
at its registered root is flock-verify's complete check (`Flock.Registered.checkComplete`, part of `leafLayer`'s
acceptance), from which `RowsSession.rowsReg_eq_portRead` proves the registered reads are the port rows. What stays
assumed is that the circuit file a session's flock-verify reads computes its Definition, read through the decodes: the
Lean has no lowering of a Definition to a circuit, and the session's relation (`CheckedRel`) is that circuit's own
evaluation. The circuit's input wires follow `PouwWindow.Lanes.wireOf?`'s map (`class_statement.unit_lowering`'s order:
the input rows one after another, wire `w` the port, row bit and field `wireOf?` gives), which makes the port rows
`decL` and `decC` read the Definition's inputs; that the circuit file lays its inputs out so is part of this premise
until the verifier reads the circuit as a templated statement. `dq` is the Program's dequant on words
(`PouwZyDequant_v2`'s word view) and `part` the honest lane partial (`h1zy.lane_partial`'s u32 words, in its order: sets
`T_0..T_{KS-1}`, `O′`, `D_1..D_{m-1}`, words `2s` the set's y-XOR and `2s + 1` its z-XOR). -/
def LeafChecks {K nw : ℕ} {Pub : Type} (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (sess : Option (Fin 32) → RowsSession)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw) : Prop :=
  (32 ∣ K → K < 2 ^ 17 → ∀ l (vals : Fin (sess (some l)).Circ.N → Bool),
      (∀ u, ∀ g ∈ (sess (some l)).P.io u, (sess (some l)).Circ.eval vals g = vals g) →
      (decL (sess (some l)) (portRead (sess (some l)) vals)).ok = true →
      LaneOk dq part (decL (sess (some l)) (portRead (sess (some l)) vals))) ∧
    ∀ vals : Fin (sess none).Circ.N → Bool,
      (∀ u, ∀ g ∈ (sess none).P.io u, (sess none).Circ.eval vals g = vals g) →
      (decC (sess none) (portRead (sess none) vals)).ok = true →
      CombineOk (decC (sess none) (portRead (sess none) vals))

end Accounting.Assumptions.TS
