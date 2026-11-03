import Proofs.Accounting.TS.Session.Group.Layer
import Proofs.Accounting.TS.Session.Lanes.Partials

/-!
# The named premise a leaf's group of linked sessions takes

`GroupRealizes`: a leaf's group of statements (its lanes, each in `n` chunks, and its combine) computes the leaf's
relation at the rows the group's sessions open. It is the group counterpart of `Accounting.Assumptions.TS.LeafChecks`
(option A's 33 statements at a 32 × 32 leaf), which states each circuit's own Definition, and it adds what turns the
statements' checks into the leaf's: that the lanes cover the leaf, that a row several statements read decodes alike
in each, and that the chunks' carries chain a lane's `K` (for the shape's Definitions). At the 32 × 32 leaf, one chunk
a lane, it is proved from the circuits computing their pieces' compositions
(`Accounting.TS.Lanes.LaneLowers`, `CombineLowers`), the lanes' cover and the decodes
(`Accounting.TS.Lanes.groupRealizes_lanes32`). At another shape nothing in Lean states the shape's Definitions yet
(`Pouw.H1Zy.laneOk` and `chunkOk` are at a 32 × 32 leaf), so it is assumed there, backed by circuit-check and
`h1zy`'s tests. Like `Accounting.Assumptions.TS`'s other session premises it reads `RowsSession`, in the proofs package,
so it sits here until C-Flock's spec is extracted.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Accounting.TS.Checked Accounting.TS.Lanes Flock.SoundAtBridge

/-- `accounting-ts-group-realizes` (a fact of a leaf's group of circuits at their layout, backed by circuit-check and
`h1zy`'s tests; proved at the 32 × 32 leaf with one chunk a lane from `LaneLowers` and `CombineLowers`,
`Accounting.TS.Lanes.groupRealizes_lanes32`). **A leaf's group of circuits realizes its relation at the rows its
sessions open**: at the group's sessions `sess k` (`k : ι`, each statement of the group), where the leaf verifier's
group check `gate` passed and each session's complete check, known-read check (at its known content `known k` and the
ports `full k` it reads in full) and file check (its ports `want k`) passed, every unit's served roots are registered,
each session has a unit, every unit reads its one-row ports at their rows (`ReadsOwnRow`), the sessions' registrants'
rows agree wherever two of them read one registered tree at one position (`RowsAgree`), the known rows hold
(`KnownHeld`), and each session's committed rows (`openedReg`) are the port rows of a wire assignment its circuit's
evaluation agrees with at every unit: the relation `rel` holds of the value `opened` the layer opens. What it asks of
the circuits is that each computes its Definition at its port rows, as `LeafChecks` does of each of the 33; the rest
(the cover, the decodes, the chunks' chaining) is what the shape's Definitions give, proved where they are stated. -/
def GroupRealizes {ι V : Type} (sess : ι → RowsSession) (known : ι → Known) (full : ι → Array String)
    (want : ι → List _root_.Flock.Registered.Port) (gate : Prop) (opened : V) (rel : V → Prop) : Prop :=
  gate →
    (∀ k, _root_.Flock.Registered.checkComplete (sess k).c (sess k).own = .ok () ∧
      _root_.Flock.KnownRead.check (sess k).c (known k ++ _root_.Flock.KnownRead.fullRows (sess k).own (full k)) =
        .ok () ∧
      PouwWindow.Lanes.checkOwn (want k) (sess k).own = .ok ()) →
    (∀ k u, (sess k).ServedRegistered u) → (∀ k, 0 < (sess k).n) → (∀ k, ReadsOwnRow (sess k)) →
    RowsAgree sess → (∀ k, KnownHeld (sess k) (known k)) →
    (∀ k, ∃ vals : Fin (sess k).Circ.N → Bool, openedReg (sess k) = portRead (sess k) vals ∧
      ∀ u, ∀ g ∈ (sess k).P.io u, (sess k).Circ.eval vals g = vals g) →
    rel opened

end Accounting.Assumptions.TS
