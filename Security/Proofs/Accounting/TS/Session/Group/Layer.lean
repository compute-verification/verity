import Proofs.Accounting.TS.Session.Lanes.Layer
import PouwWindow.KnownRows

/-!
# A leaf's proof layer through a group of linked C-Flock sessions, any shape

A drawn leaf too big for one C-Flock block is proved as a group of one-unit statements: `L` lanes, each in `n` chunks
along `K`, and one combine, `G = L n + 1` sessions (`Ix L n`: `some (l, c)` lane `l`'s chunk `c`, `none` the combine).
The 32 × 32 leaf's option-A group (`Accounting.TS.Lanes.leafLayer`) is `L = 32`, `n = 1`, indexed by `Option (Fin 32)`.

* `SessionChecks`: what the leaf verifier asks of each of the group's sessions, `leafLayer`'s acceptance at one
  statement: flock-verify `--zk --registered` accepts it, with its known-content check (`Registered.checkKnownAll`), its
  complete check (`Registered.checkComplete`) and its known-read check (`KnownRead.check`, at the known content and the
  ports read in full), and `pouw-leaf-own`'s checks of its registered file (`PouwWindow.Lanes.checkOwn`), of its draw
  (`PouwWindow.Lanes.checkFull`, every unit of a non-empty population) and of its reads (`checkOneRead`).
* `okKnown`: statement `k`'s known content, the other known ports' rows the verifier builds (`known`) and then its `o0`
  row 1 under its salt, as `PouwWindow.Lanes.knownRows` ends every statement's.
* `groupLayer`: proof `i` of round `w` is accepted where the leaf verifier's check of the tile and its chunks
  (`PouwWindow.Lanes.checkChunks T n`) passes and every one of the `T.lanes · n + 1` sessions passes `SessionChecks`; it
  opens `opened` and its relation is `rel`, or two of the sessions' registrants' records clash (`LeafColl`), or a
  session's known-content tie collides (`LeafKnownColl`).

The leaf's relation `rel` and the value `opened` are parameters: which leaf the committed rows decode to and what its
served checks are depends on the shape's Definitions, which the group's premise
(`Accounting.Assumptions.TS.GroupRealizes`) reads.
-/

namespace Accounting.TS.Group

open Accounting.TS Accounting.TS.Checked Accounting.TS.Lanes Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

/-- **A leaf's group of statements**: lane `l`'s chunk `c` (`some (l, c)`) for `L` lanes of `n` chunks, then the
combine (`none`). -/
abbrev Ix (L n : ℕ) : Type := Option (Fin L × Fin n)

/-- A group of `L` lanes of `n` chunks and a combine is `L n + 1` statements. -/
theorem card_ix (L n : ℕ) : Fintype.card (Ix L n) = L * n + 1 := by
  simp [Ix, Fintype.card_option, Fintype.card_prod, Fintype.card_fin]

/-- **What the leaf verifier asks of one of a group's sessions**: flock-verify `--zk --registered` accepts its outcome
`o`, its known-content check at `known`, its complete check and its known-read check at `known` and the ports `full`
it reads in full included, and `pouw-leaf-own`'s checks of its registered file (the ports `want`), of its draw (every
unit of a non-empty population) and of its reads (one read at each port) pass. -/
def SessionChecks (s : RowsSession) (o : s.Out) (known : Known) (full : Array String)
    (want : List _root_.Flock.Registered.Port) : Prop :=
  LiveAcceptsZKR s.I s.own o ∧
    _root_.Flock.Registered.checkKnownAll s.own known = .ok () ∧
    _root_.Flock.Registered.checkComplete s.c s.own = .ok () ∧
    _root_.Flock.KnownRead.check s.c (known ++ _root_.Flock.KnownRead.fullRows s.own full) = .ok () ∧
    PouwWindow.Lanes.checkOwn want s.own = .ok () ∧
    PouwWindow.Lanes.checkFull o.2.2.record = .ok () ∧
    PouwWindow.Lanes.checkOneRead s.own = .ok ()

/-- **A statement's known content**: the other known ports' rows `known`, then its `o0` row, the word `1`, under its
salt (`PouwWindow.Lanes.knownRows`' last entry at every statement). -/
def okKnown (known : Known) (salt : ByteArray) : Known :=
  known ++ #[("o0", #[(PouwWindow.Lanes.okRow, salt)])]

theorem okKnown_names (known : Known) (salt : ByteArray) : ∃ ks, ("o0", ks) ∈ okKnown known salt :=
  ⟨#[(PouwWindow.Lanes.okRow, salt)], Array.mem_append.2 (Or.inr (by simp))⟩

/-- **The leaf's proof layer through its group of sessions**: for a tile `T` whose `T.lanes` lanes take `n` chunks
each, proof `i` of round `w` is accepted where the leaf verifier's check of the tile and its chunks passes
(`PouwWindow.Lanes.checkChunks T n`: `T.check`, `n ≥ 1` and `32 n ∣ K`) and each of its `T.lanes · n + 1` sessions
`sess x w i k` passes `SessionChecks` at its known content (`okKnown`: `known x w i k`, then its `o0` row under
`okSalt x w i k`), the ports it reads in full (`full x w i k`) and its file's ports (`want x w i k`); it opens
`opened x w i`, and its relation is `rel x w i`, or a clash of two of its sessions' registrants' records (`LeafColl`),
or a collision of a session's known-content tie (`LeafKnownColl`). -/
noncomputable def groupLayer {Ω : Type} {S nπ : ℕ} {V : Type} (T : PouwWindow.Lanes.Tile) (n : ℕ)
    (sess : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → RowsSession) (out : ∀ x w i k, (sess x w i k).Out)
    (known : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → Known)
    (okSalt : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → ByteArray)
    (full : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → Array String)
    (want : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → List _root_.Flock.Registered.Port)
    (opened : Paths Ω S → Fin S → Fin nπ → V) (rel : Paths Ω S → Fin S → Fin nπ → V → Prop)
    (right : Paths Ω S → (Fin S → Fin nπ → V) → Prop) : ProofLayer Ω S nπ V where
  accepted x w i := PouwWindow.Lanes.checkChunks T n = .ok () ∧
    ∀ k, SessionChecks (sess x w i k) (out x w i k) (okKnown (known x w i k) (okSalt x w i k)) (full x w i k)
      (want x w i k)
  opened := opened
  rel x w i v := rel x w i v ∨ LeafColl (sess x w i) ∨
    LeafKnownColl (sess x w i) fun k => okKnown (known x w i k) (okSalt x w i k)
  right := right

end Accounting.TS.Group
