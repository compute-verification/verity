module

public import Proofs.Pouw.TS.Session.Group.Layer
public import Proofs.Pouw.TS.Session.Assumptions
public import Proofs.Flock.SoundAtBridge.Served
public import Proofs.Pouw.TS.Session.Lanes.Decode
public import Flock.KnownRead
public import Mathlib.Data.String.Basic
public import Proofs.Flock.EndToEndRegistered
public import Proofs.Flock.Soundness.Discharge.ZkLink.Rows
public import Proofs.Flock.SoundAtBridge.Reduce
public import Proofs.Flock.SoundAtBridge.Statement
public import Definitions.Pouw.TS.ProofLayer
public import Proofs.Flock.SoundAtBridge.Session
public import Proofs.Pouw.TS.Session.Held.Layer
public import Proofs.Pouw.TS.Session.Lanes.Holds
public import Proofs.Pouw.TS.Session.Lanes.Range
public import Proofs.Pouw.TS.Session.Lanes.KnownOf
public import Proofs.Pouw.TS.Session.Lanes.Canon
public import Proofs.Flock.C3.RegLevels.TopLeaf
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.C3.Compose.RowsReg
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJCustody
import Proofs.Flock.Soundness.Game
import Proofs.Pouw.RelativeSecurity

/-!
# A leaf's group of linked sessions realizes one unit, any shape (the proof)

`ResourceAccounting.TS.Group.group_soundAt`, the argument every leaf proved as a group of one-unit Flock statements shares,
whatever the group (any finite type `ι` of statements: `Ix L n`'s `L` lanes of `n` chunks and a combine, or option A's
33): a proof layer accepted only where each of the group's sessions passes the leaf verifier's per-statement checks
(`Group.SessionChecks`), whose relation the group's circuits realize at the rows the sessions open
(`ResourceAccounting.Assumptions.TS.GroupRealizes`) and admits a clash of two of the sessions' registrants' records
(`LeafColl`) and a collision of a session's known-content tie (`LeafKnownColl`), is sound at `|ι| δ`. An unsound proof
is, at one of its sessions, an unsound proof of the checked layer where its ports are named once and include `ok`
(`ResourceAccounting.TS.Checked.checkedOn_soundAt`'s `SoundAt δ`, `CheckedSessionSound`'s argument with those asked only where
they hold, which the known-read check's names give at known content naming `ok`: `names_of_check`, with full checking
asked there too) or an accepted outcome that isn't the session's write of a live run (`RowsSession.Written`), which
has probability `0` in every session's game (`RowsSession.prob_unwritten`) and so on the window (`ExecutionPhaseDesign`). At a
written outcome the leaf verifier's sample check (`PouwWindow.Lanes.checkFull`) gives full checking: the record's sample is
`subset` of every unit of a non-empty population, and the session's custody makes it the verifier's sample `subset kd`
over `n` units, so the outcome samples every unit, `kd = n` and `0 < n` (`RowsSession.full_of_check`), and every unit
reads its one-row ports at their rows (`RowsSession.readsOwnRow_of`). Every unit's served roots are registered
(`RowsSession.ServedRegistered`) wherever the one-read check (`PouwWindow.Lanes.checkOneRead`) holds, which the
sessions' reads built from `own` give (`RowsSession.servedRegistered_canon`). The sessions' registrants' rows agree
unless two records clash (`rowsAgree_or_leafColl`, from `FlockSoundness.Registered.enc_eq_of_not_clash`), and the
known rows hold unless a known-content tie collides (`knownHeld_or_coll`, from verity-verify-proof's known-content check).
Under the complete check the registered reads the checked layer opens are the port rows
(`RowsSession.rowsReg_eq_portRead`), so each session's committed rows are the port rows of a wire assignment its
circuit's evaluation agrees with, which `GroupRealizes` reads. The union bound over the group gives `|ι| δ`.

`ResourceAccounting.SecurityProofs.TS.LeafGroupSound` is it at `groupLayer` and `Ix T.lanes n` (`T.lanes · n + 1` statements,
`card_ix`); `ResourceAccounting.SecurityProofs.TS.LeafLanesSound` is it at option A's `leafLayer`
(`Proofs.Pouw.TS.LanesSound`). `ResourceAccounting.TS.Group.groupLayer_not_soundAt` is the nontrivial lemma: at a proof
accepted on every path with an opening that fails its relation and no collision in hand, the guarantee's conclusion
fails at every `δ` with `(T.lanes · n + 1) δ < 1`, so its premises are what make it hold.

This module imports `Proofs.Flock.SoundAtBridge` (through `Proofs.Pouw.TS.CheckedSound`), whose imports make `X`
a token, so `Proofs/ResourceAccounting.lean` imports it.
-/

@[expose] public section

namespace Flock.SoundAtBridge.RowsSession

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable (s : RowsSession)

end Flock.SoundAtBridge.RowsSession

namespace ResourceAccounting.TS.Lanes

open ResourceAccounting.TS.Checked Flock.SoundAtBridge

end ResourceAccounting.TS.Lanes

namespace ResourceAccounting.TS.Group

open ResourceAccounting.TS ResourceAccounting.TS.Checked ResourceAccounting.TS.Lanes Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A proof accepted on every path with an opening that fails its relation is unsound with probability 1**, so no
layer with one is sound at a bound below 1. -/
theorem not_soundAt {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} [Nonempty (Paths Ω S)] (pf : ProofLayer Ω S nπ V)
    (w : Fin S) (i : Fin nπ) (h : ∀ x, pf.Unsound x w i) {ε : ℝ} (hε : ε < 1) : ¬ pf.SoundAt ε := by
  intro hs
  have z : Pouw.RelativeSecurity.prob (fun x => ¬ pf.Unsound x w i) = 0 := by
    unfold Pouw.RelativeSecurity.prob Claims.prUniform
    simp [h]
  have e := Pouw.RelativeSecurity.prob_add_not (fun x => pf.Unsound x w i)
  linarith [hs w i]

/-- **`LeafGroupSound`'s nontrivial lemma**: at a group layer whose proof `i` of round `w` is accepted on every path
with an opening that fails its relation, and no clash of its sessions' registrants' records or of their known-content
ties, the guarantee's conclusion fails at every `δ` with `(T.lanes · n + 1) δ < 1`: the bound it states is about an
event that can happen, and the premises are what rule it out. -/
theorem groupLayer_not_soundAt {Ω : Type} [Fintype Ω] {S nπ n : ℕ} {V : Type} [Nonempty (Paths Ω S)]
    (T : PouwWindow.Lanes.Tile) (sess : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) (known : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → Known)
    (okSalt : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → ByteArray)
    (full : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → Array String)
    (want : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → List _root_.Flock.Registered.Port)
    (opened : Paths Ω S → Fin S → Fin nπ → V) (rel : Paths Ω S → Fin S → Fin nπ → V → Prop)
    (right : Paths Ω S → (Fin S → Fin nπ → V) → Prop) (w : Fin S) (i : Fin nπ)
    (hacc : ∀ x, (groupLayer T n sess out known okSalt full want opened rel right).accepted x w i)
    (hrel : ∀ x, ¬ rel x w i (opened x w i)) (hcoll : ∀ x, ¬ LeafColl (sess x w i))
    (hkcoll : ∀ x, ¬ LeafKnownColl (sess x w i) fun k => okKnown (known x w i k) (okSalt x w i k))
    {δ : ℝ} (hδ : (T.lanes * n + 1) * δ < 1) :
    ¬ (groupLayer T n sess out known okSalt full want opened rel right).SoundAt ((T.lanes * n + 1) * δ) :=
  not_soundAt _ w i (fun x => ⟨hacc x, fun h => Or.elim h (hrel x) fun h => Or.elim h (hcoll x) (hkcoll x)⟩) hδ

end ResourceAccounting.TS.Group

