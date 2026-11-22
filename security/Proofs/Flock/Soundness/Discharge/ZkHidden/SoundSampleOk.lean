module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.Defs
public import Proofs.Flock.Soundness.Discharge.ZkLink.SampleOk
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
import Proofs.Flock.Soundness.Discharge.Exec.SamplingDesignSample
import Proofs.Flock.Soundness.Discharge.Hidden.Theorems
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt

/-!
# setupHidden's link points at a sample

setupHidden, under the `--zk` tags, sets up at `mPtsOf` of the circuit's `g` and the units the sample proves, as at
public outputs (`ZkLink/SampleOk.lean`):

* `setupHidden_mPts_sampled`: setupHidden's `m_pts` at a sample is `mPtsOf` of the circuit's `g` and the units the sample
  proves, as setupH's (`Exec.setupH_mPts_sampled`), from `Hidden.setupHidden_spec`.
* `setupHidden_mPts_of_provedG` reads `g` off the session, and `setupHidden_mPts_of_provedSt` off the statement circuit
  (`HmOut.parseStmt`, a templated file's template unrolled), at a sample file proving `N₀` units.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Flock Exec ZkExec ZkLink HiddenExec Finset FlockSoundness.Refine
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {partition : Option Json} {program : Option ByteArray} {st : Flock.Stmt}

/-- **`m_pts` at a sampled hidden-output session** is `mPtsOf` of the circuit's `g` and the number of units the sample
proves. -/
theorem setupHidden_mPts_sampled {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) :
    ∃ ud, Draw.UnitSample.ofJson d = .ok ud ∧ st.mPts = mPtsOf st.c.g ud.proved.size := by
  have s := Hidden.setupHidden_spec h
  obtain ⟨pop, -, hd⟩ := setupHidden_sampled h
  obtain ⟨ud, hud, hn⟩ := sampled_n (c := { st.c with outPorts := #[] }) hd
  refine ⟨ud, hud, ?_⟩
  rw [s.mPts, s.nbl, s.blocksReal, s.n, hn]
  rfl

/-- `setupHidden_mPts_sampled` at the session's own `g`, at a sample file proving `N₀` units. -/
theorem setupHidden_mPts_of_provedG {g : ℕ} {d : Json} {N₀ : ℕ} (hg : st.c.g = g)
    (hD : ∀ D, Draw.UnitSample.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) : st.mPts = mPtsOf g N₀ := by
  obtain ⟨ud, hud, hm⟩ := setupHidden_mPts_sampled h
  rw [hm, hg, hD ud hud]

/-- `setupHidden_mPts_sampled` at the statement circuit (`HmOut.parseStmt`), on untyped tags, at a sample file proving
`N₀` units. -/
theorem setupHidden_mPts_of_provedSt {c : Circuit} {d : Json} {N₀ : ℕ} (ht : tags.typed = false)
    (hc : HmOut.parseStmt tags cf tables = .ok c) (hD : ∀ D, Draw.UnitSample.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) : st.mPts = mPtsOf c.g N₀ :=
  setupHidden_mPts_of_provedG (stmt_g_of_setupHidden ht hc h).symm hD h

variable {n : ℕ}

end FlockSoundness.Discharge.ZkHidden
