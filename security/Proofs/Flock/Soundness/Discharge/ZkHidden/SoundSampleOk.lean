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
# The challenger's sample at every sample of the live sampling design, at hidden-output `--zk`

`zk_session_soundH` takes `SampleOkZKH I dj mPts` at each of `SamplingDesign.execOS`'s samples: the sample file `dj S` proves `S`'s units,
and setupHidden, under the `--zk` tags, sets up at `mPts` link points. As at public outputs (`ZkLink/SampleOk.lean`), the
first half is custody (`SampleFileZK`) and the second is verity-verify-proof's own arithmetic.

* `setupHidden_mPts_sampled`, `setupHidden_mPts_of_proved`: setupHidden's `m_pts` at a sample is `mPtsOf` of the circuit's
  `g` and the units the sample proves, as setupH's (`Exec.setupH_mPts_sampled`, `Exec.setupH_mPts_of_proved`), from
  `Hidden.setupHidden_spec`. `setupHidden_mPts_of_provedG` reads `g` off the session, and
  `setupHidden_mPts_of_provedSt` off the statement circuit (`HmOut.parseStmt`, a templated file's template unrolled).
* `sampleOkZKH_of_exec`: from `SampleFileZK` at the live sampling design's own stratified sampling design, `SampleOkZKH` at every sample of
  `SamplingDesign.execOS`, at `mPtsOf c.g (∑ s, ks s)`: `ZkLink.sampleOkZK_of_exec`'s proof at setupHidden. `sampleOkZKH_of_execG` is
  it at any `g` every session has, and `sampleOkZKH_of_execSt` at the statement circuit's.
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

/-- `setupHidden_mPts_of_proved` at the session's own `g`. -/
theorem setupHidden_mPts_of_provedG {g : ℕ} {d : Json} {N₀ : ℕ} (hg : st.c.g = g)
    (hD : ∀ D, Draw.UnitSample.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) : st.mPts = mPtsOf g N₀ := by
  obtain ⟨ud, hud, hm⟩ := setupHidden_mPts_sampled h
  rw [hm, hg, hD ud hud]

/-- `setupHidden_mPts_of_proved` at the statement circuit (`HmOut.parseStmt`), on untyped tags. -/
theorem setupHidden_mPts_of_provedSt {c : Circuit} {d : Json} {N₀ : ℕ} (ht : tags.typed = false)
    (hc : HmOut.parseStmt tags cf tables = .ok c) (hD : ∀ D, Draw.UnitSample.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) : st.mPts = mPtsOf c.g N₀ :=
  setupHidden_mPts_of_provedG (stmt_g_of_setupHidden ht hc h).symm hD h

variable {n : ℕ}

end FlockSoundness.Discharge.ZkHidden
