module

public import Flock.Desc
public import Proofs.Flock.Soundness.Assumptions.Zk
public import Proofs.Flock.Soundness.Discharge.ZkLink.Public
public import Proofs.Flock.Soundness.Discharge.ZkReg.BindHJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.Defs
public import Proofs.Flock.Soundness.Discharge.ZkReg.DefsHJ
public import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundSampleOk
public import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJ
public import Proofs.Flock.Soundness.Registered.ReadsX
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

/-!
# One Flock session with a registrant's opening per unit (trusted)

`RowsSession` is `Flock.Guarantees.EndToEndRegisteredRows`' setting as a value, its binders in its order, with its one
registrant's opening `rd` replaced by one per unit (`rdU u`); its circuit `c` is the one its descriptor `D` unrolls
to. `EndToEndRegisteredRows` reads the input rows only where the verifier owns every registered read of `rd` at the
outcome's sample (`OwnReadsHJ` is one condition over all of them), so at an `rd` covering every unit's reads it links no input row on a sample that leaves one out. With `rdU u` the
opening of unit `u`'s reads alone (`Registered.ReadsX.restrict`, in `Proofs.Flock.Recursive.ZkReg.Off`), a sampled
proof block's input rows are linked wherever its own reads are owned, whatever else is sampled; the cost is a union bound over
the `n` units (`Flock.SoundAtBridge.RowsSession.prob_badU_le`), `n` times `EndToEndRegisteredRows`' bound.

The session's game, prover, value layer and events are those `EndToEndRegisteredRows` reads, at `rdU u` for unit
`u`:

* `G`, `τ`: the live audit and its prover; `Out`: its outcome (the registration, the sample and the transcript);
* `Xv`: the plurality value layer the rows extraction finds at the registration (`XplurZC`);
* `IncorrectAt u`: sampled proof block `u` is wrong on the registration (`incorrectRegZ`);
* `RowsAt o u`: unit `u`'s rows are right at `Xv` (`RowsRightZC`) at its own owned registered reads;
* `BadU o`: verity-verify-proof accepts and some sampled proof block is wrong or its rows are off;
* `CR`: `EndToEndRegisteredRows`' named assumptions at every unit's opening; `bound`: its bound.
-/

@[expose] public section

namespace Flock.SoundAtBridge

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal
open Classical

/-- **One Flock session in `EndToEndRegisteredRows`' setting, with a registrant's opening per unit**: its binders,
in its order (the roots `own`, then the statement `I` with its coin, then the live prover `τ`), with its one `rd`
replaced by `rdU u` for each unit `u`. -/
structure RowsSession where
  n : ℕ
  Reg : Type
  own : Array Flock.Registered.Port
  I : Inputs
  ht : I.tags.typed = false
  D : Flock.Desc
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  hU : Built D.unroll.unit
  pub : Public
  hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub
  hscope : Discharge.Layout.scopeOkH D.unroll = true
  kd : ℕ
  hdiv : kd % nTab I = 0
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : SampleSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  rdU : Fin n → FlockSoundness.Registered.ReadsX
    (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
    (rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub)
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  Ω : Type
  [instF : Fintype Ω]
  [instN : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hTags : TagsOkZHJ I
  hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ
  t' : ℝ
  ht' : 0 ≤ t'
  τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))
  qF : ℝ
  qS : ℝ
  qR : ℝ

attribute [instance] RowsSession.instF RowsSession.instN

noncomputable section

namespace RowsSession

variable (s : RowsSession)

/-- The circuit the session's descriptor unrolls to. -/
abbrev c : Flock.Circuit := s.D.unroll

/-- `HmOut.parse` reads the session's circuit from its circuit file. -/
theorem hc : HmOut.parse (zkTags s.I) s.I.circuitFile s.I.tables = .ok s.c := Flock.Desc.parse_flat s.hD

/-- The key program's partition. -/
abbrev P := (keyProg s.c s.hU s.n).partition (Discharge.Layout.outsOf s.c s.hU s.n)

/-- The key program's circuit, whose wires the value layer assigns. -/
abbrev Circ := (keyProg s.c s.hU s.n).circuit (Discharge.Layout.outsOf s.c s.hU s.n)

/-- The value binding of the session's rows. -/
abbrev vb := (rsZHJS (Reg := s.Reg) s.dj s.y₀ s.ht s.hc s.hscope s.hU s.hpub).bindingZ
  (rsZHJS_computes (Reg := s.Reg) s.dj s.y₀ s.ht s.hc s.hscope s.hU s.hpub)

end RowsSession

end

end Flock.SoundAtBridge
