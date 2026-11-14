module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Flock.Bridge.DrawnLaw
public import Proofs.Flock.Soundness.Discharge.Hidden.PortsP
public import Proofs.Flock.Soundness.Discharge.Hidden.Built
public import Proofs.Flock.Soundness.Discharge.HiddenExec.SetupP
public import Proofs.Flock.Soundness.Discharge.Layout.Drawn
public import Proofs.Flock.Soundness.Discharge.ZkBind.Rate
public import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundDrawOk

/-!
# From a setup to a session's fields

Three general facts turn a table `verify --zk` sets up into `Live`'s fields: the reference table (`y₀_of_setup`), the
public file over the circuit's empty public input ports (`pub_of_setup`), and the honest bound's rate at every table of
the plan (`hr_planZHJP`).
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.LiveFacts

open Lean (Json)
open FlockSoundness.Discharge

open Composed (inputsJ nTab) in
/-- A one-table setup at a sample file whose proved units number `kd`, that passes the ZK shape check, supplies the
reference table `Live` reads. -/
theorem y₀_of_setup (n : ℕ) {I : Exec.Inputs} {D : Flock.Desc} {kd : ℕ} {d : Json} {st : Flock.Stmt}
    (hc1 : I.count = 1) (ht : I.tags.typed = false)
    (hD : Flock.Desc.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok D)
    (hprov : ∀ D', Flock.Draw.UnitDraw.ofJson d = .ok D' → D'.proved.size = kd)
    (hst : Flock.Stmt.setupHidden (inputsJ I 0).tags (inputsJ I 0).circuitFile (inputsJ I 0).publicFile
      (inputsJ I 0).coins (inputsJ I 0).tables (some d) (inputsJ I 0).partition (inputsJ I 0).program none = .ok st)
    (hshape : Flock.Zk.shapeOk st = .ok ()) :
    ∃ (S₀ : Finset (Fin n)) (d₀ : Json),
      Nonempty (ZkHidden.DrawSetupZHJP (inputsJ I 0) none (Exec.mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀) := by
  sorry -- piece P8 (session): `y₀_of_setup` of the earlier honest session, at `some d` for `recordDraw`

open Composed (inputsJ) in
/-- A setup without the verifier's own public words supplies `Live`'s public-file fields. -/
theorem pub_of_setup {I : Exec.Inputs} {D : Flock.Desc} {pubO : Flock.Public} {n : ℕ}
    {d : Json} {st : Flock.Stmt} (hc1 : I.count = 1)
    (hD : Flock.Desc.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok D)
    (hst : Flock.Stmt.setupHidden (inputsJ I 0).tags (inputsJ I 0).circuitFile (inputsJ I 0).publicFile
      (inputsJ I 0).coins (inputsJ I 0).tables (some d) (inputsJ I 0).partition
      (inputsJ I 0).program none = .ok st)
    (hpubO : Flock.HmOut.loadPublic (ZkExec.zkTags I) D.unroll I.publicFile = .ok pubO)
    (hnO : pubO.n = n)
    (hports : Flock.PortRoots.checkOf (.ok D.unroll) (ZkExec.zkTags I) I.publicFile = .ok ()) :
    Flock.HmIn.parse D.unroll = .ok {} ∧
      ∃ pub, Flock.HmIn.loadPublic (ZkExec.zkTags I) D.unroll {} I.publicFile = .ok pub ∧
        Flock.PortRoots.dupId D.unroll pub = false ∧ pub.n = n := by
  sorry -- piece P8 (session): `pub_of_setup` of the earlier honest session

open ZkLink ZkBind ZkHidden Composed in
/-- Every table of the plan meets the honest bound's rate. -/
theorem hr_planZHJP {n mPts : ℕ} {I : Exec.Inputs} {own : Option ByteArray}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} {d₀ : Json}
    (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀) {Reg : Type} {k : ℕ} {ρ : ℝ}
    (hkρ : (2 : ℝ) ^ 23 ≤ ρ * (Real.exp 1 * k)) (S : Finset (Fin n)) (R : Reg)
    (j : Fin (planZHJP (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ := by
  sorry -- piece P8 (session): `hr_planZHJP` of the earlier honest session

end Verity.Sampled.Instance.HonestSession.LiveFacts
