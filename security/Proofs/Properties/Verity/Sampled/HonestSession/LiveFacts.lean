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
  have htJ : (inputsJ I 0).tags = ZkExec.zkTags I := by
    show Exec.tagsAt _ I.count 0 = _
    unfold Exec.tagsAt
    exact ite_eq_left (by omega)
  have hn1 : nTab I = 1 := by
    unfold nTab
    simp [hc1]
  obtain ⟨⟨sch, tPad⟩, hz⟩ := ZkHidden.schedule_ok_of_shapeOk hshape
  obtain ⟨sch₀, hf⟩ := ZkHidden.fast100_of_schedule hz
  have hst' := hst
  rw [htJ] at hst'
  have hmPts : st.mPts = Exec.mPtsOf D.unroll.g (kd / nTab I) := by
    rw [hn1, Nat.div_one]
    exact ZkHidden.setupHidden_mPts_of_proved (Flock.Desc.parse_flat hD) hprov hst'
  obtain ⟨pi', pop, -, -, hdr⟩ := Hidden.setupHidden_drawnP hst
  obtain ⟨Dd, hDd, -⟩ := Layout.drawn_spec
    (c := { st.c with outPorts := if pi'.ports.isEmpty then #[] else #[("public inputs", pi'.words)] }) hdr
  obtain ⟨xx, hxx⟩ := HiddenExec.drawSetupHP_of_setupHiddenI (I := inputsJ I 0)
    (own := none) (S := (∅ : Finset (Fin n))) (Composed.typedJ ht 0) hst hf hDd
    (fun v hv => by simp at hv) hmPts
  obtain ⟨y, -, -, -⟩ := ZkHidden.drawSetupZHJP_of xx (by rw [hxx]; exact hshape)
    (sch := sch) (tPad := tPad) (by rw [hxx]; exact hz)
  exact ⟨∅, _, ⟨y⟩⟩

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
  have htJ : (inputsJ I 0).tags = ZkExec.zkTags I := by
    show Exec.tagsAt _ I.count 0 = _
    unfold Exec.tagsAt
    exact ite_eq_left (by omega)
  rw [htJ] at hst
  have hc := HiddenExec.setupHidden_c_of_parse (Flock.Desc.parse_flat hD) hst
  obtain ⟨pi, s⟩ := Hidden.setupHidden_specP hst
  obtain ⟨pop, hlp, hown, -⟩ := s.pub
  have hpi : pi = {} := Hidden.parse_empty s.ports (Hidden.checkOwn_none hown)
  have hpp := s.ports
  rw [hc, hpi] at hpp hlp
  have hn : pop.n = n := (Flock.Bridge.loadPublic_inOut hlp hpubO).1.symm.trans hnO
  exact ⟨hpp, pop, hlp, (Flock.Bridge.v5Files_of_checks hlp hn hports).1, hn⟩

open ZkLink ZkBind ZkHidden Composed in
/-- Every table of the plan meets the honest bound's rate. -/
theorem hr_planZHJP {n mPts : ℕ} {I : Exec.Inputs} {own : Option ByteArray}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} {d₀ : Json}
    (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀) {Reg : Type} {k : ℕ} {ρ : ℝ}
    (hkρ : (2 : ℝ) ^ 23 ≤ ρ * (Real.exp 1 * k)) (S : Finset (Fin n)) (R : Reg)
    (j : Fin (planZHJP (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ := by
  unfold rateZ
  rw [show (planZHJP (Reg := Reg) dj y₀ S R).get j =
      tableZHJP dj y₀ S (Fin.cast (planZHJP_length dj y₀ S R) j) from List.get_ofFn _ _]
  refine (div_le_div_of_nonneg_right (pow_le_pow_right₀ (by norm_num) ?_) (by positivity)).trans (cap_div_le hkρ)
  unfold tableZHJP
  split_ifs with h
  · exact zkLogLenCap_23 _ _ _ (Classical.choice h).hz
  · exact zkLogLenCap_23 _ _ _ y₀.hz

end Verity.Sampled.Instance.HonestSession.LiveFacts
