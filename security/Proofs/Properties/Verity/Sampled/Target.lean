module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Flock.Bridge.V5.KSLive
public import Proofs.Flock.Bridge.Count
public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Bridge.Deployed
public import Proofs.Core.Game.Prob
public import Proofs.Properties.Verity.Sampled.Unbound
public import Proofs.Properties.Verity.Sampled.RecursiveRefused

/-! SPOT's target at the deployed device, from the coupling. -/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

namespace Live

variable (A : Live)

noncomputable abbrev stmt :=
  Flock.Bridge.stmtOf A.P A.κ (_root_.Properties.Spot.oneStageOf A.design) A.desc ⟨"", 0, []⟩ A.roots

noncomputable abbrev part := Flock.Bridge.partOf A.P A.κ (Flock.Bridge.nsOf A.desc A.qr)

noncomputable def traceAt (w : Claims.Sampling.Wit) : A.fileRead.circuit.Trace :=
  cast (congrArg Verity.Circuit.Trace A.circuit_eq.symm)
    (Flock.Bridge.traceOf A.D A.hU A.n A.κ (w.val Claims.Sampling.sha512Msg A.Cd A.stmt))

theorem extract_eq (τ : FlockSoundness.Game.Strategy A.game) :
    A.extract τ = A.traceAt (Flock.Bridge.witDep A.Cc A.Cd Claims.Sampling.sha512Msg A.roots (A.Xr τ)) :=
  rfl

end Live

def ConstsOpen (r : Verity.ExecutionPhaseFiles) (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop) : Prop :=
  ∀ g b, r.consts g = some b → ∀ v : BitVec (r.circuit.w g), (∀ i : Fin (r.circuit.w g), v.getLsbD i.val = b) → op g v

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

namespace Live

variable (A : Live)

open Classical in
theorem v5_at (L : FlockSoundness.Audit.SamplingDesign A.n) (hL : Flock.Bridge.execSel (n := A.n) A.src A.design A.n = L) (t : ℕ)
    (B : Finset ℕ) (τ : FlockSoundness.Game.Strategy (auditLive L A.Reg (liveOfZJ A.dg (planZHJP A.dj A.y₀) A.N A.wr))) :
    let σ := simZJA execArith hs512 (planZHJP A.dj A.y₀) (AcceptsZKP A.I A.own) A.wr
      (opnAtZHJP (planZHJP A.dj A.y₀) A.I A.own A.dj) A.dg (decAtZJ (planZHJP A.dj A.y₀)) (fun _ _ _ _ => true)
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) A.N
      (decodesZCJH_custodyP A.I A.own A.ht A.hcS (.subset A.kd) A.kd A.hdiv A.dj A.y₀ A.wr L A.hTags A.hCust A.hsz) τ
    let Xr := XplurZC execArith hs512 A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb (reg σ) (cont σ) A.k A.Rw
    (∀ ω j, TableCRZ execArith hs512 (planZHJP A.dj A.y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJP A.dj A.y₀) (cont σ ω)) (A.qF t) (A.qS t)) →
    LinkCRZC execArith hs512 A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb (reg σ) (cont σ) A.k A.Rw A.M (A.tL t) →
    Verity.Game.prob (fun o => LiveAcceptsZKP A.I A.own o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg A.Cd
          ((Flock.Bridge.designOf (_root_.Properties.Spot.oneStageOf A.design)).Pa A.part B) A.stmt A.part
          (Flock.Bridge.witDep A.Cc A.Cd Claims.Sampling.sha512Msg A.roots Xr))
        (auditLive L A.Reg (liveOfZJ A.dg (planZHJP A.dj A.y₀) A.N A.wr)) τ ≤
      ENNReal.ofReal ((Flock.Bridge.designOf (_root_.Properties.Spot.oneStageOf A.design)).Pa A.part B) +
        ksAvgStrictZ (planZHJP A.dj A.y₀) L A.k (A.qF t) (A.qS t) (reg σ) +
        ENNReal.ofReal ((∑ p : A.vb.Pos,
            (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb L p) : ℝ) /
              Fintype.card L.Ω) *
          (2 * (A.tL t * (1 + A.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * A.M) + A.k / (Real.exp 1 * A.Rw)) /
            (1 - A.ρ)) +
        A.rootsErr := by
  subst hL
  have hCust : FlockSoundness.Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPtsOf A.D.unroll.g (A.kd / nTab A.I))
      A.I A.own A.dj (fun S R => (planZHJP (Reg := A.Reg) A.dj A.y₀ S R).length) A.wr fun _ => A.design :=
    (congrArg (fun l => FlockSoundness.Assumptions.Zk.RecordCustodyZKJPLaw
      (mPts := mPtsOf A.D.unroll.g (A.kd / nTab A.I)) A.I A.own A.dj
      (fun S R => (planZHJP (Reg := A.Reg) A.dj A.y₀ S R).length) A.wr fun _ => l) A.hsub).mpr A.hCust
  have h := Flock.SecurityProofs.SampledKSLiveV5 A.n A.Reg A.I A.own A.ht A.D (Flock.Desc.parseStmt_of_parse A.hD)
    {} A.hpi A.pub A.hpub A.hscope (Flock.Bridge.rowsOk_of_scopeOkH A.hscope) A.hdup A.desc A.qr A.hpop A.lj A.tb
    A.hdiv A.dj A.S₀ A.d₀ A.y₀ A.wr A.dg A.N A.Ω A.Lb A.src A.hA3 hCust A.k A.Rw A.M A.hk A.hRw A.hM A.ρ A.hρ A.hr
    (A.tL t) (by unfold Live.tL; positivity) B τ (A.qF t) (A.qS t) ⟨"", 0, []⟩
  rw [dif_pos A.hTags] at h
  exact h

theorem units_eq : A.fileRead.units = A.n :=
  (Flock.Bridge.unitsOfFile_eq (Flock.Bridge.parsedOf_eq A.hD A.hpubO A.hU)).trans A.hnO

theorem ks {r : Verity.ExecutionPhaseFiles} (hr : r = Verity.ExecutionPhaseFiles.one A.fileRead)
    (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop) (hop : A.OpensUnder r op) (hc : ConstsOpen r op)
    (tr : r.circuit.Trace) (τ : FlockSoundness.Game.Strategy A.game) (htr : HEq tr (A.extract τ)) (t : ℕ)
    (hF : A.FinderCR τ t) (hI : ¬ ResourceAccounting.IntegrityHolds r.parts op tr A.ε) :
    Verity.Game.prob (fun o => LiveAcceptsZKP A.I A.own o) A.game τ ≤ A.sessionDelta t := by
  classical
  subst hr
  obtain rfl := eq_of_heq htr
  have hns : (Flock.Bridge.nsOf A.desc A.qr).sum = A.n := Flock.Bridge.nsOf_sum A.hpop
  let w := Flock.Bridge.witDep A.Cc A.Cd Claims.Sampling.sha512Msg A.roots (A.Xr τ)
  let Fs := Claims.Sampling.faulty Claims.Sampling.sha512Msg A.Cd A.stmt A.part w
  have hFs : Fs ⊆ Finset.range A.n := by
    intro v hv
    have h := (Finset.mem_filter.1 hv).1
    rwa [Flock.Bridge.blocks_length_partOf A.P hns] at h
  have hcard : A.ε * A.n < (Fs.card : ℝ) := by
    have h1 : (ResourceAccounting.unbound A.fileRead.parts op (A.traceAt w)).card ≤ Fs.card := by
      rw [← Finset.card_image_of_injective _ Fin.val_injective]
      exact Finset.card_le_card (A.unbound_subset_faulty_part op hop hc w)
    have h2 : A.ε * (A.fileRead.units : ℝ) <
        ((ResourceAccounting.unbound A.fileRead.parts op (A.traceAt w)).card : ℝ) := not_le.mp hI
    calc A.ε * A.n = A.ε * (A.fileRead.units : ℝ) := by rw [A.units_eq]
      _ < _ := h2
      _ ≤ (Fs.card : ℝ) := by exact_mod_cast h1
  have hv := A.v5_at A.L A.L_eq.symm t Fs τ hF.table hF.link
  unfold Live.sessionDelta
  have hmono : Verity.Game.prob (fun o => LiveAcceptsZKP A.I A.own o) A.game τ ≤
      Verity.Game.prob (fun o => LiveAcceptsZKP A.I A.own o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg A.Cd
          ((Flock.Bridge.designOf (_root_.Properties.Spot.oneStageOf A.design)).Pa A.part Fs) A.stmt A.part w)
        A.game τ :=
    Verity.Game.prob_mono (fun o ho => ⟨ho, fun hR =>
      lt_irrefl ((Flock.Bridge.designOf (_root_.Properties.Spot.oneStageOf A.design)).Pa A.part Fs) hR⟩) _ τ
  refine (hmono.trans hv).trans (add_le_add (add_le_add (add_le_add ?_ ?_) le_rfl) le_rfl)
  · unfold Live.sampleMiss
    rw [Flock.Bridge.pa_partOf A.P hns _ Fs hFs]
    apply ENNReal.ofReal_le_ofReal
    have : Finite {B : Finset ℕ // B ⊆ Finset.range A.n ∧ A.ε * A.n < B.card} :=
      Finite.of_injective (fun B => (⟨B.1, Finset.mem_powerset.2 B.2.1⟩ : (Finset.range A.n).powerset))
        fun a b h => Subtype.ext (by simpa using congrArg Subtype.val h)
    exact le_ciSup (f := fun B : {B : Finset ℕ // B ⊆ Finset.range A.n ∧ A.ε * A.n < B.card} =>
        (_root_.Properties.Spot.oneStageOf A.design).Pa (Flock.Bridge.nsOf A.desc A.qr) B.1)
      (Set.finite_range _).bddAbove ⟨Fs, hFs, hcard⟩
  · exact le_iSup (fun R : A.Reg => ksAvgStrictZ (planZHJP A.dj A.y₀) A.L A.k (A.qF t) (A.qS t) R) _

end Live

variable (V : Vocab d α) (k : Verity.Registration.Composite)

theorem sqrt_le_twoRun {f : Claims.Finder} (hf : Claims.StrictBirthday Verity.sha512Msg f) {q : ℝ}
    (hq : ∀ s, (f.cost s : ℝ) ≤ q) : ENNReal.ofReal (Real.sqrt (f.Pr Verity.sha512Msg)) ≤ Live.twoRun q :=
  ENNReal.ofReal_le_ofReal (Real.sqrt_le_sqrt (hf q hq))

theorem spotSound_of_targets {K : Constructions V} {S : ByteArray → Array ByteArray → Option Live}
    (hD : Deployed V K S) (hCov : Covers V k S) (hC : ∀ θ F A, S θ F = some A → CouplingTarget V k K A θ F) :
    V.SpotSound k := by
  intro θ F c σ
  by_cases hP : ∃ p, ParamsAt θ p
  swap
  · exact recursiveForm_refused V k θ hP F c σ
  unfold Vocab.error ResourceAccounting.condError
  beta_reduce
  split_ifs with hE
  swap
  · exact le_top
  obtain ⟨hfs, hpr, hFC, hUf⟩ := hE
  by_cases hI : ResourceAccounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε
  · calc Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ
          ≤ Verity.Game.prob (fun _ => False) (V.game θ F c) σ :=
            Verity.Game.prob_mono (fun o ho => ho.2 hI) _ σ
      _ = 0 := Verity.Game.prob_false _ σ
      _ ≤ _ := zero_le
  have hver : Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ :=
    Verity.Game.prob_mono (fun o ho => ho.1) _ σ
  cases hS : S θ F with
  | none => exact hver.trans ((hCov θ F hS hP c σ hfs hFC).trans (le_add_right (le_add_right le_rfl)))
  | some A =>
    have hsu := hD.setsUp θ F A hS
    have hFin := hD.finderCR θ F A hS c σ hfs hpr
    have hε : (K.live A θ F c σ).ε = (V.certOf θ).ε := hsu.share.symm
    have hks := (K.live A θ F c σ).ks (r := V.read θ F) hsu.read (V.Opens θ F c) (hD.opens θ F A hS c)
      (fun g b hb v hv => by unfold Vocab.Opens; rw [hb]; exact hv) (V.extract θ F c σ) (K.project A θ F c σ)
      (hD.extract θ F A hS c σ) (V.cost θ F c σ) hFin.session (by rw [hε]; exact hI)
    rw [Live.sessionDelta_extend] at hks
    set t := V.cost θ F c σ
    have hmsg : (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder A θ F c σ j).Pr Verity.sha512Msg))) ≤
        A.rounds * Live.twoRun (A.qM t) := by
      calc (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder A θ F c σ j).Pr Verity.sha512Msg)))
          ≤ ∑ _j : Fin A.rounds, Live.twoRun (A.qM t) :=
            Finset.sum_le_sum fun j _ => sqrt_le_twoRun (hFin.msg j) (K.msg_calls A θ F c σ j)
        _ = A.rounds * Live.twoRun (A.qM t) := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    have hrec : ENNReal.ofReal (Real.sqrt ((K.recordFinder A θ F c σ).Pr Verity.sha512Msg)) ≤ Live.twoRun (A.qM t) :=
      sqrt_le_twoRun hFin.record (K.record_calls A θ F c σ)
    have hbind : ENNReal.ofReal (Real.sqrt ((K.bindFinder A θ F c σ).Pr Verity.sha512Msg)) ≤ Live.twoRun (A.qM t) :=
      sqrt_le_twoRun hFin.bind (K.bind_calls A θ F c σ)
    have hread : ENNReal.ofReal ((K.readFinder θ F c σ).Pr Verity.sha512Msg) ≤
        ENNReal.ofReal (V.crBound θ .sampled t) := by
      rw [← hD.finder θ F c σ]; exact ENNReal.ofReal_le_ofReal (V.finderPr_le_crBound hFC)
    have hforge : ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify) ≤
        ENNReal.ofReal (V.epsForge θ (V.callsMul θ .sampled * t + V.callsAdd θ .sampled) (V.signatures θ)) := by
      rw [← hD.forger θ F c σ]
      exact ENNReal.ofReal_le_ofReal
        (hUf (fun x => V.forger_calls .sampled θ F c σ x) (fun x => V.forger_signed .sampled θ F c σ x))
    simp only [hD.loss, add_zero]
    refine hver.trans ((hC θ F A hS c σ).trans ?_)
    calc _ ≤ A.sessionDelta t + (A.rounds * Live.twoRun (A.qM t) + Live.twoRun (A.qM t) + Live.twoRun (A.qM t) +
            A.hidingTerm + ENNReal.ofReal (V.crBound θ .sampled t) +
            ENNReal.ofReal (V.epsForge θ (V.callsMul θ .sampled * t + V.callsAdd θ .sampled) (V.signatures θ))) :=
          add_le_add hks
            (add_le_add (add_le_add (add_le_add (add_le_add (add_le_add hmsg hrec) hbind) le_rfl) hread) hforge)
      _ ≤ A.delta t (V.crBound θ .sampled t) +
            ENNReal.ofReal (V.epsForge θ (V.callsMul θ .sampled * t + V.callsAdd θ .sampled) (V.signatures θ)) := by
          unfold Live.delta
          calc _ = A.sessionDelta t + A.rounds * Live.twoRun (A.qM t) + Live.twoRun (A.qM t) +
                  Live.twoRun (A.qM t) + A.hidingTerm + ENNReal.ofReal (V.crBound θ .sampled t) +
                  ENNReal.ofReal (V.epsForge θ (V.callsMul θ .sampled * t + V.callsAdd θ .sampled)
                    (V.signatures θ)) := by ring
            _ ≤ _ := add_le_add (add_le_add le_self_add le_rfl) le_rfl
      _ ≤ _ := add_le_add (hD.delta θ F A hS t) le_rfl

end Verity.Sampled
