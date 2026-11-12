module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Flock.Bridge.V5.KSLive
public import Proofs.Flock.Bridge.Count
public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Bridge.Deployed
public import Proofs.Core.Game.Prob
public import Proofs.Properties.Verity.Sampled.Unbound
public import Proofs.Properties.Verity.Sampled.RecursiveRefused

/-!
# SPOT's target at the deployed device, from the coupling

`Verity.Vocab.SpotSound` at composite `k`, for a vocabulary whose sampled fields are the deployment's sessions
(`Deployed`, `Covers`), from one target, a `Prop` the owning goal proves: **the coupling** (`CouplingTarget`, at each
session): the sampled verdict in the period's game is at most as likely as Flock's `--zk` verifier's acceptance in the
live audit at the projected strategy, plus the messages' two-continuation finders, the coin trees' hiding term, the read
finder's collision and the forger's forgery (the sample read from the log).

What the proof reads that is proved: Flock's knowledge soundness of the live session (`SampledKSLiveV5`), applied at the
session's sample (`Live.v5_at`, along `Live.L_eq`), A3 and `live-verifier` at the deployed device (`Live.deployed`,
`Live.hCust`), the sampling design's miss (`Flock.Bridge.pa_partOf`, `Flock.Bridge.nsOf_sum`), the bridge from the
committed files' unbound units to the session's incorrect proof blocks (`Live.unbound_subset_faulty_part`), and, at
parameters that don't name the one-stage form, that SPOT accepts nothing (`recursiveForm_refused`).

* `Live.ks`: at a session, a strategy of the live audit whose extracted transcript fails the claim at the session's share
  is accepted with probability at most the session's error (`Live.sessionDelta`). V5 at `B` the extracted witness's
  incorrect set gives the live acceptance at most the sampling design's miss of `B` plus the tables', binding and roots'
  errors; through the bridge `B` holds more than an `ε` share of the proof blocks, so its miss is at most the sample's
  term of the session's error (`Live.sessionDelta`).
* `spotSound_of_targets`: SPOT's target. Where `error`'s premises hold and the claim fails, the coupling bounds the
  verdict by the live acceptance at the projected strategy and the finders' terms: `Live.ks` bounds the first, the strict
  birthday bound (`StrictBirthday`) each two-continuation finder at `2t + 2h` calls, `FinderCR` the read finder at the
  published bound, and `LmsUnforgeable` the forger at `ε_forge`, so the sum is at most the session's `Live.delta` plus
  `ε_forge`, at most the published `δ` plus `ε_forge` (`Deployed.delta`, no loss term).
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

namespace Live

variable (A : Live)

/-- **The session's sampling statement**: the key program's at the session's constants and roots, under the sampling design's
one-stage design (the one `Live.extract` reads). -/
noncomputable abbrev stmt :=
  Flock.Bridge.stmtOf A.P A.κ (_root_.Properties.Spot.oneStageOf A.design) A.desc ⟨"", 0, []⟩ A.roots

/-- **The session's sampling partition**: the key program's proof blocks at the registration's strata. -/
noncomputable abbrev part := Flock.Bridge.partOf A.P A.κ (Flock.Bridge.nsOf A.desc A.qr)

/-- **The transcript a witness opens, on the committed files' circuit**: its first passing opening at each gate's claims
position under the deployed commitment, read at the circuit's gates with its constants (`Flock.Bridge.transcriptOf`). -/
noncomputable def transcriptAt (w : Claims.Sampling.Wit) : A.fileRead.circuit.Trace :=
  cast (congrArg Verity.Circuit.Trace A.circuit_eq.symm)
    (Flock.Bridge.transcriptOf A.D A.hU A.n A.κ (w.val Verity.sha512Msg A.Cd A.stmt))

theorem extract_eq (τ : FlockSoundness.Game.Strategy A.game) :
    A.extract τ = A.transcriptAt (Flock.Bridge.witDep A.Cc A.Cd Verity.sha512Msg A.roots (A.Xr τ)) :=
  rfl

end Live

/-- **The openings `op` of the committed files `r` hold at their constants**: a gate `r` fixes to `b` opens to every
value all of whose bits are `b`, as `Vocab.Opens` reads a constant. -/
def ConstsOpen (r : Verity.FileRead) (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop) : Prop :=
  ∀ g b, r.consts g = some b → ∀ v : BitVec (r.circuit.w g), (∀ i : Fin (r.circuit.w g), v.getLsbD i.val = b) → op g v

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

namespace Live

variable (A : Live)

open Classical in
/-- **V5 at the session, at any sample equal to V5's**: `SampledKSLiveV5` at the session's binders, the link finders
charged `t + w` per run and the tables' finders at `2t + 2v` and `t + v`, at a sample `L` that is `Flock.Bridge.execSel`
at the session's bytes and sampling design; at `L = A.L` (`Live.L_eq`) its simulator and extractor are the session's
(`Live.sim`, `Live.Xr`). The session's `--zk` tags are in `TagsOkZHJ` (`Live.hTags`), so V5's bound is the one it states
there, and `live-verifier` at the registration's sampling design is the sample file's (`Live.hsub`, `Live.hCust`). -/
theorem v5_at (L : FlockSoundness.Audit.Law A.n) (hL : Flock.Bridge.execSel (n := A.n) A.src A.design A.n = L) (t : ℕ)
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
        ¬ Claims.Sampling.RelS Verity.sha512Msg A.Cd
          ((Flock.Bridge.designOf (_root_.Properties.Spot.oneStageOf A.design)).Pa A.part B) A.stmt A.part
          (Flock.Bridge.witDep A.Cc A.Cd Verity.sha512Msg A.roots Xr))
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

/-- The committed files' units are the session's population. -/
theorem units_eq : A.fileRead.units = A.n :=
  (Flock.Bridge.unitsOfFile_eq (Flock.Bridge.parsedOf_eq A.hD A.hpubO A.hU)).trans A.hnO

/-- **The session's knowledge soundness at a claim that fails**: a strategy of the live audit whose extracted transcript
`tr`, on the committed files `r` the session reads, fails the claim at the session's share under openings that hold
wherever the deployed commitment opens and at the constants, is accepted by Flock's `--zk` verifier with probability at
most the session's error at `t` calls, where the session's per-finder premises hold at `t`. -/
theorem ks {r : Verity.FileRead} (hr : r = A.fileRead)
    (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop) (hop : A.OpensUnder r op) (hc : ConstsOpen r op)
    (tr : r.circuit.Trace) (τ : FlockSoundness.Game.Strategy A.game) (htr : HEq tr (A.extract τ)) (t : ℕ)
    (hF : A.FinderCR τ t) (hI : ¬ Accounting.IntegrityHolds r.parts op tr A.ε) :
    Verity.Game.prob (fun o => LiveAcceptsZKP A.I A.own o) A.game τ ≤ A.sessionDelta t := by
  classical
  subst hr
  obtain rfl := eq_of_heq htr
  have hns : (Flock.Bridge.nsOf A.desc A.qr).sum = A.n := Flock.Bridge.nsOf_sum A.hpop
  let w := Flock.Bridge.witDep A.Cc A.Cd Verity.sha512Msg A.roots (A.Xr τ)
  let Fs := Claims.Sampling.faulty Verity.sha512Msg A.Cd A.stmt A.part w
  have hFs : Fs ⊆ Finset.range A.n := by
    intro v hv
    have h := (Finset.mem_filter.1 hv).1
    rwa [Flock.Bridge.blocks_length_partOf A.P hns] at h
  have hcard : A.ε * A.n < (Fs.card : ℝ) := by
    have h1 : (Accounting.unbound A.fileRead.parts op (A.transcriptAt w)).card ≤ Fs.card := by
      rw [← Finset.card_image_of_injective _ Fin.val_injective]
      exact Finset.card_le_card (A.unbound_subset_faulty_part op hop hc w)
    have h2 : A.ε * (A.fileRead.units : ℝ) <
        ((Accounting.unbound A.fileRead.parts op (A.transcriptAt w)).card : ℝ) := not_le.mp hI
    calc A.ε * A.n = A.ε * (A.fileRead.units : ℝ) := by rw [A.units_eq]
      _ < _ := h2
      _ ≤ (Fs.card : ℝ) := by exact_mod_cast h1
  have hv := A.v5_at A.L A.L_eq.symm t Fs τ hF.table hF.link
  unfold Live.sessionDelta
  have hmono : Verity.Game.prob (fun o => LiveAcceptsZKP A.I A.own o) A.game τ ≤
      Verity.Game.prob (fun o => LiveAcceptsZKP A.I A.own o ∧
        ¬ Claims.Sampling.RelS Verity.sha512Msg A.Cd
          ((Flock.Bridge.designOf (_root_.Properties.Spot.oneStageOf A.design)).Pa A.part Fs) A.stmt A.part w)
        A.game τ :=
    Verity.Game.prob_mono (fun o ho => ⟨ho, fun hR =>
      lt_irrefl ((Flock.Bridge.designOf (_root_.Properties.Spot.oneStageOf A.design)).Pa A.part Fs) hR⟩) _ τ
  refine (hmono.trans hv).trans (add_le_add (add_le_add (add_le_add ?_ ?_) le_rfl) le_rfl)
  · unfold Live.drawMiss
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

/-- A two-continuation finder at most `q` SHA-512 calls on every coin, under the strict birthday bound, has its term at
most `Live.twoRun q`. -/
theorem sqrt_le_twoRun {f : Claims.Finder} (hf : Claims.StrictBirthday Verity.sha512Msg f) {q : ℝ}
    (hq : ∀ s, (f.cost s : ℝ) ≤ q) : ENNReal.ofReal (Real.sqrt (f.Pr Verity.sha512Msg)) ≤ Live.twoRun q :=
  ENNReal.ofReal_le_ofReal (Real.sqrt_le_sqrt (hf q hq))

/-- **SPOT's target at the deployed device**: soundness's vocabulary at `.sampled` reading the constructions `K`
at the deployment's sessions `S` (`Deployed`), which cover the one-stage form (`Covers`), meets SPOT's target
at composite `k`, from the coupling at each session (`CouplingTarget`). Off the one-stage form SPOT accepts nothing
(`recursiveForm_refused`). -/
theorem spotSound_of_targets {K : Constructions V} {S : ByteArray → Array ByteArray → Option Live}
    (hD : Deployed V K S) (hCov : Covers V k S) (hC : ∀ θ F A, S θ F = some A → CouplingTarget V k K A θ F) :
    V.SpotSound k := by
  intro θ F c σ
  by_cases hP : ∃ p, ParamsAt θ p
  swap
  · exact recursiveForm_refused V k θ hP F c σ
  unfold Vocab.error Accounting.condError
  beta_reduce
  split_ifs with hE
  swap
  · exact le_top
  obtain ⟨hfs, hpr, hFC, hUf⟩ := hE
  by_cases hI : Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε
  · calc Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ
          ≤ Verity.Game.prob (fun _ => False) (V.game θ F c) σ :=
            Verity.Game.prob_mono (fun o ho => ho.2 hI) _ σ
      _ = 0 := Verity.Game.prob_false _ σ
      _ ≤ _ := zero_le
  have hver : Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ :=
    Verity.Game.prob_mono (fun o ho => ho.1) _ σ
  cases hS : S θ F with
  | none => exact hver.trans ((hCov θ F hS hP c σ hfs).trans (le_add_right (le_add_right le_rfl)))
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
