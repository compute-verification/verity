module

public import Proofs.Properties.Verity.Sampled.Target
public import Proofs.Flock.EndToEnd.Joint.V5.Bound

/-! SPOT's subprotocol over a window of several members, each its own session, in the members' order; Flock's
bound at the window is `Flock.SecurityProofs.JointShareV5` applied, not proved again. -/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

theorem sha512Msg_eq : Verity.sha512Msg = Claims.Sampling.sha512Msg := rfl

structure Member (nJ : ℕ) where
  /-- the member's population: its public file's proof blocks -/
  n : ℕ
  I : FlockSoundness.Discharge.Exec.Inputs
  /-- the verifier's own recomputation of each proof block's public input words -/
  own : Option ByteArray
  ht : I.tags.typed = false
  D : Flock.Desc
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  hU : FlockSoundness.Built D.unroll.unit
  hpi : Flock.HmIn.parse D.unroll = .ok {}
  pub : Flock.Public
  hpub : Flock.HmIn.loadPublic (zkTags I) D.unroll {} I.publicFile = .ok pub
  hscope : FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true
  hcount : I.count ≤ 1
  /-- where the member's proof blocks sit in the window's population -/
  emb : Fin n → Fin nJ
  /-- the size whose link points the member's session has at every sample (`ExecutionPhase.hcls`) -/
  kd : ℕ
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : SampleSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  hTags : TagsOkZHJ I
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJP (Reg := ByteArray) (printedPart (n := n)) y₀ S R) j k ≤ ρ
  /-- one opening's verification, in SHA-512 calls -/
  v : ℕ
  /-- one acceptance check, in SHA-512 calls -/
  w : ℕ

namespace Member

variable {nJ : ℕ} (m : Member nJ)

def coinSpec : Flock.CoinTree.Spec := Flock.Zk.coinSpec (m.I.tags.tableNames m.I.count)

def N : ℕ := 2 * 2 ^ m.coinSpec.roundsLog

def rounds : ℕ := 2 * 2 ^ m.coinSpec.roundsLog + 2

noncomputable def toV5 (t : ℕ) : Flock.Guarantees.MemberV5 ByteArray nJ where
  n := m.n
  I := m.I
  own := m.own
  ht := m.ht
  D := m.D
  hD := m.hD
  hU := m.hU
  pi := {}
  hpi := m.hpi
  pub := m.pub
  hpub := m.hpub
  hscope := m.hscope
  hcount := m.hcount
  emb := m.emb
  kd := m.kd
  dj := printedPart
  S₀ := m.S₀
  d₀ := m.d₀
  y₀ := m.y₀
  wr := Claims.Bridge.CustodyZkP.wrZP m.I m.own printedPart
    fun S R => (planZHJP (Reg := ByteArray) printedPart m.y₀ S R).length
  dg := Flock.Sha512.hash
  N := m.N
  hTags := m.hTags
  hCust := recordCustodyZKJPLaw_part m.I m.own
    fun S R => (planZHJP (Reg := ByteArray) printedPart m.y₀ S R).length
  k := m.k
  Rw := m.Rw
  M := m.M
  hk := m.hk
  hRw := m.hRw
  hM := m.hM
  ρ := m.ρ
  hρ := m.hρ
  hr := m.hr
  t' := (t : ℝ) + m.w
  ht' := by positivity
  qF := 2 * (t : ℝ) + 2 * m.v
  qS := (t : ℝ) + m.v

def RootsFix : Prop :=
  Flock.Bridge.RootFixesDomainL m.D.unroll (Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n)

noncomputable def rootsFinder : Claims.Finder :=
  Flock.Bridge.rootsFinderL m.D.unroll (Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n)

noncomputable def rootsCalls : ℕ :=
  Flock.Bridge.rootsCostL m.D.unroll (Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n)

def RootsCR : Prop := m.rootsFinder.Pr Verity.sha512Msg ≤ (m.rootsCalls : ℝ) / 2 ^ (256 : ℝ)

noncomputable def rootsErr : ℝ≥0∞ := ENNReal.ofReal ((m.rootsCalls : ℝ) / 2 ^ (256 : ℝ))

theorem rootsFix_or_one_le (h : m.RootsCR) : m.RootsFix ∨ 1 ≤ m.rootsErr := by
  cases hc : Flock.Bridge.rootsCollL m.D.unroll (Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n) with
  | none => exact Or.inl (Flock.Bridge.fixesL_of_none hc)
  | some z =>
    have hPr : m.rootsFinder.Pr Verity.sha512Msg = 1 := by
      rw [sha512Msg_eq]
      exact Flock.Bridge.PortRoots.finderPr_one fun _ =>
        ⟨z.1, z.2, hc, Flock.Bridge.rootsCollL_collides (x := z.1) (y := z.2) hc⟩
    have h1 : (1 : ℝ) ≤ (m.rootsCalls : ℝ) / 2 ^ (256 : ℝ) := hPr.symm.le.trans h
    exact Or.inr (ENNReal.one_le_ofReal.2 h1)

noncomputable def part (S : Finset (Fin nJ)) : Finset (Fin m.n) := (m.toV5 0).part S

theorem toV5_part (t : ℕ) (S : Finset (Fin nJ)) : (m.toV5 t).part S = m.part S := rfl

end Member

structure ExecutionPhase where
  /-- the window's population: every member's proof blocks -/
  nJ : ℕ
  /-- the members, in the registration's order -/
  ms : List (Member nJ)
  samplingDesign : Flock.Draw.SamplingDesign
  ns : List ℕ
  kd : ℕ
  hns : ns.sum = nJ
  hlaw : Flock.Bridge.SamplingDesignOn samplingDesign ns kd
  /-- the device's bytes the sample reads -/
  Ω : Type
  [fΩ : Fintype Ω]
  [nΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src
  hcls : ∀ m ∈ ms, ∀ ω : Ω, Flock.Guarantees.sampleEnded src samplingDesign nJ ω = true →
    mPtsOf m.D.unroll.g (m.part ((Flock.Bridge.execSel (n := nJ) src samplingDesign nJ).draw ω)).card =
      mPtsOf m.D.unroll.g (m.kd / nTab m.I)
  /-- the claim's share: at most an `ε` share of the population's proof blocks unbound -/
  ε : ℝ
  /-- hashing one message the device or the commitment takes a digest of, in SHA-512 calls -/
  h : ℕ

attribute [instance] ExecutionPhase.fΩ ExecutionPhase.nΩ

namespace ExecutionPhase

variable (W : ExecutionPhase)

noncomputable def joint (t : ℕ) : Flock.Guarantees.JointAuditV5 where
  Reg := ByteArray
  nJ := W.nJ
  ms := W.ms.map fun m => m.toV5 t
  samplingDesign := W.samplingDesign
  ns := W.ns
  kd := W.kd
  hns := W.hns
  hlaw := W.hlaw
  Ω := W.Ω
  fintypeΩ := W.fΩ
  nonemptyΩ := W.nΩ
  Lb := W.Lb
  src := W.src
  hcls := fun k ω hω => by
    obtain ⟨m, hm, he⟩ := List.mem_map.1 (List.get_mem (W.ms.map fun m => m.toV5 t) k)
    rw [← he]
    exact W.hcls m hm ω hω

noncomputable def extend (S : Widening) : ExecutionPhase where
  __ := W
  Ω := W.Ω × S.T
  fΩ := inferInstance
  nΩ := inferInstance
  src ω := W.src ω.1
  hA3 := uniform_fst W.hA3
  hcls := fun m hm ω hω => W.hcls m hm ω.1 hω

noncomputable def L : FlockSoundness.Audit.SamplingDesign W.nJ := Flock.Bridge.execSel (n := W.nJ) W.src W.samplingDesign W.nJ

theorem joint_L (t : ℕ) : (W.joint t).L = W.L := rfl

abbrev Out : Type := (ByteArray × Bool) × Finset (Fin W.nJ) × List FlockSoundness.Discharge.Exec.Transcript

def rounds : ℕ := (W.ms.map Member.rounds).sum

def qM (t : ℕ) : ℝ := 2 * t + 2 * W.h

noncomputable def hidingTerm (_ : ExecutionPhase) : ℝ≥0∞ := 2 * (maxLeaves : ℝ≥0∞) / 2 ^ 257

noncomputable def pa (B : Finset (Fin W.nJ)) : ℝ :=
  (_root_.Properties.Spot.oneStageOf W.samplingDesign).Pa W.ns (B.map Fin.valEmbedding)

noncomputable def sampleMiss : ℝ≥0∞ :=
  ENNReal.ofReal (⨆ B : {B : Finset (Fin W.nJ) // W.ε * W.nJ < B.card}, W.pa B.1)

def RootsFix : Prop := ∀ m ∈ W.ms, m.RootsFix

def RootsCR : Prop := ∀ m ∈ W.ms, m.RootsCR

noncomputable def rootsErr : ℝ≥0∞ := (W.ms.map Member.rootsErr).sum

noncomputable def sessionDelta (t : ℕ) : ℝ≥0∞ := W.sampleMiss + (W.joint t).eps + W.rootsErr

noncomputable def delta (t : ℕ) (cr : ℝ) : ℝ≥0∞ :=
  W.sessionDelta t + W.rounds * Live.twoRun (W.qM t) + Live.twoRun (W.qM t) + Live.twoRun (W.qM t) + W.hidingTerm +
    ENNReal.ofReal cr

theorem extend_rounds (S : Widening) : (W.extend S).rounds = W.rounds := rfl

theorem extend_qM (S : Widening) (t : ℕ) : (W.extend S).qM t = W.qM t := rfl

theorem extend_hidingTerm (S : Widening) : (W.extend S).hidingTerm = W.hidingTerm := rfl

theorem rootsErr_extend (S : Widening) : (W.extend S).rootsErr = W.rootsErr := rfl

theorem rootsCR_extend (S : Widening) : (W.extend S).RootsCR ↔ W.RootsCR := Iff.rfl

end ExecutionPhase

structure JointFinderCR (w : Flock.Guarantees.JointAuditV5) [Nonempty w.EndΩ]
    (τ : FlockSoundness.Game.Strategy w.game) : Prop where
  table : ∀ k ω j, Assumptions.TableCRZAt execArith hs512
    (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ ((w.memberSamplingDesign τ k).draw ω) (reg (w.memberSim τ k))) j
    (zcStrat execArith hs512 (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (cont (w.memberSim τ k) ω))
    (w.ms.get k).qF (w.ms.get k).qS
  link : ∀ k, Assumptions.LinkCRZCAt execArith hs512 (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
    (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (reg (w.memberSim τ k)) (cont (w.memberSim τ k))
    (w.ms.get k).k (w.ms.get k).Rw (w.ms.get k).M (w.ms.get k).t'

theorem le_add_rootsErr {nJ : ℕ} {p : ℝ≥0∞} (hp : p ≤ 1) : ∀ (ms : List (Member nJ)) (x : ℝ≥0∞),
    (∀ m ∈ ms, m.RootsCR) → ((∀ m ∈ ms, m.RootsFix) → p ≤ x) → p ≤ x + (ms.map Member.rootsErr).sum
  | [], _, _, h => (h fun _ hm => absurd hm List.not_mem_nil).trans le_self_add
  | m :: ms, x, hcr, h => by
    have hm : (∀ m' ∈ ms, m'.RootsFix) → p ≤ x + m.rootsErr := fun hms => by
      rcases m.rootsFix_or_one_le (hcr m (List.mem_cons.2 (Or.inl rfl))) with hf | h1
      · refine (h fun m' hm' => ?_).trans le_self_add
        rcases List.mem_cons.1 hm' with e | hin
        · rw [e]
          exact hf
        · exact hms m' hin
      · exact hp.trans (h1.trans le_add_self)
    rw [List.map_cons, List.sum_cons, ← add_assoc]
    exact le_add_rootsErr hp ms (x + m.rootsErr) (fun m' hm' => hcr m' (List.mem_cons.2 (Or.inr hm'))) hm

theorem ExecutionPhase.ks (W : ExecutionPhase) (t : ℕ) (τ : FlockSoundness.Game.Strategy (W.joint t).game)
    (hF : ∀ _ : Nonempty (W.joint t).EndΩ, JointFinderCR (W.joint t) τ) (hcr : W.RootsCR)
    (hI : ∀ _ : Nonempty (W.joint t).EndΩ, W.RootsFix → W.ε * W.nJ < (((W.joint t).incorrectSet τ).card : ℝ)) :
    Verity.Game.prob (fun o => (W.joint t).Accepts o) (W.joint t).game τ ≤ W.sessionDelta t := by
  by_cases he : Nonempty (W.joint t).EndΩ
  · exact show _ ≤ W.sampleMiss + (W.joint t).eps + (W.ms.map Member.rootsErr).sum from
      le_add_rootsErr (Verity.Game.prob_le_one _ _ τ) W.ms (W.sampleMiss + (W.joint t).eps) hcr fun hR =>
        (Verity.Game.prob_mono (fun _ ho => ⟨ho, hI he hR⟩) _ τ).trans
          (Flock.SecurityProofs.JointShareV5 (W.joint t) W.hA3 τ (fun h => (hF h).table) (fun h => (hF h).link)
                W.ε)
  · exact ((W.joint t).prob_accepts_of_not_ends he τ).trans_le zero_le

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

structure ExecutionPhaseConstructions (V : Vocab d α) where
  coins : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Widening
  project : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    (σ : Verity.Game.Strategy (V.game θ F c)) →
      FlockSoundness.Game.Strategy ((W.extend (coins W θ F c σ)).joint (V.cost θ F c σ)).game
  readFinder : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
    Claims.Finder
  forger : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
    Hash.Assumptions.LmsForger
  msgFinder : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Fin W.rounds → Claims.Finder
  recordFinder : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Claims.Finder
  bindFinder : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Claims.Finder
  msg_calls : ∀ W θ F c σ j (x : (msgFinder W θ F c σ j).S),
    ((msgFinder W θ F c σ j).cost x : ℝ) ≤ W.qM (V.cost θ F c σ)
  record_calls : ∀ W θ F c σ (x : (recordFinder W θ F c σ).S),
    ((recordFinder W θ F c σ).cost x : ℝ) ≤ W.qM (V.cost θ F c σ)
  bind_calls : ∀ W θ F c σ (x : (bindFinder W θ F c σ).S),
    ((bindFinder W θ F c σ).cost x : ℝ) ≤ W.qM (V.cost θ F c σ)

noncomputable abbrev ExecutionPhaseConstructions.live {V : Vocab d α} (K : ExecutionPhaseConstructions V) (W : ExecutionPhase)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : ExecutionPhase :=
  W.extend (K.coins W θ F c σ)

structure ExecutionPhaseFinderCR (V : Vocab d α) (K : ExecutionPhaseConstructions V) (W : ExecutionPhase) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Prop where
  session : ∀ _ : Nonempty ((K.live W θ F c σ).joint (V.cost θ F c σ)).EndΩ,
    JointFinderCR ((K.live W θ F c σ).joint (V.cost θ F c σ)) (K.project W θ F c σ)
  msg : ∀ j, Claims.StrictBirthday Verity.sha512Msg (K.msgFinder W θ F c σ j)
  record : Claims.StrictBirthday Verity.sha512Msg (K.recordFinder W θ F c σ)
  bind : Claims.StrictBirthday Verity.sha512Msg (K.bindFinder W θ F c σ)
  roots : W.RootsCR

def ExecutionPhaseCouplingTarget (V : Vocab d α) (k : Verity.Registration.Composite) (K : ExecutionPhaseConstructions V)
    (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray) : Prop :=
  ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
      FlockSoundness.Game.prob (fun o => ((K.live W θ F c σ).joint (V.cost θ F c σ)).Accepts o)
          ((K.live W θ F c σ).joint (V.cost θ F c σ)).game (K.project W θ F c σ) +
        (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder W θ F c σ j).Pr Verity.sha512Msg)) +
          ENNReal.ofReal (Real.sqrt ((K.recordFinder W θ F c σ).Pr Verity.sha512Msg)) +
          ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr Verity.sha512Msg)) + W.hidingTerm +
          ENNReal.ofReal ((K.readFinder θ F c σ).Pr Verity.sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))

def ExecutionPhaseCovers (V : Vocab d α) (k : Verity.Registration.Composite)
    (S : ByteArray → Array ByteArray → Option ExecutionPhase) : Prop :=
  ∀ θ F, S θ F = none → (∃ p, ParamsAt θ p) → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    V.finderCR .sampled θ F c σ → V.FinderCR .sampled θ F c σ →
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤ V.delta θ .sampled (V.cost θ F c σ)

structure ExecutionPhaseDeployed (V : Vocab d α) (K : ExecutionPhaseConstructions V)
    (S : ByteArray → Array ByteArray → Option ExecutionPhase) : Prop where
  finder : ∀ θ F c σ, V.finder .sampled θ F c σ = K.readFinder θ F c σ
  forger : ∀ θ F c σ, V.forger .sampled θ F c σ = K.forger θ F c σ
  loss : ∀ θ x, V.loss .sampled θ x = 0
  claim : ∀ θ F W, S θ F = some W → W.RootsFix → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    ¬ ResourceAccounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε →
    ∀ _ : Nonempty ((K.live W θ F c σ).joint (V.cost θ F c σ)).EndΩ,
      (K.live W θ F c σ).ε * (K.live W θ F c σ).nJ <
        ((((K.live W θ F c σ).joint (V.cost θ F c σ)).incorrectSet (K.project W θ F c σ)).card : ℝ)
  finderCR : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ → ExecutionPhaseFinderCR V K W θ F c σ
  delta : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    (K.live W θ F c σ).delta (V.cost θ F c σ) (V.crBound θ .sampled (V.cost θ F c σ)) ≤
      V.delta θ .sampled (V.cost θ F c σ)

variable (V : Vocab d α) (k : Verity.Registration.Composite)

theorem spotSound_of_executionPhase {K : ExecutionPhaseConstructions V} {S : ByteArray → Array ByteArray → Option ExecutionPhase}
    (hD : ExecutionPhaseDeployed V K S) (hCov : ExecutionPhaseCovers V k S)
    (hC : ∀ θ F W, S θ F = some W → ExecutionPhaseCouplingTarget V k K W θ F) : V.SpotSound k := by
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
  | some W =>
    have hFin := hD.finderCR θ F W hS c σ hfs hpr
    have hks := (K.live W θ F c σ).ks (V.cost θ F c σ) (K.project W θ F c σ) hFin.session
      ((W.rootsCR_extend _).2 hFin.roots) (fun he hR => hD.claim θ F W hS hR c σ hI he)
    have hdel := hD.delta θ F W hS c σ
    unfold ExecutionPhase.delta at hdel
    rw [ExecutionPhase.extend_rounds, ExecutionPhase.extend_qM, ExecutionPhase.extend_hidingTerm] at hdel
    set t := V.cost θ F c σ
    have hmsg : (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder W θ F c σ j).Pr Verity.sha512Msg))) ≤
        W.rounds * Live.twoRun (W.qM t) := by
      calc (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder W θ F c σ j).Pr Verity.sha512Msg)))
          ≤ ∑ _j : Fin W.rounds, Live.twoRun (W.qM t) :=
            Finset.sum_le_sum fun j _ => sqrt_le_twoRun (hFin.msg j) (K.msg_calls W θ F c σ j)
        _ = W.rounds * Live.twoRun (W.qM t) := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    have hrec : ENNReal.ofReal (Real.sqrt ((K.recordFinder W θ F c σ).Pr Verity.sha512Msg)) ≤
        Live.twoRun (W.qM t) :=
      sqrt_le_twoRun hFin.record (K.record_calls W θ F c σ)
    have hbind : ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr Verity.sha512Msg)) ≤ Live.twoRun (W.qM t) :=
      sqrt_le_twoRun hFin.bind (K.bind_calls W θ F c σ)
    have hread : ENNReal.ofReal ((K.readFinder θ F c σ).Pr Verity.sha512Msg) ≤
        ENNReal.ofReal (V.crBound θ .sampled t) := by
      rw [← hD.finder θ F c σ]; exact ENNReal.ofReal_le_ofReal (V.finderPr_le_crBound hFC)
    have hforge : ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify) ≤
        ENNReal.ofReal (V.epsForge θ (V.callsMul θ .sampled * t + V.callsAdd θ .sampled) (V.signatures θ)) := by
      rw [← hD.forger θ F c σ]
      exact ENNReal.ofReal_le_ofReal
        (hUf (fun x => V.forger_calls .sampled θ F c σ x) (fun x => V.forger_signed .sampled θ F c σ x))
    simp only [hD.loss, add_zero]
    refine hver.trans ((hC θ F W hS c σ).trans ?_)
    calc _ ≤ (K.live W θ F c σ).sessionDelta t +
            (W.rounds * Live.twoRun (W.qM t) + Live.twoRun (W.qM t) + Live.twoRun (W.qM t) + W.hidingTerm +
              ENNReal.ofReal (V.crBound θ .sampled t) +
              ENNReal.ofReal (V.epsForge θ (V.callsMul θ .sampled * t + V.callsAdd θ .sampled) (V.signatures θ))) :=
          add_le_add hks
            (add_le_add (add_le_add (add_le_add (add_le_add (add_le_add hmsg hrec) hbind) le_rfl) hread) hforge)
      _ = (K.live W θ F c σ).sessionDelta t + W.rounds * Live.twoRun (W.qM t) + Live.twoRun (W.qM t) +
            Live.twoRun (W.qM t) + W.hidingTerm + ENNReal.ofReal (V.crBound θ .sampled t) +
            ENNReal.ofReal (V.epsForge θ (V.callsMul θ .sampled * t + V.callsAdd θ .sampled) (V.signatures θ)) := by
          ring
      _ ≤ _ := add_le_add hdel le_rfl

end Verity.Sampled
