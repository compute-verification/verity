module

public import Definitions.Pouw.Assumptions.PearlC.TTOut
public import Proofs.Pouw.Lemmas.PearlCRecords
public import Proofs.Pouw.PearlC.Game
public import Proofs.Pouw.PearlC.Instance
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.Prob

/-! g from TT_OUT, lemmas and none a guarantee: `TTOutPearlC`, the premise of `pearlCGap8192` and `pearlCGap16384`, is
falsified by the first-promotion skip. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Assumptions Pouw.Fp8Atom Matrix

theorem gapFromTTOut {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : Layout → Prop) (g₀ Ω : ℝ) (δ : ℕ → ℕ → ℝ) (hΩ : 0 < Ω)
    (hW : ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S), ∀ u < U.layout.N, ∀ act : Codes,
      P.actOK (U.layout.shape u) act → P.inertBoundOK H s U u act → P.Wref U.layout u ≤ Ω * P.credit H s U u act)
    (hTT : TTOut CM P D g₀ δ) : GGap CM P D (1 - (1 - g₀) / Ω) δ := by
  intro U hU T hT q hq A hA
  classical
  refine le_trans (Pouw.pr_mono ?_) (hTT.2 U hU T hT q hq A hA)
  intro x hwin
  unfold Wins at hwin
  have hsum : ∑ u ∈ correctSet P U x.1 x.2 (A.trace U x.1 x.2), P.Wref U.layout u ≤
      Ω * ∑ u ∈ correctSet P U x.1 x.2 (A.trace U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.trace U x.1 x.2) u) := by
    rw [mul_sum]
    refine sum_le_sum fun u hu => ?_
    unfold correctSet at hu
    rw [mem_filter] at hu
    obtain ⟨hu, c, hc, hcor⟩ := hu
    have hact : actOf (A.trace U x.1 x.2) u = c.act := by simp [actOf, hc]
    rw [hact]
    exact hW U hU x.1 x.2 u (mem_range.mp hu) c.act hcor.1 hcor.2.1
  have hden : 1 - (1 - (1 - g₀) / Ω) = (1 - g₀) / Ω := by ring
  rw [hden, div_div_eq_mul_div] at hwin
  have hscale : T * Ω / (1 - g₀) = Ω * (T / (1 - g₀)) := by ring
  rw [hscale] at hwin
  exact lt_of_mul_lt_mul_left (lt_of_lt_of_le hwin hsum) hΩ.le

theorem ttOut_mono_domain {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {P : Protocol Q R S} {D D' : Layout → Prop} {g₀ : ℝ} {δ : ℕ → ℕ → ℝ} (hD : ∀ L, D' L → D L)
    (h : TTOut CM P D g₀ δ) : TTOut CM P D' g₀ δ :=
  ⟨h.1, fun U hU => h.2 U ⟨hU.1, hD _ hU.2⟩⟩

theorem wref_le_at {Q R S : Type} (P : Protocol Q R S) (c : Costs) (ρ : ℚ) (s : Shape) (ω : ℚ) (hω0 : 0 ≤ ω)
    (hω : wrefOf Params.pi c s = ω * ((1 - ρ) * creditOf Params.pi c s)) (hacc : PearlCAccounting P Params.pi c ρ) :
    ∀ U : Workload, U.InDomain P (pearlCDomainAt Params.pi s) → ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N,
      ∀ act : Codes, P.actOK (U.layout.shape u) act → P.inertBoundOK H s' U u act →
        P.Wref U.layout u ≤ (ω : ℝ) * P.credit H s' U u act := by
  intro U hU H s' u hu act _ hcap
  have hsh : U.layout.shape u = s := hU.2.2 u hu
  have hc := hacc.2 H s' U u act hcap
  rw [hsh] at hc
  rw [hacc.1, hsh, hω]
  have e : ((ω * ((1 - ρ) * creditOf Params.pi c s) : ℚ) : ℝ) = (ω : ℝ) * (((1 - ρ) * creditOf Params.pi c s : ℚ) : ℝ) := by
    push_cast; ring
  rw [e]
  exact mul_le_mul_of_nonneg_left hc (by exact_mod_cast hω0)

theorem pearlCGapAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) (s : Shape) (ω g : ℚ) (hω0 : 0 < ω)
    (hω : wrefOf Params.pi c s = ω * ((1 - ρ) * creditOf Params.pi c s)) (hg : g = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutPearlC CM sem c ρ) :
    GGap CM (pearlCProtocol sem c ρ) (pearlCDomainAt Params.pi s) (g : ℝ) δPearlC := by
  have key := gapFromTTOut CM (pearlCProtocol sem c ρ) (pearlCDomainAt Params.pi s) (1 / 400) (ω : ℝ) δPearlC
    (by exact_mod_cast hω0) (wref_le_at _ c ρ s ω hω0.le hω (pearlCProtocol_accounting sem c ρ))
    (ttOut_mono_domain (fun _ h => h.1) hTT)
  have e : (g : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hg]; push_cast; ring
  rw [e]
  exact key

/-- Its hypothesis `TTOutPearlC` is falsified by the first-promotion skip; rev1 (`TTOutPearlCDevRev1`) is its repair. -/
theorem pearlCGap8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (hTT : TTOutPearlC CM sem Costs.adopted (1 / 400)) :
    GGap CM (pearlCProtocol sem Costs.adopted (1 / 400)) (pearlCDomainAt Params.pi sh8192)
      ((281273 / 52340000 : ℚ) : ℝ) δPearlC :=
  pearlCGapAt CM sem _ _ _ (130850 / 130473) _ (by norm_num)
    (by norm_num [wrefOf, creditOf, Params.pi, Costs.adopted, sh8192]) (by norm_num) hTT

/-- Its hypothesis `TTOutPearlC` is falsified by the first-promotion skip; rev1 (`TTOutPearlCDevRev1`) is its repair. -/
theorem pearlCGap16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (hTT : TTOutPearlC CM sem Costs.adopted (1 / 400)) :
    GGap CM (pearlCProtocol sem Costs.adopted (1 / 400)) (pearlCDomainAt Params.pi sh16384)
      ((536953 / 103540000 : ℚ) : ℝ) δPearlC :=
  pearlCGapAt CM sem _ _ _ (258850 / 258153) _ (by norm_num)
    (by norm_num [wrefOf, creditOf, Params.pi, Costs.adopted, sh16384]) (by norm_num) hTT

/-- The chain-only reading at the cap, 8192³: the credit less the forming, less a debit of `ρ·creditOf`. -/
theorem chainOnlyGap8192InertBound :
    gapOf (1 / 400) (wrefOf Params.pi Costs.adopted sh8192 /
      (creditChainOnly Params.pi Costs.adopted sh8192 - 1 / 400 * creditOf Params.pi Costs.adopted sh8192)) =
      600473 / 52340000 := by
  norm_num [gapOf, wrefOf, creditChainOnly, creditOf, Params.pi, Costs.adopted, sh8192]

theorem chainOnlyGap16384InertBound :
    gapOf (1 / 400) (wrefOf Params.pi Costs.adopted sh16384 /
      (creditChainOnly Params.pi Costs.adopted sh16384 - 1 / 400 * creditOf Params.pi Costs.adopted sh16384)) =
      856153 / 103540000 := by
  norm_num [gapOf, wrefOf, creditChainOnly, creditOf, Params.pi, Costs.adopted, sh16384]

/-- A trivial `sem`: zero noise, the exact forming chain, and unit scales and floor. Its element `1.0` is salt-live
under P3's predicate (`witness_saltLive`). -/
def witnessSem : PearlCSem Unit Unit Unit where
  noise := fun _ _ _ _ _ => ⟨fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ _ => 0⟩
  fc := FormChain.exact _
  alpha := fun _ _ => 1
  beta := fun _ _ => 1
  floor := fun _ _ => 1

/-- The witness cost model for a protocol `P`: `true` runs `P`'s honest reference at the reference's credit of the
activations it is given (at least 0), `false` commits nothing. -/
noncomputable def witnessCM (P : Protocol Unit Unit Unit) : CostModel Unit Unit Unit where
  Prog := Bool
  run := fun b H U act _ s => if b then honestTrace P H s U act else fun _ => none
  cost := fun b H U act _ s => if b then ∑ u ∈ range U.layout.N, max (P.credit H s U u (act u)) 0 else 0
  calls := fun _ _ _ _ _ _ => 0

theorem creditOf_nonneg (c : Costs) (s : Shape) : 0 ≤ creditOf Params.pi c s := by
  unfold creditOf Params.pi
  positivity

theorem creditOf_le_wref (c : Costs) (s : Shape) : creditOf Params.pi c s ≤ wrefOf Params.pi c s := by
  unfold wrefOf
  have : (0 : ℚ) ≤ (c.qa : ℚ) * s.m * s.k := by positivity
  linarith

theorem actOf_honest (P : Protocol Unit Unit Unit) (H : Unit → Unit) (s : Unit) (U : Workload) (act : ℕ → Codes) {u : ℕ}
    (hu : u < U.layout.N) : actOf (honestTrace P H s U act) u = act u := by
  simp [actOf, honestTrace, hu]

theorem witness_sum_le (P : Protocol Unit Unit Unit) (U : Workload) (H : Unit → Unit) (s : Unit) (b : Bool)
    (act : ℕ → Codes) (σ : State) :
    ∑ u ∈ correctSet P U H s ((witnessCM P).run b H U act σ s), P.credit H s U u (actOf ((witnessCM P).run b H U act σ s) u)
      ≤ (witnessCM P).cost b H U act σ s := by
  classical
  cases b with
  | false =>
    have hempty : correctSet P U H s ((witnessCM P).run false H U act σ s) = ∅ := by
      unfold correctSet
      apply filter_false_of_mem
      intro u _ h
      obtain ⟨c, hc, _⟩ := h
      simp [witnessCM] at hc
    rw [hempty, sum_empty]
    simp [witnessCM]
  | true =>
    have hsub : correctSet P U H s ((witnessCM P).run true H U act σ s) ⊆ range U.layout.N := by
      unfold correctSet
      exact filter_subset _ _
    have heq : ∀ u ∈ correctSet P U H s ((witnessCM P).run true H U act σ s),
        P.credit H s U u (actOf ((witnessCM P).run true H U act σ s) u) = P.credit H s U u (act u) := by
      intro u hu
      have hu' : u < U.layout.N := mem_range.mp (hsub hu)
      simp only [witnessCM, ↓reduceIte]
      rw [actOf_honest P H s U act hu']
    rw [sum_congr rfl heq]
    calc ∑ u ∈ correctSet P U H s ((witnessCM P).run true H U act σ s), P.credit H s U u (act u)
        ≤ ∑ u ∈ correctSet P U H s ((witnessCM P).run true H U act σ s), max (P.credit H s U u (act u)) 0 :=
          sum_le_sum fun u _ => le_max_left _ _
      _ ≤ ∑ u ∈ range U.layout.N, max (P.credit H s U u (act u)) 0 :=
          sum_le_sum_of_subset_of_nonneg hsub (fun u _ _ => le_max_right _ _)
      _ = (witnessCM P).cost true H U act σ s := by simp [witnessCM]

theorem wordQ_one : wordQ 1065353216 = some 1 := by decide +kernel

theorem codeVal_zero : codeVal 0 = 0 := by decide +kernel

/-- One in-domain workload: one unit of shape `1 × 1 × 128`, every word `1.0`. -/
def witnessWorkload : Workload := ⟨⟨1, fun _ => ⟨1, 1, 128⟩, fun _ => 0⟩, fun _ _ _ => 1065353216⟩

def witnessAct : ℕ → Codes := fun _ _ _ => 1065353216

theorem witness_flags : ∀ t < 4, Skip.wordFlags 128 (fun _ => e4m3 1) (fun _ => e4m3 1) t = false := by decide +kernel

theorem witness_rowA (H : Unit → Unit) (s : Unit) (i : Fin (witnessWorkload.layout.shape 0).m) :
    (witnessSem.unitChain H s witnessWorkload 0 (witnessAct 0)).1 i = fun _ => e4m3 1 := by
  funext l
  simp [PearlCSem.unitChain, witnessSem, FormChain.quantA, FormChain.quant, FormChain.elem, FormChain.exact, wordsVal,
    witnessAct, witnessWorkload, wordQ_one, codeVal_zero]

theorem witness_rowB (H : Unit → Unit) (s : Unit) (j : Fin (witnessWorkload.layout.shape 0).n) :
    (witnessSem.unitChain H s witnessWorkload 0 (witnessAct 0)).2 j = fun _ => e4m3 1 := by
  funext l
  simp [PearlCSem.unitChain, witnessSem, FormChain.quantB, FormChain.quant, FormChain.elem, FormChain.exact, wordsVal,
    witnessWorkload, wordQ_one, codeVal_zero]

theorem witness_unitFlags (H : Unit → Unit) (s : Unit) (i : Fin (witnessWorkload.layout.shape 0).m)
    (j : Fin (witnessWorkload.layout.shape 0).n) (t : ℕ) (ht : t < 4) :
    Skip.unitFlags (witnessSem.unitChain H s witnessWorkload 0 (witnessAct 0)).1
      (witnessSem.unitChain H s witnessWorkload 0 (witnessAct 0)).2 i j t = false := by
  unfold Skip.unitFlags
  rw [witness_rowA, witness_rowB]
  exact witness_flags t ht

theorem witness_codes_differ :
    codeVal (FormChain.atomMix.elem 1 1 1 (SaltDead.flatLine fun _ => false) (SaltDead.flatLineF fun _ => false)) ≠
      codeVal (FormChain.atomMix.elem 1 1 1 (SaltDead.flatLine fun _ => true) (SaltDead.flatLineF fun _ => false)) := by
  decide +kernel

theorem witness_saltLive (k : ℕ) (x : Fin k → ℚ) (l : Fin k) (hx : x l = 1) :
    ¬ SaltDead.saltDead witnessSem.alpha witnessSem.beta k x l := by
  intro h
  have h' := h (SaltDead.flatLine fun _ => false, SaltDead.flatLineF fun _ => false) ⟨_, _, rfl⟩
    (SaltDead.flatLine fun _ => true, SaltDead.flatLineF fun _ => false) ⟨_, _, rfl⟩
  simp only [witnessSem, hx] at h'
  exact witness_codes_differ h'

open Classical in
theorem witness_inert_work_zero (H : Unit → Unit) (s : Unit) :
    (witnessSem.unitInertWork H s witnessWorkload 0 (witnessAct 0)).atoms = 0 ∧
      (witnessSem.unitInertWork H s witnessWorkload 0 (witnessAct 0)).elems = 0 := by
  constructor
  · unfold PearlCSem.unitInertWork
    dsimp only
    apply sum_eq_zero
    intro i _
    apply sum_eq_zero
    intro j _
    rw [card_eq_zero, filter_eq_empty_iff]
    intro t ht
    have ht' : t < 4 := by simpa [witnessWorkload] using mem_range.mp ht
    rw [witness_unitFlags H s i j t ht']
    simp
  · unfold PearlCSem.unitInertWork
    dsimp only
    rw [card_eq_zero, filter_eq_empty_iff]
    intro e _
    rintro (hd | hn)
    · exact witness_saltLive _ _ _ (by simp [wordsVal, witnessAct, wordQ_one]) hd
    apply hn
    unfold Skip.needed
    have hl : (e.2 : ℕ) / 32 < 4 := by
      have := e.2.isLt
      simp only [witnessWorkload] at this
      omega
    exact ⟨⟨0, by simp [witnessWorkload]⟩, witness_unitFlags H s e.1 _ _ hl⟩

theorem witness_operandOK (p : ℕ) : witnessSem.OperandOK p 128 (fun _ _ => 1065353216) := by
  refine ⟨fun _ _ _ _ => by simp [wordQ_one], fun i => ⟨?_, ?_⟩⟩
  · have hrow : wordsVal p 128 (fun _ _ => 1065353216) i = fun _ => 1 := by
      funext l; simp [wordsVal, wordQ_one]
    rw [hrow]
    have hcard : (Finset.univ.filter fun l : Fin 128 => l.val % 8 = 0).card = 16 := by decide
    have hst : Stat.every8 128 (fun _ : Fin 128 => (1 : ℚ)) = 1 := by
      unfold Stat.every8
      rw [Finset.sum_const, hcard]
      norm_num
    have hempty : (Finset.univ.filter fun l : Fin 128 =>
        64 * Params.pi.σ ^ 2 * Stat.every8 128 (fun _ : Fin 128 => (1 : ℚ)) ≤ (fun _ : Fin 128 => (1 : ℚ)) l ^ 2) = ∅ := by
      apply filter_false_of_mem
      intro l _
      rw [hst]
      norm_num [Params.pi]
    unfold LiveRow
    rw [hempty]
    simp
  · simp [witnessSem]

/-- `TTOutPearlC` is satisfiable at a two-program witness cost model. At the real H100 cost model it is falsified by
the first-promotion skip. -/
theorem ttOutPearlCWitness :
    AdmitsRef (witnessCM (pearlCProtocol witnessSem Costs.adopted (1 / 400))) (pearlCProtocol witnessSem Costs.adopted (1 / 400))
        (pearlCDomain Params.pi) (fun _ => 0) ∧
      TTOutPearlC (witnessCM (pearlCProtocol witnessSem Costs.adopted (1 / 400))) witnessSem Costs.adopted (1 / 400) ∧
      witnessWorkload.InDomain (pearlCProtocol witnessSem Costs.adopted (1 / 400)) (pearlCDomain Params.pi) ∧
      witnessWorkload.FixedOK (pearlCProtocol witnessSem Costs.adopted (1 / 400)) ∧
      (pearlCProtocol witnessSem Costs.adopted (1 / 400)).inertBoundOK (fun _ => ()) () witnessWorkload 0 (witnessAct 0) ∧
      0 < ∑ u ∈ range witnessWorkload.layout.N,
        (pearlCProtocol witnessSem Costs.adopted (1 / 400)).credit (fun _ => ()) () witnessWorkload u (witnessAct u) := by
  set P := pearlCProtocol witnessSem Costs.adopted (1 / 400)
  have hcr : ∀ H s U u a, P.credit H s U u a =
      ((creditOf Params.pi Costs.adopted (U.layout.shape u) -
        inertWorkOf Params.pi Costs.adopted (witnessSem.unitInertWork H s U u a) : ℚ) : ℝ) := fun _ _ _ _ _ => rfl
  have hzero : inertWorkOf Params.pi Costs.adopted (witnessSem.unitInertWork (fun _ => ()) () witnessWorkload 0 (witnessAct 0)) = 0 := by
    obtain ⟨ha, he⟩ := witness_inert_work_zero (fun _ => ()) ()
    unfold inertWorkOf
    rw [ha, he]
    simp
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨true, fun U _ _ act _ H s => ⟨by simp [witnessCM], ?_, by simp [witnessCM]⟩⟩
    show (if true = true then ∑ u ∈ range U.layout.N, max (P.credit H s U u (act u)) 0 else 0) ≤ _
    simp only [↓reduceIte]
    apply sum_le_sum
    intro u _
    rw [hcr]
    show max (((creditOf Params.pi Costs.adopted (U.layout.shape u) -
        inertWorkOf Params.pi Costs.adopted (witnessSem.unitInertWork H s U u (act u)) : ℚ) : ℝ)) 0 ≤
      (wrefOf Params.pi Costs.adopted (U.layout.shape u) : ℝ)
    have h0 := inertWorkOf_nonneg Params.pi Costs.adopted (witnessSem.unitInertWork H s U u (act u))
    have h1 := creditOf_le_wref Costs.adopted (U.layout.shape u)
    have h2 := creditOf_nonneg Costs.adopted (U.layout.shape u)
    apply max_le
    · exact_mod_cast (by linarith)
    · exact_mod_cast (by linarith)
  · refine ⟨fun prog U act σ s => ⟨QTree.done ((witnessCM P).run prog (fun _ => ()) U act σ s), fun H => ?_⟩, ?_⟩
    · have hH : H = fun _ => () := funext fun _ => Subsingleton.elim _ _
      subst hH
      exact ⟨rfl, rfl⟩
    · intro U _ T hT q _ A hA
      rw [Pouw.pr_eq_zero]
      · unfold δPearlC
        positivity
      intro x hx
      have hsum := witness_sum_le P U x.1 x.2 A.online A.act (A.pre.eval x.1)
      have hcost := (hA x.1 x.2).1
      have hT' : T ≤ T / (1 - 1 / 400) := by
        rw [le_div_iff₀ (by norm_num)]
        nlinarith
      unfold Adversary.trace at hx
      linarith
  · refine ⟨by norm_num [witnessWorkload], ?_⟩
    intro u _
    refine ⟨by norm_num [witnessWorkload], by norm_num [witnessWorkload], by norm_num [witnessWorkload], ?_,
      by norm_num [witnessWorkload], rfl⟩
    show 32 * 4 ∣ 128
    norm_num
  · exact fun u _ j hj =>
      ⟨fun l hl => (witness_operandOK 1).1 j hj l hl, fun hj' => (witness_operandOK 1).2 ⟨j, hj'⟩⟩
  · show inertWorkOf Params.pi Costs.adopted (witnessSem.unitInertWork (fun _ => ()) () witnessWorkload 0 (witnessAct 0)) ≤
      1 / 400 * creditOf Params.pi Costs.adopted (witnessWorkload.layout.shape 0)
    rw [hzero]
    norm_num [creditOf, Params.pi, Costs.adopted, witnessWorkload]
  · rw [show range witnessWorkload.layout.N = range 1 from rfl, sum_range_one, hcr, hzero]
    norm_num [creditOf, Params.pi, Costs.adopted, witnessWorkload]

end Pouw.PearlC
