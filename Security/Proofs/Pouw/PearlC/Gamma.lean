import Definitions.Pouw.Assumptions.PearlC.TTOut
import Proofs.Pouw.Prob
import Proofs.Pouw.PearlC.Game
import Proofs.Pouw.PearlC.Instance
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Definitions.Pouw.Guarantees.PearlC

/-!
# γ from TT_OUT

Lemmas, none a guarantee: `TTOutPearlC`, the premise of `pearlCGamma8192` and `pearlCGamma16384`, is falsified by the
first-promotion skip (`Definitions.Pouw.Assumptions.PearlC.TTOut`), and rev1 (`TTOutPearlCDevRev1`) is its repair.

* `gammaFromTTOut`: if every in-domain unit that passes the activation rules and the debit cap has reference cost at
  most `Ω` times its credit, `TT_OUT(γ₀)` gives the γ-gap property at `γ = 1 − (1 − γ₀)/Ω` with the same error; the
  winning event implies TT_OUT's event pointwise. Units above the cap are not correct, so they enter neither sum.
* **γ is a function of the workload through its debit.** A unit's credit is `creditOf − debit(u)`, and within the cap
  `ρ` it is at least `(1 − ρ)·creditOf` (`PearlCAccounting`). `pearlCGammaAt` gives `G_γ` at any cap `ρ`, with
  `γ = 1 − (399/400)·(1 − ρ)·creditOf/W_ref`. A workload whose units' debits are at most `ρ·creditOf` loses nothing to
  that cap. The debits measured on the census and on real activations are in `docs/pouw/cheap-binding.md` §4.
* **The worst case under the deployment cap `ρ = 1/400`,** `Costs.adopted`, promotion every 4 atoms, `r = 32`:
  - `281273/52340000` (0.5374%) at 8192³ and `536953/103540000` (0.5186%) at 16384³ (`pearlCGamma8192`,
    `pearlCGamma16384`);
  - if TT_OUT credits only the chain and the clean-up: `600473/52340000` (1.1473%) at 8192³ and `856153/103540000`
    (0.8269%) at 16384³;
  - relative to the honest cost including glue (not the game's γ): `21013/2500000` (0.8405%) at 8192³ and
    `696953/103700000` (0.6721%) at 16384³ if only `α·x` runs as glue with PoUW on, and `601273/52660000` (1.1418%) and
    `95217/11540000` (0.8251%) if a clean E4M3 cast runs as well.
* **The cap's scope.** What the cap's calibration rests on, and where it holds:
  - the admission and debit figures come from 64×8 tiles at `k ≤ 4,864` (Qwen2.5-0.5B's layers, `pearlc_activations.py`),
    below the headline k. At the headline shapes, where `pearlCDomainAt` needs all of a unit's 8,192 or 16,384
    activation rows and weight rows admissible, admission is unmeasured;
  - γ does not depend on m, but it does depend on n: `A′·F_B` and the forming stop amortizing, so at an 8-row weight
    tile (n = 8, k = 8,192) γ at the cap is about 4.1%;
  - the cap is relative to `creditOf`, whose forming and `A′·F_B` terms don't scale with n. So "every aligned-spike
    family is rejected" holds at the headline shapes, not at every in-domain shape: R = 64 spikes at n = 8, k = 8,192
    have ρ ≈ 0.21% and are admitted, with their frozen atoms debited.
* **Not in these numbers: cross-group joint skips.** No replay count bounds them (`joint_cross_group`); `TTOutPearlC`
  carries them, and their measured rate is reported beside γ, not folded into it.
* `ttOutPearlCWitness`: `TTOutPearlC` is satisfiable, beside an honest reference at `W_ref` and an in-domain workload whose
  unit is within the cap and has positive credit. Its cost model has two programs and shows satisfiability only. That the
  honest H100 program fits `W_ref` under the real H100 cost model is the kernel's measured claim, and none of these
  statements asserts it.
-/

namespace Pouw.PearlC

open Finset Assumptions Pouw.Fp8Atom Matrix

theorem gammaFromTTOut {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ω : ℝ) (ε : ℕ → ℕ → ℝ) (hΩ : 0 < Ω)
    (hW : ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S), ∀ u < U.layout.N, ∀ act : Codes,
      P.actOK (U.layout.shape u) act → P.capOK H s U u act → P.Wref U.layout u ≤ Ω * P.credit H s U u act)
    (hTT : TTOut CM P D γ₀ ε) : Gγ CM P D (1 - (1 - γ₀) / Ω) ε := by
  intro U hU T hT q hq A hA
  classical
  refine le_trans (Pouw.pr_mono ?_) (hTT.2 U hU T hT q hq A hA)
  intro x hwin
  unfold Wins at hwin
  have hsum : ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), P.Wref U.layout u ≤
      Ω * ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.transcript U x.1 x.2) u) := by
    rw [mul_sum]
    refine sum_le_sum fun u hu => ?_
    unfold correctSet at hu
    rw [mem_filter] at hu
    obtain ⟨hu, c, hc, hcor⟩ := hu
    have hact : actOf (A.transcript U x.1 x.2) u = c.act := by simp [actOf, hc]
    rw [hact]
    exact hW U hU x.1 x.2 u (mem_range.mp hu) c.act hcor.1 hcor.2.1
  have hden : 1 - (1 - (1 - γ₀) / Ω) = (1 - γ₀) / Ω := by ring
  rw [hden, div_div_eq_mul_div] at hwin
  have hscale : T * Ω / (1 - γ₀) = Ω * (T / (1 - γ₀)) := by ring
  rw [hscale] at hwin
  exact lt_of_mul_lt_mul_left (lt_of_lt_of_le hwin hsum) hΩ.le

/-- TT_OUT on a domain gives it on any smaller domain. -/
theorem ttOut_mono_domain {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {P : Protocol Q R S} {D D' : Layout → Prop} {γ₀ : ℝ} {ε : ℕ → ℕ → ℝ} (hD : ∀ L, D' L → D L)
    (h : TTOut CM P D γ₀ ε) : TTOut CM P D' γ₀ ε :=
  ⟨h.1, fun U hU => h.2 U ⟨hU.1, hD _ hU.2⟩⟩

theorem wref_le_at {Q R S : Type} (P : Protocol Q R S) (c : Costs) (ρ : ℚ) (s : Shape) (ω : ℚ) (hω0 : 0 ≤ ω)
    (hω : wrefOf Params.pi c s = ω * ((1 - ρ) * creditOf Params.pi c s)) (hacc : PearlCAccounting P Params.pi c ρ) :
    ∀ U : Workload, U.InDomain P (pearlCDomainAt Params.pi s) → ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N,
      ∀ act : Codes, P.actOK (U.layout.shape u) act → P.capOK H s' U u act →
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

/-- The headline pattern: `TTOutPearlC` at costs `c` and cap `ρ` gives `G_γ` on the one-shape domain at
`γ = 1 − (399/400)/ω`, where `W_ref = ω·(1 − ρ)·creditOf`. -/
theorem pearlCGammaAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hω : wrefOf Params.pi c s = ω * ((1 - ρ) * creditOf Params.pi c s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutPearlC CM sem c ρ) :
    Gγ CM (pearlCProtocol sem c ρ) (pearlCDomainAt Params.pi s) (γ : ℝ) εPearlC := by
  have key := gammaFromTTOut CM (pearlCProtocol sem c ρ) (pearlCDomainAt Params.pi s) (1 / 400) (ω : ℝ) εPearlC
    (by exact_mod_cast hω0) (wref_le_at _ c ρ s ω hω0.le hω (pearlCProtocol_accounting sem c ρ))
    (ttOut_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- The worst case at 8192³ under the deployment cap `ρ = 1/400`. Its hypothesis, the granted `TTOutPearlC`, is
falsified by the first-promotion skip (rating D); rev1 (`TTOutPearlCDevRev1`) is its repair. -/
theorem pearlCGamma8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (hTT : TTOutPearlC CM sem Costs.adopted (1 / 400)) :
    Gγ CM (pearlCProtocol sem Costs.adopted (1 / 400)) (pearlCDomainAt Params.pi sh8192)
      ((281273 / 52340000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaAt CM sem _ _ _ (130850 / 130473) _ (by norm_num)
    (by norm_num [wrefOf, creditOf, Params.pi, Costs.adopted, sh8192]) (by norm_num) hTT

/-- The worst case at 16384³ under the deployment cap `ρ = 1/400`. Its hypothesis, the granted `TTOutPearlC`, is
falsified by the first-promotion skip (rating D); rev1 (`TTOutPearlCDevRev1`) is its repair. -/
theorem pearlCGamma16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (hTT : TTOutPearlC CM sem Costs.adopted (1 / 400)) :
    Gγ CM (pearlCProtocol sem Costs.adopted (1 / 400)) (pearlCDomainAt Params.pi sh16384)
      ((536953 / 103540000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaAt CM sem _ _ _ (258850 / 258153) _ (by norm_num)
    (by norm_num [wrefOf, creditOf, Params.pi, Costs.adopted, sh16384]) (by norm_num) hTT

/-- The chain-only reading at the cap, 8192³: the credit less the forming, less a debit of `ρ·creditOf`. -/
theorem chainOnlyGamma8192Cap :
    gammaOf (1 / 400) (wrefOf Params.pi Costs.adopted sh8192 /
      (creditChainOnly Params.pi Costs.adopted sh8192 - 1 / 400 * creditOf Params.pi Costs.adopted sh8192)) =
      600473 / 52340000 := by
  norm_num [gammaOf, wrefOf, creditChainOnly, creditOf, Params.pi, Costs.adopted, sh8192]

theorem chainOnlyGamma16384Cap :
    gammaOf (1 / 400) (wrefOf Params.pi Costs.adopted sh16384 /
      (creditChainOnly Params.pi Costs.adopted sh16384 - 1 / 400 * creditOf Params.pi Costs.adopted sh16384)) =
      856153 / 103540000 := by
  norm_num [gammaOf, wrefOf, creditChainOnly, creditOf, Params.pi, Costs.adopted, sh16384]

/-! ## TT_OUT for Pearl-C is satisfiable -/

/-- A trivial `sem`: zero noise, the exact forming chain, and unit scales and floor. Its element `1.0` is salt-live under
P3's predicate (`witness_saltLive`). The flags and
U are the shared definitions (`Skip`, `PeelExact`). -/
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
  run := fun b H U act _ s => if b then honestTranscript P H s U act else fun _ => none
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
    (hu : u < U.layout.N) : actOf (honestTranscript P H s U act) u = act u := by
  simp [actOf, honestTranscript, hu]

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

/-- The witness's activations: every word `1.0`. -/
def witnessAct : ℕ → Codes := fun _ _ _ => 1065353216

/-- P1's flags on the witness's word: all four atoms are unflagged (the kernel evaluates `jointFlags`). -/
theorem witness_flags : ∀ t < 4, Skip.wordFlags 128 (fun _ => e4m3 1) (fun _ => e4m3 1) t = false := by decide +kernel

/-- The witness's formed rows: every element is `e4m3 1` (zero noise, unit scales). -/
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

/-- No atom of the witness's word is flagged. -/
theorem witness_unitFlags (H : Unit → Unit) (s : Unit) (i : Fin (witnessWorkload.layout.shape 0).m)
    (j : Fin (witnessWorkload.layout.shape 0).n) (t : ℕ) (ht : t < 4) :
    Skip.unitFlags (witnessSem.unitChain H s witnessWorkload 0 (witnessAct 0)).1
      (witnessSem.unitChain H s witnessWorkload 0 (witnessAct 0)).2 i j t = false := by
  unfold Skip.unitFlags
  rw [witness_rowA, witness_rowB]
  exact witness_flags t ht

/-- The kernel evaluates §5's atom-mix at the witness's element `1.0` and unit scales on two flat noise pairs of
`SaltDead.support`: the codes' values are `448` and `−448`. -/
theorem witness_codes_differ :
    codeVal (FormChain.atomMix.elem 1 1 1 (SaltDead.flatLine fun _ => false) (SaltDead.flatLineF fun _ => false)) ≠
      codeVal (FormChain.atomMix.elem 1 1 1 (SaltDead.flatLine fun _ => true) (SaltDead.flatLineF fun _ => false)) := by
  decide +kernel

/-- An element `1.0` at the witness's unit scales is not salt-dead (`SaltDead.saltDead`). -/
theorem witness_saltLive (k : ℕ) (x : Fin k → ℚ) (l : Fin k) (hx : x l = 1) :
    ¬ SaltDead.saltDead witnessSem.alpha witnessSem.beta k x l := by
  intro h
  have h' := h (SaltDead.flatLine fun _ => false, SaltDead.flatLineF fun _ => false) ⟨_, _, rfl⟩
    (SaltDead.flatLine fun _ => true, SaltDead.flatLineF fun _ => false) ⟨_, _, rfl⟩
  simp only [witnessSem, hx] at h'
  exact witness_codes_differ h'

open Classical in
/-- The witness's debit is zero: no atom is flagged, no element is salt-dead, and every element is needed. -/
theorem witness_debit_zero (H : Unit → Unit) (s : Unit) :
    (witnessSem.unitDebit H s witnessWorkload 0 (witnessAct 0)).atoms = 0 ∧
      (witnessSem.unitDebit H s witnessWorkload 0 (witnessAct 0)).elems = 0 := by
  constructor
  · unfold PearlCSem.unitDebit
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
  · unfold PearlCSem.unitDebit
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
        64 * Params.pi.δ ^ 2 * Stat.every8 128 (fun _ : Fin 128 => (1 : ℚ)) ≤ (fun _ : Fin 128 => (1 : ℚ)) l ^ 2) = ∅ := by
      apply filter_false_of_mem
      intro l _
      rw [hst]
      norm_num [Params.pi]
    unfold LiveRow
    rw [hempty]
    simp
  · simp [witnessSem]

/-- `TTOutPearlC` is satisfiable at a two-program witness cost model, beside an honest reference at `W_ref` and an
in-domain unit, whose weight passes the rules, within the cap with positive credit. At the real H100 cost model the row
is falsified by the first-promotion skip (rating D); rev1 (`TTOutPearlCDevRev1`) is its repair. -/
theorem ttOutPearlCWitness :
    AdmitsRef (witnessCM (pearlCProtocol witnessSem Costs.adopted (1 / 400))) (pearlCProtocol witnessSem Costs.adopted (1 / 400))
        (pearlCDomain Params.pi) (fun _ => 0) ∧
      TTOutPearlC (witnessCM (pearlCProtocol witnessSem Costs.adopted (1 / 400))) witnessSem Costs.adopted (1 / 400) ∧
      witnessWorkload.InDomain (pearlCProtocol witnessSem Costs.adopted (1 / 400)) (pearlCDomain Params.pi) ∧
      witnessWorkload.WeightsOK (pearlCProtocol witnessSem Costs.adopted (1 / 400)) ∧
      (pearlCProtocol witnessSem Costs.adopted (1 / 400)).capOK (fun _ => ()) () witnessWorkload 0 (witnessAct 0) ∧
      0 < ∑ u ∈ range witnessWorkload.layout.N,
        (pearlCProtocol witnessSem Costs.adopted (1 / 400)).credit (fun _ => ()) () witnessWorkload u (witnessAct u) := by
  set P := pearlCProtocol witnessSem Costs.adopted (1 / 400)
  have hcr : ∀ H s U u a, P.credit H s U u a =
      ((creditOf Params.pi Costs.adopted (U.layout.shape u) -
        debitOf Params.pi Costs.adopted (witnessSem.unitDebit H s U u a) : ℚ) : ℝ) := fun _ _ _ _ _ => rfl
  have hzero : debitOf Params.pi Costs.adopted (witnessSem.unitDebit (fun _ => ()) () witnessWorkload 0 (witnessAct 0)) = 0 := by
    obtain ⟨ha, he⟩ := witness_debit_zero (fun _ => ()) ()
    unfold debitOf
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
        debitOf Params.pi Costs.adopted (witnessSem.unitDebit H s U u (act u)) : ℚ) : ℝ)) 0 ≤
      (wrefOf Params.pi Costs.adopted (U.layout.shape u) : ℝ)
    have h0 := debitOf_nonneg Params.pi Costs.adopted (witnessSem.unitDebit H s U u (act u))
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
      · unfold εPearlC
        positivity
      intro x hx
      have hsum := witness_sum_le P U x.1 x.2 A.online A.act (A.pre.eval x.1)
      have hcost := (hA x.1 x.2).1
      have hT' : T ≤ T / (1 - 1 / 400) := by
        rw [le_div_iff₀ (by norm_num)]
        nlinarith
      unfold Adversary.transcript at hx
      linarith
  · refine ⟨by norm_num [witnessWorkload], ?_⟩
    intro u _
    refine ⟨by norm_num [witnessWorkload], by norm_num [witnessWorkload], by norm_num [witnessWorkload], ?_,
      by norm_num [witnessWorkload], rfl⟩
    show 32 * 4 ∣ 128
    norm_num
  · exact fun u _ j hj =>
      ⟨fun l hl => (witness_operandOK 1).1 j hj l hl, fun hj' => (witness_operandOK 1).2 ⟨j, hj'⟩⟩
  · show debitOf Params.pi Costs.adopted (witnessSem.unitDebit (fun _ => ()) () witnessWorkload 0 (witnessAct 0)) ≤
      1 / 400 * creditOf Params.pi Costs.adopted (witnessWorkload.layout.shape 0)
    rw [hzero]
    norm_num [creditOf, Params.pi, Costs.adopted, witnessWorkload]
  · rw [show range witnessWorkload.layout.N = range 1 from rfl, sum_range_one, hcr, hzero]
    norm_num [creditOf, Params.pi, Costs.adopted, witnessWorkload]

end Pouw.PearlC
