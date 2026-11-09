module

public import Definitions.NetworkCertifier.Guarantees.Checked
public import Proofs.NetworkCertifier.SampledCharge
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.KS
import Proofs.NetworkCertifier.Counting
import Proofs.NetworkCertifier.PooledLeak
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.SampledLeak

/-!
# The leak bounds with checked inputs

**The count.** A sequence accepted with a valid run at an incorrect set `F` of proof blocks of no class sampled `all` maps
injectively to its inputs off the checked and registered ones, the bit at which `holdAll` holds each committed gate of
`F`'s units, and its jitter code (`encard_acceptedChecked_le`): two sequences with one code agree step by step, since at
step `t` each egress row is the run's over the whole horizon (`rows_full`, by causality), and the check pins it from the
ingress through `t`, the inputs off `chk` and the held values (`CheckedUnique`). The count is
`2^#(freeInputs \ (reg ∪ chk)) · 2^(faultyBits F) · D^n`, and over the possible incorrect sets
`2^#(freeInputs \ (reg ∪ chk)) · #faultySets · 2^β · D^n`, `β` the most bits one commits (`encard_accSeqsChecked_le`).

**The relation.** On `RAcc δ`, the trace run with the incorrect proof blocks' committed gates held at the trace's values
(`heldAdv`) is the run the count reads at the inputs pinned to the trace (`inputOf_full`), so where it is `valid`,
`RLeakChecked δ` holds (`rLeakChecked_of`). Otherwise the check missed, which `CheckedWitness` bounds by `εchk`, so
`KSLinOn` moves from `RAcc δ` to `RLeakChecked δ` at `κ + εchk` (`ksLinOn_rLeakChecked`).

**The monitoring period.** `PooledLeak`'s count and escape bound at any per-window relation (`encard_pooledFam_le`,
`pr_pooled_le`) at `LeakAtChecked`, with `b` the most committed bits of one of the period's proof blocks.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset

/-! ### The run over the whole horizon -/

/-- **The full ingress reads the trace**: on the operator's moves over the whole horizon the trace carries, every input
reads the trace's value (an ingress gate the ingress the trace holds, every other input the trace's value). -/
theorem inputOf_full {L K S : ℕ} {lay : BoundaryGates L S} {C : Prog} {ref : ℕ → Option Bool}
    {op : ℕ → List (Obs L K S) → List Bool} {k : ℕ} {o : List (Obs L K S)} (hcar : Carries lay C ref op k o) (g : ℕ) :
    lay.inputOf k ((List.range k).map fun i => op i (o.take i)) (traceVal C ref) g = traceVal C ref g := by
  cases k with
  | zero => simp [BoundaryGates.inputOf]
  | succ n => exact SampledAuditProof.inputOf_eq hcar (Nat.lt_succ_self n) fun t' h1 h2 => absurd h2 (by omega)

/-- **A step's egress rows are the whole run's**: by causality, step `t`'s rows on the ingress up to `t` are those of the
run on the ingress over the whole horizon. -/
theorem rows_full {L K S : ℕ} (C : Prog) (lay : BoundaryGates L S) (n : ℕ) (hfits : lay.Fits C n)
    (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)) (p : (ℕ → Bool) × (ℕ → Option Bool)) {t : ℕ}
    (ht : t < n) :
    (circuitProgram C lay n).rows t (ingressUpTo op o t) p =
      fun ℓ => rowOf (valOver C (lay.inputOf n (ingressUpTo op o (n - 1)) p.1) p.2) (lay.egress t ℓ) := by
  funext ℓ
  simp only [circuitProgram]
  apply rowOf_congr
  intro σ e he
  apply valOver_congr_inp C p.2 e
  intro i hi
  apply inputOf_eq_of_causal hfits ht he (length_ingressUpTo op o t)
    (by rw [length_ingressUpTo]; omega) (by rw [length_ingressUpTo]; omega) _ p.1 hi
  intro j hj
  rw [ingressUpTo_getD op o [] hj, ingressUpTo_getD op o [] (by omega)]

theorem exists_of_mem_faultyGateSet {C : Prog} {P : Partition} {F : Finset ℕ} {g : ℕ} (h : g ∈ faultyGateSet C P F) :
    ∃ v ∈ F, FaultyGate C P {v} g := by
  simp only [faultyGateSet, Finset.mem_biUnion, List.mem_toFinset, List.mem_filter] at h
  obtain ⟨v, hv, hg, hb⟩ := h
  rcases hu : P.blocks[v]? with _ | u
  · rw [List.getD_eq_getElem?_getD, hu, Option.getD_none] at hg
    simp at hg
  · rw [List.getD_eq_getElem?_getD, hu, Option.getD_some] at hg hb
    exact ⟨v, hv, v, Finset.mem_singleton_self v, u, hu, hg, hb⟩

theorem disjoint_of_mem_faultySets {d : Design} {P : Partition} {δ : ℝ} {F : Finset ℕ} (h : F ∈ faultySets d P δ) :
    Disjoint F (allSlots d P) := by
  classical
  simp only [faultySets, Finset.mem_filter] at h
  exact h.2.2

/-! ### The count -/

/-- **One incorrect set's count with a valid run.** At a check that pins `chk` and an incorrect set `F` of proof blocks of
no class sampled `all`, the sequences accepted with a valid run at the inputs pinned at `R` number at most
`2^#(freeInputs \ (R ∪ chk)) · 2^(faultyBits F) · D^n`. -/
theorem encard_acceptedChecked_le {L K S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : BoundaryGates L S) (n : ℕ)
    (hfits : lay.Fits C n) (chk : Finset ℕ) (valid : (ℕ → Bool) → Prop) (hchk : CheckedUnique C P d lay n chk valid)
    (R : Finset ℕ) (ρ : ℕ → Bool) (F : Finset ℕ) (hF : Disjoint F (allSlots d P))
    (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ) :
    {o : List (Obs L K S) | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧
        AcceptedChecked C P lay n (AsReg C lay n R ρ) valid F op o}.encard ≤
      ((2 ^ (freeInputs C lay n \ (R ∪ chk)).card * (2 ^ faultyBits C P F * D ^ n) : ℕ) : ℕ∞) := by
  classical
  set Sq := {o : List (Obs L K S) | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧
    AcceptedChecked C P lay n (AsReg C lay n R ρ) valid F op o} with hSq
  set G : Finset ℕ := faultyGateSet C P F with hGdef
  set A : Finset ℕ := freeInputs C lay n \ (R ∪ chk) with hA
  let adv : List (Obs L K S) → (ℕ → Bool) × (ℕ → Option Bool) := fun o =>
    if h : AcceptedChecked C P lay n (AsReg C lay n R ρ) valid F op o then (h.choose, h.choose_spec.2.choose)
    else (fun _ => false, fun _ => none)
  have adv_spec : ∀ o ∈ Sq, (adv o).1 ∈ AsReg C lay n R ρ ∧ (∀ g b, (adv o).2 g = some b → g ∈ G) ∧
      valid (valOver C (lay.inputOf n ((List.range n).map fun i => op i (o.take i)) (adv o).1) (adv o).2) ∧
      Accepted (circuitProgram C lay n) op n o (adv o) := by
    intro o ho
    have hacc : AcceptedChecked C P lay n (AsReg C lay n R ρ) valid F op o := ho.2.2
    simp only [adv, hacc, ↓reduceDIte]
    exact ⟨hacc.choose_spec.1, fun g b hgb => mem_faultyGateSet (hacc.choose_spec.2.choose_spec.1 g b hgb),
      hacc.choose_spec.2.choose_spec.2.1, hacc.choose_spec.2.choose_spec.2.2⟩
  have hvalid : ∀ o ∈ Sq, 0 < n → valid (valOver C (lay.inputOf n (ingressUpTo op o (n - 1)) (adv o).1)
      (holdAll C lay n op o (adv o).1 (adv o).2 G)) := by
    intro o ho hn
    obtain ⟨-, hG1, hval, -⟩ := adv_spec o ho
    have hfull : (List.range n).map (fun i => op i (o.take i)) = ingressUpTo op o (n - 1) := by
      simp only [ingressUpTo, Nat.sub_add_cancel (show 1 ≤ n by omega)]
    have heq : valOver C (lay.inputOf n (ingressUpTo op o (n - 1)) (adv o).1) (adv o).2 =
        valOver C (lay.inputOf n (ingressUpTo op o (n - 1)) (adv o).1) (holdAll C lay n op o (adv o).1 (adv o).2 G) :=
      funext fun g => valOver_hold C (lay.inputOf n (ingressUpTo op o (n - 1)) (adv o).1)
        (lay.inputOf n (ingressUpTo op o (n - 1)) (adv o).1) (adv o).2 G hG1 g (fun _ _ => rfl) g
        Relation.ReflTransGen.refl
    rw [← heq, ← hfull]
    exact hval
  let code : List (Obs L K S) → (↥A → Bool) × (↥G → Bool) × (Fin n → ℕ) := fun o =>
    ((fun g => (adv o).1 g),
      (fun g => valOver C (lay.inputOf n (ingressUpTo op o (n - 1)) (adv o).1) (adv o).2 g), jitterCode o)
  have hinj : Set.InjOn code Sq := by
    intro o ho o' ho' h
    simp only [code, Prod.mk.injEq] at h
    obtain ⟨ha, hv, hj⟩ := h
    obtain ⟨haR, hG1, -, acc⟩ := adv_spec o ho
    obtain ⟨haR', hG2, -, acc'⟩ := adv_spec o' ho'
    have hoff : ∀ g ∉ chk, (adv o).1 g = (adv o').1 g := by
      intro g hg
      by_cases hf : g ∈ freeInputs C lay n
      · by_cases hr : g ∈ R
        · rw [haR.1 g hr hf, haR'.1 g hr hf]
        · exact congrFun ha ⟨g, Finset.mem_sdiff.2 ⟨hf, fun h => (Finset.mem_union.1 h).elim hr hg⟩⟩
      · rw [haR.2 g hf, haR'.2 g hf]
    have hhold : holdAll C lay n op o (adv o).1 (adv o).2 G = holdAll C lay n op o' (adv o').1 (adv o').2 G := by
      funext g
      simp only [holdAll]
      split_ifs with hg
      · exact congrArg some (congrFun hv ⟨g, hg⟩)
      · rfl
    have hsupp : ∀ g b, holdAll C lay n op o (adv o).1 (adv o).2 G g = some b →
        ∃ v ∉ allSlots d P, FaultyGate C P {v} g := by
      intro g b hgb
      by_cases hg : g ∈ G
      · obtain ⟨v, hv, hfv⟩ := exists_of_mem_faultyGateSet hg
        exact ⟨v, Finset.disjoint_left.1 hF hv, hfv⟩
      · simp [holdAll, hg] at hgb
    apply eq_of_stepwise ho.1 ho'.1
    intro t ht hpre
    obtain ⟨h1, hr, hs⟩ := acc t ht
    obtain ⟨h2, hr', hs'⟩ := acc' t ht
    have hn : 0 < n := by omega
    have hlen : (ingressUpTo op o (n - 1)).length = n := by rw [length_ingressUpTo]; omega
    have hlen' : (ingressUpTo op o' (n - 1)).length = n := by rw [length_ingressUpTo]; omega
    have hpreF : ∀ j ≤ t, (ingressUpTo op o (n - 1)).getD j [] = (ingressUpTo op o' (n - 1)).getD j [] := by
      intro j hj
      rw [ingressUpTo_getD op o [] (by omega), ingressUpTo_getD op o' [] (by omega),
        ← ingressUpTo_getD op o [] hj, ← ingressUpTo_getD op o' [] hj, ingressUpTo_congr op hpre]
    have hv1 := hvalid o ho hn
    have hv2 := hvalid o' ho' hn
    rw [← hhold] at hv2
    apply Obs.ext'
    · calc o[t].rows = (circuitProgram C lay n).rows t (ingressUpTo op o t) (adv o) := hr
        _ = (circuitProgram C lay n).rows t (ingressUpTo op o t)
              ((adv o).1, holdAll C lay n op o (adv o).1 (adv o).2 G) :=
            rows_holdAll C lay n hfits op o (adv o) G hG1 ht
        _ = fun ℓ => rowOf (valOver C (lay.inputOf n (ingressUpTo op o (n - 1)) (adv o).1)
              (holdAll C lay n op o (adv o).1 (adv o).2 G)) (lay.egress t ℓ) :=
            rows_full C lay n hfits op o ((adv o).1, holdAll C lay n op o (adv o).1 (adv o).2 G) ht
        _ = fun ℓ => rowOf (valOver C (lay.inputOf n (ingressUpTo op o' (n - 1)) (adv o').1)
              (holdAll C lay n op o (adv o).1 (adv o).2 G)) (lay.egress t ℓ) :=
            funext (hchk t ht _ _ _ _ _ hlen hlen' hpreF hoff hsupp hv1 hv2)
        _ = fun ℓ => rowOf (valOver C (lay.inputOf n (ingressUpTo op o' (n - 1)) (adv o').1)
              (holdAll C lay n op o' (adv o').1 (adv o').2 G)) (lay.egress t ℓ) := by rw [hhold]
        _ = (circuitProgram C lay n).rows t (ingressUpTo op o' t)
              ((adv o').1, holdAll C lay n op o' (adv o').1 (adv o').2 G) :=
            (rows_full C lay n hfits op o' ((adv o').1, holdAll C lay n op o' (adv o').1 (adv o').2 G) ht).symm
        _ = (circuitProgram C lay n).rows t (ingressUpTo op o' t) (adv o') :=
            (rows_holdAll C lay n hfits op o' (adv o') G hG2 ht).symm
        _ = o'[t].rows := hr'.symm
    · exact jitter_eq_of_jitterCode hj ht h1 h2
    · funext κ; rw [hs κ, hs' κ]
  let T : Finset ((↥A → Bool) × (↥G → Bool) × (Fin n → ℕ)) :=
    univ ×ˢ univ ×ˢ Fintype.piFinset fun _ : Fin n => range D
  have hsub : code '' Sq ⊆ (T : Set ((↥A → Bool) × (↥G → Bool) × (Fin n → ℕ))) := by
    rintro _ ⟨o, ho, rfl⟩
    simp only [T, Finset.coe_product, Finset.coe_univ, Set.mem_prod, Set.mem_univ, true_and, Finset.mem_coe]
    exact jitterCode_mem o ho.1 ho.2.1
  calc Sq.encard = (code '' Sq).encard := hinj.encard_image.symm
    _ ≤ (T : Set ((↥A → Bool) × (↥G → Bool) × (Fin n → ℕ))).encard := Set.encard_le_encard hsub
    _ = ((2 ^ A.card * (2 ^ G.card * D ^ n) : ℕ) : ℕ∞) := by
        rw [Set.encard_coe_eq_coe_finsetCard]
        congr 1
        simp [T, Fintype.card_piFinset]
    _ ≤ ((2 ^ A.card * (2 ^ faultyBits C P F * D ^ n) : ℕ) : ℕ∞) := by
        have h := Nat.mul_le_mul_right (D ^ n) (Nat.pow_le_pow_right (n := 2) (by norm_num) (card_faultyGateSet_le C P F))
        exact_mod_cast Nat.mul_le_mul_left _ h

/-- **The count with a valid run over the possible incorrect sets**: `2^#(freeInputs \ (R ∪ chk)) · #faultySets · 2^β · D^n`,
`β` the most committed bits of a possible incorrect set. -/
theorem encard_accSeqsChecked_le {L K S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : BoundaryGates L S) (n : ℕ)
    (hfits : lay.Fits C n) (chk : Finset ℕ) (valid : (ℕ → Bool) → Prop) (hchk : CheckedUnique C P d lay n chk valid)
    (R : Finset ℕ) (ρ : ℕ → Bool) (δ : ℝ) (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ) :
    (AccSeqsChecked C P d lay n (AsReg C lay n R ρ) valid δ op D).encard ≤
      ((2 ^ (freeInputs C lay n \ (R ∪ chk)).card * (faultySets d P δ).card *
        2 ^ ((faultySets d P δ).sup (faultyBits C P)) * D ^ n : ℕ) : ℕ∞) := by
  classical
  let Sq : Finset ℕ → Set (List (Obs L K S)) := fun F =>
    {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ AcceptedChecked C P lay n (AsReg C lay n R ρ) valid F op o}
  set β := (faultySets d P δ).sup (faultyBits C P) with hβ
  set c := (freeInputs C lay n \ (R ∪ chk)).card with hc
  have hsub : AccSeqsChecked C P d lay n (AsReg C lay n R ρ) valid δ op D ⊆ ⋃ F ∈ faultySets d P δ, Sq F := by
    rintro o ⟨hl, hD, F, hF, hacc⟩
    exact Set.mem_biUnion hF ⟨hl, hD, hacc⟩
  calc _ ≤ (⋃ F ∈ faultySets d P δ, Sq F).encard := Set.encard_le_encard hsub
    _ ≤ ∑ F ∈ faultySets d P δ, (Sq F).encard := encard_biUnion_le _ _
    _ ≤ ∑ _F ∈ faultySets d P δ, ((2 ^ c * (2 ^ β * D ^ n) : ℕ) : ℕ∞) := by
        apply Finset.sum_le_sum
        intro F hF
        calc (Sq F).encard ≤ ((2 ^ c * (2 ^ faultyBits C P F * D ^ n) : ℕ) : ℕ∞) :=
              encard_acceptedChecked_le C P d lay n hfits chk valid hchk R ρ F (disjoint_of_mem_faultySets hF) op D
          _ ≤ ((2 ^ c * (2 ^ β * D ^ n) : ℕ) : ℕ∞) := by
              have h := Nat.mul_le_mul_right (D ^ n)
                (Nat.pow_le_pow_right (n := 2) (by norm_num) (Finset.le_sup (f := faultyBits C P) hF))
              exact_mod_cast Nat.mul_le_mul_left _ h
    _ = _ := by
        rw [Finset.sum_const, nsmul_eq_mul]
        norm_cast
        ring

/-- **A window's count with a valid run**, at a check that pins its checked inputs. -/
theorem encard_leakAtChecked_le {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) (s : AuditStatement)
    (P : Partition) (hchk : CheckedUnique s.prog P s.design (lay s) (k s) (chk s) (valid s)) (ρ : ℕ → Bool)
    (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ) :
    {o | LeakAtChecked lay k reg valid Dj s P ρ op F o}.encard ≤
      ((2 ^ (freeInputs s.prog (lay s) (k s) \ (reg s ∪ chk s)).card * (2 ^ faultyBits s.prog P F * Dj ^ k s) : ℕ) :
        ℕ∞) := by
  by_cases hfits : (lay s).Fits s.prog (k s)
  · by_cases hF : Disjoint F (allSlots s.design P)
    · calc _ ≤ {o : List (Obs L K S) | o.length = k s ∧ (∀ x ∈ o, x.jitter < Dj) ∧
              AcceptedChecked s.prog P (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) (valid s) F op o}.encard :=
            Set.encard_le_encard fun o h => h.2.2
        _ ≤ _ :=
            encard_acceptedChecked_le s.prog P s.design (lay s) (k s) hfits (chk s) (valid s) hchk (reg s) ρ F hF op Dj
    · have : {o | LeakAtChecked lay k reg valid Dj s P ρ op F o} = ∅ :=
        Set.eq_empty_iff_forall_notMem.2 fun o h => hF h.2.1
      rw [this, Set.encard_empty]
      exact zero_le
  · have : {o | LeakAtChecked lay k reg valid Dj s P ρ op F o} = ∅ :=
      Set.eq_empty_iff_forall_notMem.2 fun o h => hfits h.1
    rw [this, Set.encard_empty]
    exact zero_le

/-! ### The relation -/

/-- **A valid trace run gives the leak relation with a valid run**: on the audit's scope, at a witness in `RelS δ` whose
incorrect set holds no proof block of a class sampled `all` and whose trace run is `valid`, every sequence the trace
carries is accepted with a valid run, at the inputs pinned to the trace at the registered ones and the trace's values
held at the incorrect set's committed gates. -/
theorem rLeakChecked_of {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin} (H : Msg → Msg)
    (D : Claims.SamplingGeneric.Compiled Coin A) {L K S : ℕ}
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) {δ : ℝ} (reg : AuditStatement → Finset ℕ)
    (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) {x : Msg} (hx : ScopeAcc D lay k x) {w : Wit}
    {s : AuditStatement} {P : Partition} (hs : D.parse x = some s) (hP : D.partition s = some P)
    (hRel : RelS H D.commit δ s P w) (hdisj : Disjoint (faulty H D.commit s P w) (allSlots s.design P))
    (hv : valid s (valOver s.prog (traceVal s.prog (w.val H D.commit s))
      (heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s)))) :
    RLeakChecked H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) valid Dj
      x w := by
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hin⟩, s₂, hs₂, hfit⟩ := hx
  rw [hs] at hs₁ hs₂
  obtain rfl := Option.some.inj hs₁
  obtain rfl := Option.some.inj hs₂
  rw [hP] at hP₁
  obtain rfl := Option.some.inj hP₁
  obtain ⟨ι, hι⟩ := SampledAuditProof.correct_of_not_faulty H D.commit s P w
  have hF := faulty_mem_faultySets hRel hdisj
  let a : ℕ → Bool := fun g =>
    if g ∈ freeInputs s.prog (lay s) (k s) then traceVal s.prog (w.val H D.commit s) g else false
  refine ⟨s, P, hs, hP, fun op o hcar hcov hl hD => ⟨hl, hD, _, hF, a, ⟨fun g _ hg => if_pos hg, fun g hg => if_neg hg⟩,
    heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s),
    fun g b h => SampledAuditProof.faultyGate_of_heldAdv h, ?_, ?_⟩⟩
  · have hinp : valOver s.prog (traceVal s.prog (w.val H D.commit s))
        (heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s)) =
        valOver s.prog ((lay s).inputOf (k s) ((List.range (k s)).map fun i => op i (o.take i)) a)
          (heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s)) :=
      funext (valOver_congr_input s.prog _ fun i hi => by
        rw [← inputOf_full hcar i]
        exact inputOf_restrict s.prog (lay s) (k s) _ _ hi)
    rw [← hinp]
    exact hv
  · intro t ht
    obtain ⟨h, hr, hst⟩ := SampledAuditProof.accepted_of hin.1 hin.2.1 hfit ι hι hcar hcov t ht
    refine ⟨h, ?_, hst⟩
    have hv' : valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) (traceVal s.prog (w.val H D.commit s)))
        (heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s)) =
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) a)
          (heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s)) :=
      funext (valOver_congr_input s.prog _ fun i hi => inputOf_restrict s.prog (lay s) (k s) _ _ hi)
    rw [hr]
    simp only [circuitProgram, hv']

/-- **The leak relation with a valid run gives the leak relation**: a sequence accepted with a valid run is accepted. -/
theorem rLeak_of_rLeakChecked {H : Msg → Msg} {Cm : CommitmentScheme Bool} {parse : Msg → Option AuditStatement}
    {partition : AuditStatement → Option Partition} {L K S : ℕ} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {δ : ℝ} {As : AuditStatement → (ℕ → Bool) → Set (ℕ → Bool)}
    {valid : AuditStatement → (ℕ → Bool) → Prop} {Dj : ℕ} {x : Msg} {w : Wit}
    (h : RLeakChecked H Cm parse partition L K S lay k δ As valid Dj x w) :
    RLeak H Cm parse partition L K S lay k δ As Dj x w := by
  obtain ⟨s, P, hs, hP, h⟩ := h
  refine ⟨s, P, hs, hP, fun op o hC hCov hl hD => ?_⟩
  obtain ⟨hl', hD', F, hF, a, ha, free, hfree, -, hacc⟩ := h op o hC hCov hl hD
  exact ⟨hl', hD', F, hF, a, ha, free, hfree, hacc⟩

/-- **`KSLinOn` moves from `RAcc δ` to the leak relation with a valid run**, on the statements where the check holds on
the trace's run except with probability `εchk` (`CheckedWitness`), at `κ + εchk`. -/
theorem ksLinOn_rLeakChecked {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Claims.SamplingGeneric.Arg Coin} (H : Msg → Msg) (D : Claims.SamplingGeneric.Compiled Coin A) {L K S : ℕ}
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) {δ : ℝ} (reg : AuditStatement → Finset ℕ)
    (valid : AuditStatement → (ℕ → Bool) → Prop) (εchk : Msg → ℝ) (Dj : ℕ)
    (h : KSLinOn H D.V (ScopeAcc D lay k) (RAcc H D.commit D.parse D.partition L K S lay k δ) D.E (D.κ δ) D.γ
      (fun _ => 1) (Bext H D.V D.E)) :
    KSLinOn H D.V (fun x => ScopeAcc D lay k x ∧ CheckedWitness H D valid εchk x)
      (RLeakChecked H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) valid Dj)
      D.E (fun x => D.κ δ x + εchk x) D.γ (fun _ => 1) (Bext H D.V D.E) := by
  intro x hx Ω _ _ P
  have h1 := h x hx.1 Ω P
  have h2 := hx.2 Ω P
  unfold PrBad at h1 ⊢
  unfold PrCheckMiss at h2
  refine (Claims.Sampling.Pr_or_le (fun r hr => ?_) h1 h2).trans (le_of_eq ?_)
  · beta_reduce at hr ⊢
    obtain ⟨hacc, hng⟩ := hr
    generalize (D.E.eval H P x r.1 r.2.1 r.2.2).1 = ow at hng ⊢
    rcases ow with _ | w
    · exact Or.inl ⟨hacc, fun hR => by simp only [Rel.ext] at hR⟩
    · by_cases hR : RAcc H D.commit D.parse D.partition L K S lay k δ x w
      · obtain ⟨s, Pt, hs, hP, hRel, hdisj, -⟩ := hR
        by_cases hv : valid s (valOver s.prog (traceVal s.prog (w.val H D.commit s))
            (heldAdv s.prog Pt (faulty H D.commit s Pt w) (w.val H D.commit s)))
        · exact absurd (rLeakChecked_of H D lay k reg valid Dj hx.1 hs hP hRel hdisj hv) hng
        · exact Or.inr ⟨hacc, w, s, Pt, rfl, hs, hP, hdisj, hv⟩
      · exact Or.inl ⟨hacc, hR⟩
  · beta_reduce
    ring

/-! ### The guarantees -/

/-- **The leak bound with checked inputs**: `Guarantees.SampledLeakChecked`, from `ksLinOn_rLeakChecked` at
`KSAccepted`'s bound (`SecurityProofs.SampledAuditAccepted`) and `encard_accSeqsChecked_le` on the statements whose
layout fits. -/
theorem sampledLeakChecked : Guarantees.SampledLeakChecked := by
  intro Coin _ _ _ A H D hks L K S lay k δ hδ reg chk valid εchk Dj
  refine ⟨ksLinOn_rLeakChecked H D lay k reg valid εchk Dj
      (SecurityProofs.SampledAuditAccepted H D L K S lay k hks δ hδ),
    fun x hx s P hs hP hchk op ρ => ?_⟩
  obtain ⟨s', hs', hfits⟩ := hx.2
  rw [Option.some_injective _ (hs'.symm.trans hs)] at hfits
  exact encard_accSeqsChecked_le s.prog P s.design (lay s) (k s) hfits (chk s) (valid s) hchk (reg s) ρ δ op Dj

/-- **The leak through the network certifier over a monitoring period, with checked inputs**:
`Guarantees.PooledLeakChecked`, from `pr_pooled_le` at the leak relations with a valid run and `encard_pooledFam_le`
at `LeakAtChecked`'s count, `b` the most committed bits of one of the period's proof blocks. -/
theorem pooledLeakChecked : Guarantees.PooledLeakChecked := by
  intro L K S lay k reg chk valid Dj m sw Pw ρ op n hchk Ω _ _ Conn Cell Srv Dev Z _ arr f tap run htrust cap Ss Zs
    schedOf srvOf dec hch acc W had e e' he q hq0 hq1 hesc hleak Fmax
  obtain ⟨-, hfit, hs, hdz⟩ := hch
  obtain ⟨hacc, hW, hWn⟩ := had
  refine ⟨fun z hz => pr_pooled_le (acc z) (hacc z) (W z) (hW z) e e' he q hq0 hq1 (hesc z hz)
    (fun i ω => LeakAtChecked lay k reg valid Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i)
      ((run tap (schedOf z ω, srvOf z ω)).2 i)) (hleak z hz) Fmax, fun ω => ?_⟩
  exact encard_pooledFam_le (fun i => LeakAtChecked lay k reg valid Dj (sw i) (Pw i) (ρ i) (op i)) n
    (univ.sup fun i => (range (n i)).sup fun v => faultyBits (sw i).prog (Pw i) {v})
    (fun i => (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i))).card)
    (fun i => Dj ^ k (sw i)) (fun i F => faultyBits (sw i).prog (Pw i) F)
    (fun i F hF => faultyBits_le_card_mul _ _ hF fun v hv =>
      (Finset.le_sup (f := fun v => faultyBits (sw i).prog (Pw i) {v}) (mem_range.2 hv)).trans
        (Finset.le_sup (f := fun i => (range (n i)).sup fun v => faultyBits (sw i).prog (Pw i) {v}) (mem_univ i)))
    (fun i F _ => encard_leakAtChecked_le lay k reg chk valid Dj (sw i) (Pw i) (hchk i) (ρ i) (op i) F)
    arr f tap run htrust cap Ss hfit Zs (fun z => schedOf z ω) (fun z => srvOf z ω) (fun z hz => hs z hz ω) (dec ω)
    (fun z hz => hdz z hz ω) (fun z => W z ω) (fun z i => hWn z ω i) Fmax

end NetTiming
