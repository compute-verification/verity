module

public import Definitions.NetworkCertifier.Guarantees.Carried
public import Proofs.NetworkCertifier.Checked
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
import Proofs.Hash.RowBinding
import Proofs.NetworkCertifier.Counting
import Proofs.NetworkCertifier.PooledLeak
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.SampledLeak

/-!
# The leak bounds over a carried pool

**One run's guarantee** (`sampledLeakCarried`). The class's run from the trace (`rLeakChecked_of`: the trace's inputs, and
its values held at the incorrect set's committed gates) reads the trace's values at opening gates that are inputs and
leaves the trace run's values at the closing gates (`rLeakCarried_of`), so `KSLinOn` moves to the relation at those rows
as it moves to `RLeakChecked`. At a fixed opened row, a sequence accepted from it is accepted with the opening gates
registered at its values, so the class's count there charges none of them (`encard_accSeqsCarried_le`).

**One run's step** (`carried_step`). Two runs accepted with a valid run at one incorrect set, whose inputs agree off the
checked ones and whose held values and jitter codes agree, see one observation sequence, by the class proof's induction
over steps, and leave one row at the closing gates (`CheckedClosing`).

**The count** (`encard_carried_le`). A period's observations map injectively to each run's inputs off its checked,
registered and opening gates, its held values and its jitter code. By induction over the runs, two sequences with one
code open one row at each run: the row the record rule names, the run before's closing row (one row, by the induction)
or `genesis`. The opening gates then read one row, so the runs' inputs agree off the checked gates and the step applies.

**The monitoring period** (`encard_pooledCarried_le`, `pooledLeakCarried`). At each pattern of restarts the count is
`encard_pooledJoint_le` at that joint count, and the secrets whose audits all accept and that meet the good event at the
pattern serving picked for them are in the union of those sets over the registered patterns. The escape bound is
`pr_pooled_le` at each run's leak relation, its opening root, and no collision at its opening, which together give rows
that chain at the pattern picked (`Hash.BindingProofs.rowSeg_bind`).

**The empty pool** (`pooledLeakChecked_of_carried`). With no opening or closing gate, one pattern that restarts no run,
and every row empty, a run is accepted from its row exactly when it is accepted with a valid run, every run opens the
empty row's root, and no opening collides (`poolColl_empty`). So the period's bound there is the bound with checked
inputs over windows that each hold a pool of their own: the escape bound loses its collision term, and the count over
every secret meeting the good event is `encard_pooledCarried_le` at the one pattern.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset

/-! ### One run -/

/-- **A free input reads the advice** at an ingress over the horizon. -/
theorem inputOf_free {L S : ℕ} {C : Prog} {lay : BoundaryGates L S} {k : ℕ} (f : ℕ → List Bool) (a : ℕ → Bool)
    {g : ℕ} (hg : g ∈ freeInputs C lay k) : lay.inputOf k ((List.range k).map f) a g = a g := by
  have hany : (List.range k).any (fun i => (lay.ingress i).contains g) = false := (Finset.mem_filter.1 hg).2.2
  have hnone : (List.range ((List.range k).map f).length).findSome?
      (fun i => ((lay.ingress i).findIdx? (· == g)).map (i, ·)) = none := by
    rw [List.findSome?_eq_none_iff]
    intro i hi
    rw [Option.map_eq_none_iff, List.findIdx?_eq_none_iff]
    intro x hx
    have hik : i < k := by simpa using hi
    have hnot : ¬(lay.ingress i).contains g = true := List.any_eq_false.1 hany i (List.mem_range.2 hik)
    rw [List.contains_iff_mem] at hnot
    rw [beq_eq_false_iff_ne]
    rintro rfl
    exact hnot hx
  unfold BoundaryGates.inputOf
  rw [hnone, hany]
  simp

/-- A run's witness: its run `p`, accepted with a valid run at the inputs `As`, from the row `rin` at the opening gates,
leaving `rout` at the closing gates. -/
def CarriedRun {L K S : ℕ} (C : Prog) (P : Partition) (lay : BoundaryGates L S) (k : ℕ) (As : Set (ℕ → Bool))
    (valid : (ℕ → Bool) → Prop) (prow nrow : List ℕ) (rin rout : Hash.Row) (F : Finset ℕ)
    (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)) (p : (ℕ → Bool) × (ℕ → Option Bool)) : Prop :=
  p.1 ∈ As ∧ (∀ g b, p.2 g = some b → FaultyGate C P F g) ∧
    valid (valOver C (lay.inputOf k ((List.range k).map fun i => op i (o.take i)) p.1) p.2) ∧
    prow.map (lay.inputOf k ((List.range k).map fun i => op i (o.take i)) p.1) = rin ∧
    nrow.map (valOver C (lay.inputOf k ((List.range k).map fun i => op i (o.take i)) p.1) p.2) = rout ∧
    Accepted (circuitProgram C lay k) op k o p

theorem acceptedCarried_iff {L K S : ℕ} {C : Prog} {P : Partition} {lay : BoundaryGates L S} {k : ℕ}
    {As : Set (ℕ → Bool)} {valid : (ℕ → Bool) → Prop} {prow nrow : List ℕ} {rin rout : Hash.Row} {F : Finset ℕ}
    {op : ℕ → List (Obs L K S) → List Bool} {o : List (Obs L K S)} :
    AcceptedCarried C P lay k As valid prow nrow rin rout F op o ↔
      ∃ p, CarriedRun C P lay k As valid prow nrow rin rout F op o p := by
  constructor
  · rintro ⟨a, ha, free, hfree, hv, hp, hn, hacc⟩
    exact ⟨(a, free), ha, hfree, hv, hp, hn, hacc⟩
  · rintro ⟨⟨a, free⟩, ha, hfree, hv, hp, hn, hacc⟩
    exact ⟨a, ha, free, hfree, hv, hp, hn, hacc⟩

/-- **One run's step**: two runs accepted with a valid run at one incorrect set `F` of proof blocks of no class sampled
`all`, whose inputs agree off the checked inputs, whose values at `F`'s committed gates agree, and whose jitter codes
agree, see one observation sequence and leave one row at the closing gates. -/
theorem carried_step {L K S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : BoundaryGates L S) (n : ℕ)
    (hfits : lay.Fits C n) (chk : Finset ℕ) (valid : (ℕ → Bool) → Prop) (hchk : CheckedUnique C P d lay n chk valid)
    (nrow : List ℕ) (hcls : CheckedClosing C P d lay n chk valid nrow) (F : Finset ℕ)
    (hF : Disjoint F (allSlots d P)) (op : ℕ → List (Obs L K S) → List Bool) {o o' : List (Obs L K S)}
    (ho : o.length = n) (ho' : o'.length = n) {p p' : (ℕ → Bool) × (ℕ → Option Bool)}
    (hG : ∀ g b, p.2 g = some b → g ∈ faultyGateSet C P F)
    (hG' : ∀ g b, p'.2 g = some b → g ∈ faultyGateSet C P F)
    (hval : valid (valOver C (lay.inputOf n ((List.range n).map fun i => op i (o.take i)) p.1) p.2))
    (hval' : valid (valOver C (lay.inputOf n ((List.range n).map fun i => op i (o'.take i)) p'.1) p'.2))
    (hacc : Accepted (circuitProgram C lay n) op n o p) (hacc' : Accepted (circuitProgram C lay n) op n o' p')
    (hoff : ∀ g ∉ chk, p.1 g = p'.1 g)
    (hheld : ∀ g ∈ faultyGateSet C P F,
      valOver C (lay.inputOf n ((List.range n).map fun i => op i (o.take i)) p.1) p.2 g =
        valOver C (lay.inputOf n ((List.range n).map fun i => op i (o'.take i)) p'.1) p'.2 g)
    (hj : jitterCode (n := n) o = jitterCode o') :
    o = o' ∧ nrow.map (valOver C (lay.inputOf n ((List.range n).map fun i => op i (o.take i)) p.1) p.2) =
      nrow.map (valOver C (lay.inputOf n ((List.range n).map fun i => op i (o'.take i)) p'.1) p'.2) := by
  classical
  obtain ⟨hf, hhf⟩ : ∃ hf : ℕ → Option Bool, hf = fun g => if g ∈ faultyGateSet C P F then
      some (valOver C (lay.inputOf n ((List.range n).map fun i => op i (o.take i)) p.1) p.2 g) else none :=
    ⟨_, rfl⟩
  have hhf' : hf = fun g => if g ∈ faultyGateSet C P F then
      some (valOver C (lay.inputOf n ((List.range n).map fun i => op i (o'.take i)) p'.1) p'.2 g) else none := by
    rw [hhf]
    funext g
    split_ifs with hg
    · rw [hheld g hg]
    · rfl
  have hrun : valOver C (lay.inputOf n ((List.range n).map fun i => op i (o.take i)) p.1) p.2 =
      valOver C (lay.inputOf n ((List.range n).map fun i => op i (o.take i)) p.1) hf := by
    rw [hhf]
    exact funext fun g => valOver_hold C _ _ p.2 (faultyGateSet C P F) hG g (fun _ _ => rfl) g
      Relation.ReflTransGen.refl
  have hrun' : valOver C (lay.inputOf n ((List.range n).map fun i => op i (o'.take i)) p'.1) p'.2 =
      valOver C (lay.inputOf n ((List.range n).map fun i => op i (o'.take i)) p'.1) hf := by
    rw [hhf']
    exact funext fun g => valOver_hold C _ _ p'.2 (faultyGateSet C P F) hG' g (fun _ _ => rfl) g
      Relation.ReflTransGen.refl
  have hsupp : ∀ g b, hf g = some b → ∃ v ∉ allSlots d P, FaultyGate C P {v} g := by
    intro g b hgb
    by_cases hg : g ∈ faultyGateSet C P F
    · obtain ⟨v, hv, hfv⟩ := exists_of_mem_faultyGateSet hg
      exact ⟨v, Finset.disjoint_left.1 hF hv, hfv⟩
    · simp [hhf, hg] at hgb
  have hv1 : valid (valOver C (lay.inputOf n ((List.range n).map fun i => op i (o.take i)) p.1) hf) := by
    rw [← hrun]
    exact hval
  have hv2 : valid (valOver C (lay.inputOf n ((List.range n).map fun i => op i (o'.take i)) p'.1) hf) := by
    rw [← hrun']
    exact hval'
  have hoo : o = o' := by
    rcases Nat.eq_zero_or_pos n with hn | hn
    · rw [List.eq_nil_of_length_eq_zero (ho.trans hn), List.eq_nil_of_length_eq_zero (ho'.trans hn)]
    have hfull : (List.range n).map (fun i => op i (o.take i)) = ingressUpTo op o (n - 1) := by
      simp only [ingressUpTo, Nat.sub_add_cancel (show 1 ≤ n by omega)]
    have hfull' : (List.range n).map (fun i => op i (o'.take i)) = ingressUpTo op o' (n - 1) := by
      simp only [ingressUpTo, Nat.sub_add_cancel (show 1 ≤ n by omega)]
    have hhold : holdAll C lay n op o p.1 p.2 (faultyGateSet C P F) = hf := by
      rw [hhf]
      funext g
      simp only [holdAll, hfull]
    have hhold' : holdAll C lay n op o' p'.1 p'.2 (faultyGateSet C P F) = hf := by
      rw [hhf']
      funext g
      simp only [holdAll, hfull']
    rw [hfull] at hv1
    rw [hfull'] at hv2
    apply eq_of_stepwise ho ho'
    intro t ht hpre
    obtain ⟨h1, hr, hs⟩ := hacc t ht
    obtain ⟨h2, hr', hs'⟩ := hacc' t ht
    have hlen : (ingressUpTo op o (n - 1)).length = n := by rw [length_ingressUpTo]; omega
    have hlen' : (ingressUpTo op o' (n - 1)).length = n := by rw [length_ingressUpTo]; omega
    have hpreF : ∀ j ≤ t, (ingressUpTo op o (n - 1)).getD j [] = (ingressUpTo op o' (n - 1)).getD j [] := by
      intro j hj
      rw [ingressUpTo_getD op o [] (by omega), ingressUpTo_getD op o' [] (by omega),
        ← ingressUpTo_getD op o [] hj, ← ingressUpTo_getD op o' [] hj, ingressUpTo_congr op hpre]
    apply Obs.ext'
    · calc o[t].rows = (circuitProgram C lay n).rows t (ingressUpTo op o t) p := hr
        _ = (circuitProgram C lay n).rows t (ingressUpTo op o t)
              (p.1, holdAll C lay n op o p.1 p.2 (faultyGateSet C P F)) :=
            rows_holdAll C lay n hfits op o p (faultyGateSet C P F) hG ht
        _ = fun ℓ => rowOf (valOver C (lay.inputOf n (ingressUpTo op o (n - 1)) p.1)
              (holdAll C lay n op o p.1 p.2 (faultyGateSet C P F))) (lay.egress t ℓ) :=
            rows_full C lay n hfits op o (p.1, holdAll C lay n op o p.1 p.2 (faultyGateSet C P F)) ht
        _ = fun ℓ => rowOf (valOver C (lay.inputOf n (ingressUpTo op o (n - 1)) p.1) hf) (lay.egress t ℓ) := by
            rw [hhold]
        _ = fun ℓ => rowOf (valOver C (lay.inputOf n (ingressUpTo op o' (n - 1)) p'.1) hf) (lay.egress t ℓ) :=
            funext (hchk t ht _ _ _ _ _ hlen hlen' hpreF hoff hsupp hv1 hv2)
        _ = fun ℓ => rowOf (valOver C (lay.inputOf n (ingressUpTo op o' (n - 1)) p'.1)
              (holdAll C lay n op o' p'.1 p'.2 (faultyGateSet C P F))) (lay.egress t ℓ) := by
            rw [hhold']
        _ = (circuitProgram C lay n).rows t (ingressUpTo op o' t)
              (p'.1, holdAll C lay n op o' p'.1 p'.2 (faultyGateSet C P F)) :=
            (rows_full C lay n hfits op o' (p'.1, holdAll C lay n op o' p'.1 p'.2 (faultyGateSet C P F)) ht).symm
        _ = (circuitProgram C lay n).rows t (ingressUpTo op o' t) p' :=
            (rows_holdAll C lay n hfits op o' p' (faultyGateSet C P F) hG' ht).symm
        _ = o'[t].rows := hr'.symm
    · exact jitter_eq_of_jitterCode hj ht h1 h2
    · funext κ; rw [hs κ, hs' κ]
  subst hoo
  refine ⟨rfl, ?_⟩
  rw [hrun, hrun']
  exact hcls _ p.1 p'.1 hf (by simp) hoff hsupp hv1 hv2

/-! ### One run's guarantee -/

/-- **A valid trace run gives the leak relation from the trace's rows**: `rLeakChecked_of`'s run (the trace's inputs, and
its values held at the incorrect set's committed gates) reads the trace's values at opening gates that are inputs, and
leaves the trace run's values at the closing gates. -/
theorem rLeakCarried_of {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin} (H : Msg → Msg)
    (D : Claims.SamplingGeneric.Compiled Coin A) {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) {δ : ℝ} (reg : AuditStatement → Finset ℕ)
    (valid : AuditStatement → (ℕ → Bool) → Prop) (prow nrow : AuditStatement → List ℕ) (hopen : OpensAtInputs prow)
    (Dj : ℕ) {x : Msg} (hx : ScopeAcc D lay k x) {w : Wit} {s : AuditStatement} {P : Partition}
    (hs : D.parse x = some s) (hP : D.partition s = some P) (hRel : RelS H D.commit δ s P w)
    (hdisj : Disjoint (faulty H D.commit s P w) (allSlots s.design P))
    (hv : valid s (valOver s.prog (traceVal s.prog (w.val H D.commit s))
      (heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s)))) :
    RLeakCarried H D.commit D.parse D.partition L K S lay k δ reg valid prow nrow Dj x w := by
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
  refine ⟨s, P, hs, hP, fun op o hcar hcov _ _ => ?_⟩
  have hinp : valOver s.prog (traceVal s.prog (w.val H D.commit s))
      (heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s)) =
      valOver s.prog ((lay s).inputOf (k s) ((List.range (k s)).map fun i => op i (o.take i)) a)
        (heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s)) :=
    funext (valOver_congr_input s.prog _ fun i hi => by
      rw [← inputOf_full hcar i]
      exact inputOf_restrict s.prog (lay s) (k s) _ _ hi)
  refine ⟨_, hF, a, ⟨fun g _ hg => if_pos hg, fun g hg => if_neg hg⟩,
    heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s),
    fun g b h => SampledAuditProof.faultyGate_of_heldAdv h, by rw [← hinp]; exact hv, ?_, by rw [← hinp], ?_⟩
  · refine List.map_congr_left fun g hg => ?_
    rw [← inputOf_full hcar g]
    exact (inputOf_restrict s.prog (lay s) (k s) _ _ (hopen s g hg)).symm
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

/-- **The leak relation from the trace's rows gives the leak relation**: a sequence accepted with a valid run from a row
is accepted. -/
theorem rLeak_of_rLeakCarried {H : Msg → Msg} {Cm : CommitmentScheme Bool} {parse : Msg → Option AuditStatement}
    {partition : AuditStatement → Option Partition} {L K S : ℕ} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {δ : ℝ} {reg : AuditStatement → Finset ℕ} {valid : AuditStatement → (ℕ → Bool) → Prop}
    {prow nrow : AuditStatement → List ℕ} {Dj : ℕ} {x : Msg} {w : Wit}
    (h : RLeakCarried H Cm parse partition L K S lay k δ reg valid prow nrow Dj x w) :
    RLeak H Cm parse partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj x w := by
  obtain ⟨s, P, hs, hP, h⟩ := h
  refine ⟨s, P, hs, hP, fun op o hC hCov hl hD => ?_⟩
  obtain ⟨F, hF, a, ha, free, hfree, -, -, -, hacc⟩ := h op o hC hCov hl hD
  exact ⟨hl, hD, F, hF, a, ha, free, hfree, hacc⟩

/-- **`KSLinOn` moves from `RAcc δ` to the leak relation from the trace's rows**, on the statements where the check holds
on the trace's run except with probability `εchk` (`CheckedWitness`), at `κ + εchk`: `ksLinOn_rLeakChecked`'s argument
at `rLeakCarried_of`. -/
theorem ksLinOn_rLeakCarried {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Claims.SamplingGeneric.Arg Coin} (H : Msg → Msg) (D : Claims.SamplingGeneric.Compiled Coin A) {L K S : ℕ}
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) {δ : ℝ} (reg : AuditStatement → Finset ℕ)
    (valid : AuditStatement → (ℕ → Bool) → Prop) (prow nrow : AuditStatement → List ℕ) (hopen : OpensAtInputs prow)
    (εchk : Msg → ℝ) (Dj : ℕ)
    (h : KSLinOn H D.V (ScopeAcc D lay k) (RAcc H D.commit D.parse D.partition L K S lay k δ) D.E (D.κ δ) D.γ
      (fun _ => 1) (Bext H D.V D.E)) :
    KSLinOn H D.V (fun x => ScopeAcc D lay k x ∧ CheckedWitness H D valid εchk x)
      (RLeakCarried H D.commit D.parse D.partition L K S lay k δ reg valid prow nrow Dj)
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
        · exact absurd (rLeakCarried_of H D lay k reg valid prow nrow hopen Dj hx.1 hs hP hRel hdisj hv) hng
        · exact Or.inr ⟨hacc, w, s, Pt, rfl, hs, hP, hdisj, hv⟩
      · exact Or.inl ⟨hacc, hR⟩
  · beta_reduce
    ring

/-- **The sequences accepted from a row number at most the class's count with the opening gates registered**: a sequence
accepted from `rin` is accepted at the inputs pinned at `reg ∪ prow`, `prow`'s free inputs at `rin`'s values, so
`encard_accSeqsChecked_le` there charges no free input of `prow`. -/
theorem encard_accSeqsCarried_le {L K S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : BoundaryGates L S) (n : ℕ)
    (hfits : lay.Fits C n) (chk : Finset ℕ) (valid : (ℕ → Bool) → Prop) (hchk : CheckedUnique C P d lay n chk valid)
    (reg : Finset ℕ) (ρ : ℕ → Bool) (prow nrow : List ℕ) (rin : Hash.Row) (δ : ℝ)
    (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ) :
    (AccSeqsCarried C P d lay n (AsReg C lay n reg ρ) valid prow nrow rin δ op D).encard ≤
      ((2 ^ (freeInputs C lay n \ (reg ∪ chk ∪ prow.toFinset)).card * (faultySets d P δ).card *
        2 ^ ((faultySets d P δ).sup (faultyBits C P)) * D ^ n : ℕ) : ℕ∞) := by
  classical
  let ρ' : ℕ → Bool := fun g => if g ∈ prow then rin.getD (prow.idxOf g) false else ρ g
  have hsub : AccSeqsCarried C P d lay n (AsReg C lay n reg ρ) valid prow nrow rin δ op D ⊆
      AccSeqsChecked C P d lay n (AsReg C lay n (reg ∪ prow.toFinset) ρ') valid δ op D := by
    rintro o ⟨hl, hD, F, hF, rout, a, ⟨hreg, hoff⟩, free, hfree, hv, hp, -, hacc⟩
    refine ⟨hl, hD, F, hF, a, ⟨fun g hg hgf => ?_, hoff⟩, free, hfree, hv, hacc⟩
    by_cases hgp : g ∈ prow
    · have hlt : prow.idxOf g < prow.length := List.idxOf_lt_length_iff.2 hgp
      simp only [ρ', if_pos hgp]
      rw [← hp, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
      simp [inputOf_free (fun i => op i (o.take i)) a hgf]
    · simp only [ρ', if_neg hgp]
      rcases Finset.mem_union.1 hg with hr | hr
      · exact hreg g hr hgf
      · exact absurd (List.mem_toFinset.1 hr) hgp
  calc _ ≤ (AccSeqsChecked C P d lay n (AsReg C lay n (reg ∪ prow.toFinset) ρ') valid δ op D).encard :=
        Set.encard_le_encard hsub
    _ ≤ _ := encard_accSeqsChecked_le C P d lay n hfits chk valid hchk (reg ∪ prow.toFinset) ρ' δ op D
    _ = _ := by rw [Finset.union_right_comm]

/-- **One carried run's leak bound**: `Guarantees.SampledLeakCarried`, from `ksLinOn_rLeakCarried` at `KSAccepted`'s
bound (`SecurityProofs.SampledAuditAccepted`) and `encard_accSeqsCarried_le` on the statements whose layout fits. -/
theorem sampledLeakCarried : Guarantees.SampledLeakCarried := by
  intro Coin _ _ _ A H D hks L K S lay k δ hδ reg chk valid prow nrow hopen εchk Dj
  refine ⟨ksLinOn_rLeakCarried H D lay k reg valid prow nrow hopen εchk Dj
      (SecurityProofs.SampledAuditAccepted H D L K S lay k hks δ hδ), fun x hx s P hs hP hchk op ρ rin => ?_⟩
  obtain ⟨s', hs', hfits⟩ := hx.2
  rw [Option.some_injective _ (hs'.symm.trans hs)] at hfits
  exact encard_accSeqsCarried_le s.prog P s.design (lay s) (k s) hfits (chk s) (valid s) hchk (reg s) ρ (prow s)
    (nrow s) rin δ op Dj

/-! ### The period -/

/-- **Prefix stability across runs**: run `i`'s operator reads only the runs before it, so two observation sequences
that agree on those runs give it one input, whatever run `i` and the runs after it saw. -/
theorem runsBefore_congr {m : ℕ} {α : Type} {o o' : Fin m → α} {i : Fin m}
    (h : ∀ j : Fin m, j.val < i.val → o j = o' j) : runsBefore o i = runsBefore o' i := by
  apply List.ext_getElem
  · simp [runsBefore]
  · intro j h1 h2
    have hj : j < i.val :=
      lt_of_lt_of_le h1 (by simp only [runsBefore, List.length_take]; exact min_le_left _ _)
    simp only [runsBefore, List.getElem_take, List.getElem_ofFn]
    exact h ⟨j, lt_trans hj i.isLt⟩ hj

/-- **The count over a carried pool, at one family of incorrect sets**: the period's observations meeting the carried
good event at the sets `F i` number at most
`∏_i 2^#(freeInputs_i \ (reg_i ∪ chk_i ∪ prow_i)) · 2^(faultyBits_i (F i)) · Dj^k_i`. A run's code is its inputs off its
checked, registered and opening gates, its values at its incorrect set's committed gates, and its jitter code. By
induction over the runs, two sequences with one code open one row at each run (the row the run before left, one row by
the induction, or `genesis`), so see one observation sequence there and leave one row (`carried_step`). -/
theorem encard_carried_le {L K S m : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (prow nrow : AuditStatement → List ℕ) (Dj : ℕ) (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i)))
    (hcls : ∀ i, CheckedClosing (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i))
      (valid (sw i)) (nrow (sw i)))
    (chains : Fin m → Bool) (genesis : Fin m → Hash.Row) (F : Fin m → Finset ℕ) :
    {o : Fin m → List (Obs L K S) | ∃ rin rout : Fin m → Hash.Row, PoolChained chains genesis rin rout ∧
        ∀ i, LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (rin i)
          (rout i) (F i) (o i)}.encard ≤
      ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
          (reg (sw i) ∪ chk (sw i) ∪ (prow (sw i)).toFinset)).card *
        (2 ^ faultyBits (sw i).prog (Pw i) (F i) * Dj ^ k (sw i))) : ℕ) : ℕ∞) := by
  classical
  set Sq := {o : Fin m → List (Obs L K S) | ∃ rin rout : Fin m → Hash.Row, PoolChained chains genesis rin rout ∧
    ∀ i, LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (rin i)
      (rout i) (F i) (o i)} with hSq
  have hex : ∀ o ∈ Sq, ∃ q : (Fin m → Hash.Row) × (Fin m → Hash.Row) × (Fin m → (ℕ → Bool) × (ℕ → Option Bool)),
      PoolChained chains genesis q.1 q.2.1 ∧ ∀ i, (o i).length = k (sw i) ∧ (∀ y ∈ o i, y.jitter < Dj) ∧
        (lay (sw i)).Fits (sw i).prog (k (sw i)) ∧ Disjoint (F i) (allSlots (sw i).design (Pw i)) ∧
        CarriedRun (sw i).prog (Pw i) (lay (sw i)) (k (sw i))
          (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) (valid (sw i)) (prow (sw i)) (nrow (sw i))
          (q.1 i) (q.2.1 i) (F i) (op i (runsBefore o i)) (o i) (q.2.2 i) := by
    rintro o ⟨rin, rout, hch, hl⟩
    choose p hp using fun i => acceptedCarried_iff.1 (hl i).2.2.2.2
    exact ⟨(rin, rout, p), hch, fun i => ⟨(hl i).2.2.1, (hl i).2.2.2.1, (hl i).1, (hl i).2.1, hp i⟩⟩
  let pick : (Fin m → List (Obs L K S)) →
      (Fin m → Hash.Row) × (Fin m → Hash.Row) × (Fin m → (ℕ → Bool) × (ℕ → Option Bool)) := fun o =>
    if h : o ∈ Sq then (hex o h).choose else (fun _ => [], fun _ => [], fun _ => (fun _ => false, fun _ => none))
  have hpick : ∀ o (ho : o ∈ Sq), pick o = (hex o ho).choose := fun o ho => by
    simp only [pick, ho, ↓reduceDIte]
  let code : (Fin m → List (Obs L K S)) → (i : Fin m) →
      (↥(freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i) ∪ (prow (sw i)).toFinset)) →
        Bool) × (↥(faultyGateSet (sw i).prog (Pw i) (F i)) → Bool) × (Fin (k (sw i)) → ℕ) := fun o i =>
    ((fun g => ((pick o).2.2 i).1 g),
      (fun g => valOver (sw i).prog ((lay (sw i)).inputOf (k (sw i))
        ((List.range (k (sw i))).map fun t => op i (runsBefore o i) t ((o i).take t)) ((pick o).2.2 i).1)
        ((pick o).2.2 i).2 g),
      jitterCode (o i))
  have hinj : Set.InjOn code Sq := by
    intro o ho o' ho' h
    obtain ⟨hch, hrun⟩ := (hex o ho).choose_spec
    obtain ⟨hch', hrun'⟩ := (hex o' ho').choose_spec
    rw [← hpick o ho] at hch hrun
    rw [← hpick o' ho'] at hch' hrun'
    have step : ∀ i : Fin m, (∀ i' : Fin m, i'.val < i.val → o i' = o' i' ∧ (pick o).2.1 i' = (pick o').2.1 i') →
        o i = o' i ∧ (pick o).2.1 i = (pick o').2.1 i := by
      intro i hprev
      have hrin : (pick o).1 i = (pick o').1 i := by
        rw [hch i, hch' i]
        unfold poolOpen
        split_ifs with hc <;> first
          | rfl
          | exact (hprev _ (by have := hc.1; show i.val - 1 < i.val; omega)).2
      have hrb : runsBefore o i = runsBefore o' i := runsBefore_congr fun j hj => (hprev j hj).1
      obtain ⟨hl, -, hfit, hdisj, ha, hfree, hval, hpin, hpout, hacc⟩ := hrun i
      obtain ⟨hl', -, -, -, ha', hfree', hval', hpin', hpout', hacc'⟩ := hrun' i
      have hc := congrFun h i
      simp only [code, Prod.mk.injEq] at hc
      obtain ⟨hca, hcv, hcj⟩ := hc
      rw [hrb] at hval hpin hpout hacc hcv
      have hoff : ∀ g ∉ chk (sw i), ((pick o).2.2 i).1 g = ((pick o').2.2 i).1 g := by
        intro g hg
        by_cases hf : g ∈ freeInputs (sw i).prog (lay (sw i)) (k (sw i))
        · by_cases hr : g ∈ reg (sw i)
          · rw [ha.1 g hr hf, ha'.1 g hr hf]
          · by_cases hpr : g ∈ prow (sw i)
            · have hx := List.map_inj_left.1 ((hpin.trans hrin).trans hpin'.symm) g hpr
              rwa [inputOf_free _ _ hf, inputOf_free _ _ hf] at hx
            · exact congrFun hca ⟨g, Finset.mem_sdiff.2 ⟨hf, by simp [hr, hg, hpr]⟩⟩
        · rw [ha.2 g hf, ha'.2 g hf]
      have hheld : ∀ g ∈ faultyGateSet (sw i).prog (Pw i) (F i),
          valOver (sw i).prog ((lay (sw i)).inputOf (k (sw i))
            ((List.range (k (sw i))).map fun t => op i (runsBefore o' i) t ((o i).take t)) ((pick o).2.2 i).1)
            ((pick o).2.2 i).2 g =
          valOver (sw i).prog ((lay (sw i)).inputOf (k (sw i))
            ((List.range (k (sw i))).map fun t => op i (runsBefore o' i) t ((o' i).take t)) ((pick o').2.2 i).1)
            ((pick o').2.2 i).2 g :=
        fun g hg => congrFun hcv ⟨g, hg⟩
      obtain ⟨hoi, hcl⟩ := carried_step (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) hfit (chk (sw i))
        (valid (sw i)) (hchk i) (nrow (sw i)) (hcls i) (F i) hdisj (op i (runsBefore o' i)) hl hl'
        (fun g b hgb => mem_faultyGateSet (hfree g b hgb)) (fun g b hgb => mem_faultyGateSet (hfree' g b hgb))
        hval hval' hacc hacc' hoff hheld hcj
      exact ⟨hoi, hpout.symm.trans (hcl.trans hpout')⟩
    have key : ∀ j, ∀ i : Fin m, i.val < j → o i = o' i ∧ (pick o).2.1 i = (pick o').2.1 i := by
      intro j
      induction j with
      | zero => intro i hi; omega
      | succ j ih =>
        intro i hi
        by_cases hlt : i.val < j
        · exact ih i hlt
        · exact step i fun i' h' => ih i' (by omega)
    funext i
    exact (key (i.val + 1) i (Nat.lt_succ_self _)).1
  let T : Finset ((i : Fin m) →
      (↥(freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i) ∪ (prow (sw i)).toFinset)) →
        Bool) × (↥(faultyGateSet (sw i).prog (Pw i) (F i)) → Bool) × (Fin (k (sw i)) → ℕ)) :=
    Fintype.piFinset fun i => univ ×ˢ univ ×ˢ Fintype.piFinset fun _ : Fin (k (sw i)) => range Dj
  have hsub : code '' Sq ⊆ (T : Set _) := by
    rintro _ ⟨o, ho, rfl⟩
    simp only [T, Finset.mem_coe, Fintype.mem_piFinset]
    intro i
    obtain ⟨-, hrun⟩ := (hex o ho).choose_spec
    obtain ⟨hl, hjit, -⟩ := hrun i
    simp only [Finset.mem_product, Finset.mem_univ, true_and]
    exact jitterCode_mem (o i) hl hjit
  calc Sq.encard = (code '' Sq).encard := hinj.encard_image.symm
    _ ≤ (T : Set _).encard := Set.encard_le_encard hsub
    _ = ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
            (reg (sw i) ∪ chk (sw i) ∪ (prow (sw i)).toFinset)).card *
          (2 ^ (faultyGateSet (sw i).prog (Pw i) (F i)).card * Dj ^ k (sw i))) : ℕ) : ℕ∞) := by
        rw [Set.encard_coe_eq_coe_finsetCard]
        congr 1
        simp only [T, Fintype.card_piFinset]
        refine Finset.prod_congr rfl fun i _ => ?_
        simp
    _ ≤ _ := by
        have h : ∀ i ∈ (univ : Finset (Fin m)),
            2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
              (reg (sw i) ∪ chk (sw i) ∪ (prow (sw i)).toFinset)).card *
              (2 ^ (faultyGateSet (sw i).prog (Pw i) (F i)).card * Dj ^ k (sw i)) ≤
            2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
              (reg (sw i) ∪ chk (sw i) ∪ (prow (sw i)).toFinset)).card *
              (2 ^ faultyBits (sw i).prog (Pw i) (F i) * Dj ^ k (sw i)) := fun i _ =>
          Nat.mul_le_mul_left _ (Nat.mul_le_mul_right _
            (Nat.pow_le_pow_right (by norm_num) (card_faultyGateSet_le _ _ _)))
        exact_mod_cast Finset.prod_le_prod h

/-- **The count over a carried pool at one pattern of restarts**: at a sample of the coins, the secrets meeting the
carried good event at the restarts `chains` and opened rows `genesis` number at most `PooledLeakCarried`'s count. It is
`encard_pooledJoint_le` at `encard_carried_le`, `b` the most committed bits of one of the period's proof blocks. -/
theorem encard_pooledCarried_le {L K S m : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (prow nrow : AuditStatement → List ℕ) (Dj : ℕ) (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool) (n : Fin m → ℕ)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i)))
    (hcls : ∀ i, CheckedClosing (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i))
      (nrow (sw i)))
    (chains : Fin m → Bool) (genesis : Fin m → Hash.Row) {Ω Conn Cell Srv Dev Z : Type} [Fintype Conn]
    (arr : Srv → Conn → Arrivals Cell) (f : RecordFormat Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (Zs : Set Z)
    (schedOf : Z → (Fin m → Ω) → CertifierSchedule Conn) (srvOf : Z → (Fin m → Ω) → Srv)
    (dec : (Fin m → Ω) → (Conn → List (ℕ × CellView Cell)) → Z)
    (hch : PeriodChannel f cap Ss Zs schedOf srvOf dec fun p => (run tap p).1)
    (W : Z → (Fin m → Ω) → Fin m → Finset ℕ) (hWn : ∀ z ω i, W z ω i ⊆ range (n i)) (Fmax : ℕ) (ω : Fin m → Ω) :
    {z | z ∈ Zs ∧ PooledGoodCarried lay k reg valid prow nrow Dj sw Pw ρ op chains genesis Fmax (W z ω)
        (run tap (schedOf z ω, srvOf z ω)).2}.encard ≤
      Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax *
        2 ^ ((univ.sup fun i => (range (n i)).sup fun v => faultyBits (sw i).prog (Pw i) {v}) * Fmax) *
        ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
            (reg (sw i) ∪ chk (sw i) ∪ (prow (sw i)).toFinset)).card * Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
  obtain ⟨-, hfit, hs, hdz⟩ := hch
  exact encard_pooledJoint_le
    (fun F o => ∃ rin rout : Fin m → Hash.Row, PoolChained chains genesis rin rout ∧
      ∀ i, LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (rin i)
        (rout i) (F i) (o i)) n
    (univ.sup fun i => (range (n i)).sup fun v => faultyBits (sw i).prog (Pw i) {v})
    (fun i => (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
      (reg (sw i) ∪ chk (sw i) ∪ (prow (sw i)).toFinset)).card)
    (fun i => Dj ^ k (sw i)) (fun i F => faultyBits (sw i).prog (Pw i) F)
    (fun i F hF => faultyBits_le_card_mul _ _ hF fun v hv =>
      (Finset.le_sup (f := fun v => faultyBits (sw i).prog (Pw i) {v}) (mem_range.2 hv)).trans
        (Finset.le_sup (f := fun i => (range (n i)).sup fun v => faultyBits (sw i).prog (Pw i) {v}) (mem_univ i)))
    (fun F _ => encard_carried_le lay k reg chk valid prow nrow Dj sw Pw ρ op hchk hcls chains genesis F)
    arr f tap run htrust cap Ss hfit Zs (fun z => schedOf z ω) (fun z => srvOf z ω) (fun z hz => hs z hz ω) (dec ω)
    (fun z hz => hdz z hz ω) (fun z => W z ω) (fun z i => hWn z ω i) Fmax

/-- **The leak through the network certifier over a monitoring period whose runs carry the pool**:
`Guarantees.PooledLeakCarried`. The escape bound is `pr_pooled_le` at each run's leak relation and no collision at its
opening, at the pattern serving picked. Where every audit accepts, the roots the record rule names there
(`CarriedRoots`) and those two give rows that chain, since equal roots are equal rows or a collision
(`Hash.BindingProofs.rowSeg_bind`). The secrets whose audits all accept and that meet the good event at the pattern
picked for them are in the union over `Cs` of those meeting it at each pattern (`Finset.set_encard_biUnion_le`), each
counted by `encard_pooledCarried_le`. -/
theorem pooledLeakCarried : Guarantees.PooledLeakCarried := by
  intro L K S G lay k reg chk valid prow nrow Dj m sw Pw ρ op n hchk hcls Ω _ _ Conn Cell Srv Dev Z _ arr f tap run
    htrust cap Ss Zs schedOf srvOf dec hch Cs chains genesis acc hcs hroot W had rin rout e e' he q hq0 hq1 hesc hleak
    Fmax
  refine ⟨fun z hz => ?_, fun ω => ?_⟩
  · obtain ⟨hacc, hW, -⟩ := had
    let obs : (Fin m → Ω) → Fin m → List (Obs L K S) := fun ω => (run tap (schedOf z ω, srvOf z ω)).2
    let Good : Fin m → (Fin m → Ω) → Prop := fun i ω =>
      LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs ω) i)) (rin z ω i)
          (rout z ω i) (W z ω i) (obs ω i) ∧
        ¬PoolColl G (chains z ω) (genesis z ω) (rin z ω) (rout z ω) i
    have hleak' : ∀ i, Pr (fun ω => acc z i ω (ω i) ∧ ¬Good i ω) ≤
        e' i + Pr (fun ω => PoolColl G (chains z ω) (genesis z ω) (rin z ω) (rout z ω) i) := by
      intro i
      refine Pr_or_le (fun ω h => ?_) (hleak z hz i) le_rfl
      obtain ⟨ha, hng⟩ := h
      by_cases h1 : LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs ω) i))
          (rin z ω i) (rout z ω i) (W z ω i) (obs ω i)
      · exact Or.inr (Classical.byContradiction fun h2 => hng ⟨h1, h2⟩)
      · exact Or.inl ⟨ha, h1⟩
    calc Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
          ¬PooledGoodCarried lay k reg valid prow nrow Dj sw Pw ρ op (chains z ω) (genesis z ω) Fmax (W z ω)
            (obs ω))
        ≤ Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧ ¬(∑ i, (W z ω i).card < Fmax ∧ ∀ i, Good i ω)) := by
          refine Pr_mono fun ω h => ⟨h.1, fun hg => h.2 ?_⟩
          obtain ⟨hlt, hgood⟩ := hg
          have hd := hroot z hz ω h.1 (obs ω) (W z ω) (rin z ω) (rout z ω) fun i => (hgood i).1
          exact ⟨hlt, rin z ω, rout z ω,
            fun i => (Hash.BindingProofs.rowSeg_bind G (hd i)).resolve_right (hgood i).2, fun i => (hgood i).1⟩
      _ ≤ (1 - q) ^ Fmax + ∑ i, e i +
            ∑ i, (e' i + Pr (fun ω => PoolColl G (chains z ω) (genesis z ω) (rin z ω) (rout z ω) i)) :=
          pr_pooled_le (acc z) (hacc z) (W z) (hW z) e
            (fun i => e' i + Pr (fun ω => PoolColl G (chains z ω) (genesis z ω) (rin z ω) (rout z ω) i)) he q hq0
            hq1 (hesc z hz) Good hleak' Fmax
      _ = _ := by rw [Finset.sum_add_distrib]; ring
  · let A : (Fin m → Bool) × (Fin m → Hash.Row) → Set Z := fun c =>
      {z | z ∈ Zs ∧ PooledGoodCarried lay k reg valid prow nrow Dj sw Pw ρ op c.1 c.2 Fmax (W z ω)
        (run tap (schedOf z ω, srvOf z ω)).2}
    refine le_trans (Set.encard_le_encard (t := ⋃ c ∈ Cs, A c) fun z hz =>
      Set.mem_iUnion₂.2 ⟨_, hcs z hz.1 ω hz.2.1, hz.1, hz.2.2⟩) ?_
    refine (Finset.set_encard_biUnion_le Cs A).trans ((Finset.sum_le_card_nsmul Cs _ _ fun c _ => ?_).trans
      (nsmul_eq_mul _ _).le)
    exact encard_pooledCarried_le lay k reg chk valid prow nrow Dj sw Pw ρ op n hchk hcls c.1 c.2 arr f tap run htrust
      cap Ss Zs schedOf srvOf dec hch W had.2.2 Fmax ω

/-! ### The empty pool -/

/-- **No collision at an opening of the empty pool**: the digests of the empty row hash one message, twice. -/
theorem poolColl_empty (G : Hash.Msg → Hash.Digest) {m : ℕ} (chains : Fin m → Bool) (i : Fin m) :
    ¬PoolColl G chains (fun _ => []) (fun _ => []) (fun _ => []) i := by
  rintro ⟨x, hx, y, hy, hne, -⟩
  have hseg : Hash.RowSeg.segmentMsgs [] = [] := by
    simp [Hash.RowSeg.segmentMsgs, Hash.RowSeg.segCount, Hash.RowSeg.SEG_BITS]
  have hopen : poolOpen chains (fun _ => ([] : Hash.Row)) (fun _ => []) i = [] := by
    unfold poolOpen
    split_ifs <;> rfl
  simp [Hash.RowSeg.hashed, hseg, hopen] at hx hy
  exact hne (hx.trans hy.symm)

/-- **A run accepted from a row is accepted with a valid run**, whatever its rows. -/
theorem leakAtChecked_of_leakAtCarried {L K S : ℕ} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {reg : AuditStatement → Finset ℕ} {valid : AuditStatement → (ℕ → Bool) → Prop}
    {prow nrow : AuditStatement → List ℕ} {Dj : ℕ} {s : AuditStatement} {P : Partition} {ρ : ℕ → Bool}
    {op : ℕ → List (Obs L K S) → List Bool} {rin rout : Hash.Row} {F : Finset ℕ} {o : List (Obs L K S)}
    (h : LeakAtCarried lay k reg valid prow nrow Dj s P ρ op rin rout F o) :
    LeakAtChecked lay k reg valid Dj s P ρ op F o := by
  obtain ⟨hfit, hdisj, hl, hD, a, ha, free, hfree, hv, -, -, hacc⟩ := h
  exact ⟨hfit, hdisj, hl, hD, a, ha, free, hfree, hv, hacc⟩

/-- **A run accepted with a valid run is accepted from the empty row**, with no opening or closing gate. -/
theorem leakAtCarried_of_leakAtChecked {L K S : ℕ} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {reg : AuditStatement → Finset ℕ} {valid : AuditStatement → (ℕ → Bool) → Prop} {Dj : ℕ}
    {s : AuditStatement} {P : Partition} {ρ : ℕ → Bool} {op : ℕ → List (Obs L K S) → List Bool} {F : Finset ℕ}
    {o : List (Obs L K S)} (h : LeakAtChecked lay k reg valid Dj s P ρ op F o) :
    LeakAtCarried lay k reg valid (fun _ => []) (fun _ => []) Dj s P ρ op [] [] F o := by
  obtain ⟨hfit, hdisj, hl, hD, a, ha, free, hfree, hv, hacc⟩ := h
  exact ⟨hfit, hdisj, hl, hD, a, ha, free, hfree, hv, rfl, rfl, hacc⟩

/-- **A period's good event over a pool gives its good event with valid runs**, whatever its rows and restarts, at an
operator that doesn't read the runs before it. -/
theorem pooledGoodChecked_of_carried {L K S m : ℕ} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {reg : AuditStatement → Finset ℕ} {valid : AuditStatement → (ℕ → Bool) → Prop}
    {prow nrow : AuditStatement → List ℕ} {Dj : ℕ} {sw : Fin m → AuditStatement} {Pw : Fin m → Partition}
    {ρ : Fin m → ℕ → Bool} {op : Fin m → ℕ → List (Obs L K S) → List Bool} {chains : Fin m → Bool}
    {genesis : Fin m → Hash.Row} {Fmax : ℕ} {W : Fin m → Finset ℕ} {o : Fin m → List (Obs L K S)}
    (h : PooledGoodCarried lay k reg valid prow nrow Dj sw Pw ρ (fun i _ => op i) chains genesis Fmax W o) :
    PooledGoodChecked lay k reg valid Dj sw Pw ρ op Fmax W o := by
  obtain ⟨hlt, _, _, -, hl⟩ := h
  exact ⟨hlt, fun i => leakAtChecked_of_leakAtCarried (hl i)⟩

/-- **A period's good event with valid runs is its good event over the empty pool**, at the pattern that restarts no
run. -/
theorem pooledGoodCarried_of_checked {L K S m : ℕ} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {reg : AuditStatement → Finset ℕ} {valid : AuditStatement → (ℕ → Bool) → Prop} {Dj : ℕ}
    {sw : Fin m → AuditStatement} {Pw : Fin m → Partition} {ρ : Fin m → ℕ → Bool}
    {op : Fin m → ℕ → List (Obs L K S) → List Bool} {Fmax : ℕ} {W : Fin m → Finset ℕ} {o : Fin m → List (Obs L K S)}
    (h : PooledGoodChecked lay k reg valid Dj sw Pw ρ op Fmax W o) :
    PooledGoodCarried lay k reg valid (fun _ => []) (fun _ => []) Dj sw Pw ρ (fun i _ => op i) (fun _ => false)
      (fun _ => []) Fmax W o :=
  ⟨h.1, fun _ => [], fun _ => [], fun _ => by simp [poolOpen], fun i => leakAtCarried_of_leakAtChecked (h.2 i)⟩

/-- **The leak through the network certifier over a monitoring period, with the checked inputs left out, over windows
that each hold a pool of their own**: `pooledLeakCarried` at the empty pool, where no run opens or closes a row and the
one pattern restarts none. There a period's good event with valid runs is its good event over the pool, every run opens
the empty row's root, and no opening collides (`poolColl_empty`), so the escape bound is `PooledLeakAccepted`'s with
`(1 - q)^Fmax`; the count over every secret meeting the good event is `encard_pooledCarried_le` at the one pattern. -/
theorem pooledLeakChecked_of_carried {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) {m : ℕ}
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → ℕ → List (Obs L K S) → List Bool) (n : Fin m → ℕ)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i)))
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (Zs : Set Z)
    (schedOf : Z → (Fin m → Ω) → CertifierSchedule Conn) (srvOf : Z → (Fin m → Ω) → Srv)
    (dec : (Fin m → Ω) → (Conn → List (ℕ × CellView Cell)) → Z)
    (hch : PeriodChannel f cap Ss Zs schedOf srvOf dec fun p => (run tap p).1)
    (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop) (W : Z → (Fin m → Ω) → Fin m → Finset ℕ)
    (had : PeriodAdaptive n acc W) (e e' : Fin m → ℝ) (he : ∀ i, 0 ≤ e i) (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hesc : PeriodEscapes Zs acc W q e)
    (hleak : PeriodLeaksChecked lay k reg valid Dj sw Pw ρ op Zs acc W
      (fun z ω => (run tap (schedOf z ω, srvOf z ω)).2) e')
    (Fmax : ℕ) :
    (∀ z ∈ Zs, Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
        ¬PooledGoodChecked lay k reg valid Dj sw Pw ρ op Fmax (W z ω) (run tap (schedOf z ω, srvOf z ω)).2) ≤
        (1 - q) ^ Fmax + ∑ i, e i + ∑ i, e' i) ∧
      ∀ ω, {z | z ∈ Zs ∧
          PooledGoodChecked lay k reg valid Dj sw Pw ρ op Fmax (W z ω) (run tap (schedOf z ω, srvOf z ω)).2}.encard ≤
        Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax *
          2 ^ ((univ.sup fun i => (range (n i)).sup fun v => faultyBits (sw i).prog (Pw i) {v}) * Fmax) *
          ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i))).card *
            Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
  have hcls : ∀ i, CheckedClosing (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i))
      (valid (sw i)) [] := fun _ => by unfold CheckedClosing; intros; rfl
  refine ⟨fun z hz => ?_, fun ω => ?_⟩
  · have hroot : ∀ z ∈ Zs, ∀ ω, (∀ i, acc z i ω (ω i)) →
        CarriedRoots (fun _ _ => 0) lay k reg valid (fun _ => []) (fun _ => []) Dj sw Pw ρ (fun i _ => op i)
          (fun _ => false) (fun _ => []) := by
      intro _ _ _ _ _ _ _ _ h i
      obtain ⟨-, -, -, -, _, -, _, -, -, hp, -, -⟩ := h i
      simp only [List.map_nil] at hp
      simp [poolOpen, ← hp]
    have hleak' : PeriodLeaksCarried lay k reg valid (fun _ => []) (fun _ => []) Dj sw Pw ρ (fun i _ => op i) Zs acc W
        (fun _ _ _ => []) (fun _ _ _ => []) (fun z ω => (run tap (schedOf z ω, srvOf z ω)).2) e' := by
      intro z hz i
      refine le_trans (Pr_mono fun ω hω => ?_) (hleak z hz i)
      exact ⟨hω.1, fun h' => hω.2 (leakAtCarried_of_leakAtChecked h')⟩
    have h := (pooledLeakCarried (fun _ _ => 0) lay k reg chk valid (fun _ => []) (fun _ => []) Dj sw Pw ρ
      (fun i _ => op i) n hchk hcls arr f tap run htrust cap Ss Zs schedOf srvOf dec hch
      {(fun _ => false, fun _ => [])} (fun _ _ _ => false) (fun _ _ _ => []) acc
      (fun _ _ _ _ => Finset.mem_singleton_self _) hroot W had (fun _ _ _ => []) (fun _ _ _ => []) e e' he q hq0 hq1
      hesc hleak' Fmax).1 z hz
    simp only [poolColl_empty, Pr_false, Finset.sum_const_zero, add_zero] at h
    refine le_trans (Pr_mono fun ω hω => ?_) h
    exact ⟨hω.1, fun hg => hω.2 (pooledGoodChecked_of_carried hg)⟩
  · have h := encard_pooledCarried_le lay k reg chk valid (fun _ => []) (fun _ => []) Dj sw Pw ρ (fun i _ => op i) n
      hchk hcls (fun _ => false) (fun _ => []) arr f tap run htrust cap Ss Zs schedOf srvOf dec hch W had.2.2 Fmax ω
    simp only [List.toFinset_nil, Finset.union_empty] at h
    refine le_trans (Set.encard_le_encard fun z hz => ?_) h
    exact ⟨hz.1, pooledGoodCarried_of_checked hz.2⟩

end NetTiming
