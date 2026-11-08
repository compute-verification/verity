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
import Proofs.NetworkCertifier.SampledLeak

/-!
# The leak bound over a carried pool

**One run's step** (`carried_step`). Two runs accepted with a valid run at one incorrect set, whose inputs agree off the
checked ones and whose held values and jitter codes agree, see one observation sequence, by the class proof's induction
over steps, and leave one row at the closing gates (`CheckedClosing`).

**The count** (`encard_carried_le`). A period's observations map injectively to each run's inputs off its checked,
registered and opening gates, its held values and its jitter code. By induction over the runs, two sequences with one
code open one row at each run: the row the record rule names, the run before's closing row (one row, by the induction)
or `genesis`. The opening gates then read one row, so the runs' inputs agree off the checked gates and the step applies.

**The monitoring period** (`pooledLeakCarried`). The count is `encard_pooledJoint_le` at that joint count. The escape bound
is `pr_pooled_le` at each run's leak relation, its opening root, and no collision at its opening, which together give
rows that chain (`Hash.BindingProofs.rowSeg_bind`).
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

/-- **The leak through the network certifier over a monitoring period whose runs carry the pool**:
`Guarantees.PooledLeakCarried`. The count is `encard_pooledJoint_le` at `encard_carried_le`, `b` the most committed bits
of one of the period's proof blocks. The escape bound is `pr_pooled_le` at each run's leak relation, its opening root
and no collision at its opening: those give rows that chain, since equal roots are equal rows or a collision
(`Hash.BindingProofs.rowSeg_bind`). -/
theorem pooledLeakCarried : Guarantees.PooledLeakCarried := by
  intro L K S G lay k reg chk valid prow nrow Dj m sw Pw ρ op n hchk hcls chains genesis Ω _ _ Conn Cell Srv Dev Z _
    arr f tap run htrust cap Ss Zs schedOf srvOf dec hch acc W had rin rout e e' eo he q hq0 hq1 hesc hleak hroot Fmax
  obtain ⟨-, hfit, hs, hdz⟩ := hch
  obtain ⟨hacc, hW, hWn⟩ := had
  refine ⟨fun z hz => ?_, fun ω => ?_⟩
  · let obs : (Fin m → Ω) → Fin m → List (Obs L K S) := fun ω => (run tap (schedOf z ω, srvOf z ω)).2
    let Good : Fin m → (Fin m → Ω) → Prop := fun i ω =>
      LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs ω) i)) (rin z ω i)
          (rout z ω i) (W z ω i) (obs ω i) ∧
        Hash.RowSeg.digest G (rin z ω i) = Hash.RowSeg.digest G (poolOpen chains genesis (rout z ω) i) ∧
        ¬PoolColl G chains genesis (rin z ω) (rout z ω) i
    have hleak' : ∀ i, Pr (fun ω => acc z i ω (ω i) ∧ ¬Good i ω) ≤
        e' i + eo i + Pr (fun ω => PoolColl G chains genesis (rin z ω) (rout z ω) i) := by
      intro i
      have hr : Pr (fun ω => (acc z i ω (ω i) ∧
            Hash.RowSeg.digest G (rin z ω i) ≠ Hash.RowSeg.digest G (poolOpen chains genesis (rout z ω) i)) ∨
            PoolColl G chains genesis (rin z ω) (rout z ω) i) ≤
          eo i + Pr (fun ω => PoolColl G chains genesis (rin z ω) (rout z ω) i) :=
        Pr_or_le (fun ω h => h) (hroot z hz i) le_rfl
      rw [add_assoc]
      refine Pr_or_le (fun ω h => ?_) (hleak z hz i) hr
      obtain ⟨ha, hng⟩ := h
      by_cases h1 : LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs ω) i))
          (rin z ω i) (rout z ω i) (W z ω i) (obs ω i)
      · right
        by_cases h2 : Hash.RowSeg.digest G (rin z ω i) = Hash.RowSeg.digest G (poolOpen chains genesis (rout z ω) i)
        · exact Or.inr (Classical.byContradiction fun h3 => hng ⟨h1, h2, h3⟩)
        · exact Or.inl ⟨ha, h2⟩
      · exact Or.inl ⟨ha, h1⟩
    calc Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
          ¬PooledGoodCarried lay k reg valid prow nrow Dj sw Pw ρ op chains genesis Fmax (W z ω) (obs ω))
        ≤ Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧ ¬(∑ i, (W z ω i).card < Fmax ∧ ∀ i, Good i ω)) := by
          refine Pr_mono fun ω h => ⟨h.1, fun hg => h.2 ?_⟩
          obtain ⟨hlt, hgood⟩ := hg
          refine ⟨hlt, rin z ω, rout z ω, fun i => ?_, fun i => (hgood i).1⟩
          obtain ⟨-, hd, hnc⟩ := hgood i
          exact (Hash.BindingProofs.rowSeg_bind G hd).resolve_right hnc
      _ ≤ (1 - q) ^ Fmax + ∑ i, e i +
            ∑ i, (e' i + eo i + Pr (fun ω => PoolColl G chains genesis (rin z ω) (rout z ω) i)) :=
          pr_pooled_le (acc z) (hacc z) (W z) (hW z) e
            (fun i => e' i + eo i + Pr (fun ω => PoolColl G chains genesis (rin z ω) (rout z ω) i)) he q hq0 hq1
            (hesc z hz) Good hleak' Fmax
      _ = _ := by rw [Finset.sum_add_distrib, Finset.sum_add_distrib]; ring
  · exact encard_pooledJoint_le
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

end NetTiming
