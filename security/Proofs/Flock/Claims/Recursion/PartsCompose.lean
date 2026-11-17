module

public import Proofs.Flock.Claims.Recursion.BoundRoute
public import Proofs.Flock.Claims.Recursion.PartAccepts
public import Std.Tactic.BVDecide.Normalize.BitVec
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.DeployChecked

/-! V\*'s parts' checks together make `IVStar` accept (`ivstar_of_parts`); a coin-dependent check no part's lowering fact reads makes the premises unsatisfiable, not the conclusion unsound. -/

@[expose] public section

namespace Claims.Recursion.VStarB

open Claims

theorem allOC_of_forall (H : Msg → Msg) : ∀ ps : List (OC Bool), (∀ p ∈ ps, (p.eval H).1 = true) →
    ((allOC ps).eval H).1 = true
  | [], _ => rfl
  | p :: ps, h => by
    show (OC.eval H (bindOC p fun b => if b then allOC ps else .ret false)).1 = true
    rw [eval_bindOC, h p (by simp)]
    exact allOC_of_forall H ps fun q hq => h q (by simp [hq])

theorem forall_of_allOC (H : Msg → Msg) : ∀ ps : List (OC Bool), ((allOC ps).eval H).1 = true →
    ∀ p ∈ ps, (p.eval H).1 = true
  | [], _, _, hp => absurd hp (List.not_mem_nil)
  | p :: ps, h, q, hq => by
    have h' : (OC.eval H (bindOC p fun b => if b then allOC ps else .ret false)).1 = true := h
    rw [eval_bindOC] at h'
    cases hb : (p.eval H).1 with
    | false =>
      rw [hb] at h'
      exact absurd h' (by simp [OC.eval])
    | true =>
      rw [hb] at h'
      rcases List.mem_cons.1 hq with rfl | hq
      · exact hb
      · exact forall_of_allOC H ps h' q hq

theorem eval_bindOC_of {α β : Type} (H : Msg → Msg) (p : OC α) (f : α → OC β) {a : α} (hp : (p.eval H).1 = a) :
    (OC.eval H (bindOC p f)).1 = (OC.eval H (f a)).1 := by
  rw [eval_bindOC, hp]

section Checks

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

structure ElemSplit (H : Msg → Msg) (Bd : Bound Coin Col T) (Q : Type) where
  checks : Msg → List Q
  holds : Q → Msg → List Msg → List Coin → Prop
  sound : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ms.length = Bd.DE.V.k x → cs.length = ms.length →
    (∀ q ∈ checks x, holds q x ms cs) → ((Bd.agrees x ms cs).eval H).1 = true

def StepOk (H : Msg → Msg) (Bd : Bound Coin Col T) (x : Msg) (ms : List Msg) (cs : List Coin) (j : ℕ) : Prop :=
  ((Bd.DE.V.next x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
    if j + 1 < ms.length then .msg (Bd.ans (cs.getD j default)) else .acc

def stepSplit (H : Msg → Msg) (Bd : Bound Coin Col T) : ElemSplit H Bd ℕ where
  checks x := List.range (Bd.DE.V.k x)
  holds j x ms cs := StepOk H Bd x ms cs j
  sound x ms cs hk _ h := by
    unfold Bound.agrees
    refine allOC_of_forall H _ fun p hp => ?_
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hp
    rw [eval_bindOC_of H _ _ (h j (by rw [List.mem_range] at hj ⊢; omega))]
    simp [OC.eval]

def ElemSplit.refine {H : Msg → Msg} {Bd : Bound Coin Col T} {Q : Type} (S : ElemSplit H Bd Q) {R : Type}
    (checksR : Q → Msg → List R) (holdsR : Q → R → Msg → List Msg → List Coin → Prop)
    (soundR : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ms.length = Bd.DE.V.k x → cs.length = ms.length →
      ∀ q ∈ S.checks x, (∀ r ∈ checksR q x, holdsR q r x ms cs) → S.holds q x ms cs) :
    ElemSplit H Bd (Q × R) where
  checks x := (S.checks x).flatMap fun q => (checksR q x).map fun r => (q, r)
  holds p x ms cs := holdsR p.1 p.2 x ms cs
  sound x ms cs hk hc h := S.sound x ms cs hk hc fun q hq => soundR x ms cs hk hc q hq fun r hr =>
    h (q, r) (List.mem_flatMap.2 ⟨q, hq, List.mem_map.2 ⟨r, hr, rfl⟩⟩)

inductive IVCheck (Q : Type) where
  | shape
  | len (j : ℕ)
  | salt (j i : ℕ)
  | row (j i : ℕ)
  | elem (q : Q)

def commitAt (a : Msg) (i : ℕ) : Msg := (a.drop (128 * i)).take 128

def IVCheck.Holds {Q : Type} (H : Msg → Msg) (Bd : Bound Coin Col T) (S : ElemSplit H Bd Q) (x : Msg) (c : ℕ → Coin)
    (ms : List Msg) : IVCheck Q → Prop
  | .shape => 0 < Bd.DE.V.k x ∧ ∃ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) ∧
      os.length + 1 = Bd.DE.V.k x ∧ ∀ j < os.length, (os.getD j []).length = Bd.nAt x j
  | .len j => ∀ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) →
      (ms.getD j []).length = 128 * (os.getD j []).length
  | .salt j i => ∀ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) → ∀ o,
      (os.getD j [])[i]? = some o → H (Bd.saltPrefix ++ List.ofFn o.2) = (commitAt (ms.getD j []) i).drop 64
  | .row j i => ∀ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) → ∀ o,
      (os.getD j [])[i]? = some o → H (Bd.rowIn o.1) = xorMask ((commitAt (ms.getD j []) i).take 64) (Bd.mask o.2)
  | .elem q => ∀ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) → os.length + 1 = Bd.DE.V.k x →
      S.holds q x (Bd.opened os last) ((List.range (Bd.DE.V.k x)).map c)

def ivChecks {Q : Type} {H : Msg → Msg} (Bd : Bound Coin Col T) (S : ElemSplit H Bd Q) (x : Msg) : List (IVCheck Q) :=
  .shape :: ((List.range (Bd.DE.V.k x - 1)).flatMap fun j =>
    .len j :: (List.range (Bd.nAt x j)).flatMap fun i => [.salt j i, .row j i]) ++ (S.checks x).map .elem

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem mem_ivChecks {Q : Type} {H : Msg → Msg} (Bd : Bound Coin Col T) (S : ElemSplit H Bd Q) (x : Msg)
    {q : IVCheck Q} (hq : q ∈ ivChecks Bd S x) :
    q = .shape ∨ (∃ j < Bd.DE.V.k x - 1, q = .len j) ∨
      (∃ j < Bd.DE.V.k x - 1, ∃ i < Bd.nAt x j, q = .salt j i ∨ q = .row j i) ∨ ∃ q' ∈ S.checks x, q = .elem q' := by
  rcases List.mem_cons.1 hq with h | hq
  · exact Or.inl h
  rcases List.mem_append.1 hq with hq | hq
  · obtain ⟨j, hj, hq⟩ := List.mem_flatMap.1 hq
    have hj := List.mem_range.1 hj
    rcases List.mem_cons.1 hq with h | hq
    · exact Or.inr (Or.inl ⟨j, hj, h⟩)
    obtain ⟨i, hi, hq⟩ := List.mem_flatMap.1 hq
    have hi := List.mem_range.1 hi
    rcases List.mem_cons.1 hq with h | hq
    · exact Or.inr (Or.inr (Or.inl ⟨j, hj, i, hi, Or.inl h⟩))
    rcases List.mem_cons.1 hq with h | hq
    · exact Or.inr (Or.inr (Or.inl ⟨j, hj, i, hi, Or.inr h⟩))
    · exact absurd hq List.not_mem_nil
  · obtain ⟨q', hq', h⟩ := List.mem_map.1 hq
    exact Or.inr (Or.inr (Or.inr ⟨q', hq', h.symm⟩))

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem mem_ivChecks_elem {Q : Type} {H : Msg → Msg} (Bd : Bound Coin Col T) (S : ElemSplit H Bd Q) (x : Msg) {q : Q}
    (hq : IVCheck.elem q ∈ ivChecks Bd S x) : q ∈ S.checks x := by
  rcases mem_ivChecks Bd S x hq with h | ⟨_, _, h⟩ | ⟨_, _, _, _, h | h⟩ | ⟨q', hq', h⟩
  · cases h
  · cases h
  · cases h
  · cases h
  · cases h
    exact hq'

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem ivstar_next_acc (H : Msg → Msg) (Bd : Bound Coin Col T) (x : Msg) (as : List Msg) (cs : List Coin)
    (hlen : as.length = Bd.DE.V.k x) (os : List (List (Msg × (Fin Bd.saltLen → UInt8)))) (last : Msg)
    (hfin : Bd.final (as.getLastD []) = some (os, last)) (hos : os.length + 1 = as.length)
    (hcnt : ∀ j < os.length, (os.getD j []).length = Bd.nAt x j)
    (hopen : ∀ p ∈ as.dropLast.zip os, ((Bd.opensRound p.1 p.2).eval H).1 = true)
    (hagr : ((Bd.agrees x (Bd.opened os last) cs).eval H).1 = true) :
    ((Bd.IVStar.next x as cs).eval H).1 = .acc := by
  have h1 : ¬ as.length < Bd.DE.V.k x := by omega
  simp only [Bound.IVStar, h1, ↓reduceIte, hfin]
  rw [ite_eq_left ⟨hos, hcnt⟩, eval_bindOC_of H _ _ (allOC_of_forall H _ fun p hp => by
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hp
    exact hopen q hq)]
  rw [ite_eq_left rfl, eval_bindOC_of H _ _ hagr]
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem opensRound_of_checks (H : Msg → Msg) (Bd : Bound Coin Col T) (a : Msg)
    (os : List (Msg × (Fin Bd.saltLen → UInt8))) (hl : a.length = 128 * os.length)
    (hs : ∀ i o, os[i]? = some o → H (Bd.saltPrefix ++ List.ofFn o.2) = (commitAt a i).drop 64)
    (hr : ∀ i o, os[i]? = some o → H (Bd.rowIn o.1) = xorMask ((commitAt a i).take 64) (Bd.mask o.2)) :
    ((Bd.opensRound a os).eval H).1 = true := by
  unfold Bound.opensRound
  rw [ite_eq_left hl]
  refine allOC_of_forall H _ fun p hp => ?_
  obtain ⟨i, hi, rfl⟩ := List.mem_mapIdx.1 hp
  have ho : os[i]? = some os[i] := List.getElem?_eq_getElem hi
  simp only [Bound.opens, OC.eval]
  exact decide_eq_true ⟨hs i _ ho, hr i _ ho⟩

theorem ivstar_acc_of_checks {Q : Type} (H : Msg → Msg) (Bd : Bound Coin Col T) (S : ElemSplit H Bd Q) (x : Msg)
    (c : ℕ → Coin) (ms : List Msg) (h : ∀ q ∈ ivChecks Bd S x, q.Holds H Bd S x c ms) :
    (run H (replay ms) Bd.IVStar x () fun i => c i.val).out = .acc := by
  obtain ⟨hk, os, last, hfin, hos, hcnt⟩ := h .shape (List.mem_cons_self ..)
  have hmem : ∀ q, q ∈ ((List.range (Bd.DE.V.k x - 1)).flatMap fun j =>
      IVCheck.len j :: (List.range (Bd.nAt x j)).flatMap fun i => [IVCheck.salt j i, IVCheck.row j i]) →
      q.Holds H Bd S x c ms := fun q hq =>
    h q (List.mem_cons_of_mem _ (List.mem_append_left _ hq))
  rw [Bd.run_ivstar H x ms _ hk]
  show (Bd.decAt H x ms _).1 = .acc
  unfold Bound.decAt
  obtain ⟨n, hn⟩ : ∃ n, Bd.DE.V.k x = n + 1 := ⟨Bd.DE.V.k x - 1, by omega⟩
  have hcs : (List.range (Bd.DE.V.k x)).map (coinSeq (k := Bd.IVStar.k x) fun i => c i.val) =
      (List.range (Bd.DE.V.k x)).map c :=
    List.map_congr_left fun i hi => by
      rw [List.mem_range] at hi
      simp [coinSeq, show i < Bd.IVStar.k x from hi]
  rw [hcs]
  refine ivstar_next_acc H Bd x _ _ (by simp) os last ?_ (by simp [hos]) hcnt ?_ ?_
  · rw [hn, List.range_succ, List.map_append, List.map_singleton, List.getLastD_concat]
    rw [hn, Nat.add_sub_cancel] at hfin
    exact hfin
  · intro p hp
    obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hp
    simp only [List.length_zip, List.length_dropLast, List.length_map, List.length_range] at hj
    have hjo : j < os.length := by omega
    have hjk : j < Bd.DE.V.k x - 1 := by omega
    have e1 : ((List.range (Bd.DE.V.k x)).map fun i => ms.getD i []).dropLast[j]'(by simp; omega) = ms.getD j [] := by
      rw [List.getElem_dropLast]
      simp
    have e2 : os[j] = os.getD j [] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjo]
    rw [List.getElem_zip, e1, e2]
    have hjm : ∀ q, q ∈ IVCheck.len j :: (List.range (Bd.nAt x j)).flatMap (fun i => [IVCheck.salt j i,
        IVCheck.row j i]) → q.Holds H Bd S x c ms := fun q hq =>
      hmem q (List.mem_flatMap.2 ⟨j, List.mem_range.2 hjk, hq⟩)
    have hnj : (os.getD j []).length = Bd.nAt x j := hcnt j hjo
    refine opensRound_of_checks H Bd _ _ (hjm _ (List.mem_cons_self ..) os last hfin) ?_ ?_
    · intro i o hi
      have hil : i < Bd.nAt x j := hnj ▸ (List.getElem?_eq_some_iff.1 hi).1
      exact hjm (.salt j i) (List.mem_cons_of_mem _ (List.mem_flatMap.2 ⟨i, List.mem_range.2 hil, by simp⟩))
        os last hfin o hi
    · intro i o hi
      have hil : i < Bd.nAt x j := hnj ▸ (List.getElem?_eq_some_iff.1 hi).1
      exact hjm (.row j i) (List.mem_cons_of_mem _ (List.mem_flatMap.2 ⟨i, List.mem_range.2 hil, by simp⟩))
        os last hfin o hi
  · refine S.sound x _ _ (by simp [Bound.opened, hos]) (by simp [Bound.opened, hos]) fun q hq => ?_
    exact h (.elem q) (List.mem_cons_of_mem _ (List.mem_append_right _ (List.mem_map.2 ⟨q, hq, rfl⟩))) os last hfin
      hos

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem ivstar_next_acc_inv (H : Msg → Msg) (Bd : Bound Coin Col T) (x : Msg) (as : List Msg) (cs : List Coin)
    (hlen : as.length = Bd.DE.V.k x) (h : ((Bd.IVStar.next x as cs).eval H).1 = .acc) :
    ∃ os last, Bd.final (as.getLastD []) = some (os, last) ∧ os.length + 1 = as.length ∧
      (∀ j < os.length, (os.getD j []).length = Bd.nAt x j) ∧
      (∀ p ∈ as.dropLast.zip os, ((Bd.opensRound p.1 p.2).eval H).1 = true) ∧
      ((Bd.agrees x (Bd.opened os last) cs).eval H).1 = true := by
  have h1 : ¬ as.length < Bd.DE.V.k x := by omega
  simp only [Bound.IVStar, h1, ↓reduceIte] at h
  split at h
  · simp [OC.eval] at h
  · rename_i os last hfin
    split at h
    · rename_i hc
      rw [eval_bindOC] at h
      cases ho : (OC.eval H (allOC ((as.dropLast.zip os).map fun p => Bd.opensRound p.1 p.2))).1 with
      | false =>
        rw [ho] at h
        simp [OC.eval] at h
      | true =>
        rw [ho] at h
        simp only [↓reduceIte] at h
        rw [eval_bindOC] at h
        cases ha : (OC.eval H (Bd.agrees x (Bd.opened os last) cs)).1 with
        | false =>
          rw [ha] at h
          simp [OC.eval] at h
        | true =>
          refine ⟨os, last, hfin, hc.1, hc.2, fun p hp => ?_, ha⟩
          exact forall_of_allOC H _ ho _ (List.mem_map.2 ⟨p, hp, rfl⟩)
    · simp [OC.eval] at h

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem checks_of_opensRound (H : Msg → Msg) (Bd : Bound Coin Col T) (a : Msg)
    (os : List (Msg × (Fin Bd.saltLen → UInt8))) (h : ((Bd.opensRound a os).eval H).1 = true) :
    a.length = 128 * os.length ∧ ∀ i o, os[i]? = some o →
      H (Bd.saltPrefix ++ List.ofFn o.2) = (commitAt a i).drop 64 ∧
        H (Bd.rowIn o.1) = xorMask ((commitAt a i).take 64) (Bd.mask o.2) := by
  unfold Bound.opensRound at h
  split at h
  · rename_i hl
    refine ⟨hl, fun i o hi => ?_⟩
    obtain ⟨hil, rfl⟩ := List.getElem?_eq_some_iff.1 hi
    have hp := forall_of_allOC H _ h _ (List.mem_mapIdx.2 ⟨i, hil, rfl⟩)
    simp only [Bound.opens, OC.eval] at hp
    exact of_decide_eq_true hp
  · simp [OC.eval] at h

theorem ivstar_acc_iff_checks (H : Msg → Msg) (Bd : Bound Coin Col T) (x : Msg) (c : ℕ → Coin) (ms : List Msg) :
    (run H (replay ms) Bd.IVStar x () fun i => c i.val).out = .acc ↔
      ∀ q ∈ ivChecks Bd (stepSplit H Bd) x, q.Holds H Bd (stepSplit H Bd) x c ms := by
  refine ⟨fun h => ?_, ivstar_acc_of_checks H Bd _ x c ms⟩
  rcases Nat.eq_zero_or_pos (Bd.DE.V.k x) with hk | hk
  · rw [Bd.run_ivstar_zero H x ms _ hk] at h
    cases h
  rw [Bd.run_ivstar H x ms _ hk] at h
  change (Bd.decAt H x ms _).1 = .acc at h
  unfold Bound.decAt at h
  obtain ⟨n, hn⟩ : ∃ n, Bd.DE.V.k x = n + 1 := ⟨Bd.DE.V.k x - 1, by omega⟩
  have hcs : (List.range (Bd.DE.V.k x)).map (coinSeq (k := Bd.IVStar.k x) fun i => c i.val) =
      (List.range (Bd.DE.V.k x)).map c :=
    List.map_congr_left fun i hi => by
      rw [List.mem_range] at hi
      simp [coinSeq, show i < Bd.IVStar.k x from hi]
  rw [hcs] at h
  obtain ⟨os, last, hfin, hos, hcnt, hopen, hagr⟩ := ivstar_next_acc_inv H Bd x _ _ (by simp) h
  have hfin' : Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) := by
    rw [hn, List.range_succ, List.map_append, List.map_singleton, List.getLastD_concat] at hfin
    rw [hn, Nat.add_sub_cancel]
    exact hfin
  simp only [List.length_map, List.length_range] at hos
  have hround : ∀ j < os.length, ((Bd.opensRound (ms.getD j []) (os.getD j [])).eval H).1 = true := by
    intro j hj
    have hjl : j < (((List.range (Bd.DE.V.k x)).map fun i => ms.getD i []).dropLast.zip os).length := by
      simp only [List.length_zip, List.length_dropLast, List.length_map, List.length_range]
      omega
    have hp := hopen _ (List.getElem_mem hjl)
    rw [List.getElem_zip, List.getElem_dropLast] at hp
    simpa [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj] using hp
  intro q hq
  rcases mem_ivChecks Bd _ x hq with rfl | ⟨j, hj, rfl⟩ | ⟨j, hj, i, hi, rfl | rfl⟩ | ⟨j, hj, rfl⟩
  · exact ⟨hk, os, last, hfin', hos, hcnt⟩
  · intro os' last' hf'
    rw [hfin'] at hf'
    cases hf'
    exact (checks_of_opensRound H Bd _ _ (hround j (by omega))).1
  · intro os' last' hf' o ho
    rw [hfin'] at hf'
    cases hf'
    exact ((checks_of_opensRound H Bd _ _ (hround j (by omega))).2 i o ho).1
  · intro os' last' hf' o ho
    rw [hfin'] at hf'
    cases hf'
    exact ((checks_of_opensRound H Bd _ _ (hround j (by omega))).2 i o ho).2
  · intro os' last' hf' _
    rw [hfin'] at hf'
    cases hf'
    have hjk : j < (Bd.opened os last).length := by
      have := List.mem_range.1 hj
      simp only [Bound.opened, List.length_append, List.length_map, List.length_singleton]
      omega
    have hp := forall_of_allOC H _ hagr _ (List.mem_map.2 ⟨j, List.mem_range.2 hjk, rfl⟩)
    rw [eval_bindOC] at hp
    show StepOk H Bd x _ _ j
    unfold StepOk
    exact of_decide_eq_true hp

end Checks

section Compose

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

def OwnedBy {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T) (shape : Form → Prop)
    (S : ElemSplit H Bd Q) (own : Msg → IVCheck Q → Option ℕ) (PC : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) :
    Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ q ∈ ivChecks Bd S (xI D x), ∀ s,
    own x q = some s → s < D.nS x ∧ ∀ c ms, PC s x c ms → q.Holds H Bd S (xI D x) c ms

def StmtFixes {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (S : ElemSplit H Bd Q) (own : Msg → IVCheck Q → Option ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ q ∈ ivChecks Bd S (xI D x), own x q = none →
    ∀ (c : ℕ → Coin) (ws : List Flock.Wit), ws.length = D.nS x → (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
      q.Holds H Bd S (xI D x) c (L.msgs x ws)

/-- Its escape is never taken. -/
theorem ivstar_of_parts {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (S : ElemSplit H Bd Q) (own : Msg → IVCheck Q → Option ℕ)
    (PC : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (hown : OwnedBy H D Bd shape S own PC)
    (hfix : StmtFixes H D Bd L shape S own) : DecompB H D Bd.IVStar L shape PC := by
  intro x st hx hf c ws hl hT hC
  refine Or.inl (ivstar_acc_of_checks H Bd S (xI D x) c _ fun q hq => ?_)
  cases ho : own x q with
  | none => exact hfix x st hx hf q hq ho c ws hl hT
  | some s =>
    obtain ⟨hs, hq'⟩ := hown x st hx hf q hq s ho
    exact hq' c _ (hC s hs)

def PartComputes {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T) (S : ElemSplit H Bd Q)
    (own : Msg → IVCheck Q → Option ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin) (ms : List Msg) : Prop :=
  ∀ q ∈ ivChecks Bd S (xI D x), own x q = some s → q.Holds H Bd S (xI D x) c ms

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem ownedBy_partComputes {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T)
    (shape : Form → Prop) (S : ElemSplit H Bd Q) (own : Msg → IVCheck Q → Option ℕ)
    (hcov : ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ q ∈ ivChecks Bd S (xI D x), ∀ s,
      own x q = some s → s < D.nS x) :
    OwnedBy H D Bd shape S own (PartComputes H D Bd S own) :=
  fun x st hx hf q hq s ho => ⟨hcov x st hx hf q hq s ho, fun _ _ h => h q hq ho⟩

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem partComputes_of_ownedBy {Q : Type} {H : Msg → Msg} {D : Deploy Coin} {Bd : Bound Coin Col T}
    {shape : Form → Prop} {S : ElemSplit H Bd Q} {own : Msg → IVCheck Q → Option ℕ}
    {PC : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} (hown : OwnedBy H D Bd shape S own PC) {s : ℕ} {x : Msg}
    {st : CompiledStatement} (hx : D.parse x = some st) (hf : shape st.form) {c : ℕ → Coin} {ms : List Msg}
    (h : PC s x c ms) : PartComputes H D Bd S own s x c ms :=
  fun q hq ho => (hown x st hx hf q hq s ho).2 c ms h

theorem decompB_partAccepts_of_parts {Q Col' T' : Type} (H : Msg → Msg) (D : Deploy Coin)
    (DO : Flock.Deploy Coin Col' T') (Bd : Bound Coin Col T) (L : LayoutB) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) (S : ElemSplit H Bd Q)
    (own : Msg → IVCheck Q → Option ℕ) (PC : ℕ → Msg → (ℕ → Coin) → List Msg → Prop)
    (hlow : ∀ s x c ms, PartAccepts H D DO L shape shows cmt s x c ms → PC s x c ms)
    (hown : OwnedBy H D Bd shape S own PC) (hfix : StmtFixes H D Bd L shape S own) :
    DecompB H D Bd.IVStar L shape (PartAccepts H D DO L shape shows cmt) :=
  decompB_mono hlow (ivstar_of_parts H D Bd L shape S own PC hown hfix)

def firstHolder (D : Deploy Coin) (L : LayoutB) (x : Msg) (j : ℕ) : Option ℕ :=
  (List.range (D.nS x)).find? fun s => (L.msgAt x s j).isSome

def ownHolders {Q : Type} (D : Deploy Coin) (L : LayoutB) (ownE : Msg → Q → Option ℕ) (x : Msg) :
    IVCheck Q → Option ℕ
  | .shape => none
  | .len _ => none
  | .salt j _ => firstHolder D L x j
  | .row j _ => firstHolder D L x j
  | .elem q => ownE x q

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem firstHolder_lt {D : Deploy Coin} {L : LayoutB} {x : Msg} {j s : ℕ} (h : firstHolder D L x j = some s) :
    s < D.nS x :=
  List.mem_range.1 (List.mem_of_find?_eq_some h)

end Compose

end Claims.Recursion.VStarB

namespace Claims.Recursion.VStarB

open Claims

section Inventory

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem ownedBy_holders {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (S : ElemSplit H Bd Q) (ownE : Msg → Q → Option ℕ)
    (hE : ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ q ∈ S.checks (xI D x), ∀ s,
      ownE x q = some s → s < D.nS x) :
    OwnedBy H D Bd shape S (ownHolders D L ownE) (PartComputes H D Bd S (ownHolders D L ownE)) := by
  refine ownedBy_partComputes H D Bd shape S _ fun x st hx hf q hq s ho => ?_
  cases q with
  | shape => cases ho
  | len _ => cases ho
  | salt j _ => exact firstHolder_lt ho
  | row j _ => exact firstHolder_lt ho
  | elem q =>
    exact hE x st hx hf q (mem_ivChecks_elem Bd S _ hq) s ho

end Inventory

end Claims.Recursion.VStarB

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB

structure LastSplit (H : Msg → Msg) (nextE : Msg → List Msg → List Coin → Claims.OC Out) (R : Type) where
  checks : Msg → List R
  holds : R → Msg → List Msg → List Coin → Prop
  sound : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ms.length = (FlockDeploy.V false).k x →
    cs.length = ms.length → (∀ r ∈ checks x, holds r x ms cs) → ((nextE x ms cs).eval H).1 = .acc

inductive DeployCheck (R : Type) where
  | setup
  | tag (j : ℕ)
  | last (r : R)

def DeployCheck.Holds {H : Msg → Msg} {nextE : Msg → List Msg → List Coin → Claims.OC Out} {R : Type}
    (LS : LastSplit H nextE R) (x : Msg) (ms : List Msg) (cs : List Coin) : DeployCheck R → Prop
  | .setup => (FlockDeploy.setupOf false x).isSome
  | .tag j => ∃ a, ms[j]? = some (0 :: a)
  | .last r => LS.holds r x ms cs

theorem getLast?_take_succ {α : Type} (l : List α) {j : ℕ} (hj : j < l.length) :
    (l.take (j + 1)).getLast? = l[j]? := by
  rw [List.getLast?_eq_getElem?, List.length_take, Nat.min_eq_left (by omega : j + 1 ≤ l.length), Nat.add_sub_cancel,
    List.getElem?_take_of_lt (by omega)]

theorem bdDeployed_next (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed rel nextE colsE wE join rowIn final w).DE.V.next = nextE := rfl

noncomputable def deploySplit (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hcoin : ∀ x as cs c, FlockDeploy.act false x as cs = .coin c → ((nextE x as cs).eval H).1 = .msg c.bytes)
    {R : Type} (LS : LastSplit H nextE R) :
    ElemSplit H (bdDeployed rel nextE colsE wE join rowIn final w) (DeployCheck R) where
  checks x := .setup :: ((List.range ((FlockDeploy.V false).k x - 1)).map .tag ++ (LS.checks x).map .last)
  holds q x ms cs := q.Holds LS x ms cs
  sound x ms cs hk hc h := by
    refine (stepSplit H (bdDeployed rel nextE colsE wE join rowIn final w)).sound x ms cs hk hc fun j hj => ?_
    have hk' : ms.length = (FlockDeploy.V false).k x := hk
    have hj' : j < ms.length := by
      rw [hk']
      exact List.mem_range.1 hj
    show StepOk H (bdDeployed rel nextE colsE wE join rowIn final w) x ms cs j
    unfold StepOk
    rw [bdDeployed_next, bdDeployed_ans]
    by_cases hl : j + 1 < ms.length
    · rw [ite_eq_left hl]
      obtain ⟨⟨s, spec, sts⟩, hsp⟩ := Option.isSome_iff_exists.1 (h .setup (List.mem_cons_self ..))
      have hjk : j < (FlockDeploy.V false).k x - 1 := by omega
      obtain ⟨a, ha⟩ := h (.tag j) (List.mem_cons_of_mem _ (List.mem_append_left _
        (List.mem_map.2 ⟨j, List.mem_range.2 hjk, rfl⟩)))
      have hjc : j < cs.length := by omega
      obtain ⟨cj, hcj⟩ : ∃ cj, cs[j]? = some cj := ⟨_, List.getElem?_eq_getElem hjc⟩
      rw [List.getD_eq_getElem?_getD, hcj, Option.getD_some]
      have hm1 : (ms.take (j + 1)).getLast? = some (0 :: a) := (getLast?_take_succ ms hj').trans ha
      have hc1 : (cs.take (j + 1)).getLast? = some cj := (getLast?_take_succ cs hjc).trans hcj
      refine hcoin x _ _ _ ?_
      simp only [FlockDeploy.act, hsp, hm1, hc1]
    · rw [ite_eq_right hl]
      have e1 : ms.take (j + 1) = ms := List.take_of_length_le (by omega)
      have e2 : cs.take (j + 1) = cs := List.take_of_length_le (by omega)
      rw [e1, e2]
      exact LS.sound x ms cs hk' hc fun r hr =>
        h (.last r) (List.mem_cons_of_mem _ (List.mem_append_right _ (List.mem_map.2 ⟨r, hr, rfl⟩)))

theorem tag_of_join {join : List Msg → Msg} (hj : ∀ rs, ∃ a, join rs = 0 :: a) {saltLen : ℕ}
    (os : List (List (Msg × (Fin saltLen → UInt8)))) (last : Msg) {j : ℕ} (hjo : j < os.length) :
    ∃ a, (os.map (fun o => join (o.map Prod.fst)) ++ [last])[j]? = some (0 :: a) := by
  obtain ⟨a, ha⟩ := hj ((os[j]).map Prod.fst)
  refine ⟨a, ?_⟩
  rw [List.getElem?_append_left (by simpa using hjo), List.getElem?_map, List.getElem?_eq_getElem hjo]
  simp [ha]

theorem next_coin_fst (H : Msg → Msg) (x : Msg) (as : List Msg) (cs : List Coin) (c : Coin)
    (h : FlockDeploy.act false x as cs = .coin c) : ((FlockDeploy.next false x as cs).eval H).1 = .msg c.bytes := by
  rw [FlockDeploy.next_coin H h]

def ownDeploy {R : Type} (ownR : Msg → R → Option ℕ) (x : Msg) : DeployCheck R → Option ℕ
  | .setup => none
  | .tag _ => none
  | .last r => ownR x r

theorem ownedBy_deploy (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hcoin : ∀ x as cs c, FlockDeploy.act false x as cs = .coin c → ((nextE x as cs).eval H).1 = .msg c.bytes)
    {R : Type} (LS : LastSplit H nextE R) (ownR : Msg → R → Option ℕ) (D : Claims.Recursion.Deploy Coin)
    (L : LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ r ∈ LS.checks (Claims.Recursion.xI D x), ∀ s, ownR x r = some s → s < D.nS x) :
    let S := deploySplit H rel nextE colsE wE join rowIn final w hcoin LS
    OwnedBy H D (bdDeployed rel nextE colsE wE join rowIn final w) shape S (ownHolders D L (ownDeploy ownR))
      (PartComputes H D (bdDeployed rel nextE colsE wE join rowIn final w) S (ownHolders D L (ownDeploy ownR))) := by
  intro S
  refine ownedBy_holders H D _ L shape S _ fun x st hx hf q hq s ho => ?_
  cases q with
  | setup => cases ho
  | tag _ => cases ho
  | last r =>
    have hr : r ∈ LS.checks (Claims.Recursion.xI D x) := by
      rcases List.mem_cons.1 hq with h | hq
      · cases h
      rcases List.mem_append.1 hq with hq | hq
      · obtain ⟨_, _, h⟩ := List.mem_map.1 hq
        cases h
      · obtain ⟨r', hr', h⟩ := List.mem_map.1 hq
        cases h
        exact hr'
    exact hR x st hx hf r hr s ho

theorem setup_fixed (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hcoin : ∀ x as cs c, FlockDeploy.act false x as cs = .coin c → ((nextE x as cs).eval H).1 = .msg c.bytes)
    {R : Type} (LS : LastSplit H nextE R) (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : LayoutB) (IV : Claims.Verifier Coin) {x : Msg} {st : Claims.Recursion.CompiledStatement}
    (hx : (recBound rel' b L IV).parse x = some st) (c : ℕ → Coin) (ms : List Msg) :
    (IVCheck.elem DeployCheck.setup : IVCheck (DeployCheck R)).Holds H (bdDeployed rel nextE colsE wE join rowIn final w)
      (deploySplit H rel nextE colsE wE join rowIn final w hcoin LS) (Claims.Recursion.xI (recBound rel' b L IV) x)
      c ms := by
  intro _ _ _ _
  obtain ⟨s, spec, sts, h, -⟩ := parse_innerOk hx
  show (FlockDeploy.setupOf false (xIOf x)).isSome = true
  rw [h]
  rfl

theorem tag_fixed (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) {join : List Msg → Msg}
    (hj : ∀ rs, ∃ a, join rs = 0 :: a)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hcoin : ∀ x as cs c, FlockDeploy.act false x as cs = .coin c → ((nextE x as cs).eval H).1 = .msg c.bytes)
    {R : Type} (LS : LastSplit H nextE R) (x : Msg) (c : ℕ → Coin) (ms : List Msg) {j : ℕ}
    (hjk : j < (FlockDeploy.V false).k x - 1) :
    (IVCheck.elem (DeployCheck.tag j) : IVCheck (DeployCheck R)).Holds H
      (bdDeployed rel nextE colsE wE join rowIn final w) (deploySplit H rel nextE colsE wE join rowIn final w hcoin LS)
      x c ms := by
  intro os last _ hos
  have hos' : os.length + 1 = (FlockDeploy.V false).k x := hos
  exact tag_of_join hj os last (by omega)

theorem stmtFixes_deploy (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) {join : List Msg → Msg}
    (hj : ∀ rs, ∃ a, join rs = 0 :: a)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hcoin : ∀ x as cs c, FlockDeploy.act false x as cs = .coin c → ((nextE x as cs).eval H).1 = .msg c.bytes)
    {R : Type} (LS : LastSplit H nextE R) (ownR : Msg → R → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel nextE colsE wE join rowIn final w)
          (deploySplit H rel nextE colsE wE join rowIn final w hcoin LS) (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy ownR) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel nextE colsE wE join rowIn final w)
            (deploySplit H rel nextE colsE wE join rowIn final w hcoin LS) (Claims.Recursion.xI (recBound rel' b L IV) x)
            c (L.msgs x ws)) :
    StmtFixes H (recBound rel' b L IV) (bdDeployed rel nextE colsE wE join rowIn final w) L shape
      (deploySplit H rel nextE colsE wE join rowIn final w hcoin LS)
      (ownHolders (recBound rel' b L IV) L (ownDeploy ownR)) := by
  intro x st hx hf q hq ho c ws hl hT
  cases q with
  | elem r =>
    cases r with
    | setup => exact setup_fixed H rel nextE colsE wE join rowIn final w hcoin LS rel' b L IV hx c _
    | tag j =>
      have hjm := mem_ivChecks_elem _ _ _ hq
      have hjk : j < (FlockDeploy.V false).k (Claims.Recursion.xI (recBound rel' b L IV) x) - 1 := by
        rcases List.mem_cons.1 hjm with h | h
        · cases h
        rcases List.mem_append.1 h with h | h
        · obtain ⟨j', hj', h⟩ := List.mem_map.1 h
          cases h
          exact List.mem_range.1 hj'
        · obtain ⟨_, _, h⟩ := List.mem_map.1 h
          cases h
      exact tag_fixed H rel nextE colsE wE hj rowIn final w hcoin LS _ c _ hjk
    | last r' => exact hrest x st hx hf _ hq ho (fun _ h => ⟨r', (IVCheck.elem.inj h).symm⟩) c ws hl hT
  | shape => exact hrest x st hx hf _ hq ho (fun _ h => by cases h) c ws hl hT
  | len _ => exact hrest x st hx hf _ hq ho (fun _ h => by cases h) c ws hl hT
  | salt _ _ => exact hrest x st hx hf _ hq ho (fun _ h => by cases h) c ws hl hT
  | row _ _ => exact hrest x st hx hf _ hq ho (fun _ h => by cases h) c ws hl hT

/-- This bounds no number by itself: its premises are open. -/
theorem ksLin2_deployed_composed (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    ∀ {Q : Type} (S : ElemSplit FlockDeploy.sha512 Bd Q) (own : Msg → IVCheck Q → Option ℕ)
      (PC : ℕ → Msg → (ℕ → Coin) → List Msg → Prop),
    (∀ s x c ms, PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
      Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s x c ms → PC s x c ms) →
    OwnedBy FlockDeploy.sha512 D Bd Claims.Recursion.Form.Dense S own PC →
    StmtFixes FlockDeploy.sha512 D Bd L Claims.Recursion.Form.Dense S own →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D Q S own PC hlow hown hfix
  refine ksLin2_deployed_partAccepts b aux nOf hd recRest rest wE join final w (fun shape hsh => ?_)
  obtain rfl := List.mem_singleton.1 hsh
  exact decompB_partAccepts_of_parts _ _ _ _ _ _ _ _ S own PC hlow hown hfix

/-- It reads `StrictBirthday`, `ExpectedCR` and `ksLin2_deployed_composed`'s open premises: a bound on a number
under named premises, not a proved security level. -/
theorem bitsLin2_deployed_composed (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    ∀ {Q : Type} (S : ElemSplit FlockDeploy.sha512 Bd Q) (own : Msg → IVCheck Q → Option ℕ)
      (PC : ℕ → Msg → (ℕ → Coin) → List Msg → Prop),
    (∀ s x c ms, PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
      Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s x c ms → PC s x c ms) →
    OwnedBy FlockDeploy.sha512 D Bd Claims.Recursion.Form.Dense S own PC →
    StmtFixes FlockDeploy.sha512 D Bd L Claims.Recursion.Form.Dense S own →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D Q S own PC hlow hown hfix
  refine bitsLin2_deployed_partAccepts b aux nOf hd recRest rest wE join final w (fun shape hsh => ?_)
  obtain rfl := List.mem_singleton.1 hsh
  exact decompB_partAccepts_of_parts _ _ _ _ _ _ _ _ S own PC hlow hown hfix

end RecDeploy
