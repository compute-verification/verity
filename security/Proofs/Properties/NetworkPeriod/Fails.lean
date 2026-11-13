module

public import Definitions.NetworkCertifier.Guarantees.MonitoringPeriodFails
public import Proofs.Properties.NetworkPeriod
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Nat.Choose.Basic

/-!
# The period check's `K` at the terms of the failure forms

`NetworkCheck.Period`'s terms charge every connection's cells (`capsTerm`), in `cells` and, in the rows form, again in
`schedules`. `MonitoringPeriodLeakFailsAtMost`'s count has neither factor: it has the registered parameters' count and the fail
patterns of the period's failed connections. These lemmas say what the check's `K` bounds at the terms that count:

* `clockKBoundsHeld`: the clock form's `terms` at no caps bounds the count with no cells factor;
* `rowsTermsFails`: the rows form's `rowsTerms` with `schedules` the registered parameters' count and `cells` the
  failures' term (`failsTerm`: `⌈log2⌉` of `failsAtMost`, the patterns of at most `fails` of the connections at the
  widest cap, which is 0 with no failure);
* `failsAtMostCovers`: `failsAtMost` covers `failPatterns`, summed up to `fails`;
* `rowsKBoundsFails`, `rowsKBoundsFailPatterns`: `K` at `rowsTermsFails` bounds the count, at the check's number for
  the patterns and at `failPatterns`.

The check does not compute `rowsTermsFails`: its `rowsTerms` is unchanged, and these are lemmas about `K` at another
record.
-/

@[expose] public section

namespace NetTiming.PeriodTerms

open NetworkCheck.Period Properties.NetworkPeriod

/-- The widest cap. -/
def capMax (cells : Array ℕ) : ℕ := cells.foldl max 0

/-- One step of `failsAtMost`'s loop at `j`: the state is `C(n, j)`, `cap^j` and the sum so far. -/
def failsStep (n cap : ℕ) (s : ℕ × ℕ × ℕ) (j : ℕ) : ℕ × ℕ × ℕ :=
  (s.1 * (n - j) / (j + 1), s.2.1 * cap, s.2.2 + s.1 * (n - j) / (j + 1) * (s.2.1 * cap))

/-- `Σ_{j ≤ b} C(n, j) · cap^j`: the fail patterns of at most `b` of `n` connections, each failing at one of `cap`
cells (`atMost`'s loop with each term weighted, as a fold). -/
def failsAtMost (n cap b : ℕ) : ℕ := ((List.range (min b n)).foldl (failsStep n cap) (1, 1, 1)).2.2

/-- **The failures' term**: `⌈log2 Σ_{j ≤ fails} C(N, j) · capMax^j⌉` over the caps' `N` connections, 0 with no
failure. -/
def failsTerm (cells : Array ℕ) (fails : ℕ) : ℕ := clog2 (failsAtMost cells.size (capMax cells) fails)

/-- The rows form's terms with each length read once: `rowsTerms` with `schedules` the registered parameters' count
(`thetas`, one under one fixed schedule) and `cells` the term of the period's `fails` failed connections. -/
def rowsTermsFails (facts : Array Facts) (scan : Scan) (scans : Array (Facts × ℕ × ℕ)) (cells : Array ℕ)
    (thetas fails jitter windows : ℕ) (charged : Array Facts := #[]) : Public :=
  { rowsTerms facts scan scans cells jitter windows charged with
    cells := failsTerm cells fails, schedules := clog2 thetas }

/-- **The clock form's `K` at no caps bounds the count with no cells factor**: `#Ss` at `syncs ^ windows` and each
audit's factors at the check's numbers. -/
theorem clockKBoundsHeld (jitter syncs windows : ℕ) {n : ℕ} (a : Fin n → Audit) (w : Fin n → Facts) :
    syncs ^ windows * ∏ i, (2 ^ (a i).free * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * (a i).bits) * jitter ^ (a i).steps) ≤
      2 ^ (terms (Array.ofFn a) (Array.ofFn w) #[] jitter syncs windows).K := by
  have h := clockKBoundsCount jitter syncs windows (fun j : Fin 0 => j.elim0) a w
  rw [Finset.univ_eq_empty, Finset.prod_empty, one_mul, Array.ofFn_zero] at h
  exact h

theorem K_failsOfFn (facts : Array Facts) (scan : Scan) (jitter windows thetas fails : ℕ) {m n : ℕ} (caps : Fin m → ℕ)
    (w : Fin n → Facts) (tied faulty : Fin n → ℕ) (charged : Array Facts) :
    (rowsTermsFails facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) thetas fails jitter
        windows charged).K =
      clog2 thetas + failsTerm (Array.ofFn caps) fails + ∑ i, (scan.terms (w i) (tied i) (faulty i) jitter).bits := by
  simp only [Public.K, rowsTermsFails, rowsTerms, Array.map_ofFn]
  rw [foldl_ofFn]
  rfl

/-- **The rows form's `K` at `fails` failures bounds the count of the runs with at most `fails`**, at the check's
number for the patterns. For the caps `caps` of `m` connections, `thetas` registered parameters and `fails` failed
connections: `failsAtMost`, times `thetas`, times each window's factors at the check's numbers, is at most `2 ^ K`.
With no failure and one registered parameter the first two factors are one and their terms zero. -/
theorem rowsKBoundsFails (facts : Array Facts) (scan : Scan) (jitter windows thetas fails : ℕ) {m n : ℕ}
    (caps : Fin m → ℕ) (w : Fin n → Facts) (tied faulty : Fin n → ℕ) (charged : Array Facts) :
    failsAtMost m (capMax (Array.ofFn caps)) fails *
        (thetas * ∏ i, (2 ^ (scan.free + tied i) * atMost (w i).units (w i).wrong *
          2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps * 2 ^ faulty i)) ≤
      2 ^ (rowsTermsFails facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) thetas fails
        jitter windows charged).K := by
  have hf : failsAtMost m (capMax (Array.ofFn caps)) fails ≤ 2 ^ failsTerm (Array.ofFn caps) fails := by
    unfold failsTerm
    rw [Array.size_ofFn]
    exact le_two_pow_clog2 _
  have hw : ∏ i, (2 ^ (scan.free + tied i) * atMost (w i).units (w i).wrong *
      2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps * 2 ^ faulty i) ≤
      2 ^ ∑ i, (scan.terms (w i) (tied i) (faulty i) jitter).bits := by
    rw [← Finset.prod_pow_eq_pow_sum]
    refine Finset.prod_le_prod fun i _ => ?_
    simp only [AuditTerms.bits, Scan.terms, pow_add]
    exact Nat.mul_le_mul
      (Nat.mul_le_mul (Nat.mul_le_mul (Nat.mul_le_mul le_rfl (le_two_pow_clog2 _)) le_rfl) (le_two_pow_clog2 _)) le_rfl
  rw [K_failsOfFn, pow_add, pow_add, mul_comm (2 ^ clog2 thetas), mul_assoc]
  exact Nat.mul_le_mul hf (Nat.mul_le_mul (le_two_pow_clog2 _) hw)

/-- The loop's state after `t ≤ n` steps: `C(n, t)`, `cap^t` and `Σ_{i ≤ t} C(n, i) · cap^i`. -/
theorem failsFold (n cap t : ℕ) (ht : t ≤ n) :
    (List.range t).foldl (failsStep n cap) (1, 1, 1) =
      (n.choose t, cap ^ t, ∑ i ∈ Finset.range (t + 1), n.choose i * cap ^ i) := by
  induction t with
  | zero => simp
  | succ t ih =>
    have hc : n.choose t * (n - t) / (t + 1) = n.choose (t + 1) := by
      rw [← Nat.choose_succ_right_eq]
      exact Nat.mul_div_cancel _ (Nat.succ_pos t)
    rw [List.range_succ, List.foldl_append, ih (Nat.le_of_succ_le ht), List.foldl_cons, List.foldl_nil]
    simp only [failsStep, hc, Finset.sum_range_succ (fun i => n.choose i * cap ^ i) (t + 1), pow_succ]

/-- The loop is the sum it names. -/
theorem failsAtMost_eq (n cap b : ℕ) :
    failsAtMost n cap b = ∑ i ∈ Finset.range (min b n + 1), n.choose i * cap ^ i := by
  simp only [failsAtMost, failsFold n cap (min b n) (Nat.min_le_right b n)]

theorem foldl_max_ge (l : List ℕ) (a : ℕ) : a ≤ l.foldl max a ∧ ∀ x ∈ l, x ≤ l.foldl max a := by
  induction l generalizing a with
  | nil => simp
  | cons y l ih =>
    rw [List.foldl_cons]
    obtain ⟨h1, h2⟩ := ih (max a y)
    refine ⟨(le_max_left a y).trans h1, fun x hx => ?_⟩
    rcases List.mem_cons.1 hx with h | h
    · rw [h]
      exact (le_max_right a y).trans h1
    · exact h2 x h

theorem le_capMax {m : ℕ} (caps : Fin m → ℕ) (c : Fin m) : caps c ≤ capMax (Array.ofFn caps) := by
  unfold capMax
  rw [← Array.foldl_toList]
  exact (foldl_max_ge _ 0).2 _ (Array.mem_toList_iff.mpr (Array.mem_ofFn.2 ⟨c, rfl⟩))

/-- **The check's number covers the fail patterns**: the patterns of at most `fails` of the `m` connections, each
failing below its cap (`failPatterns`, summed up to `fails`), number at most `failsAtMost` at the widest cap. -/
theorem failsAtMostCovers {m : ℕ} (caps : Fin m → ℕ) (fails : ℕ) :
    ∑ i ∈ Finset.range (fails + 1), failPatterns caps i ≤ failsAtMost m (capMax (Array.ofFn caps)) fails := by
  rw [failsAtMost_eq]
  simp only [failPatterns]
  set M := capMax (Array.ofFn caps) with hM
  have hlevel : ∀ i, ∑ J ∈ (Finset.univ : Finset (Fin m)).powersetCard i, ∏ c ∈ J, caps c ≤ m.choose i * M ^ i := by
    intro i
    calc ∑ J ∈ (Finset.univ : Finset (Fin m)).powersetCard i, ∏ c ∈ J, caps c
        ≤ ∑ _J ∈ (Finset.univ : Finset (Fin m)).powersetCard i, M ^ i := by
          refine Finset.sum_le_sum fun J hJ => ?_
          calc ∏ c ∈ J, caps c ≤ ∏ _c ∈ J, M := Finset.prod_le_prod fun c _ => le_capMax caps c
            _ = M ^ i := by rw [Finset.prod_const, (Finset.mem_powersetCard.1 hJ).2]
      _ = m.choose i * M ^ i := by
          rw [Finset.sum_const, Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
  calc ∑ i ∈ Finset.range (fails + 1), ∑ J ∈ (Finset.univ : Finset (Fin m)).powersetCard i, ∏ c ∈ J, caps c
      ≤ ∑ i ∈ Finset.range (fails + 1), m.choose i * M ^ i := Finset.sum_le_sum fun i _ => hlevel i
    _ ≤ ∑ i ∈ Finset.range (min fails m + 1), m.choose i * M ^ i := by
        rcases Nat.le_total fails m with h | h
        · exact le_of_eq (by rw [Nat.min_eq_left h])
        · rw [Nat.min_eq_right h]
          have hsub : Finset.range (m + 1) ⊆ Finset.range (fails + 1) :=
            Finset.range_subset_range.2 (Nat.succ_le_succ h)
          refine le_of_eq (Finset.sum_subset hsub fun i _ hi => ?_).symm
          have hlt : m < i := by
            have := Finset.mem_range.not.1 hi
            omega
          rw [Nat.choose_eq_zero_of_lt hlt, zero_mul]

/-- **The rows form's `K` at `fails` failures bounds `MonitoringPeriodLeakFailsAtMost`'s factors at the check's numbers**: the
fail patterns of at most `fails` connections below their caps (`Σ_{i ≤ fails} failPatterns caps i`), times `thetas`
registered parameters, times each window's factors, is at most `2 ^ K`. -/
theorem rowsKBoundsFailPatterns (facts : Array Facts) (scan : Scan) (jitter windows thetas fails : ℕ) {m n : ℕ}
    (caps : Fin m → ℕ) (w : Fin n → Facts) (tied faulty : Fin n → ℕ) (charged : Array Facts) :
    (∑ i ∈ Finset.range (fails + 1), failPatterns caps i) *
        (thetas * ∏ i, (2 ^ (scan.free + tied i) * atMost (w i).units (w i).wrong *
          2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps * 2 ^ faulty i)) ≤
      2 ^ (rowsTermsFails facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) thetas fails
        jitter windows charged).K :=
  (Nat.mul_le_mul_right _ (failsAtMostCovers caps fails)).trans
    (rowsKBoundsFails facts scan jitter windows thetas fails caps w tied faulty charged)

end NetTiming.PeriodTerms
