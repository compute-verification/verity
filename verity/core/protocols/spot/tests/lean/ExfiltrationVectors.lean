import Proofs.Flock.Soundness.Audit.Influence
import Proofs.Flock.Soundness.Audit.StratifiedMiss
import Mathlib.Algebra.BigOperators.Fin

/-!
# `audit_exfiltration`'s bound, evaluated by Lean

Vectors for `verity_one_stage.consumers.exfiltration_bound` and `location_bits` (`tests/fixtures/exfiltration_lean.json`),
from the soundness package's `audit_exfiltration` on the uniform `k`-subset of `n` units (`Law.subset n k`), at
`δ = 2^-e` and `b` output bits per unit:

* `K` is the least `K` that meets `audit_exfiltration`'s `hk : miss (K + 1) < δ`. By `Law.subset_miss`,
  `miss (K + 1) = C(n - (K + 1), k) / C(n, k)`, so `hk` is `C(n - (K + 1), k) · 2^e < C(n, k)`.
* `H = b · K` is its `hH`'s harm bound: a set `B` with `δ ≤ escape B = miss |B|` (`Law.subset_escape`) has
  `|B| ≤ K`, so it holds at most `b · K` output bits.
* `admissible = ∑ j ∈ range (K + 1), n.choose j`, and the bound's first term is `numerator n H K / |M|` with
  `numerator n H K = 2 ^ H * admissible`. `audit_exfiltration_numerator` checks that `numerator` is that term.

Regenerate with `tests/lean/generate.sh`.
-/

namespace OneStageExfiltrationVectors

open FlockSoundness FlockSoundness.Audit FlockSoundness.Game Finset
open scoped ENNReal

/-- The numerator of `audit_exfiltration`'s bound. -/
def numerator (n H K : ℕ) : ℕ := 2 ^ H * ∑ j ∈ range (K + 1), n.choose j

/-- `audit_exfiltration` itself, its bound read through `numerator`: this fails to build if the bound's term changes. -/
theorem audit_exfiltration_numerator {C : Circuit} {n : ℕ} (P : Partition C n) {L : Law n} {Reg : Type}
    {session : Finset (Fin n) → Reg → Game Bool} {X : (R : Reg) → Cont L Reg session R → (Fin C.N → Bool)}
    {ext : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Option (Fin C.N → Bool)}
    {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞} {D : Finset (Fin C.N)} {M : Type} [Fintype M] [Nonempty M]
    [DecidableEq M] (hks : (P.analysis X ext).KnowledgeSound εks) (hlink : (P.analysis X ext).LinkSound δlink)
    (hin : Partition.AnchorsSound X a δin) (hD : D ⊆ P.committed) (hb : Fin n → ℕ)
    (K : Finset (Fin n) → Finset (Fin C.N)) (hK : ∀ B, K B ⊆ P.committed) (hsep : ∀ B, P.Separates (K B) B D)
    (hw : ∀ B, (K B).card ≤ ∑ u ∈ B, hb u) {δ : ℝ≥0∞} {H k : ℕ} (hH : IsHarmBound L (fun u => (hb u : ℝ)) δ H)
    (hk : L.miss (k + 1) < δ) (σ : M → Strategy (audit L Reg session)) (g : (D → Bool) → M) :
    avg (fun m => prob (fun o => o.1 = true ∧ g (Partition.out D (X (reg (σ m)) (cont (σ m)))) = m)
        (audit L Reg session) (σ m)) ≤
      (numerator n H k : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) + (δ + εks + δlink + δin) :=
  P.audit_exfiltration hks hlink hin hD hb K hK hsep hw hH hk σ g

/-- `C(m, k)` as `m.descFactorial k / k!`, which the kernel evaluates in `k` steps (`Nat.choose` unfolds Pascal's rule). -/
def chooseD (m k : ℕ) : ℕ := m.descFactorial k / k.factorial

theorem chooseD_eq (m k : ℕ) : chooseD m k = m.choose k := (Nat.choose_eq_descFactorial_div_factorial m k).symm

/-- From `K`, with `c = C(n - (K + 1), k)` and `f` steps left: the first count whose successor meets
`c · 2^e < top = C(n, k)`. The next `c` is `c · (m - k) / m` for `m = n - (K + 1)`. -/
def search (n k e top : ℕ) : ℕ → ℕ → ℕ → ℕ
  | 0, K, _ => K
  | f + 1, K, c =>
    if c * 2 ^ e < top then K else search n k e top f (K + 1) (c * (n - (K + 1) - k) / (n - (K + 1)))

/-- The vector's count, the least `K` whose successor meets `hk`: `grid_meets` checks it, not this search. -/
def countBound (n k e : ℕ) : ℕ := search n k e (chooseD n k) n 0 (chooseD (n - 1) k)

/-- What the kernel checks of a vector: `k ≤ n`, `K + 1 ≤ n` and `C(n - (K + 1), k) · 2^e < C(n, k) ≤ C(n - K, k) · 2^e`. -/
def meets (n k e K : ℕ) : Bool :=
  k ≤ n && K + 1 ≤ n && chooseD (n - (K + 1)) k * 2 ^ e < chooseD n k && chooseD n k ≤ chooseD (n - K) k * 2 ^ e

/-- `meets` is `miss (K + 1) < 2^-e ≤ miss K` for the uniform `k`-subset: `K` meets `audit_exfiltration`'s `hk`, and it is
the least that does, since `miss` is antitone. -/
theorem miss_of_meets {n k e K : ℕ} (h : meets n k e K = true) :
    ∃ hk : k ≤ n, (Law.subset n k hk).miss (K + 1) < ((2 ^ e : ℕ) : ℝ≥0∞)⁻¹ ∧
      ((2 ^ e : ℕ) : ℝ≥0∞)⁻¹ ≤ (Law.subset n k hk).miss K := by
  simp only [meets, Bool.and_eq_true, decide_eq_true_eq, chooseD_eq] at h
  obtain ⟨⟨⟨hk, hK⟩, hlt⟩, hle⟩ := h
  have hB : (n.choose k : ℝ≥0∞) ≠ 0 := by exact_mod_cast (Nat.choose_pos hk).ne'
  have hc : ((2 ^ e : ℕ) : ℝ≥0∞) ≠ 0 := by exact_mod_cast (pow_pos two_pos e).ne'
  have ht : ∀ x : ℕ, (x : ℝ≥0∞) ≠ ⊤ := fun x => ENNReal.natCast_ne_top x
  refine ⟨hk, ?_, ?_⟩
  · rw [Law.subset_miss hk hK, ENNReal.div_lt_iff (Or.inl hB) (Or.inl (ht _)), ← ENNReal.div_eq_inv_mul,
      ENNReal.lt_div_iff_mul_lt (Or.inl hc) (Or.inl (ht _))]
    exact_mod_cast hlt
  · rw [Law.subset_miss hk (by omega), ENNReal.le_div_iff_mul_le (Or.inl hB) (Or.inl (ht _)), ← ENNReal.div_eq_inv_mul,
      ENNReal.div_le_iff_le_mul (Or.inl hc) (Or.inl (ht _))]
    exact_mod_cast hle

/-- `(n, k, e, b)`: `n` units, `k` drawn, `δ = 2^-e`, `b` output bits per unit. -/
def grid : List (ℕ × ℕ × ℕ × ℕ) :=
  [(16, 1, 4), (16, 4, 4), (16, 8, 10), (64, 4, 4), (64, 8, 10), (64, 16, 20), (512, 4, 10), (512, 16, 10),
    (512, 64, 20), (4096, 38, 10), (4096, 128, 20), (65536, 1024, 20)].flatMap fun (n, k, e) =>
      [(n, k, e, 1), (n, k, e, 16)]

/-- Every vector's count meets `meets`, evaluated by the kernel. -/
theorem grid_meets : grid.all (fun (n, k, e, _) => meets n k e (countBound n k e)) = true := by decide +kernel

/-- **Each vector's `K` meets `hk`, and no smaller count does**: `miss (K + 1) < 2^-e ≤ miss K`. -/
theorem grid_counts : ∀ c ∈ grid, ∃ hk : c.2.1 ≤ c.1,
    (Law.subset c.1 c.2.1 hk).miss (countBound c.1 c.2.1 c.2.2.1 + 1) < ((2 ^ c.2.2.1 : ℕ) : ℝ≥0∞)⁻¹ ∧
      ((2 ^ c.2.2.1 : ℕ) : ℝ≥0∞)⁻¹ ≤ (Law.subset c.1 c.2.1 hk).miss (countBound c.1 c.2.1 c.2.2.1) :=
  fun c hc => miss_of_meets (List.all_eq_true.1 grid_meets c hc)

/-! ## Stratified laws

A stratified vector's law has strata of `N_s` units drawing `k_s` (`Law.stratified`). A fill `c` that some `τ ∈ (0, 1]`
separates attains its `miss (∑ c)` (`Law.stratified_miss_eq_greedy`): every factor `(N_s − j − k_s)/(N_s − j)` the fill takes
is at least `τ`, and every factor it leaves is at most `τ`. `searchS` fills greedily in exact integers, taking the largest
next factor by cross-multiplication, so the factors it takes never rise and `τ` is the last one. `meetsS` checks both
fills and their `τ` in ℕ, so exactly: a fill ordered in floating point can fail to be separated at a near-tie, and then its
escape can be below `miss`.
-/

/-- `τ = p/q` separates stratum `(N, k)` filled to `c`: the last factor taken, at `c − 1`, is at least `τ`, and the next, at
`c`, at most `τ`, cross-multiplied. -/
def sepOk (p q : ℕ) : ℕ × ℕ × ℕ → Bool
  | (N, k, c) => decide (k ≤ N ∧ c ≤ N ∧ (0 < c → p * (N - (c - 1)) ≤ q * (N - (c - 1) - k)) ∧
      (c < N → q * (N - c - k) ≤ p * (N - c)))

/-- What the kernel checks of a stratified vector with rows `(N_s, k_s, c_s, c'_s)`: the fill `c` of `K + 1` units is
separated by `τ` and the fill `c'` of `K` by `τ'`, both in `(0, 1]`, and
`∏ C(N_s − c_s, k_s) · 2^e < ∏ C(N_s, k_s) ≤ ∏ C(N_s − c'_s, k_s) · 2^e`. -/
def meetsS (e K : ℕ) (r : List (ℕ × ℕ × ℕ × ℕ)) (τ τ' : ℕ × ℕ) : Bool :=
  decide ((r.map fun x => x.2.2.1).sum = K + 1 ∧ (r.map fun x => x.2.2.2).sum = K ∧
    0 < τ.1 ∧ τ.1 ≤ τ.2 ∧ 0 < τ'.1 ∧ τ'.1 ≤ τ'.2) &&
  r.all (fun x => sepOk τ.1 τ.2 (x.1, x.2.1, x.2.2.1) && sepOk τ'.1 τ'.2 (x.1, x.2.1, x.2.2.2)) &&
  decide ((r.map fun x => chooseD (x.1 - x.2.2.1) x.2.1).prod * 2 ^ e < (r.map fun x => chooseD x.1 x.2.1).prod) &&
  decide ((r.map fun x => chooseD x.1 x.2.1).prod ≤ (r.map fun x => chooseD (x.1 - x.2.2.2) x.2.1).prod * 2 ^ e)

theorem natCast_div_le_div {p q x y : ℕ} (hq : q ≠ 0) (hy : y ≠ 0) (h : p * y ≤ x * q) :
    (p : ℝ≥0∞) / q ≤ (x : ℝ≥0∞) / y := by
  rw [ENNReal.div_le_iff (by exact_mod_cast hq) (ENNReal.natCast_ne_top _), div_eq_mul_inv, mul_right_comm,
    ← div_eq_mul_inv, ENNReal.le_div_iff_mul_le (Or.inl (by exact_mod_cast hy)) (Or.inl (ENNReal.natCast_ne_top _))]
  exact_mod_cast h

theorem prod_natCast_div {ι : Type*} (s : Finset ι) (c d : ι → ℕ) :
    ∏ i ∈ s, ((c i : ℝ≥0∞) / (d i : ℝ≥0∞)) = ((∏ i ∈ s, c i : ℕ) : ℝ≥0∞) / ((∏ i ∈ s, d i : ℕ) : ℝ≥0∞) := by
  simp only [div_eq_mul_inv, Finset.prod_mul_distrib, Nat.cast_prod]
  rw [ENNReal.prod_inv_distrib]
  exact fun i _ j _ _ => Or.inr (ENNReal.natCast_ne_top _)

/-- A fill `c` that `τ = p/q` separates, in ℕ, attains `miss (∑ c) = ∏ C(n_s − c_s, k_s) / ∏ C(n_s, k_s)`. -/
theorem miss_eq_of_sep {n m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (Law.stratum σ s).card)
    (c : Fin m → ℕ) {p q : ℕ} (hp : 0 < p) (hpq : p ≤ q)
    (hs : ∀ s, c s ≤ (Law.stratum σ s).card ∧
      (0 < c s → p * ((Law.stratum σ s).card - (c s - 1)) ≤ q * ((Law.stratum σ s).card - (c s - 1) - k s)) ∧
      (c s < (Law.stratum σ s).card → q * ((Law.stratum σ s).card - c s - k s) ≤ p * ((Law.stratum σ s).card - c s))) :
    (Law.stratified σ k hk).miss (∑ s, c s) =
      ((∏ s, ((Law.stratum σ s).card - c s).choose (k s) : ℕ) : ℝ≥0∞) /
        ((∏ s, (Law.stratum σ s).card.choose (k s) : ℕ) : ℝ≥0∞) := by
  have hq : q ≠ 0 := by omega
  rw [Law.stratified_miss_eq_greedy σ k hk c (fun s => (hs s).1) (τ := (p : ℝ≥0∞) / q)
    (ENNReal.div_pos_iff.2 ⟨by exact_mod_cast hp.ne', ENNReal.natCast_ne_top _⟩)
    (ENNReal.div_le_of_le_mul (by rw [one_mul]; exact_mod_cast hpq))
    (fun s h => natCast_div_le_div hq (by have := (hs s).1; omega) (by rw [mul_comm _ q]; exact (hs s).2.1 h))
    (fun s h => natCast_div_le_div (by omega) hq (by rw [mul_comm]; exact (hs s).2.2 h))]
  exact prod_natCast_div _ _ _

/-- **`meetsS` is `miss (K + 1) < 2^-e ≤ miss K`** for every stratified law whose strata are the rows' sizes and draws. -/
theorem miss_of_meetsS {e K : ℕ} {r : List (ℕ × ℕ × ℕ × ℕ)} {τ τ' : ℕ × ℕ} (h : meetsS e K r τ τ' = true)
    {n : ℕ} (σ : Fin n → Fin r.length) (hσ : ∀ s, (Law.stratum σ s).card = r[s.1].1) :
    ∃ hk : ∀ s, r[s.1].2.1 ≤ (Law.stratum σ s).card,
      (Law.stratified σ (fun s => r[s.1].2.1) hk).miss (K + 1) < ((2 ^ e : ℕ) : ℝ≥0∞)⁻¹ ∧
        ((2 ^ e : ℕ) : ℝ≥0∞)⁻¹ ≤ (Law.stratified σ (fun s => r[s.1].2.1) hk).miss K := by
  simp only [meetsS, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨hs1, hs2, hp, hpq, hp', hpq'⟩, hall⟩, hlt⟩, hle⟩ := h
  have row : ∀ s : Fin r.length, sepOk τ.1 τ.2 (r[s.1].1, r[s.1].2.1, r[s.1].2.2.1) = true ∧
      sepOk τ'.1 τ'.2 (r[s.1].1, r[s.1].2.1, r[s.1].2.2.2) = true := fun s => by
    simpa only [Bool.and_eq_true] using hall _ (List.getElem_mem s.2)
  simp only [sepOk, decide_eq_true_eq] at row
  have hk : ∀ s, r[s.1].2.1 ≤ (Law.stratum σ s).card := fun s => (hσ s) ▸ (row s).1.1
  have hc : ((2 ^ e : ℕ) : ℝ≥0∞) ≠ 0 := by exact_mod_cast (pow_pos two_pos e).ne'
  have ht : ∀ x : ℕ, (x : ℝ≥0∞) ≠ ⊤ := fun x => ENNReal.natCast_ne_top x
  have hD : ((∏ s : Fin r.length, (Law.stratum σ s).card.choose r[s.1].2.1 : ℕ) : ℝ≥0∞) ≠ 0 := by
    exact_mod_cast Finset.prod_ne_zero_iff.2 fun s _ => (Nat.choose_pos (hk s)).ne'
  have eD : (∏ s : Fin r.length, (Law.stratum σ s).card.choose r[s.1].2.1) = (r.map fun x => chooseD x.1 x.2.1).prod := by
    simp only [hσ, chooseD_eq]; exact Fin.prod_univ_fun_getElem r fun x => x.1.choose x.2.1
  refine ⟨hk, ?_, ?_⟩
  · have e1 := miss_eq_of_sep σ _ hk (fun s => r[s.1].2.2.1) hp hpq fun s => by rw [hσ]; exact (row s).1.2
    rw [Fin.sum_univ_fun_getElem r fun x => x.2.2.1, hs1] at e1
    have eN : (∏ s : Fin r.length, ((Law.stratum σ s).card - r[s.1].2.2.1).choose r[s.1].2.1) =
        (r.map fun x => chooseD (x.1 - x.2.2.1) x.2.1).prod := by
      simp only [hσ, chooseD_eq]; exact Fin.prod_univ_fun_getElem r fun x => (x.1 - x.2.2.1).choose x.2.1
    rw [e1, ENNReal.div_lt_iff (Or.inl hD) (Or.inl (ht _)), ← ENNReal.div_eq_inv_mul,
      ENNReal.lt_div_iff_mul_lt (Or.inl hc) (Or.inl (ht _)), eN, eD]
    exact_mod_cast hlt
  · have e1 := miss_eq_of_sep σ _ hk (fun s => r[s.1].2.2.2) hp' hpq' fun s => by rw [hσ]; exact (row s).2.2
    rw [Fin.sum_univ_fun_getElem r fun x => x.2.2.2, hs2] at e1
    have eN : (∏ s : Fin r.length, ((Law.stratum σ s).card - r[s.1].2.2.2).choose r[s.1].2.1) =
        (r.map fun x => chooseD (x.1 - x.2.2.2) x.2.1).prod := by
      simp only [hσ, chooseD_eq]; exact Fin.prod_univ_fun_getElem r fun x => (x.1 - x.2.2.2).choose x.2.1
    rw [e1, ENNReal.le_div_iff_mul_le (Or.inl hD) (Or.inl (ht _)), ← ENNReal.div_eq_inv_mul,
      ENNReal.div_le_iff_le_mul (Or.inl hc) (Or.inl (ht _)), eN, eD]
    exact_mod_cast hle

/-- The stratum not yet full whose next factor `(N − c − k)/(N − c)` is largest, the first on ties: its index and that
factor, compared by cross-multiplication. -/
def best : List (ℕ × ℕ × ℕ) → ℕ → Option (ℕ × ℕ × ℕ) → Option (ℕ × ℕ × ℕ)
  | [], _, acc => acc
  | (N, k, c) :: t, i, acc =>
    best t (i + 1) <| if c < N then
      match acc with
      | some (j, p, q) => if p * (N - c) < (N - c - k) * q then some (i, N - c - k, N - c) else some (j, p, q)
      | none => some (i, N - c - k, N - c)
    else acc

def bump : List (ℕ × ℕ × ℕ) → ℕ → List (ℕ × ℕ × ℕ)
  | [], _ => []
  | (N, k, c) :: t, 0 => (N, k, c + 1) :: t
  | x :: t, i + 1 => x :: bump t i

/-- The greedy fill from rows `(N, k, c)` of `K` units, whose escape is `P / D` and whose last factor is `τ`, one unit at
a time until the escape drops below `2^-e`: the count `K`, the fill of `K + 1` with its last factor, and the fill of `K`
with its own. `C(N − c − 1, k) = C(N − c, k) · (N − c − k) / (N − c)` exactly. -/
def searchS (e D : ℕ) : ℕ → List (ℕ × ℕ × ℕ) → ℕ × ℕ → ℕ → ℕ →
    Option (ℕ × List (ℕ × ℕ × ℕ) × (ℕ × ℕ) × List (ℕ × ℕ × ℕ) × (ℕ × ℕ))
  | 0, _, _, _, _ => none
  | f + 1, r, τ, K, P =>
    match best r 0 none with
    | none => none
    | some (i, p, q) =>
      if P * p / q * 2 ^ e < D then some (K, bump r i, (p, q), r, τ)
      else searchS e D f (bump r i) (p, q) (K + 1) (P * p / q)

/-- A stratified vector's certificate: `K`, the rows `(N_s, k_s, c_s, c'_s)` and the two `τ`, from `searchS`, which the
kernel checks with `meetsS`, not this search. -/
def certS (sh : List (ℕ × ℕ)) (e : ℕ) : Option (ℕ × List (ℕ × ℕ × ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ)) :=
  let D := (sh.map fun x => chooseD x.1 x.2).prod
  (searchS e D (sh.map (·.1)).sum (sh.map fun x => (x.1, x.2, 0)) (1, 1) 0 D).map fun (K, next, τ, cur, τ') =>
    (K, List.zipWith (fun x y => (x.1, x.2.1, x.2.2, y.2.2)) next cur, τ, τ')

/-- `(strata, e, b)`: strata `(N_s, k_s)`, `δ = 2^-e`, `b` output bits per unit. Exact ties at every step, near-ties,
floor-like strata of one or two draws, and a stratum that draws nothing. -/
def gridS : List (List (ℕ × ℕ) × ℕ × ℕ) :=
  [([(1000, 10), (2000, 20)], 10), ([(1000, 10), (2001, 20)], 10), ([(999, 10), (2000, 20)], 20),
    ([(10000, 100), (10001, 100)], 10), ([(3, 1), (6, 2), (9, 3)], 4), ([(287, 1), (1148, 2), (287, 1), (4096, 38)], 20),
    ([(64, 0), (512, 16)], 10), ([(16, 1), (16, 1), (16, 1), (16, 1)], 4)].flatMap fun (sh, e) =>
      [(sh, e, 1), (sh, e, 16)]

/-- Every stratified vector's certificate has the vector's strata and meets `meetsS`, evaluated by the kernel. -/
theorem gridS_meets : gridS.all (fun (sh, e, _) => match certS sh e with
    | some (K, r, τ, τ') => decide (r.map (fun x => (x.1, x.2.1)) = sh) && meetsS e K r τ τ'
    | none => false) = true := by decide +kernel

/-- **Each stratified vector's `K` meets `hk`, and no smaller count does**, for every stratified law with the vector's
strata: `miss (K + 1) < 2^-e ≤ miss K`. -/
theorem gridS_counts : ∀ v ∈ gridS, ∃ K r τ τ', certS v.1 v.2.1 = some (K, r, τ, τ') ∧
    r.map (fun x => (x.1, x.2.1)) = v.1 ∧ ∀ {n : ℕ} (σ : Fin n → Fin r.length), (∀ s, (Law.stratum σ s).card = r[s.1].1) →
      ∃ hk : ∀ s, r[s.1].2.1 ≤ (Law.stratum σ s).card,
        (Law.stratified σ (fun s => r[s.1].2.1) hk).miss (K + 1) < ((2 ^ v.2.1 : ℕ) : ℝ≥0∞)⁻¹ ∧
          ((2 ^ v.2.1 : ℕ) : ℝ≥0∞)⁻¹ ≤ (Law.stratified σ (fun s => r[s.1].2.1) hk).miss K := by
  intro v hv
  have h := List.all_eq_true.1 gridS_meets v hv
  obtain ⟨sh, e, b⟩ := v
  simp only at h ⊢
  cases hc : certS sh e with
  | none => rw [hc] at h; exact absurd h Bool.false_ne_true
  | some c =>
    obtain ⟨K, r, τ, τ'⟩ := c
    rw [hc] at h
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    exact ⟨K, r, τ, τ', rfl, h.1, fun σ hσ => miss_of_meetsS h.2 σ hσ⟩

/-- **An unseparated fill is refused.** At the near-tie `(10000, 100), (10001, 100)`, moving one of the 678 units across
the strata leaves the factor `9563/9663` in the second, above the `9561/9661` it took last in the first, so no `τ`
separates it. Its escape is below `miss 678`, by a factor of about `1 + 2·10⁻⁶`. -/
example : meetsS 10 677 [(10000, 100, 340, 338), (10001, 100, 338, 339)] (9561, 9661) (9563, 9663) = false := by
  decide +kernel

/-- `∑ j ∈ range (K + 1), n.choose j` and `n.choose K`, each term from the last: `C(n, j + 1) = C(n, j) · (n − j) / (j + 1)`. -/
def sumChoose (n : ℕ) : ℕ → ℕ × ℕ
  | 0 => (1, 1)
  | K + 1 => let (s, c) := sumChoose n K; (s + c * (n - K) / (K + 1), c * (n - K) / (K + 1))

theorem sumChoose_eq (n K : ℕ) : sumChoose n K = (∑ j ∈ range (K + 1), n.choose j, n.choose K) := by
  induction K with
  | zero => simp [sumChoose]
  | succ K ih =>
    have hc : n.choose K * (n - K) / (K + 1) = n.choose (K + 1) := by
      rw [← Nat.choose_succ_right_eq]; exact Nat.mul_div_cancel _ (Nat.succ_pos K)
    simp only [sumChoose, ih, hc, Finset.sum_range_succ _ (K + 1)]

open Lean in
run_meta do
  for d in [``grid_counts, ``gridS_counts, ``sumChoose_eq, ``audit_exfiltration_numerator] do
    for a in ← collectAxioms d do
      unless a ∈ [``propext, ``Classical.choice, ``Quot.sound] do
        throwError "{d} uses {a}, beyond propext, Classical.choice and Quot.sound"

def row : ℕ × ℕ × ℕ × ℕ → String
  | (n, k, e, b) =>
    let K := countBound n k e
    "{" ++ s!"\"n\": {n}, \"k\": {k}, \"delta_log2\": -{e}, \"bits\": {b}, \"K\": {K}, \"H\": {b * K}, " ++
      s!"\"admissible\": {∑ j ∈ range (K + 1), n.choose j}" ++ "}"

/-- A stratified vector's row: its strata, `K`, `H = b · K`, the two fills and their `τ` as `[p, q]`, and `admissible`
(`sumChoose_eq`), over `n = ∑ N_s` units. -/
def rowS : List (ℕ × ℕ) × ℕ × ℕ → String
  | (sh, e, b) =>
    match certS sh e with
    | some (K, r, τ, τ') =>
      "{" ++ s!"\"strata\": {sh.map fun x => [x.1, x.2]}, \"delta_log2\": -{e}, \"bits\": {b}, \"K\": {K}, " ++
        s!"\"H\": {b * K}, \"fill_next\": {r.map (·.2.2.1)}, \"tau_next\": [{τ.1}, {τ.2}], " ++
        s!"\"fill\": {r.map (·.2.2.2)}, \"tau\": [{τ'.1}, {τ'.2}], " ++
        s!"\"admissible\": {(sumChoose (sh.map (·.1)).sum K).1}" ++ "}"
    | none => "null"

end OneStageExfiltrationVectors

open OneStageExfiltrationVectors in
def main : IO Unit :=
  IO.println ("{\"source\": \"tests/lean/ExfiltrationVectors.lean via tests/lean/generate.sh\", " ++
    "\"statement\": \"FlockSoundness.Audit.Partition.audit_exfiltration\", \"vectors\": [\n" ++
    ",\n".intercalate (grid.map row) ++ "\n], \"stratified\": [\n" ++ ",\n".intercalate (gridS.map rowS) ++ "\n]}")
