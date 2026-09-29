import FlockSoundness.Audit.Influence

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

open Lean in
run_meta do
  for d in [``grid_counts, ``audit_exfiltration_numerator] do
    for a in ← collectAxioms d do
      unless a ∈ [``propext, ``Classical.choice, ``Quot.sound] do
        throwError "{d} uses {a}, beyond propext, Classical.choice and Quot.sound"

def row : ℕ × ℕ × ℕ × ℕ → String
  | (n, k, e, b) =>
    let K := countBound n k e
    "{" ++ s!"\"n\": {n}, \"k\": {k}, \"delta_log2\": -{e}, \"bits\": {b}, \"K\": {K}, \"H\": {b * K}, " ++
      s!"\"admissible\": {∑ j ∈ range (K + 1), n.choose j}" ++ "}"

end OneStageExfiltrationVectors

open OneStageExfiltrationVectors in
def main : IO Unit :=
  IO.println ("{\"source\": \"tests/lean/ExfiltrationVectors.lean via tests/lean/generate.sh\", " ++
    "\"statement\": \"FlockSoundness.Audit.Partition.audit_exfiltration\", \"vectors\": [\n" ++
    ",\n".intercalate (grid.map row) ++ "\n]}")
