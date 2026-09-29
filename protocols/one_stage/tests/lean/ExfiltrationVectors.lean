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

/-- The least `K` with `C(n - (K + 1), k) · 2^e < C(n, k)`, which is `(Law.subset n k).miss (K + 1) < 2^-e`. -/
def countBound (n k e : ℕ) : ℕ :=
  ((List.range (n + 1)).find? fun K => (n - (K + 1)).choose k * 2 ^ e < n.choose k).getD n

/-- `(n, k, e, b)`: `n` units, `k` drawn, `δ = 2^-e`, `b` output bits per unit. -/
def grid : List (ℕ × ℕ × ℕ × ℕ) :=
  [(16, 1, 4), (16, 4, 4), (16, 8, 10), (64, 4, 4), (64, 8, 10), (64, 16, 20), (512, 4, 10), (512, 16, 10),
    (512, 64, 20), (4096, 38, 10), (4096, 128, 20), (65536, 1024, 20)].flatMap fun (n, k, e) =>
      [(n, k, e, 1), (n, k, e, 16)]

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
