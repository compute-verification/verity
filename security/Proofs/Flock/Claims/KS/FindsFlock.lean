module

public import Proofs.Flock.Claims.KS.Finds
public import Proofs.Flock.Claims.KS.Flock
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob

/-!
# Flock: what `E^rep` finds, its one wait finds at half the rate

`Flock.E₁` is Flock's extractor with one wait in place of its `m′`: the same accepted reruns per wait, the same
truncation, decoder and check. From one wait's bound (`Flock.OneWait`), under `Flock.KS_of_oneWait`'s side
conditions, the extractor returns a witness with a property `G` of the run's coins, on an accepted run, with
probability at most twice the one-wait extractor's, plus `2·κ + √(γ·Pr[B])` (`Flock.finds_of_oneWait`, the claim
`Claims.Finds`): `ERep_finds` at the per-tape bound `KS_of_oneWait` reads, with `Escapes.le` on the escapes.

A bound on what the one-wait extractor's witness holds is then a bound on what the extractor's holds. The one-wait
extractor makes at most `1 + k` runs' worth of calls where the extractor's first-hit count reads `1 + m′·k`.
-/

@[expose] public section

namespace Claims

namespace Flock

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **Flock's extractor with one wait**: `E^rep` at a decoder with one wait for `k` accepted reruns among `K`
samples, where `Flock.E` makes `m′`. -/
noncomputable def E₁ (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) : Extractor Coin Wit D.V :=
  ERep D.V (fun _ => 1) D.k D.cap (fun v => [D.cols v]) (assemble D.parse D.table decode D.read) (check D.parse)

/-- **Flock: what the extractor finds, its one wait finds at half the rate**, from one wait's bound, under
`KS_of_oneWait`'s side conditions, at every decoder that meets the list-decoding spec. -/
theorem finds_of_oneWait (H : Msg → Msg) (D : Deploy Coin Col T) (hK : ∀ x, 0 < D.cap x)
    (hk : ∀ x, (8 ≤ D.k x ∧ Splits D x) ∨ NoAcc H D.V x) (hεt : ∀ x, 0 ≤ D.εtable x) (h : OneWait H D)
    (decode : List (ℕ × Col) → List T) (hlw : ListWithin D.len D.enc D.θ decode) :
    Finds H D.V (E D decode) (E₁ D decode) (κ D) (γ D) := by
  intro x Ω _ _ P G
  have hκ0 := (κ_pos D x (hεt x)).le
  have hs0 := Real.sqrt_nonneg (γ D x * PrB H D.V P x)
  rcases hk x with ⟨hk8, hsp⟩ | hno
  swap
  · rw [Pr_congr (q := fun _ => False) fun s => ⟨fun hs => hno Ω P s.1 s.2.1 hs.1, False.elim⟩, Pr_false]
    have h1 := Pr_nonneg fun s : Ω × (Fin (D.V.k x) → Coin) × (E₁ D decode).C x =>
      (run H P D.V x s.1 s.2.1).out = .acc ∧ ∃ w, ((E₁ D decode).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w
    linarith
  obtain ⟨e, hfit, hw⟩ := h x Ω P
  have hwait : ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (D.k x)
        (Assembles (fun v => [D.cols v]) (assemble D.parse D.table decode D.read x) (check D.parse x)) /
          accPr H D.V P x ω ^ D.k x) ≤
        (D.εtable x + e.term ω) + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω := by
    intro ω hε
    refine le_trans ?_ ((hw ω hε).trans (le_of_eq (by ring)))
    refine mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right ?_ (pow_nonneg hε.le _)) hε.le
    unfold accFail
    exact Pr_mono fun σ ⟨hacc, hno⟩ => ⟨hacc, fun hex => hno (assembles_of_extracts D decode hlw x _ hsp hex)⟩
  have hb : Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((E D decode).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w) ≤
      2 * Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E₁ D decode).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((E₁ D decode).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w) +
        (4 * Ex (fun ω => D.εtable x + e.term ω) + 2 * D.k x / D.cap x + 2 * Real.exp (-(D.k x : ℝ) / 8)) :=
    ERep_finds H D.V D.waits D.k D.cap (fun v => [D.cols v]) (assemble D.parse D.table decode D.read)
      (check D.parse) x P (hK x) hk8 (fun ω => D.εtable x + e.term ω)
      (fun ω => add_nonneg (hεt x) (e.term_nonneg ω)) (rate_le D x (by omega)) hwait G
  have hEx : 4 * Ex (fun ω => D.εtable x + e.term ω) ≤
      4 * D.εtable x + Real.sqrt (γ D x * PrB H D.V P x) := by
    rw [Ex_add, Ex_const]
    have := e.le (D.weight x) hfit
    unfold γ
    linarith
  have hκ : 4 * D.εtable x + 2 * D.k x / D.cap x + 2 * Real.exp (-(D.k x : ℝ) / 8) ≤ 2 * κ D x := by
    unfold κ
    have h1 : (0 : ℝ) < 2 ^ (-(D.waits x : ℝ)) := Real.rpow_pos_of_pos (by norm_num) _
    have h2 : (0 : ℝ) ≤ 2 * D.k x / D.cap x := by positivity
    have h3 := hεt x
    linarith
  linarith

end Flock

end Claims
