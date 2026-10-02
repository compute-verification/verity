import FlockSoundness.ZK.GK.Theorem
import FlockSoundness.ZK.AdaptivePrefinal

/-!
# Zero knowledge: Lemma C as Theorem GK's `PrefinalClose`

`Session.adaptive_prefinal_hm96` (Lemma C) bounds, against a deterministic `V*`, every event of the real prover's
pre-final transcript on one witness by its probability on another plus `2·|L|·δ₁`. `prefinalClose_hm96` reads it as the
model's `PrefinalClose`: when the model's runs are the prover's runs on randomness `(ω, (y_l)_l)`, a run's view before
`V*`'s final message being that transcript on the witness (`real`) or the dummy witness (`dummy`), `PrefinalClose` holds
at `δ_C = 2·|L|·δ₁`, the hiding of `hm96` (`Hm96Hiding`) being the only assumption.
-/

namespace FlockSoundness.ZK.GK

open Finset Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- An `ℝ≥0∞` bound between two probabilities, as reals. -/
theorem toReal_le_add {a b ε : ℝ≥0∞} (hb : b ≠ ⊤) (hε : ε ≠ ⊤) (h : a ≤ b + ε) : a.toReal ≤ b.toReal + ε.toReal := by
  rw [← ENNReal.toReal_add hb hε]
  exact ENNReal.toReal_mono (ENNReal.add_ne_top.2 ⟨hb, hε⟩) h

section Real

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt] [AddCommGroup B]
  [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B]
  [Fintype Hi] [DecidableEq Hi] {J : ℕ}

/-- **Lemma C is `PrefinalClose`.** In the setting of `Session.adaptive_prefinal_hm96` (the session's tables `T c` at the
coins `c` opens, the prover `P` given by `msgs` of its leaves, its world-`W₀` form `hI`), for witnesses `w` and `w₀` carrying
the public region data, and a model whose runs are the prover's runs on `(ω, (y_l)_l)` against `V*`'s responder `Vr`, with
`V*`'s view before its final message the prover's pre-final transcript on `w` (`hR`) and on `w₀` (`hD`): `PrefinalClose`
holds at `2·|L|·δ₁`. -/
theorem prefinalClose_hm96 {V Ωs : Type} [Fintype Ωs] [Nonempty Ωs] {n : ℕ} {Cn M Cc Ω L : Type} [Inhabited M]
    [DecidableEq M] [Fintype Ω] [Nonempty Ω] [Fintype L] [DecidableEq L]
    (T : (Fin n → Cn) → Fin J → Table F K W Salt (B × Salt) D Hi VL)
    (rep : ∀ c j, Fin (T c j).nC → Fin 2) (My : Salt → B) (cs : Salt → Cc) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (H : B × Cc → D)
    (msgs : (Fin J → W) → Ω → (Fin n → Cn) → (L → D) → Fin n → M) (dig : (Fin J → W) → Ω → (Fin n → Cn) → L → B)
    (P : (Fin J → W) → ZK.Strategy (Ω × (L → B × Cc)) M Cn n)
    (hP : ∀ w ω p c, (P w).msg (ω, p) c = msgs w ω c fun l => H (dig w ω c l + (p l).1, (p l).2))
    (e : ∀ c, Ω × (L → B × Salt) → ∀ j, (T c j).Rand) (k : (Fin n → Cn) → ℕ)
    (he : ∀ c r, Nat.card {x // e c x = r} = k c) (out : ∀ c, (∀ j, (T c j).Cut) → Fin n → M)
    (hI : ∀ w ω u c, msgs w ω c (fun l => H ((u l).1, cs (u l).2))
      = out c fun j => (T c j).cut (Session.sent (T c) (rep c) j) ((T c j).preView (w j) (e c (ω, u) j)))
    (w w₀ : Fin J → W) (hreg : ∀ c j, (T c j).sReg (w j) = (T c j).pubReg)
    (hreg₀ : ∀ c j, (T c j).sReg (w₀ j) = (T c j).pubReg) (Vr : Responder M Cn n)
    (m : Model V (Fin n → M) (Ω × (L → Salt)) (Ω × (L → Salt)) Ωs)
    (hR : ∀ p, m.pre (m.real p) = run (P w) Vr (p.1, fun l => (My (p.2 l), cs (p.2 l))))
    (hD : ∀ p, m.pre (m.dummy p) = run (P w₀) Vr (p.1, fun l => (My (p.2 l), cs (p.2 l)))) :
    m.PrefinalClose (2 * Fintype.card L * δ₁.toReal) := by
  intro U
  have h1 := Session.adaptive_prefinal_hm96 T rep My cs hT1 H msgs dig P hP e k he out hI w w₀ hreg hreg₀ Vr U
  have h2 := Session.adaptive_prefinal_hm96 T rep My cs hT1 H msgs dig P hP e k he out hI w₀ w hreg₀ hreg Vr U
  have eR : avgR (fun p => ind U (m.pre (m.real p))) =
      (prCoin fun p : Ω × (L → Salt) => U (run (P w) Vr (p.1, fun l => (My (p.2 l), cs (p.2 l))))).toReal := by
    rw [← avgR_ind_eq_prCoin]
    congr 1
    funext p
    show Game.ind U (m.pre (m.real p)) = Game.ind _ p
    rw [hR]
    rfl
  have eD : avgR (fun p => ind U (m.pre (m.dummy p))) =
      (prCoin fun p : Ω × (L → Salt) => U (run (P w₀) Vr (p.1, fun l => (My (p.2 l), cs (p.2 l))))).toReal := by
    rw [← avgR_ind_eq_prCoin]
    congr 1
    funext p
    show Game.ind U (m.pre (m.dummy p)) = Game.ind _ p
    rw [hD]
    rfl
  have hε : 2 * (Fintype.card L : ℝ≥0∞) * δ₁ ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hεr : (2 * (Fintype.card L : ℝ≥0∞) * δ₁).toReal = 2 * Fintype.card L * δ₁.toReal := by
    simp [ENNReal.toReal_mul]
  have a := toReal_le_add (prCoin_ne_top _) hε h1
  have b := toReal_le_add (prCoin_ne_top _) hε h2
  rw [hεr] at a b
  rw [eR, eD, abs_le]
  constructor <;> linarith

end Real

end FlockSoundness.ZK.GK
