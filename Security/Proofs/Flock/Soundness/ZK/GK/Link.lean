import Proofs.Flock.Soundness.ZK.GK.Theorem
import Proofs.Flock.Soundness.ZK.AdaptivePrefinal

/-!
# Zero knowledge: Lemmas B and C as Theorem GK's two distances

`Session.adaptive_prefinal_hm96` (Lemma C) bounds, against a deterministic `V*`, every event of the real prover's
pre-final transcript on one witness by its probability on another plus `2·|L|·δ₁`. `prefinalClose_hm96` reads it as the
model's `PrefinalClose`: when the model's runs are the prover's runs on randomness `(ω, (y_l)_l)`, a run's view before
`V*`'s final message being that transcript on the witness (`real`) or the dummy witness (`dummy`), `PrefinalClose` holds
at `δ_C = 2·|L|·δ₁`, the hiding of `hm96` (`Hm96Hiding`) being the only assumption.
`Session.session_shvzk_hm96` (Lemma B) bounds, at fixed coins that are non-degenerate for every table, every event of the
session's view under the real prover by its probability under the tables' `S_shvzk` plus `Σ_j 2·|Hid_j|·δ₁`, both ways.
`rewindClose_hm96` reads it as the model's `RewindClose`: the coins of a rewind are those `V*` committed to in the first
run, so the tables are a function of that run (`T ω₀`); when they are non-degenerate for every completed first run, each
rewind's view is the tuple of the tables' views (the simulator's on fresh dummy openings, hybrid 1's on the real
witness), and the session has at most `N_hid` hidden leaves, `RewindClose` holds at `δ_B = 2·N_hid·δ₁`.
`gk_simulate_hm96` is Theorem GK on the keyed coin tree with both distances discharged this way: its cryptographic
assumptions are `Hm96Hiding` at `δ₁` and `SHA512CRStrict` for the finder, and
`ε = 3·2|L|δ₁ + 8t·2·N_hid·δ₁ + 3^{-t} + t/M + q²/2^513`.
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

/-- Through a map whose fibers have one size, a uniform average is the average on the target. -/
theorem avgR_comp_fiber {Ω R : Type} [Fintype Ω] [Nonempty Ω] [Fintype R] (e : Ω → R) {k : ℕ}
    (he : ∀ r, Nat.card {x // e x = r} = k) (f : R → ℝ) : avgR (fun x => f (e x)) = avgR f := by
  classical
  have hk : ∀ r, #{x | e x = r} = k := fun r => by rw [← he r, Nat.card_eq_fintype_card, Fintype.card_subtype]
  have hs : ∀ g : R → ℝ, ∑ x, g (e x) = k * ∑ r, g r := fun g => by
    rw [← Finset.sum_fiberwise Finset.univ e (fun x => g (e x)), Finset.mul_sum]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [Finset.sum_congr rfl (fun x hx => by rw [(Finset.mem_filter.1 hx).2]), Finset.sum_const, hk r, nsmul_eq_mul]
  have hc : (Fintype.card Ω : ℝ) = k * Fintype.card R := by simpa using hs fun _ => 1
  have hk0 : (k : ℝ) ≠ 0 := by
    intro h0
    rw [h0, zero_mul] at hc
    exact Fintype.card_ne_zero (by exact_mod_cast hc)
  rw [avgR, avgR, hs f, hc, mul_div_mul_left _ _ hk0]

section Rewind

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2] [AddCommGroup Salt]
  [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt]
  [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi] {J : ℕ}

/-- **Lemma B is `RewindClose`.** A rewind from the completed dummy run `dummy ω₀` keeps the coins `V*` committed to in it,
so the session's tables there are a function of that run (`T ω₀`, with their hm96 leaves `Mh ω₀`, of one hiding pair
`(My, cs)`). When they are non-degenerate for every completed first run (`hrank`, `hpad`, `hW`, `hpads`, `hβ`) and the
witness `w` carries the public region data and satisfies the inner constraints there (`hreg`, `hinner`); the simulator's
rewind is the tuple of the tables' `S_shvzk` on the dummy witness `w0` (`hS`) and hybrid 1's the tuple of the real
prover's views on `w` (`hR`), each on randomness read through a map with fibers of one size (`eS`, `eR`); and the
session has at most `nH` hidden leaves: `RewindClose` holds at `2·nH·δ₁`. -/
theorem rewindClose_hm96 {V P Ω Ωd Ωs : Type} [Fintype Ω] [Nonempty Ω] [Fintype Ωd] [Nonempty Ωd] [Fintype Ωs]
    [Nonempty Ωs] (m : Model V P Ω Ωd Ωs) (T : Ωd → Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type}
    (Mh : ∀ ω₀ j, (T ω₀ j).Hm96 C) (My : Salt → B) (cs : Salt → C) (hMy : ∀ ω₀ j, (Mh ω₀ j).My = My)
    (hcs : ∀ ω₀ j, (Mh ω₀ j).cs = cs) {δ₁ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (w w0 : Fin J → W)
    (hrank : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j, (T ω₀ j).Rank)
    (hpad : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j, (T ω₀ j).PadOnto)
    (hW : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j r, Function.Bijective fun p : F × F => (T ω₀ j).Wr r p.1 p.2)
    (hpads : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j, (T ω₀ j).PadsOnto)
    (hβ : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j r, (T ω₀ j).β r ≠ 0)
    (hreg : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j, (T ω₀ j).sReg (w j) = (T ω₀ j).pubReg)
    (hinner : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j, (T ω₀ j).InnerHolds (w j))
    (lift : ∀ ω₀, (∀ j, (T ω₀ j).View) → V)
    (eS : ∀ ω₀, Ωs → ∀ j, (T ω₀ j).SimRand × ((T ω₀ j).Hid → Salt)) (kS : Ωd → ℕ)
    (heS : ∀ ω₀ r, Nat.card {σ // eS ω₀ σ = r} = kS ω₀)
    (eR : ∀ ω₀, Ω → ∀ j, (T ω₀ j).Rand × ((T ω₀ j).Hid → Salt)) (kR : Ωd → ℕ)
    (heR : ∀ ω₀ r, Nat.card {ω // eR ω₀ ω = r} = kR ω₀)
    (hS : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ σ,
      m.rwS (m.dummy ω₀) σ = lift ω₀ fun j => (Mh ω₀ j).simR (w0 j) (eS ω₀ σ j).1 (eS ω₀ σ j).2)
    (hR : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ ω,
      m.rwR (m.dummy ω₀) ω = lift ω₀ fun j => (Mh ω₀ j).viewR (w j) (eR ω₀ ω j).1 (eR ω₀ ω j).2)
    {nH : ℕ} (hnH : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∑ j, Fintype.card (T ω₀ j).Hid ≤ nH) :
    m.RewindClose (2 * nH * δ₁.toReal) := by
  refine ⟨by positivity, fun ω₀ hc U => ?_⟩
  have hb := Session.session_shvzk_hm96 (T ω₀) (Mh ω₀) (fun j => by rw [hMy, hcs]; exact hT1) (hrank ω₀ hc)
    (hpad ω₀ hc) (hW ω₀ hc) (hpads ω₀ hc) (hβ ω₀ hc) w (hreg ω₀ hc) (hinner ω₀ hc) w0 (fun v => U (lift ω₀ v))
  have : Nonempty (∀ j, (T ω₀ j).SimRand × ((T ω₀ j).Hid → Salt)) := ⟨eS ω₀ (Classical.arbitrary _)⟩
  have : Nonempty (∀ j, (T ω₀ j).Rand × ((T ω₀ j).Hid → Salt)) := ⟨eR ω₀ (Classical.arbitrary _)⟩
  have eSim : avgR (fun σ => ind U (m.rwS (m.dummy ω₀) σ)) = (prCoin fun p : ∀ j, (T ω₀ j).SimRand × ((T ω₀ j).Hid → Salt) =>
      U (lift ω₀ fun j => (Mh ω₀ j).simR (w0 j) (p j).1 (p j).2)).toReal := by
    have e : (fun σ => ind U (m.rwS (m.dummy ω₀) σ)) = fun σ =>
        Game.ind (fun p : ∀ j, (T ω₀ j).SimRand × ((T ω₀ j).Hid → Salt) =>
          U (lift ω₀ fun j => (Mh ω₀ j).simR (w0 j) (p j).1 (p j).2)) (eS ω₀ σ) := by
      funext σ
      rw [hS ω₀ hc σ]
      rfl
    rw [e, avgR_comp_fiber (eS ω₀) (heS ω₀), avgR_ind_eq_prCoin]
  have eReal : avgR (fun ω => ind U (m.rwR (m.dummy ω₀) ω)) = (prCoin fun p : ∀ j, (T ω₀ j).Rand × ((T ω₀ j).Hid → Salt) =>
      U (lift ω₀ fun j => (Mh ω₀ j).viewR (w j) (p j).1 (p j).2)).toReal := by
    have e : (fun ω => ind U (m.rwR (m.dummy ω₀) ω)) = fun ω =>
        Game.ind (fun p : ∀ j, (T ω₀ j).Rand × ((T ω₀ j).Hid → Salt) =>
          U (lift ω₀ fun j => (Mh ω₀ j).viewR (w j) (p j).1 (p j).2)) (eR ω₀ ω) := by
      funext ω
      rw [hR ω₀ hc ω]
      rfl
    rw [e, avgR_comp_fiber (eR ω₀) (heR ω₀), avgR_ind_eq_prCoin]
  have hle : ∑ j, 2 * (Fintype.card (T ω₀ j).Hid : ℝ≥0∞) * δ₁ ≤ 2 * nH * δ₁ := by
    rw [← Finset.sum_mul, ← Finset.mul_sum]
    gcongr
    exact_mod_cast hnH ω₀ hc
  have htop : (2 * nH * δ₁ : ℝ≥0∞) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hε : (∑ j, 2 * (Fintype.card (T ω₀ j).Hid : ℝ≥0∞) * δ₁) ≠ ⊤ := ne_top_of_le_ne_top htop hle
  have hεr : (∑ j, 2 * (Fintype.card (T ω₀ j).Hid : ℝ≥0∞) * δ₁).toReal ≤ 2 * nH * δ₁.toReal := by
    have := ENNReal.toReal_mono htop hle
    simpa [ENNReal.toReal_mul] using this
  have a := toReal_le_add (prCoin_ne_top _) hε hb.1
  have b := toReal_le_add (prCoin_ne_top _) hε hb.2
  rw [eSim, eReal, abs_le]
  constructor <;> linarith

/-- **Theorem GK with Lemmas B and C** (`coin-tree/hm96-sha512/v2`): for a deterministic `V*` (responder `Vr`, keyed coin
tree `CT`, session nonce `ν`, root and openings read from its view), the simulator's output is within
`3·2|L|δ₁ + 8t·2·nH·δ₁ + 3^{-t} + t/M + q²/2^513` of `V*`'s real view on every test with values in `[0, 1]`. Its
cryptographic assumptions are `Hm96Hiding` at `δ₁` (one pair `(My, cs)` for every hm96 leaf) and `SHA512CRStrict` for the
finder at budget `q`; everything else is a hypothesis about the protocol's runs: Lemma C's setting with the model's
pre-final views (`hRp`, `hDp`), Lemma B's tables at the coins `coins v₀` of each completed first run (non-degenerate there)
with the model's rewinds (`hS`, `hR`), the coin conflict `hbad`, and the model's own fields. -/
theorem gk_simulate_hm96 {V Ωs : Type} [Fintype Ωs] [Nonempty Ωs] {n : ℕ} {Cn M C Ω L : Type} [Inhabited M]
    [DecidableEq M] [Fintype Ω] [Nonempty Ω] [Fintype L] [DecidableEq L]
    (Tc : (Fin n → Cn) → Fin J → Table F K W Salt (B × Salt) D Hi VL) (coins : V → Fin n → Cn)
    (rep : ∀ c j, Fin (Tc c j).nC → Fin 2) (My : Salt → B) (cs : Salt → C) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (Hl : B × C → D)
    (msgs : (Fin J → W) → Ω → (Fin n → Cn) → (L → D) → Fin n → M) (dig : (Fin J → W) → Ω → (Fin n → Cn) → L → B)
    (P : (Fin J → W) → ZK.Strategy (Ω × (L → B × C)) M Cn n)
    (hP : ∀ w ω p c, (P w).msg (ω, p) c = msgs w ω c fun l => Hl (dig w ω c l + (p l).1, (p l).2))
    (e : ∀ c, Ω × (L → B × Salt) → ∀ j, (Tc c j).Rand) (k : (Fin n → Cn) → ℕ)
    (he : ∀ c r, Nat.card {x // e c x = r} = k c) (out : ∀ c, (∀ j, (Tc c j).Cut) → Fin n → M)
    (hI : ∀ w ω u c, msgs w ω c (fun l => Hl ((u l).1, cs (u l).2))
      = out c fun j => (Tc c j).cut (Session.sent (Tc c) (rep c) j) ((Tc c j).preView (w j) (e c (ω, u) j)))
    (w w₀ : Fin J → W) (hreg : ∀ c j, (Tc c j).sReg (w j) = (Tc c j).pubReg)
    (hreg₀ : ∀ c j, (Tc c j).sReg (w₀ j) = (Tc c j).pubReg) (Vr : Responder M Cn n)
    (m : Model V (Fin n → M) (Ω × (L → Salt)) (Ω × (L → Salt)) Ωs)
    (hRp : ∀ p, m.pre (m.real p) = run (P w) Vr (p.1, fun l => (My (p.2 l), cs (p.2 l))))
    (hDp : ∀ p, m.pre (m.dummy p) = run (P w₀) Vr (p.1, fun l => (My (p.2 l), cs (p.2 l))))
    (Mh : ∀ c j, (Tc c j).Hm96 C) (hMy : ∀ c j, (Mh c j).My = My) (hcs : ∀ c j, (Mh c j).cs = cs)
    (hrank : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j, (Tc (coins (m.dummy ω₀)) j).Rank)
    (hpad : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j, (Tc (coins (m.dummy ω₀)) j).PadOnto)
    (hW : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j r,
      Function.Bijective fun p : F × F => (Tc (coins (m.dummy ω₀)) j).Wr r p.1 p.2)
    (hpads : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j, (Tc (coins (m.dummy ω₀)) j).PadsOnto)
    (hβ : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j r, (Tc (coins (m.dummy ω₀)) j).β r ≠ 0)
    (hinner : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ j, (Tc (coins (m.dummy ω₀)) j).InnerHolds (w j))
    (lift : ∀ ω₀, (∀ j, (Tc (coins (m.dummy ω₀)) j).View) → V)
    (eS : ∀ ω₀, Ωs → ∀ j, (Tc (coins (m.dummy ω₀)) j).SimRand × ((Tc (coins (m.dummy ω₀)) j).Hid → Salt))
    (kS : Ω × (L → Salt) → ℕ) (heS : ∀ ω₀ r, Nat.card {σ // eS ω₀ σ = r} = kS ω₀)
    (eR : ∀ ω₀, Ω × (L → Salt) → ∀ j, (Tc (coins (m.dummy ω₀)) j).Rand × ((Tc (coins (m.dummy ω₀)) j).Hid → Salt))
    (kR : Ω × (L → Salt) → ℕ) (heR : ∀ ω₀ r, Nat.card {ω // eR ω₀ ω = r} = kR ω₀)
    (hS : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ σ, m.rwS (m.dummy ω₀) σ =
      lift ω₀ fun j => (Mh (coins (m.dummy ω₀)) j).simR (w₀ j) (eS ω₀ σ j).1 (eS ω₀ σ j).2)
    (hR : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ ω, m.rwR (m.dummy ω₀) ω =
      lift ω₀ fun j => (Mh (coins (m.dummy ω₀)) j).viewR (w j) (eR ω₀ ω j).1 (eR ω₀ ω j).2)
    {nH : ℕ} (hnH : ∀ ω₀, m.comp (m.dummy ω₀) = true → ∑ j, Fintype.card (Tc (coins (m.dummy ω₀)) j).Hid ≤ nH)
    {Dg Coin Sl : Type} [DecidableEq Dg] [DecidableEq Coin] (H : List UInt8 → Dg) (CT : CoinTree Dg Coin Sl)
    (ν : V → List UInt8) (root : V → Dg) (opens : V → List (CoinOpening Dg Coin Sl))
    (hbad : ∀ v₀ v, m.bad v₀ v = conflicts (fun x => H (ν v₀ ++ x)) CT (root v₀) (opens v₀) (opens v))
    (t N : ℕ) (hN : 0 < N) {q : ℝ}
    (hcr : Assumptions.SHA512CRStrict H
      (m.finder (fun v₀ v => bindCollKeyed H CT (ν v₀) (root v₀) (opens v₀) (opens v)) t N).toGame
      (m.finder (fun v₀ v => bindCollKeyed H CT (ν v₀) (root v₀) (opens v₀) (opens v)) t N).strat id (fun _ => q) q)
    (φ : V → ℝ) (h0 : ∀ v, 0 ≤ φ v) (h1 : ∀ v, φ v ≤ 1) :
    |Rand.ev (fun o => Out.val φ o.1) (m.sim t N) - avgR (fun ω => φ (m.real ω))| ≤
      3 * (2 * Fintype.card L * δ₁.toReal) + 8 * t * (2 * nH * δ₁.toReal) + (1 / 3) ^ t + t / N
        + q ^ 2 / 2 ^ (513 : ℝ) :=
  m.gk_simulate_coins_keyed H CT ν root opens hbad
    (rewindClose_hm96 m (fun ω₀ => Tc (coins (m.dummy ω₀))) (fun ω₀ => Mh (coins (m.dummy ω₀))) My cs
      (fun _ => hMy _) (fun _ => hcs _) hT1 hδ₁ w w₀ hrank hpad hW hpads hβ (fun _ _ => hreg _) hinner lift eS kS heS
      eR kR heR hS hR hnH)
    (prefinalClose_hm96 Tc rep My cs hT1 hδ₁ Hl msgs dig P hP e k he out hI w w₀ hreg hreg₀ Vr m hRp hDp)
    t N hN hcr φ h0 h1

end Rewind

end FlockSoundness.ZK.GK
