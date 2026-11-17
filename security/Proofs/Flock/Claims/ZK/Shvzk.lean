module

public import Definitions.Flock.ZK.Cut
public import Definitions.Flock.ZK.RealView
public import Proofs.Flock.Claims.ZK.GK
public import Proofs.Flock.Soundness.ZK.Adaptivity
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Star.Real
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
import Mathlib.Tactic.ENatToNat
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
import Proofs.Flock.Claims.ZK.Prefix
import Proofs.Flock.Soundness.ZK.AdaptivePrefinal
import Proofs.Flock.Soundness.ZK.GK.Link
import Proofs.Flock.Soundness.ZK.RealLeaves
import Proofs.Flock.Soundness.ZK.Session
import VCVio.Prelude.Core

/-! Special honest-verifier zero knowledge at every coin vector: `SHVZK` from Lemmas B and C, with
`InnerHolds` a hypothesis where the prover goes on. -/

@[expose] public section

namespace Claims.ZKMal

open Claims

def oneRound {Ω M : Type} (f : Ω → M) : FlockSoundness.ZK.Strategy Ω M Unit 1 where
  msg ω _ _ := f ω
  causal _ _ _ _ _ := rfl

def unitResp (M : Type) : FlockSoundness.ZK.Responder M Unit 1 := ⟨fun _ _ => (), fun _ _ _ _ => rfl⟩

theorem run_oneRound {Ω M : Type} [Inhabited M] (f : Ω → M) (V : FlockSoundness.ZK.Responder M Unit 1) (ω : Ω) :
    FlockSoundness.ZK.run (oneRound f) V ω 0 = f ω := rfl

namespace Proto

variable {W : Type}

def dVec (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (r : G.Rand) (c : Fin (G.K x) → G.Coin) : List Msg :=
  (List.range (G.K x + 1)).map fun j => ((G.dmsg x ν r ((List.ofFn c).take j)).eval H).1

theorem shvzk_at_hm96 (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    {F K Wt Salt B D Hi VL Cc : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt] [AddCommGroup B]
    [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt]
    [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi] {J : ℕ}
    (T : Fin J → FlockSoundness.ZK.Table F K Wt Salt (B × Salt) D Hi VL) (Mh : ∀ j, (T j).Hm96 Cc) {δ₁ : ENNReal}
    (hT1 : ∀ j, Claims.Hm96Hiding (Mh j).My (Mh j).cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) (wt wt₀ : Fin J → Wt) (hreg : ∀ j, (T j).sReg (wt j) = (T j).pubReg)
    (hinner : ∀ j, (T j).InnerHolds (wt j)) (lift : (∀ j, (T j).View) → List Msg)
    (eR : G.Rand → ∀ j, (T j).Rand × ((T j).Hid → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (eS : G.SimRand x → ∀ j, (T j).SimRand × ((T j).Hid → Salt)) (kS : ℕ)
    (heS : ∀ p, Nat.card {s // eS s = p} = kS)
    (hR : ∀ r, G.realVec H x w ν r c = lift fun j => (Mh j).viewR (wt j) (eR r j).1 (eR r j).2)
    (hS : ∀ s, ((G.sim x ν c s).eval H).1 = lift fun j => (Mh j).simR (wt₀ j) (eS s j).1 (eS s j).2)
    {nH : ℕ} (hnH : ∑ j, Fintype.card (T j).Hid ≤ nH) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1)|
      ≤ 2 * nH * δ₁.toReal := by
  have hb := FlockSoundness.ZK.Session.session_shvzk_hm96 T Mh hT1 hrank hpad hW hpads hβ wt hreg hinner wt₀
    (fun v => E (lift v))
  have : Nonempty (∀ j, (T j).SimRand × ((T j).Hid → Salt)) := ⟨eS (Classical.arbitrary _)⟩
  have : Nonempty (∀ j, (T j).Rand × ((T j).Hid → Salt)) := ⟨eR (Classical.arbitrary _)⟩
  have eReal : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) =>
        E (lift fun j => (Mh j).viewR (wt j) (p j).1 (p j).2)).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    simp only [hR]
  have eSim : Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1) =
      (FlockSoundness.Game.prCoin fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) =>
        E (lift fun j => (Mh j).simR (wt₀ j) (p j).1 (p j).2)).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eS heS]
    simp only [hS]
  have hle : ∑ j, 2 * (Fintype.card (T j).Hid : ENNReal) * δ₁ ≤ 2 * nH * δ₁ := by
    rw [← Finset.sum_mul, ← Finset.mul_sum]
    gcongr
    exact_mod_cast hnH
  have htop : (2 * nH * δ₁ : ENNReal) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hε : (∑ j, 2 * (Fintype.card (T j).Hid : ENNReal) * δ₁) ≠ ⊤ := ne_top_of_le_ne_top htop hle
  have hεr : (∑ j, 2 * (Fintype.card (T j).Hid : ENNReal) * δ₁).toReal ≤ 2 * nH * δ₁.toReal := by
    have := ENNReal.toReal_mono htop hle
    simpa [ENNReal.toReal_mul] using this
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε hb.1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε hb.2
  rw [eReal, eSim, abs_le]
  constructor <;> linarith

theorem shvzk_refused_hm96 (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    {F K Wt Salt B D Hi VL Cc Ω L : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt] [AddCommGroup B]
    [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B]
    [Fintype Hi] [DecidableEq Hi] [Fintype Ω] [Nonempty Ω] [Fintype L] [DecidableEq L] {J : ℕ}
    (T : Fin J → FlockSoundness.ZK.Table F K Wt Salt (B × Salt) D Hi VL) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (Hl : B × Cc → D) (msgs : (Fin J → Wt) → Ω → (L → D) → List Msg)
    (dig : (Fin J → Wt) → Ω → L → B) (e : Ω × (L → B × Salt) → ∀ j, (T j).Rand) (k : ℕ)
    (he : ∀ r, Nat.card {u // e u = r} = k) (out : (∀ j, (T j).Cut) → List Msg)
    (hI : ∀ wt ω u, msgs wt ω (fun l => Hl ((u l).1, cs (u l).2))
      = out fun j => (T j).cut (FlockSoundness.ZK.Session.sent T rep j) ((T j).preView (wt j) (e (ω, u) j)))
    (wt wt₀ : Fin J → Wt) (hreg : ∀ j, (T j).sReg (wt j) = (T j).pubReg)
    (hreg₀ : ∀ j, (T j).sReg (wt₀ j) = (T j).pubReg)
    (eR : G.Rand → Ω × (L → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (hR : ∀ r, G.realVec H x w ν r c =
      msgs wt (eR r).1 fun l => Hl (dig wt (eR r).1 l + My ((eR r).2 l), cs ((eR r).2 l)))
    (hD : ∀ r, G.dVec H x ν r c =
      msgs wt₀ (eR r).1 fun l => Hl (dig wt₀ (eR r).1 l + My ((eR r).2 l), cs ((eR r).2 l)))
    (eS : G.SimRand x → G.Rand) (kS : ℕ) (heS : ∀ r, Nat.card {s // eS s = r} = kS)
    (hS : ∀ s, ((G.sim x ν c s).eval H).1 = G.dVec H x ν (eS s) c) {n : ℕ} (hn : Fintype.card L ≤ n)
    (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1)|
      ≤ 2 * n * δ₁.toReal := by
  have : Nonempty Salt := ⟨0⟩
  have h1 := FlockSoundness.ZK.Session.adaptive_prefinal_hm96 (n := 1) (Cn := Unit) (fun _ => T) (fun _ => rep) My cs
    hT1 Hl (fun wt' ω _ ls _ => msgs wt' ω ls) (fun wt' ω _ l => dig wt' ω l)
    (fun wt' => oneRound fun p => msgs wt' p.1 fun l => Hl (dig wt' p.1 l + (p.2 l).1, (p.2 l).2))
    (fun _ _ _ _ => rfl) (fun _ => e) (fun _ => k) (fun _ => he) (fun _ cut _ => out cut)
    (fun wt' ω u _ => funext fun _ => hI wt' ω u) wt wt₀ (fun _ => hreg) (fun _ => hreg₀) (unitResp _)
    (fun τ => E (τ 0))
  have h2 := FlockSoundness.ZK.Session.adaptive_prefinal_hm96 (n := 1) (Cn := Unit) (fun _ => T) (fun _ => rep) My cs
    hT1 Hl (fun wt' ω _ ls _ => msgs wt' ω ls) (fun wt' ω _ l => dig wt' ω l)
    (fun wt' => oneRound fun p => msgs wt' p.1 fun l => Hl (dig wt' p.1 l + (p.2 l).1, (p.2 l).2))
    (fun _ _ _ _ => rfl) (fun _ => e) (fun _ => k) (fun _ => he) (fun _ cut _ => out cut)
    (fun wt' ω u _ => funext fun _ => hI wt' ω u) wt₀ wt (fun _ => hreg₀) (fun _ => hreg) (unitResp _)
    (fun τ => E (τ 0))
  simp only [run_oneRound] at h1 h2
  have eW : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (msgs wt p.1 fun l => Hl (dig wt p.1 l + My (p.2 l), cs (p.2 l)))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    simp only [hR]
  have eD : Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (msgs wt₀ p.1 fun l => Hl (dig wt₀ p.1 l + My (p.2 l), cs (p.2 l)))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR, ← Pr_comp_fiber eS heS]
    simp only [hS, hD]
  have hε : 2 * (Fintype.card L : ENNReal) * δ₁ ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hεr : (2 * (Fintype.card L : ENNReal) * δ₁).toReal = 2 * Fintype.card L * δ₁.toReal := by
    simp [ENNReal.toReal_mul]
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h2
  rw [hεr] at a b
  have hn' : 2 * (Fintype.card L : ℝ) * δ₁.toReal ≤ 2 * n * δ₁.toReal := by
    have : (Fintype.card L : ℝ) ≤ n := by exact_mod_cast hn
    have := ENNReal.toReal_nonneg (a := δ₁)
    nlinarith
  rw [eW, eD, abs_le]
  constructor <;> linarith

end Proto

end Claims.ZKMal
