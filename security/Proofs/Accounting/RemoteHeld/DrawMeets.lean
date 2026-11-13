module

public import Proofs.Accounting.RemoteHeld.Meets
public import Definitions.Accounting.Assumptions.RemoteDraw
public import Definitions.Pous.Model.StreamedWipe
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
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
import Mathlib.RingTheory.WittVector.IsPoly
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Accounting.Bandwidth.Chip
import Proofs.Accounting.RateAudit.Pooled
import Proofs.Accounting.RemoteHeld.Draw
import Proofs.Pouw.Prob

/-!
# The sampled guarantees' premises meet, and their conclusions bite

Each guarantee of `RemoteHeld.Draw` (`PooledSample`, `LiveCoinsSampled`, `HeldSampled`, `HeldOverTimeSampled`,
`HeldOverTimeLinksSampled`) has its premises met at an honest instance (`*_honest`) and its conclusion false for an audit
that accepts everything (`*_nontrivial`).

* **Met.** The honest session (`honest`): window `t` registers `U t` receipts, every one correct, whatever the coins,
  its draw's fallbacks `0`. It is `Live`, `Refused` at any receipts (every opening is one of the window's receipts,
  `openings_zero_subset`), `Counted` on any trace that finds nothing wrong, and its distinguishers gain nothing
  (`Draw.prfAdvantage_honest`). Over `Meets.burst` on its coins (a node bringing absent blocks back by every route) it
  meets every premise of the memory-line forms too, at any rates, W2's measured ones among them (`Meets.w2_meets`).
* **Bites.** The toy (`toy`): one window of one receipt, none correct, sampled once, on an audit that accepts every
  outcome with that receipt wrong (`acceptAll`), its node holding nothing and bringing nothing in, inside the device
  model at no rates, bursts or chip blocks (`acceptAll_inside`). It is `Live` and `Counted`; no opening of its
  receipts verifies (`unsalted`: every salt empty), so the binding term is `0`; its draw always opens the wrong receipt,
  so every advantage is `0`; and its pooled bound is `0`. Each conclusion then claims probability at most
  `δ + 2^-64` (and `δs`) for an event of probability `1`.
-/

@[expose] public section

namespace Accounting.SecurityProofs.Remote.DrawMeets

open Finset Accounting.Rate.Pooled
open Pous (pr)
open scoped ENNReal
open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions Accounting.SecurityProofs.Remote.Meets

/-! ## The honest session -/

theorem offsets_zero_lt {m U k : ℕ} (b : Blocks m) (hk : k ≤ U) :
    ∀ l, l < k → l + offsets U b (fun _ => 0) l < U := by
  intro l hl
  simp only [offsets, offset]
  split_ifs with hlm
  · cases h : Accounting.Actual.wordIndex (U - l) (word (U - l) (b ⟨l, hlm⟩)) with
    | none =>
      simp
      omega
    | some o =>
      have : o.val < U - l := o.isLt
      simp
      omega
  · omega

/-- With fallbacks `0`, a window of `U` opens receipts among its `U`, accepted or not. -/
theorem openings_zero_subset {m U k : ℕ} (b : Blocks m) (hk : k ≤ U) :
    openings U k b (fun _ => 0) ⊆ range U := by
  intro q hq
  obtain ⟨p, hp, rfl⟩ := mem_image.1 hq
  exact mem_range.2 (Draw.arrange_lt hk (offsets_zero_lt b hk) p (by have := mem_range.1 hp; omega))

variable {n m : ℕ} {k : Fin n → ℕ} {κ : ℕ} {X : Type}

/-- **The honest session**: window `t` registers `U t` receipts, every one correct, whatever the coins. -/
def honest (k : Fin n → ℕ) (κ : ℕ) (X : Type) (U : Fin n → Fin (m + 1)) : Session n m k κ X where
  size t _ := U t
  correct t _ := range (U t)

/-- The honest session is `Live`, `Refused` at any receipts with fallbacks `0`, `Counted` on a trace with no wrong
receipt on an accepting outcome, and its distinguishers gain nothing. -/
theorem honest_meets [Fintype X] (U : Fin n → Fin (m + 1)) (prf : Fin n → Pous.Bits κ → Blocks m)
    (T : Trace (DrawCoins n κ X)) (R : Receipts n κ X) (hT : ∀ ω, T.accepts ω → T.incorrect ω = 0) :
    Live (honest k κ X U) ∧ Refused (honest k κ X U) prf (fun _ _ _ => 0) T R ∧ Counted (honest k κ X U) T ∧
      ∀ δ t, Pous.StreamedWipe.prfAdvantage (prf t) ((honest k κ X U).distinguisher prf δ t) = 0 :=
  ⟨fun _ _ _ _ _ => ⟨rfl, rfl⟩,
    fun ω _ t _ hp => Or.inl (openings_zero_subset (U := (U t : ℕ)) (k := min (k t) (U t : ℕ)) (prf t (ω.2 t))
      (min_le_right _ _) hp),
    fun ω ha => (hT ω ha).trans_le (Nat.zero_le _),
    fun δ t => Draw.prfAdvantage_honest (honest k κ X U) (fun _ _ => subset_rfl) prf δ t⟩

/-- `Meets.burst` on a session's coins. -/
abbrev burstOn (n κ : ℕ) (X : Type) (N D t₁ Br Bd Bl : ℕ) : Trace (DrawCoins n κ X) := burst N D t₁ Br Bd Bl

theorem burst_incorrect {Ω : Type} {N D t₁ Br Bd Bl : ℕ} (ω : Ω) : (burst (Ω := Ω) N D t₁ Br Bd Bl).incorrect ω = 0 := by
  show ∑ p ∈ range 2, ((univ : Finset (Fin N)) \ univ).card = 0
  simp

/-- **`PooledSample`'s premises meet** at the honest session, on any trace with no wrong receipt on an accepting outcome,
with no advantage. -/
theorem pooledSample_honest [Fintype X] (U : Fin n → Fin (m + 1)) (prf : Fin n → Pous.Bits κ → Blocks m)
    (T : Trace (DrawCoins n κ X)) (R : Receipts n κ X) (hT : ∀ ω, T.accepts ω → T.incorrect ω = 0) :
    Live (honest k κ X U) ∧ Refused (honest k κ X U) prf (fun _ _ _ => 0) T R ∧ Counted (honest k κ X U) T ∧
      ∀ δ t, Pous.StreamedWipe.prfAdvantage (prf t) ((honest k κ X U).distinguisher prf δ t) = 0 :=
  honest_meets U prf T R hT

/-- **`LiveCoinsSampled`'s premises meet** at the honest session, at `W` its pooled bound, which reads no coin. -/
theorem liveCoinsSampled_honest [Fintype X] (U : Fin n → Fin (m + 1)) (prf : Fin n → Pous.Bits κ → Blocks m)
    (T : Trace (DrawCoins n κ X)) (R : Receipts n κ X) (hT : ∀ ω, T.accepts ω → T.incorrect ω = 0) (δ : ℝ) :
    Live (honest k κ X U) ∧ Refused (honest k κ X U) prf (fun _ _ _ => 0) T R ∧ Counted (honest k κ X U) T ∧
      ∀ ω, T.accepts ω →
        (honest k κ X U).pooled δ ω ≤ pooled n (fun t => (U t : ℕ)) (fun t => min (k t) (U t : ℕ)) δ :=
  let ⟨hL, hR, hC, _⟩ := honest_meets (k := k) U prf T R hT
  ⟨hL, hR, hC, fun _ _ => le_rfl⟩

/-- **`HeldSampled`'s premises meet** at the honest session over `Meets.burst` on its coins, with no link block. -/
theorem heldSampled_honest [Fintype X] (U : Fin n → Fin (m + 1)) (prf : Fin n → Pous.Bits κ → Blocks m)
    (R : Receipts n κ X) {N D t₁ Br Bd S : ℕ} {s E β : ℝ} (hs : 0 ≤ s) (hE : 0 < E) (hβ : 0 ≤ β)
    (ht : t₁ + 1 < D) :
    Identity (burstOn n κ X N D t₁ Br Bd 0) ∧ SquaringCeiling (burstOn n κ X N D t₁ Br Bd 0) s E Br ∧
      Disks (burstOn n κ X N D t₁ Br Bd 0) β Bd ∧ ChipCeiling (burstOn n κ X N D t₁ Br Bd 0) S ∧
      Routed (burstOn n κ X N D t₁ Br Bd 0) ∧ Causal (burstOn n κ X N D t₁ Br Bd 0) 0 (2 * D) ∧
      SketchBound (burstOn n κ X N D t₁ Br Bd 0) 0 ∧ Live (honest k κ X U) ∧
      Refused (honest k κ X U) prf (fun _ _ _ => 0) (burstOn n κ X N D t₁ Br Bd 0) R ∧
      Counted (honest k κ X U) (burstOn n κ X N D t₁ Br Bd 0) := by
  obtain ⟨hI, hsq, hd, hS, hr, hc, hk, -⟩ :=
    burst_meets_held (Ω := DrawCoins n κ X) (N := N) (D := D) (t₁ := t₁) (Br := Br) (Bd := Bd) hs hE.le hβ S ht
  obtain ⟨hL, hR, hC, -⟩ :=
    honest_meets (k := k) U prf (burstOn n κ X N D t₁ Br Bd 0) R fun ω _ => burst_incorrect ω
  exact ⟨hI, hsq, hd, hS, hr, hc, hk, hL, hR, hC⟩

/-- **`HeldOverTimeSampled`'s premises meet** at the honest session over `Meets.burst` on its coins, with no link
block, paced by `Meets.burstPace`. -/
theorem heldOverTimeSampled_honest [Fintype X] (U : Fin n → Fin (m + 1)) (prf : Fin n → Pous.Bits κ → Blocks m)
    (R : Receipts n κ X) {N D t₁ Br Bd S : ℕ} {s E β : ℝ} (hs : 0 ≤ s) (hE : 0 < E) (hβ : 0 ≤ β)
    (ht : t₁ + 1 < D) :
    Identity (burstOn n κ X N D t₁ Br Bd 0) ∧ SquaringCeiling (burstOn n κ X N D t₁ Br Bd 0) s E Br ∧
      Disks (burstOn n κ X N D t₁ Br Bd 0) β Bd ∧ ChipCeiling (burstOn n κ X N D t₁ Br Bd 0) S ∧
      Routed (burstOn n κ X N D t₁ Br Bd 0) ∧ Paced (burstOn n κ X N D t₁ Br Bd 0) (burstPace D) ∧
      Covers (burstOn n κ X N D t₁ Br Bd 0) ∧ SketchBound (burstOn n κ X N D t₁ Br Bd 0) 0 ∧
      Live (honest k κ X U) ∧ Refused (honest k κ X U) prf (fun _ _ _ => 0) (burstOn n κ X N D t₁ Br Bd 0) R ∧
      Counted (honest k κ X U) (burstOn n κ X N D t₁ Br Bd 0) := by
  obtain ⟨-, -, hsq, hd, hS, hr, -, hp, hcov, hk, -⟩ :=
    burst_meets (Ω := DrawCoins n κ X) (N := N) (D := D) (t₁ := t₁) (Br := Br) (Bd := Bd) (Bl := 0) (βl := 0)
      hs hE.le hβ le_rfl S ht
  obtain ⟨hL, hR, hC, -⟩ :=
    honest_meets (k := k) U prf (burstOn n κ X N D t₁ Br Bd 0) R fun ω _ => burst_incorrect ω
  exact ⟨burst_identity, hsq, hd, hS, hr, hp, hcov, hk, hL, hR, hC⟩

/-- **`HeldOverTimeLinksSampled`'s premises meet** at the honest session over `Meets.burst` on its coins, its node
bringing blocks back by rebuilds, disk reads and links at once, paced by `Meets.burstPace`. -/
theorem heldOverTimeLinksSampled_honest [Fintype X] (U : Fin n → Fin (m + 1)) (prf : Fin n → Pous.Bits κ → Blocks m)
    (R : Receipts n κ X) {N D t₁ Br Bd Bl S : ℕ} {s E β βl : ℝ} (hs : 0 ≤ s) (hE : 0 < E) (hβ : 0 ≤ β)
    (hβl : 0 ≤ βl) (ht : t₁ + 1 < D) :
    Links (burstOn n κ X N D t₁ Br Bd Bl) βl Bl ∧ SquaringCeiling (burstOn n κ X N D t₁ Br Bd Bl) s E Br ∧
      Disks (burstOn n κ X N D t₁ Br Bd Bl) β Bd ∧ ChipCeiling (burstOn n κ X N D t₁ Br Bd Bl) S ∧
      Routed (burstOn n κ X N D t₁ Br Bd Bl) ∧ Paced (burstOn n κ X N D t₁ Br Bd Bl) (burstPace D) ∧
      Covers (burstOn n κ X N D t₁ Br Bd Bl) ∧ SketchBound (burstOn n κ X N D t₁ Br Bd Bl) 0 ∧
      Live (honest k κ X U) ∧ Refused (honest k κ X U) prf (fun _ _ _ => 0) (burstOn n κ X N D t₁ Br Bd Bl) R ∧
      Counted (honest k κ X U) (burstOn n κ X N D t₁ Br Bd Bl) := by
  obtain ⟨-, hl, hsq, hd, hS, hr, -, hp, hcov, hk, -⟩ :=
    burst_meets (Ω := DrawCoins n κ X) (N := N) (D := D) (t₁ := t₁) (Br := Br) (Bd := Bd) (Bl := Bl)
      hs hE.le hβ hβl S ht
  obtain ⟨hL, hR, hC, -⟩ :=
    honest_meets (k := k) U prf (burstOn n κ X N D t₁ Br Bd Bl) R fun ω _ => burst_incorrect ω
  exact ⟨hl, hsq, hd, hS, hr, hp, hcov, hk, hL, hR, hC⟩

/-! ## The toy that an audit accepting everything gets wrong -/

/-- One window of one receipt, none correct, sampled once. -/
def toy : Session 1 1 (fun _ => 1) 1 Unit where
  size _ _ := ⟨1, by norm_num⟩
  correct _ _ := ∅

/-- An audit that accepts every outcome, its one pass's one block wrong and its node holding nothing. -/
def acceptAll : Trace (DrawCoins 1 1 Unit) where
  blocks := 1
  passes := 1
  accepts _ := True
  tick _ _ := 0
  plan _ _ := univ
  correct _ _ := ∅
  mem _ _ := ∅
  chip _ _ := ∅
  rebuilt _ _ := 0
  disk _ _ := 0
  link _ _ := 0

/-- Receipts no opening verifies: every salt is empty. -/
def unsalted : Receipts 1 1 Unit where
  hash _ _ := 0
  verifier _ _ _ := ⟨⟨fun _ => 0, 0, fun _ => 0⟩, [], .v2⟩
  root _ _ _ _ := 0
  rank _ _ _ := 0
  opened _ _ _ := ⟨[], [], []⟩

/-- Its credited span is `[0, 1)`, with no credited group. -/
def toyPace : Pace (DrawCoins 1 1 Unit) where
  credited := 0
  first _ _ := 0
  released _ _ := 1
  ended _ _ := 0

theorem toy_live : Live toy := fun _ _ _ _ _ => ⟨rfl, rfl⟩

theorem acceptAll_incorrect (ω : DrawCoins 1 1 Unit) : acceptAll.incorrect ω = 1 := by
  show ∑ _p ∈ range 1, ((univ : Finset (Fin 1)) \ ∅).card = 1
  simp

theorem acceptAll_planned (ω : DrawCoins 1 1 Unit) : acceptAll.planned ω = 1 := by
  show ∑ _p ∈ range 1, (univ : Finset (Fin 1)).card = 1
  simp

theorem acceptAll_held (ω : DrawCoins 1 1 Unit) : acceptAll.held ω = 0 := by
  show ∑ _p ∈ range 1, (∅ : Finset (Fin 1)).card = 0
  simp

theorem acceptAll_heldOver (ω : DrawCoins 1 1 Unit) (u v : ℕ) : acceptAll.heldOver ω u v = 0 := by
  show ∑ _t ∈ Ico u v, (∅ : Finset (Fin 1)).card = 0
  simp

/-- The audit that accepts everything is inside the device model at no rates, bursts or chip blocks: its node brings
no block in by any route and holds none on its chips. -/
theorem acceptAll_inside :
    Identity acceptAll ∧ Links acceptAll 0 0 ∧ SquaringCeiling acceptAll 0 1 0 ∧ Disks acceptAll 0 0 ∧
      ChipCeiling acceptAll 0 ∧ Routed acceptAll := by
  refine ⟨fun _ _ => rfl, fun ω u v _ => ?_, fun ω u v _ => ?_, fun ω u v _ => ?_, fun _ _ => ?_, fun _ _ => ?_⟩
  · show ((«over» (fun _ _ => 0) ω u v : ℕ) : ℝ) ≤ 0 * ((v - u : ℕ) : ℝ) + 0
    simp [«over»]
  · show ((«over» (fun _ _ => 0) ω u v : ℕ) : ℝ) ≤ 0 / 1 * ((v - u : ℕ) : ℝ) + 0
    simp [«over»]
  · show ((«over» (fun _ _ => 0) ω u v : ℕ) : ℝ) ≤ 0 * ((v - u : ℕ) : ℝ) + 0
    simp [«over»]
  · show (∅ : Finset (Fin 1)).card ≤ 0
    simp
  · show (((∅ : Finset (Fin 1)) ∪ ∅) \ (∅ ∪ ∅)).card ≤ 0 + 0 + 0
    simp

theorem toy_good (t : Fin 1) (ω : DrawCoins 1 1 Unit) : toy.good t ω = 0 := by
  show ((range 1).filter (· ∈ (∅ : Finset ℕ))).card = 0
  simp

theorem toy_counted : Counted toy acceptAll := fun ω _ => by
  rw [acceptAll_incorrect]
  show 1 ≤ ∑ t : Fin 1, (1 - toy.good t ω)
  simp [toy_good]

theorem pooled_one (δ : ℝ) : pooled 1 (fun _ => 1) (fun _ => 1) δ ≤ 0 := by
  refine Real.sSup_nonpos ?_
  rintro _ ⟨f, ⟨hf, -⟩, rfl⟩
  have h := (hf 0).2
  norm_num at h
  simpa using h

theorem toy_pooled (δ : ℝ) (ω : DrawCoins 1 1 Unit) : toy.pooled δ ω ≤ 0 := pooled_one δ

theorem fill_nonpos {Ω : Type} (K : Pace Ω) (ω : Ω) (N : ℕ) {W : ℝ} (hW : W ≤ 0) : K.fill ω N W ≤ 0 := by
  refine Real.sSup_nonpos ?_
  rintro _ ⟨x, ⟨hx, hs⟩, rfl⟩
  have h0 : ∀ g ∈ range (K.credited + 1), x g = 0 := fun g hg =>
    le_antisymm ((single_le_sum (fun i _ => (hx i).1) hg).trans (hs.trans hW)) (hx g).1
  show ∑ g ∈ range (K.credited + 1), K.rate ω g * x g ≤ 0
  exact (sum_eq_zero fun g hg => by rw [h0 g hg, mul_zero]).le

theorem wordIndex_one_isSome {w : ℕ} (u : Pous.Bits w) : (Accounting.Actual.wordIndex 1 u).isSome = true := by
  have h : Pous.bitsVal u < 2 ^ w - 2 ^ w % 1 := by
    rw [Nat.mod_one, Nat.sub_zero]
    exact (Pous.bitsEquiv w u).isLt
  unfold Accounting.Actual.wordIndex
  rw [dite_eq_left h]
  rfl

theorem accepted_one {m : ℕ} (b : Blocks m) : accepted 1 1 b := fun i hi => by
  have h0 : (i : ℕ) = 0 := by omega
  rw [h0, Nat.sub_zero]
  exact wordIndex_one_isSome _

/-- The toy's draw always opens its one receipt, which is wrong. -/
theorem toy_lands (t : Fin 1) (ω : DrawCoins 1 1 Unit) (b : Blocks 1) : ¬ toy.lands t ω b := fun h => by
  have h' : accepted 1 1 b → openings 1 1 b (fun _ => 0) ⊆ ∅ := h
  have hmem := h' (accepted_one b)
    (mem_image_of_mem (arrange (offsets 1 b fun _ => 0) 1) (mem_range.2 Nat.one_pos))
  simp at hmem

theorem toy_advantage (prf : Fin 1 → Pous.Bits 1 → Blocks 1) (δ : ℝ) (t : Fin 1) :
    Pous.StreamedWipe.prfAdvantage (prf t) (toy.distinguisher prf δ t) = 0 := by
  have h0 : ∀ y, ¬ toy.distinguisher prf δ t y := fun y h => toy_lands t y.1.1 y.2 h.2.1
  unfold Pous.StreamedWipe.prfAdvantage
  exact le_antisymm (tsub_le_self.trans ((Bandwidth.pr_mono (Q := fun _ => False) fun _ hx => h0 _ hx).trans
    (le_of_eq pr_false))) zero_le

theorem unsalted_binding (D : Session 1 1 (fun _ => 1) 1 Unit) (prf : Fin 1 → Pous.Bits 1 → Blocks 1)
    (rest : DrawCoins 1 1 Unit → Fin 1 → ℕ → ℕ) : unsalted.binding D prf rest = 0 := by
  refine le_antisymm ((Bandwidth.pr_mono (Q := fun _ => False) fun ω hω => ?_).trans (le_of_eq pr_false)) zero_le
  obtain ⟨t, p, -, hmv⟩ := hω
  have h := hmv.1.2.1
  have h0 : (unsalted.opened t ω p).salt.length = 0 := rfl
  rw [h0] at h
  exact absurd h (by simp [Hash.Hm96.SALT_BYTES])

theorem toy_error (prf : Fin 1 → Pous.Bits 1 → Blocks 1) (rest : DrawCoins 1 1 Unit → Fin 1 → ℕ → ℕ) (δ : ℝ) :
    δ + ((∑ _t : Fin 1, 1 : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (toy.distinguisher prf δ t)).toReal +
        (unsalted.binding toy prf rest).toReal = δ + 1 / 2 ^ 64 := by
  rw [unsalted_binding, sum_eq_zero fun t _ => toy_advantage prf δ t]
  simp

theorem not_holds_acceptAll {ε : ℝ} (hε : ε < 1) {P : DrawCoins 1 1 Unit → Prop} (hP : ∀ ω, ¬ P ω) :
    ¬ acceptAll.Holds ε P := fun h => by
  have h1 : Pouw.pr (fun ω => acceptAll.accepts ω ∧ ¬ P ω) = 1 := Pouw.pr_eq_one fun ω => ⟨trivial, hP ω⟩
  have h2 : Pouw.pr (fun ω => acceptAll.accepts ω ∧ ¬ P ω) ≤ ε := h
  linarith

/-- **`PooledSample`'s conclusion bites**: on the toy, `Live` and `Counted`, an audit that accepts everything has more
wrong receipts than the pooled bound with probability `1`, above `PooledSample`'s error whenever `δ + 2^-64 < 1`. -/
theorem pooledSample_nontrivial (prf : Fin 1 → Pous.Bits 1 → Blocks 1) (rest : DrawCoins 1 1 Unit → Fin 1 → ℕ → ℕ)
    {δ : ℝ} (hδ : δ + 1 / 2 ^ 64 < 1) :
    (∀ ω, acceptAll.accepts ω) ∧ Live toy ∧ Counted toy acceptAll ∧
      ¬ acceptAll.Holds (δ + ((∑ _t : Fin 1, 1 : ℕ) : ℝ) / 2 ^ 64 +
          (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (toy.distinguisher prf δ t)).toReal +
          (unsalted.binding toy prf rest).toReal)
        (fun ω => (acceptAll.incorrect ω : ℝ) ≤ toy.pooled δ ω) := by
  refine ⟨fun _ => trivial, toy_live, toy_counted, ?_⟩
  rw [toy_error]
  refine not_holds_acceptAll hδ fun ω h => ?_
  rw [acceptAll_incorrect] at h
  exact absurd (h.trans (toy_pooled δ ω)) (by norm_num)

/-- **`LiveCoinsSampled`'s conclusion bites**: on the toy, at `W = 0`, the toy's pooled bound on every outcome. -/
theorem liveCoinsSampled_nontrivial (prf : Fin 1 → Pous.Bits 1 → Blocks 1) (rest : DrawCoins 1 1 Unit → Fin 1 → ℕ → ℕ)
    {δ : ℝ} (hδ : δ + 1 / 2 ^ 64 < 1) :
    (∀ ω, acceptAll.accepts ω) ∧ Live toy ∧ Counted toy acceptAll ∧ (∀ ω, acceptAll.accepts ω → toy.pooled δ ω ≤ 0) ∧
      ¬ LiveCoins acceptAll 0 (δ + ((∑ _t : Fin 1, 1 : ℕ) : ℝ) / 2 ^ 64 +
          (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (toy.distinguisher prf δ t)).toReal +
          (unsalted.binding toy prf rest).toReal) := by
  refine ⟨fun _ => trivial, toy_live, toy_counted, fun ω _ => toy_pooled δ ω, ?_⟩
  unfold LiveCoins
  rw [toy_error]
  refine not_holds_acceptAll hδ fun ω h => ?_
  rw [acceptAll_incorrect] at h
  exact absurd h (by norm_num)

/-- **`HeldSampled`'s conclusion bites**: on the toy, `Live` and `Counted`, an audit that accepts everything, its node
inside the device model at no rates, bursts or chip blocks (`acceptAll_inside`), credits over an instant a planned
block its node never held, with probability `1`. -/
theorem heldSampled_nontrivial (prf : Fin 1 → Pous.Bits 1 → Blocks 1) (rest : DrawCoins 1 1 Unit → Fin 1 → ℕ → ℕ)
    {δs δc : ℝ} (hδ : δs + (δc + 1 / 2 ^ 64) < 1) :
    (∀ ω, acceptAll.accepts ω) ∧ Identity acceptAll ∧ SquaringCeiling acceptAll 0 1 0 ∧ Disks acceptAll 0 0 ∧
      ChipCeiling acceptAll 0 ∧ Routed acceptAll ∧ Live toy ∧ Counted toy acceptAll ∧
      ¬ acceptAll.Holds (δs + (δc + ((∑ _t : Fin 1, 1 : ℕ) : ℝ) / 2 ^ 64 +
          (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (toy.distinguisher prf δc t)).toReal +
          (unsalted.binding toy prf rest).toReal)) fun ω =>
        (acceptAll.planned ω : ℝ) - toy.pooled δc ω - (((0 : ℝ) / 1 + 0) * (((0 : ℕ) : ℝ) - (0 : ℕ)) + (0 + 0)) -
            acceptAll.passes * (0 : ℕ) ≤ acceptAll.held ω := by
  obtain ⟨hI, -, hsq, hd, hS, hr⟩ := acceptAll_inside
  refine ⟨fun _ => trivial, hI, hsq, hd, hS, hr, toy_live, toy_counted, ?_⟩
  rw [toy_error]
  refine not_holds_acceptAll hδ fun ω h => ?_
  have hp := toy_pooled δc ω
  rw [acceptAll_planned, acceptAll_held] at h
  norm_num at h
  linarith

/-- **`HeldOverTimeSampled`'s conclusion bites**: as `heldSampled_nontrivial`, over the span `[0, 1)` (`toyPace`), the
audit credits a block-tick its node never held, with probability `1`. -/
theorem heldOverTimeSampled_nontrivial (prf : Fin 1 → Pous.Bits 1 → Blocks 1)
    (rest : DrawCoins 1 1 Unit → Fin 1 → ℕ → ℕ) {δs δc : ℝ} (hδ : δs + (δc + 1 / 2 ^ 64) < 1) :
    (∀ ω, acceptAll.accepts ω) ∧ Identity acceptAll ∧ SquaringCeiling acceptAll 0 1 0 ∧ Disks acceptAll 0 0 ∧
      ChipCeiling acceptAll 0 ∧ Routed acceptAll ∧ Live toy ∧ Counted toy acceptAll ∧
      ¬ acceptAll.Holds (δs + (δc + ((∑ _t : Fin 1, 1 : ℕ) : ℝ) / 2 ^ 64 +
          (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (toy.distinguisher prf δc t)).toReal +
          (unsalted.binding toy prf rest).toReal)) fun ω =>
        ((toyPace.stop ω - toyPace.start ω : ℕ) : ℝ) * ((acceptAll.blocks : ℝ) - (0 : ℕ))
            - 2 * toyPace.longest ω * (((0 : ℝ) / 1 + 0) * (((toyPace.stop ω - toyPace.start ω : ℕ) : ℝ) +
              2 * toyPace.longest ω) + (0 + 0))
            - toyPace.fill ω acceptAll.blocks (toy.pooled δc ω)
          ≤ acceptAll.heldOver ω (toyPace.start ω) (toyPace.stop ω) := by
  obtain ⟨hI, -, hsq, hd, hS, hr⟩ := acceptAll_inside
  refine ⟨fun _ => trivial, hI, hsq, hd, hS, hr, toy_live, toy_counted, ?_⟩
  rw [toy_error]
  refine not_holds_acceptAll hδ fun ω h => ?_
  have hf := fill_nonpos toyPace ω acceptAll.blocks (toy_pooled δc ω)
  have h1 : toyPace.stop ω - toyPace.start ω = 1 := rfl
  have h2 : acceptAll.blocks = 1 := rfl
  rw [h1, acceptAll_heldOver] at h
  rw [h2] at h hf
  norm_num at h
  linarith

/-- **`HeldOverTimeLinksSampled`'s conclusion bites**: as `heldOverTimeSampled_nontrivial`, with no link rate or burst
either, which the audit's node meets too (`acceptAll_inside`). -/
theorem heldOverTimeLinksSampled_nontrivial (prf : Fin 1 → Pous.Bits 1 → Blocks 1)
    (rest : DrawCoins 1 1 Unit → Fin 1 → ℕ → ℕ) {δs δc : ℝ} (hδ : δs + (δc + 1 / 2 ^ 64) < 1) :
    (∀ ω, acceptAll.accepts ω) ∧ Links acceptAll 0 0 ∧ SquaringCeiling acceptAll 0 1 0 ∧ Disks acceptAll 0 0 ∧
      ChipCeiling acceptAll 0 ∧ Routed acceptAll ∧ Live toy ∧ Counted toy acceptAll ∧
      ¬ acceptAll.Holds (δs + (δc + ((∑ _t : Fin 1, 1 : ℕ) : ℝ) / 2 ^ 64 +
          (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (toy.distinguisher prf δc t)).toReal +
          (unsalted.binding toy prf rest).toReal)) fun ω =>
        ((toyPace.stop ω - toyPace.start ω : ℕ) : ℝ) * ((acceptAll.blocks : ℝ) - (0 : ℕ))
            - 2 * toyPace.longest ω * (((0 : ℝ) / 1 + 0 + 0) * (((toyPace.stop ω - toyPace.start ω : ℕ) : ℝ) +
              2 * toyPace.longest ω) + (0 + 0 + 0))
            - toyPace.fill ω acceptAll.blocks (toy.pooled δc ω)
          ≤ acceptAll.heldOver ω (toyPace.start ω) (toyPace.stop ω) := by
  obtain ⟨-, hl, hsq, hd, hS, hr⟩ := acceptAll_inside
  refine ⟨fun _ => trivial, hl, hsq, hd, hS, hr, toy_live, toy_counted, ?_⟩
  rw [toy_error]
  refine not_holds_acceptAll hδ fun ω h => ?_
  have hf := fill_nonpos toyPace ω acceptAll.blocks (toy_pooled δc ω)
  have h1 : toyPace.stop ω - toyPace.start ω = 1 := rfl
  have h2 : acceptAll.blocks = 1 := rfl
  rw [h1, acceptAll_heldOver] at h
  rw [h2] at h hf
  norm_num at h
  linarith

end Accounting.SecurityProofs.Remote.DrawMeets
