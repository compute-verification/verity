module

public import Batteries.Data.ByteArray
public import Definitions.Flock.Game
public import Proofs.Flock.Claims.KS.ForkSim
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Soundness.Game
import VCVio.Prelude.Core

/-! A fork at a table node (`Lock.forkE`) is at most the fork after the coins on which the strategy reaches that node (`forkC`). -/

@[expose] public section

namespace FlockSoundness.Game.Lock

theorem expect_le_of_get {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) (td : T × D) {f f' : β → ℝ}
    (h : ∀ i a b, R i a b → get i slot = some td → f b ≤ f' b) :
    ∀ {i : I} {g : Game α} {g' : Game β}, Lock T D get R i g g' → ∀ σ' : Strategy g',
      get i slot = some td → expect f g' σ' ≤ expect f' g' σ'
  | _, _, _, .ret hr, _, hi => h _ _ _ hr hi
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi => by
      simp only [expect]
      exact div_le_div_of_nonneg_right
        (Finset.sum_le_sum fun c _ => expect_le_of_get plur slot td h (L c) (σ' c) hi) (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ', hi => expect_le_of_get plur slot td h (L σ'.1) σ'.2 hi
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' hf _ hk L, σ', hi => by
      have hs : slot ≠ s := fun e => by rw [← e, hi] at hf; exact Option.some_ne_none _ hf
      exact expect_le_of_get plur slot td h (L (plur s σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2 (by rw [hk _ _ slot hs, hi])

theorem expect₂_le_of_get {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) (td : T × D) {F F' : β → β → ℝ}
    (h : ∀ i₁ a₁ b₁ i₂ a₂ b₂, R i₁ a₁ b₁ → R i₂ a₂ b₂ → get i₁ slot = some td → get i₂ slot = some td →
      F b₁ b₂ ≤ F' b₁ b₂)
    {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g') (hi : get i slot = some td) :
    expect₂ F g' σ' ≤ expect₂ F' g' σ' := by
  unfold expect₂
  refine expect_le_of_get (get := get) (R := R) plur slot td (fun i₁ a₁ b₁ h₁ hi₁ => ?_) L σ' hi
  exact expect_le_of_get (get := get) (R := R) plur slot td
    (fun i₂ a₂ b₂ h₂ hi₂ => h i₁ a₁ b₁ i₂ a₂ b₂ h₁ h₂ hi₁ hi₂) L σ' hi

inductive SlotAt {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) (Φ : D → β → β → ℝ) :
    ℕ → {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Strategy g' → Prop
  | ret {j : ℕ} {i : I} {a : α} {b : β} (h : R i a b) (σ' : Strategy (.ret b)) :
      SlotAt plur slot Φ j (Lock.ret h) σ'
  | coin {j : ℕ} {i : I} (C : Type) [Fintype C] [Nonempty C] {k : C → Game α} {k' : C → Game β}
      {L : (c : C) → Lock T D get R i (k c) (k' c)} {σ' : Strategy (.coin C k')} :
      (∀ c, SlotAt plur slot Φ j (L c) (σ' c)) → SlotAt plur slot Φ (j + 1) (Lock.coin C L) σ'
  | send {j : ℕ} {i : I} (M : Type) {k : M → Game α} {k' : M → Game β}
      {L : (m : M) → Lock T D get R i (k m) (k' m)} {σ' : Strategy (.send M k')} :
      SlotAt plur slot Φ j (L σ'.1) σ'.2 → SlotAt plur slot Φ j (Lock.send M L) σ'
  | other {j : ℕ} {i : I} (s : ℕ) {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i s = none} {hs : ∀ t d, get (upd t d) s = some (t, d)}
      {hk : ∀ t d s', s' ≠ s → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} {σ' : Strategy (.send D k')}
      (hne : s ≠ slot) :
      SlotAt plur slot Φ j (L (plur s σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2 →
        SlotAt plur slot Φ j (Lock.table s upd hf hs hk L) σ'
  | here {i : I} {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i slot = none} {hs : ∀ t d, get (upd t d) slot = some (t, d)}
      {hk : ∀ t d s', s' ≠ slot → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} (σ' : Strategy (.send D k')) :
      SlotAt plur slot Φ 0 (Lock.table slot upd hf hs hk L) σ'
  | nul {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} :
      forkE plur slot Φ L σ' = 0 → SlotAt plur slot Φ j L σ'

theorem forkE_le_forkC {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) {Φ : D → β → β → ℝ} {F : β → β → ℝ} (hF : ∀ b b', 0 ≤ F b b')
    (hΦ : ∀ t d i₁ a₁ b₁ i₂ a₂ b₂, R i₁ a₁ b₁ → R i₂ a₂ b₂ → get i₁ slot = some (t, d) →
      get i₂ slot = some (t, d) → Φ d b₁ b₂ ≤ F b₁ b₂)
    {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'}
    (h : SlotAt plur slot Φ j L σ') : forkE plur slot Φ L σ' ≤ forkC F j g' σ' := by
  induction h with
  | ret _ _ => exact forkC_nonneg hF _ _ _
  | coin C _ ih =>
      rw [forkC_coin]
      simp only [forkE]
      exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => ih c) (Nat.cast_nonneg _)
  | send M _ ih => rw [forkC_send]; exact ih
  | other s hne _ ih =>
      simp only [forkE, hne, ↓reduceIte]
      rw [forkC_send]
      exact ih
  | @here i upd k k' hf hs hk L σ' =>
      simp only [forkE, ↓reduceIte]
      rw [forkC_zero]
      exact expect₂_le_of_get (get := get) (R := R) plur slot (plur slot σ'.1 (k' σ'.1) σ'.2, σ'.1)
        (fun i₁ a₁ b₁ i₂ a₂ b₂ h₁ h₂ hi₁ hi₂ => hΦ _ _ i₁ a₁ b₁ i₂ a₂ b₂ h₁ h₂ hi₁ hi₂)
        (L (plur slot σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2 (hs _ _)
  | nul h0 => rw [h0]; exact forkC_nonneg hF _ _ _

theorem Aligned.mono_self {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R R' : I → α → β → Prop}
    (h : ∀ i a b, R i a b → R' i a b) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g'), Aligned (Lock.mono h L) L
  | _, _, _, .ret hr => .ret (h _ _ _ hr) hr
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L => .coin C fun c => Aligned.mono_self h (L c)
  | _, _, _, .send M L => .send M fun m => Aligned.mono_self h (L m)
  | _, _, _, .table slot _ _ _ _ L => .table slot fun t d => Aligned.mono_self h (L t d)

theorem forkE_mapL {T D I α α₂ β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {R₂ : I → α₂ → β → Prop} (plur : Chooser T D β) (slot : ℕ) (Φ : D → β → β → ℝ) (f : α → α₂)
    (h : ∀ i a b, R i a b → R₂ i (f a) b) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      forkE plur slot Φ (Lock.mapL f h L) σ' = forkE plur slot Φ L σ'
  | _, _, _, .ret _, _ => rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [Lock.mapL, forkE]
      exact congrArg (· / _) (Finset.sum_congr rfl fun c _ => forkE_mapL plur slot Φ f h (L c) (σ' c))
  | _, _, _, .send _ L, σ' => by
      simp only [Lock.mapL, forkE]
      exact forkE_mapL plur slot Φ f h (L σ'.1) σ'.2
  | _, _, _, .table s _ _ _ _ L, σ' => by
      simp only [Lock.mapL, forkE]
      split_ifs
      · rfl
      · exact forkE_mapL plur slot Φ f h _ σ'.2

theorem SlotAt.mono {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R R' : I → α → β → Prop}
    {plur : Chooser T D β} {slot : ℕ} {Φ : D → β → β → ℝ} (h : ∀ i a b, R i a b → R' i a b) {j : ℕ} {i : I}
    {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} (hs : SlotAt plur slot Φ j L σ') :
    SlotAt plur slot Φ j (Lock.mono h L) σ' := by
  induction hs with
  | ret hr σ' => exact .ret (h _ _ _ hr) σ'
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here σ' => exact .here σ'
  | nul h0 => exact .nul ((forkE_congr _ _ _ (Aligned.mono_self h _) _).trans h0)

theorem SlotAt.mapL {T D I α α₂ β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {R₂ : I → α₂ → β → Prop} {plur : Chooser T D β} {slot : ℕ} {Φ : D → β → β → ℝ} (f : α → α₂)
    (h : ∀ i a b, R i a b → R₂ i (f a) b) {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'}
    {σ' : Strategy g'} (hs : SlotAt plur slot Φ j L σ') : SlotAt plur slot Φ j (Lock.mapL f h L) σ' := by
  induction hs with
  | ret hr σ' => exact .ret (h _ _ _ hr) σ'
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here σ' => exact .here σ'
  | nul h0 => exact .nul ((forkE_mapL _ _ _ f h _ _).trans h0)

def plurMap {T D β β₂ : Type} (f' : β → β₂) (plur : Chooser T D β₂) : Chooser T D β :=
  fun s d g σ' => plur s d (g.map f') (Strategy.map f' g σ')

theorem expect₂_map {β β₂ : Type} (F : β₂ → β₂ → ℝ) (f' : β → β₂) (g : Game β) (σ' : Strategy g) :
    expect₂ F (g.map f') (Strategy.map f' g σ') = expect₂ (fun b b' => F (f' b) (f' b')) g σ' := by
  unfold expect₂
  rw [expect_map]
  simp only [expect_map]

theorem forkE_map {T D I α α₂ β β₂ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {R₂ : I → α₂ → β₂ → Prop} (plur : Chooser T D β₂) (slot : ℕ) (Φ : D → β₂ → β₂ → ℝ) (f : α → α₂)
    (f' : β → β₂) (h : ∀ i a b, R i a b → R₂ i (f a) (f' b)) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      forkE plur slot Φ (Lock.map f f' h L) (Strategy.map f' g' σ') =
        forkE (plurMap f' plur) slot (fun d b b' => Φ d (f' b) (f' b')) L σ'
  | _, _, _, .ret _, _ => rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [Lock.map, Lock.bind, Strategy.map, Strategy.bind, forkE]
      exact congrArg (· / _) (Finset.sum_congr rfl fun c _ => forkE_map plur slot Φ f f' h (L c) (σ' c))
  | _, _, _, .send _ L, σ' => by
      simp only [Lock.map, Lock.bind, Strategy.map, Strategy.bind, forkE]
      exact forkE_map plur slot Φ f f' h (L σ'.1) σ'.2
  | _, _, _, .table s _ _ _ _ L, σ' => by
      simp only [Lock.map, Lock.bind, Strategy.map, Strategy.bind, forkE]
      split_ifs
      · exact expect₂_map _ f' _ σ'.2
      · exact forkE_map plur slot Φ f f' h _ σ'.2

theorem SlotAt.map {T D I α α₂ β β₂ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {R₂ : I → α₂ → β₂ → Prop} {plur : Chooser T D β₂} {slot : ℕ} {Φ : D → β₂ → β₂ → ℝ} (f : α → α₂)
    (f' : β → β₂) (h : ∀ i a b, R i a b → R₂ i (f a) (f' b)) {j : ℕ} {i : I} {g : Game α} {g' : Game β}
    {L : Lock T D get R i g g'} {σ' : Strategy g'}
    (hs : SlotAt (plurMap f' plur) slot (fun d b b' => Φ d (f' b) (f' b')) j L σ') :
    SlotAt plur slot Φ j (Lock.map f f' h L) (Strategy.map f' g' σ') := by
  induction hs with
  | ret hr σ' => exact .ret (h _ _ _ hr) _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here σ' => exact .here _
  | nul h0 => exact .nul ((forkE_map _ _ _ f f' h _ _).trans h0)

theorem Strategy.map_ofMap {β β₂ : Type} (f' : β → β₂) :
    ∀ (g : Game β) (σ' : Strategy (g.map f')), Strategy.map f' g (Strategy.ofMap f' g σ') = σ'
  | .ret _, _ => rfl
  | .send _ k, σ' => by
      show (⟨σ'.1, Strategy.map f' (k σ'.1) (Strategy.ofMap f' (k σ'.1) σ'.2)⟩ :
        Σ m, Strategy ((k m).map f')) = σ'
      rw [Strategy.map_ofMap f' (k σ'.1) σ'.2]
      rfl
  | @Game.coin _ _ _ _ k, σ' => funext fun c => Strategy.map_ofMap f' (k c) (σ' c)

theorem interleave_ret {α γ : Type} (c : γ) : ∀ g : Game α, interleave g (.ret c) = g.map (·, c)
  | .ret _ => rfl
  | .send _ _ => rfl
  | @Game.coin _ _ _ _ _ => rfl

theorem SlotAt.leftSame_ret {T D I α β γ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {plur : Chooser T D (β × γ)} {slot : ℕ} {Φ : D → β × γ → β × γ → ℝ} {j : ℕ} {i : I} {g : Game α}
    {g' : Game β} (c : γ) (L : Lock T D get R i g g') (σ' : Strategy (interleave g' (.ret c)))
    (hs : SlotAt (plurMap (·, c) plur) slot (fun d b b' => Φ d (b, c) (b', c)) j L
      (Strategy.ofMap (·, c) g' (cast (congrArg Strategy (interleave_ret c g')) σ'))) :
    SlotAt plur slot Φ j (leftSame L (.ret c)) σ' := by
  have key := SlotAt.map (R₂ := fun i a b => R i a.1 b.1 ∧ a.2 = b.2) (fun a => (a, c)) (fun b => (b, c))
    (fun _ _ _ hr => ⟨hr, rfl⟩) hs
  rw [Strategy.map_ofMap] at key
  cases L <;> exact key

theorem SlotAt.batchLock_one {T D I ζ : Type} {get : I → ℕ → Option (T × D)} {R : I → ζ → ζ → Prop}
    {plur : Chooser T D (List ζ)} {slot : ℕ} {Φ : D → List ζ → List ζ → ℝ} {j : ℕ} {i : I} {g g' : Game ζ}
    (d : ζ) (L : Lock T D get R i g g') (σ' : Strategy (batch [g']))
    (hs : SlotAt (plurMap (·, []) (plurMap (fun x : ζ × List ζ => x.1 :: x.2) plur)) slot
      (fun e b b' => Φ e [b] [b']) j L
      (Strategy.ofMap (·, []) g' (cast (congrArg Strategy (interleave_ret [] g'))
        (Strategy.ofMap (fun x : ζ × List ζ => x.1 :: x.2) (g'.interleave (.ret [])) σ')))) :
    SlotAt plur slot Φ j (batchLock d L [] []) σ' := by
  have key := SlotAt.map (R₂ := fun i a b => R i (a.getD 0 d) (b.getD 0 d))
    (fun x : ζ × List ζ => x.1 :: x.2) (fun x : ζ × List ζ => x.1 :: x.2) (fun _ _ _ h => h.1)
    (SlotAt.leftSame_ret [] L _ hs)
  exact Strategy.map_ofMap (fun x : ζ × List ζ => x.1 :: x.2) (g'.interleave (.ret [])) σ' ▸ key

theorem answerWith_send {β Op M : Type} (nul : Op) {k : M → Game β} (hg : AllInh (.send M k))
    (σ : Strategy (.send M k)) : ∃ (m : M) (τ : Strategy (k m)), (∀ e : Op = M, m = cast e nul) ∧
      Claims.Bridge.answerWith nul (.send M k) σ = ⟨m, Claims.Bridge.answerWith nul (k m) τ⟩ := by
  by_cases hc : ∃ e : Op = M, Nonempty (Strategy (k (cast e nul)))
  · refine ⟨cast hc.fst nul, Classical.choice hc.snd, fun _ => rfl, ?_⟩
    rw [Claims.Bridge.answerWith, dite_eq_left_of_eq_true (eq_true hc)]
  · refine ⟨σ.1, σ.2, fun e => absurd ⟨e, (hg.send_k _).strategy⟩ hc, ?_⟩
    rw [Claims.Bridge.answerWith, dite_eq_right_of_eq_false (eq_false hc)]

theorem expect_answerWith {α Op : Type} (nul : Op) {f : α → ℝ} :
    ∀ {g : Game α}, AllInh g → (∀ a, Claims.Bridge.OutN nul g a → f a = 0) → ∀ σ : Strategy g,
      expect f g (Claims.Bridge.answerWith nul g σ) = 0
  | .ret a, _, h, _ => h a (.ret a)
  | .send M k, hg, h, σ => by
      obtain ⟨m, τ, hm, he⟩ := answerWith_send nul hg σ
      rw [he]
      exact expect_answerWith nul (hg.send_k m) (fun a ha => h a (.send m hm ha)) τ
  | @Game.coin _ C _ _ k, hg, h, σ => by
      simp only [Claims.Bridge.answerWith, expect]
      rw [Finset.sum_eq_zero fun c _ =>
        expect_answerWith nul (hg.coin_k c) (fun a ha => h a (.coin c ha)) (σ c), zero_div]

theorem expect₂_answerWith {α Op : Type} (nul : Op) {F : α → α → ℝ} {g : Game α} (hg : AllInh g)
    (h : ∀ a a', Claims.Bridge.OutN nul g a → Claims.Bridge.OutN nul g a' → F a a' = 0) (σ : Strategy g) :
    expect₂ F g (Claims.Bridge.answerWith nul g σ) = 0 :=
  expect_answerWith nul hg (fun a ha => expect_answerWith nul hg (fun a' ha' => h a a' ha ha') σ) σ

theorem forkE_answerWith {T D I α β Op : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) {Φ : D → β → β → ℝ} (nul : Op) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g'), AllInh g' →
      (∀ d b b', Claims.Bridge.OutN nul g' b → Claims.Bridge.OutN nul g' b' → Φ d b b' = 0) →
      ∀ σ : Strategy g', forkE plur slot Φ L (Claims.Bridge.answerWith nul g' σ) = 0
  | _, _, _, .ret _, _, _, _ => rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, hg, hΦ, σ => by
      simp only [Claims.Bridge.answerWith, forkE]
      rw [Finset.sum_eq_zero fun c _ => forkE_answerWith plur slot nul (L c) (hg.coin_k c)
        (fun d b b' hb hb' => hΦ d b b' (.coin c hb) (.coin c hb')) (σ c), zero_div]
  | _, _, _, .send M L, hg, hΦ, σ => by
      obtain ⟨m, τ, hm, he⟩ := answerWith_send nul hg σ
      rw [he]
      exact forkE_answerWith plur slot nul (L m) (hg.send_k m)
        (fun d b b' hb hb' => hΦ d b b' (.send m hm hb) (.send m hm hb')) τ
  | _, _, _, .table s _ _ _ _ L, hg, hΦ, σ => by
      obtain ⟨m, τ, hm, he⟩ := answerWith_send nul hg σ
      rw [he]
      simp only [forkE]
      split_ifs
      · exact expect₂_answerWith nul (hg.send_k m)
          (fun b b' hb hb' => hΦ m b b' (.send m hm hb) (.send m hm hb')) τ
      · exact forkE_answerWith plur slot nul _ (hg.send_k m)
          (fun d b b' hb hb' => hΦ d b b' (.send m hm hb) (.send m hm hb')) τ

end FlockSoundness.Game.Lock
