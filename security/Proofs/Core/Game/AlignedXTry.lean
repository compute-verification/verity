module

public import Proofs.Core.Game.Lock

/-!
# Scratch: locksteps aligned across tables, and the fork measure's independence of the chooser

`AlignedX` is `Aligned` with its table case quantified over two tables. `forkE_alignedX`: on two such locksteps the fork
measure is the same for any two choosers.
-/

@[expose] public section

namespace Verity

namespace Game

namespace Lock

variable {α β T D I : Type} {get : I → ℕ → Option (T × D)}

/-- Two locksteps of one compiled game with the same table slots, whose continuations at each table node are aligned
for any two tables. -/
inductive AlignedX {R₁ R₂ : I → α → β → Prop} :
    {i₁ i₂ : I} → {g₁ g₂ : Game α} → {g' : Game β} →
      Lock T D get R₁ i₁ g₁ g' → Lock T D get R₂ i₂ g₂ g' → Prop
  | ret {i₁ i₂ : I} {a₁ a₂ : α} {b : β} (h₁ : R₁ i₁ a₁ b) (h₂ : R₂ i₂ a₂ b) :
      AlignedX (Lock.ret h₁) (Lock.ret h₂)
  | coin {i₁ i₂ : I} (C : Type) [Fintype C] [Nonempty C] {k₁ k₂ : C → Game α} {k' : C → Game β}
      {L₁ : (c : C) → Lock T D get R₁ i₁ (k₁ c) (k' c)}
      {L₂ : (c : C) → Lock T D get R₂ i₂ (k₂ c) (k' c)} :
      (∀ c, AlignedX (L₁ c) (L₂ c)) → AlignedX (Lock.coin C L₁) (Lock.coin C L₂)
  | send {i₁ i₂ : I} (M : Type) {k₁ k₂ : M → Game α} {k' : M → Game β}
      {L₁ : (m : M) → Lock T D get R₁ i₁ (k₁ m) (k' m)}
      {L₂ : (m : M) → Lock T D get R₂ i₂ (k₂ m) (k' m)} :
      (∀ m, AlignedX (L₁ m) (L₂ m)) → AlignedX (Lock.send M L₁) (Lock.send M L₂)
  | table {i₁ i₂ : I} (slot : ℕ) {upd₁ upd₂ : T → D → I} {k₁ k₂ : T → Game α} {k' : D → Game β}
      {hf₁ : get i₁ slot = none} {hs₁ : ∀ t d, get (upd₁ t d) slot = some (t, d)}
      {hk₁ : ∀ t d s, s ≠ slot → get (upd₁ t d) s = get i₁ s}
      {hf₂ : get i₂ slot = none} {hs₂ : ∀ t d, get (upd₂ t d) slot = some (t, d)}
      {hk₂ : ∀ t d s, s ≠ slot → get (upd₂ t d) s = get i₂ s}
      {L₁ : (t : T) → (d : D) → Lock T D get R₁ (upd₁ t d) (k₁ t) (k' d)}
      {L₂ : (t : T) → (d : D) → Lock T D get R₂ (upd₂ t d) (k₂ t) (k' d)} :
      (∀ t₁ t₂ d, AlignedX (L₁ t₁ d) (L₂ t₂ d)) →
        AlignedX (Lock.table slot upd₁ hf₁ hs₁ hk₁ L₁) (Lock.table slot upd₂ hf₂ hs₂ hk₂ L₂)

theorem AlignedX.toAligned {R₁ R₂ : I → α → β → Prop} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : AlignedX L₁ L₂) :
    Aligned L₁ L₂ := by
  induction h with
  | ret h₁ h₂ => exact .ret h₁ h₂
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s fun t d => ih t t d

/-- **The fork measure does not read the chooser on locksteps aligned across tables.** -/
theorem forkE_alignedX (plur₁ plur₂ : Chooser T D β) (slot : ℕ) (Φ : D → β → β → ℝ)
    {R₁ R₂ : I → α → β → Prop} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : AlignedX L₁ L₂) :
    ∀ σ' : Strategy g', forkE plur₁ slot Φ L₁ σ' = forkE plur₂ slot Φ L₂ σ' := by
  induction h with
  | ret => intro σ'; rfl
  | coin C _ ih =>
      intro σ'
      simp only [forkE]
      exact congrArg (· / _) (Finset.sum_congr rfl fun c _ => ih c (σ' c))
  | send M _ ih => intro σ'; exact ih σ'.1 σ'.2
  | table s _ ih =>
      intro σ'
      simp only [forkE]
      split_ifs
      · rfl
      · exact ih _ _ σ'.1 σ'.2

/-- At one lockstep aligned with itself across tables, any chooser gives the constant chooser's fork measure. -/
theorem forkE_const {R : I → α → β → Prop} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'}
    (h : AlignedX L L) (plur : Chooser T D β) (t₀ : T) (slot : ℕ) (Φ : D → β → β → ℝ) (σ' : Strategy g') :
    forkE plur slot Φ L σ' = forkE (fun _ _ _ _ => t₀) slot Φ L σ' :=
  forkE_alignedX plur _ slot Φ h σ'

end Lock

end Game

end Verity
