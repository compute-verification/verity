import Proofs.Flock.Claims.KS.ForkAlign
import Proofs.Flock.Soundness.Discharge.Exec.Shape

/-!
# A slot's distance in coins, independent of the strategy

`SlotAt` asks of one strategy that it reach a slot's table node after a number of coins. Where every path of a lockstep
meets that node after the same number of coins, or never meets it, the alignment holds against every strategy:

* `NoSlot slot L`: `L` has no table node of `slot`; `forkE_noSlot`: the fork there is `0`, so `SlotAt` holds by `nul`
  (`SlotAt.of_noSlot`). `noSlot_of_get_none`: an index that never holds the slot has no such node; `noSlot_same`: the
  same game on both sides has no table node; `NoSlot.mono`, `NoSlot.bind`: weakening and sequencing keep it.
* `Dist slot j L`: every path of `L` meets the node after exactly `j` coins, or meets no such node; `SlotAt.of_dist`:
  then `SlotAt` holds at `j` against every strategy. `Dist.mono`, `Dist.bind`: weakening keeps it, and a lockstep
  with no such node and an exact shape (`Exact`) in front adds the shape's coins (`coinsK`).
-/

namespace FlockSoundness.Game.Lock

open FlockSoundness.Discharge.Exec

/-- **No table node of `slot`** in `L`. -/
inductive NoSlot {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} (slot : ℕ) :
    {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Prop
  | ret {i : I} {a : α} {b : β} (h : R i a b) : NoSlot slot (Lock.ret h)
  | coin {i : I} (C : Type) [Fintype C] [Nonempty C] {k : C → Game α} {k' : C → Game β}
      {L : (c : C) → Lock T D get R i (k c) (k' c)} :
      (∀ c, NoSlot slot (L c)) → NoSlot slot (Lock.coin C L)
  | send {i : I} (M : Type) {k : M → Game α} {k' : M → Game β} {L : (m : M) → Lock T D get R i (k m) (k' m)} :
      (∀ m, NoSlot slot (L m)) → NoSlot slot (Lock.send M L)
  | table {i : I} (s : ℕ) {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i s = none} {hs : ∀ t d, get (upd t d) s = some (t, d)}
      {hk : ∀ t d s', s' ≠ s → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} (hne : s ≠ slot) :
      (∀ t d, NoSlot slot (L t d)) → NoSlot slot (Lock.table s upd hf hs hk L)

/-- **With no table node of `slot`, the fork there is `0`.** -/
theorem forkE_noSlot {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) (Φ : D → β → β → ℝ) {i : I} {g : Game α} {g' : Game β}
    {L : Lock T D get R i g g'} (h : NoSlot slot L) (σ' : Strategy g') : forkE plur slot Φ L σ' = 0 := by
  induction h with
  | ret _ => rfl
  | coin C _ ih => simp only [forkE, ih, Finset.sum_const_zero, zero_div]
  | send M _ ih => simp only [forkE]; exact ih _ _
  | table s hne _ ih => simp only [forkE, hne, ↓reduceIte]; exact ih _ _ _

theorem SlotAt.of_noSlot {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {plur : Chooser T D β} {slot : ℕ} {Φ : D → β → β → ℝ} {j : ℕ} {i : I} {g : Game α} {g' : Game β}
    {L : Lock T D get R i g g'} (h : NoSlot slot L) (σ' : Strategy g') : SlotAt plur slot Φ j L σ' :=
  .nul (forkE_noSlot plur slot Φ h σ')

/-- **An index that never holds `slot` has no table node of it.** -/
theorem noSlot_of_get_none {T D I α β : Type} [Nonempty T] [Nonempty D] {get : I → ℕ → Option (T × D)}
    {R : I → α → β → Prop} {slot : ℕ} (hg : ∀ i, get i slot = none) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g'), NoSlot slot L
  | _, _, _, .ret hr => .ret hr
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L => .coin C fun c => noSlot_of_get_none hg (L c)
  | _, _, _, .send M L => .send M fun m => noSlot_of_get_none hg (L m)
  | _, _, _, .table s upd _ hs _ L =>
      .table s (fun e => by
          obtain ⟨t⟩ := ‹Nonempty T›
          obtain ⟨d⟩ := ‹Nonempty D›
          have := hs t d
          rw [e, hg] at this
          cases this)
        fun t d => noSlot_of_get_none hg (L t d)

/-- **The same game on both sides has no table node.** -/
theorem noSlot_same {T D I α : Type} {get : I → ℕ → Option (T × D)} {slot : ℕ} {i : I} :
    ∀ g : Game α, NoSlot slot (Lock.same (T := T) (D := D) (get := get) (i := i) g)
  | .ret _ => .ret _
  | @Game.coin _ C _ _ k => .coin C fun c => noSlot_same (k c)
  | .send M k => .send M fun m => noSlot_same (k m)

theorem NoSlot.mono {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R R' : I → α → β → Prop}
    {slot : ℕ} (h : ∀ i a b, R i a b → R' i a b) {i : I} {g : Game α} {g' : Game β}
    {L : Lock T D get R i g g'} (hL : NoSlot slot L) : NoSlot slot (Lock.mono h L) := by
  induction hL with
  | ret hr => exact .ret _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s hne _ ih => exact .table s hne ih

/-- **Sequencing keeps no table node of `slot`.** -/
theorem NoSlot.bind {T D I α β α₂ β₂ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {R₂ : I → α₂ → β₂ → Prop} {f : α → Game α₂} {f' : β → Game β₂} {slot : ℕ}
    {h : ∀ i a b, R i a b → Lock T D get R₂ i (f a) (f' b)} (hK : ∀ i a b (hr : R i a b), NoSlot slot (h i a b hr))
    {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (hL : NoSlot slot L) :
    NoSlot slot (Lock.bind h L) := by
  induction hL with
  | ret hr => exact hK _ _ _ hr
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s hne _ ih => exact .table s hne ih

/-- **Every path meets `slot`'s table node after exactly `j` coins**, or meets no such node. -/
inductive Dist {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} (slot : ℕ) :
    ℕ → {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Prop
  | coin {j : ℕ} {i : I} (C : Type) [Fintype C] [Nonempty C] {k : C → Game α} {k' : C → Game β}
      {L : (c : C) → Lock T D get R i (k c) (k' c)} :
      (∀ c, Dist slot j (L c)) → Dist slot (j + 1) (Lock.coin C L)
  | send {j : ℕ} {i : I} (M : Type) {k : M → Game α} {k' : M → Game β}
      {L : (m : M) → Lock T D get R i (k m) (k' m)} :
      (∀ m, Dist slot j (L m)) → Dist slot j (Lock.send M L)
  | other {j : ℕ} {i : I} (s : ℕ) {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i s = none} {hs : ∀ t d, get (upd t d) s = some (t, d)}
      {hk : ∀ t d s', s' ≠ s → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} (hne : s ≠ slot) :
      (∀ t d, Dist slot j (L t d)) → Dist slot j (Lock.table s upd hf hs hk L)
  | here {i : I} {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i slot = none} {hs : ∀ t d, get (upd t d) slot = some (t, d)}
      {hk : ∀ t d s', s' ≠ slot → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} :
      Dist slot 0 (Lock.table slot upd hf hs hk L)
  | noSlot {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} :
      NoSlot slot L → Dist slot j L

/-- **A distance aligns the fork against every strategy.** -/
theorem SlotAt.of_dist {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {plur : Chooser T D β} {slot : ℕ} {Φ : D → β → β → ℝ} {j : ℕ} {i : I} {g : Game α} {g' : Game β}
    {L : Lock T D get R i g g'} (h : Dist slot j L) (σ' : Strategy g') : SlotAt plur slot Φ j L σ' := by
  induction h with
  | coin C _ ih => exact .coin C fun c => ih c (σ' c)
  | send M _ ih => exact .send M (ih _ _)
  | other s hne _ ih => exact .other s hne (ih _ _ _)
  | here => exact .here σ'
  | noSlot h => exact .of_noSlot h σ'

theorem Dist.mono {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R R' : I → α → β → Prop}
    {slot : ℕ} (h : ∀ i a b, R i a b → R' i a b) {j : ℕ} {i : I} {g : Game α} {g' : Game β}
    {L : Lock T D get R i g g'} (hL : Dist slot j L) : Dist slot j (Lock.mono h L) := by
  induction hL with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (h0.mono h)

/-- The number of coins in a shape. -/
def coinsK : List Kind → ℕ
  | [] => 0
  | .msg _ :: σ => coinsK σ
  | .coin _ :: σ => coinsK σ + 1

theorem coinsK_append : ∀ σ τ : List Kind, coinsK (σ ++ τ) = coinsK σ + coinsK τ
  | [], τ => by simp [coinsK]
  | .msg _ :: σ, τ => by simp only [List.cons_append, coinsK]; exact coinsK_append σ τ
  | .coin _ :: σ, τ => by simp only [List.cons_append, coinsK]; rw [coinsK_append σ τ]; omega

/-- **A distance after an exact shape with no table node of the slot** adds the shape's coins. -/
theorem Dist.bind {T D I α β α₂ β₂ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {R₂ : I → α₂ → β₂ → Prop} {f : α → Game α₂} {f' : β → Game β₂} {slot j : ℕ}
    {h : ∀ i a b, R i a b → Lock T D get R₂ i (f a) (f' b)} (hK : ∀ i a b (hr : R i a b), Dist slot j (h i a b hr))
    {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (hL : NoSlot slot L) (σ : List Kind)
    (hσ : Exact g' σ) : Dist slot (j + coinsK σ) (Lock.bind h L) := by
  induction hL generalizing σ with
  | ret hr =>
      cases hσ
      exact hK _ _ _ hr
  | coin C _ ih =>
      cases hσ with
      | coin hk =>
          simp only [coinsK, ← Nat.add_assoc]
          exact .coin C fun c => ih c _ (hk c)
  | send M _ ih =>
      cases hσ with
      | send hk =>
          simp only [coinsK]
          exact .send M fun m => ih m _ (hk m)
  | table s hne _ ih =>
      cases hσ with
      | send hk =>
          simp only [coinsK]
          exact .other s hne fun t d => ih t d _ (hk d)

theorem Dist.bind_eq {T D I α β α₂ β₂ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {R₂ : I → α₂ → β₂ → Prop} {f : α → Game α₂} {f' : β → Game β₂} {slot j j' : ℕ}
    {h : ∀ i a b, R i a b → Lock T D get R₂ i (f a) (f' b)} (hK : ∀ i a b (hr : R i a b), Dist slot j (h i a b hr))
    {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (hL : NoSlot slot L) {σ : List Kind}
    (hσ : Exact g' σ) (hj : j' = j + coinsK σ) : Dist slot j' (Lock.bind h L) :=
  hj ▸ Dist.bind hK hL σ hσ

theorem Dist.slot_eq {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {slot slot' j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (h : slot = slot')
    (hL : Dist slot j L) : Dist slot' j L :=
  h ▸ hL

end FlockSoundness.Game.Lock
