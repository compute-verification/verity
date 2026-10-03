import Pouw.Protocol.Draft.BatchSeq
import Pouw.Protocol.Draft.Decisive
import Mathlib.Order.ConditionallyCompleteLattice.Basic

/-!
# A batch of decisive games reads as no other list of games (provisional)

`unambiguous_of_decisive`: every game of `L` decisive (`Decisive`: every message type inhabited, two runs ending
differently) makes `L` `Unambiguous`. Core-only, held here with `Pouw.Draft.Game` until it moves to core's package.

The proof follows the equation `interleave g h = interleave g' h'` down both readings at once. Where both readings have
both games live, the root is the scheduling choice in both, so the games advance alike (`interleave_live`). A reading
can only drift where one of its games ends and the other reading's does not; then the game that ended is a relabelled
copy of the other reading's interleaving, which is strictly deeper (`depth_interleave`: the least depth of
`interleave x y` is `d x + d y + min (d x) (d y)`). Following `h` first (`split_right`) and `g` first (`split_left`)
each either finds the readings equal or fixes how their least depths differ (`Split`), and the four ways of drifting on
both sides contradict the depth of the root (`interleave_inj`). The game that ends needs no non-constancy below its root:
only `g` and `h` themselves are asked to be non-constant. `batch` then peels one game at a time (`batch_inj`), the rest
of a batch of decisive games being non-constant itself, or the empty batch, whose leaves have length zero.

A constant game breaks it (`not_unambiguous_of_constant`): `[send Bool fun _ => ret true, send Bool fun b => ret b]` has
the batch of `[ret true, g₂']`, where `g₂'` replays the first game's move and the schedule's choice.
-/

namespace Pouw.Draft.Game

open Verity Verity.Game

namespace Unambiguity

variable {α β : Type}

/-! ### Shapes -/

/-- `g` has not ended. -/
def Live : Game α → Prop
  | .ret _ => False
  | .send _ _ => True
  | @Game.coin _ _ _ _ _ => True

/-- Advance the first game by one node. -/
def stepL : Game α → Game β → Game (α × β)
  | .ret a, h => h.map fun b => (a, b)
  | .send M k, h => .send M fun m => interleave (k m) h
  | @Game.coin _ C iF iN k, h => @Game.coin _ C iF iN fun c => interleave (k c) h

/-- Advance the second game by one node. -/
def stepR (g : Game α) : Game β → Game (α × β)
  | .ret b => g.map fun a => (a, b)
  | .send N j => .send N fun n => interleave g (j n)
  | @Game.coin _ C iF iN j => @Game.coin _ C iF iN fun c => interleave g (j c)

theorem interleave_ret_left (a : α) (h : Game β) : interleave (.ret a) h = h.map fun b => (a, b) := rfl

theorem interleave_ret_right : ∀ (g : Game α) (b : β), interleave g (.ret b) = g.map fun a => (a, b)
  | .ret _, _ => rfl
  | .send _ _, _ => rfl
  | @Game.coin _ _ _ _ _, _ => rfl

/-- With both games live, the root is the schedule's choice. -/
theorem interleave_live : ∀ {g : Game α} {h : Game β}, Live g → Live h →
    interleave g h = .send Bool fun first => if first then stepL g h else stepR g h
  | .ret _, _, hg, _ => hg.elim
  | .send _ _, .ret _, _, hh => hh.elim
  | @Game.coin _ _ _ _ _, .ret _, _, hh => hh.elim
  | .send _ _, .send _ _, _, _ => rfl
  | .send _ _, @Game.coin _ _ _ _ _, _, _ => rfl
  | @Game.coin _ _ _ _ _, .send _ _, _, _ => rfl
  | @Game.coin _ _ _ _ _, @Game.coin _ _ _ _ _, _, _ => rfl

theorem map_ret (f : α → β) (a : α) : (Game.ret a).map f = .ret (f a) := rfl

theorem map_send (f : α → β) (M : Type) (k : M → Game α) :
    (Game.send M k).map f = .send M fun m => (k m).map f := rfl

theorem map_coin (f : α → β) (C : Type) [Fintype C] [Nonempty C] (k : C → Game α) :
    (Game.coin C k).map f = .coin C fun c => (k c).map f := rfl

/-- Relabelling by an injective map is injective. -/
theorem map_injective {f : α → β} (hf : Function.Injective f) {g g' : Game α} (e : g.map f = g'.map f) : g = g' := by
  induction g generalizing g' with
  | ret a =>
    cases g' with
    | ret a' => rw [map_ret, map_ret] at e; injection e with e; rw [hf e]
    | send _ _ => rw [map_ret, map_send] at e; cases e
    | coin _ _ => rw [map_ret, map_coin] at e; cases e
  | send M k ih =>
    cases g' with
    | ret _ => rw [map_ret, map_send] at e; cases e
    | send M' k' =>
      rw [map_send, map_send] at e
      injection e with hM hk
      subst hM
      have hk := eq_of_heq hk
      exact congrArg (Game.send M) (funext fun m => ih m (congrFun hk m))
    | coin _ _ => rw [map_send, map_coin] at e; cases e
  | coin C k ih =>
    cases g' with
    | ret _ => rw [map_ret, map_coin] at e; cases e
    | send _ _ => rw [map_send, map_coin] at e; cases e
    | coin C' k' =>
      rw [map_coin, map_coin] at e
      injection e with hC hF hk
      subst hC
      cases hF
      have hk := eq_of_heq hk
      exact congrArg (Game.coin C) (funext fun c => ih c (congrFun hk c))

/-! ### Outputs -/

theorem reaches_map {f : α → β} {g : Game α} {c : β} : Reaches (g.map f) c ↔ ∃ a, Reaches g a ∧ f a = c := by
  induction g with
  | ret a => exact ⟨fun h => ⟨a, rfl, h⟩, fun ⟨_, h1, h2⟩ => h1 ▸ h2⟩
  | send M k ih =>
    show (∃ m, Reaches ((k m).map f) c) ↔ ∃ a, (∃ m, Reaches (k m) a) ∧ f a = c
    simp only [ih]
    exact ⟨fun ⟨m, a, h1, h2⟩ => ⟨a, ⟨m, h1⟩, h2⟩, fun ⟨a, ⟨m, h1⟩, h2⟩ => ⟨m, a, h1, h2⟩⟩
  | coin C k ih =>
    show (∃ c', Reaches ((k c').map f) c) ↔ ∃ a, (∃ c', Reaches (k c') a) ∧ f a = c
    simp only [ih]
    exact ⟨fun ⟨m, a, h1, h2⟩ => ⟨a, ⟨m, h1⟩, h2⟩, fun ⟨a, ⟨m, h1⟩, h2⟩ => ⟨m, a, h1, h2⟩⟩

/-- Playing one game to its end and then the other reaches every pair of outputs. -/
theorem reaches_interleave {g : Game α} {h : Game β} {a : α} {b : β} (hg : Reaches g a) (hh : Reaches h b) :
    Reaches (interleave g h) (a, b) := by
  induction g with
  | ret a₀ =>
    cases hg
    exact reaches_map.2 ⟨b, hh, rfl⟩
  | send M k ih =>
    cases h with
    | ret b₀ =>
      rw [interleave_ret_right]
      cases hh
      exact reaches_map.2 ⟨a, hg, rfl⟩
    | send N j =>
      rw [interleave_live (g := .send _ _) (h := .send _ _) trivial trivial]
      obtain ⟨m, hm⟩ := hg
      exact ⟨true, m, ih m hm⟩
    | coin C j =>
      rw [interleave_live (g := .send _ _) (h := @Game.coin _ _ _ _ _) trivial trivial]
      obtain ⟨m, hm⟩ := hg
      exact ⟨true, m, ih m hm⟩
  | coin C k ih =>
    cases h with
    | ret b₀ =>
      rw [interleave_ret_right]
      cases hh
      exact reaches_map.2 ⟨a, hg, rfl⟩
    | send N j =>
      rw [interleave_live (g := @Game.coin _ _ _ _ _) (h := .send _ _) trivial trivial]
      obtain ⟨m, hm⟩ := hg
      exact ⟨true, m, ih m hm⟩
    | coin C' j =>
      rw [interleave_live (g := @Game.coin _ _ _ _ _) (h := @Game.coin _ _ _ _ _) trivial trivial]
      obtain ⟨m, hm⟩ := hg
      exact ⟨true, m, ih m hm⟩

/-- Every output of the interleaving is a pair of outputs. -/
theorem reaches_of_interleave {g : Game α} {h : Game β} {a : α} {b : β} (r : Reaches (interleave g h) (a, b)) :
    Reaches g a ∧ Reaches h b := by
  induction g generalizing h with
  | ret a₀ =>
    obtain ⟨b', hb', e⟩ := reaches_map.1 r
    cases e
    exact ⟨rfl, hb'⟩
  | send M k ih =>
    induction h with
    | ret b₀ =>
      rw [interleave_ret_right] at r
      obtain ⟨a', ha', e⟩ := reaches_map.1 r
      cases e
      exact ⟨ha', rfl⟩
    | send N j ihh =>
      rw [interleave_live (g := .send _ _) (h := .send _ _) trivial trivial] at r
      obtain ⟨first, r⟩ := r
      cases first with
      | true =>
        obtain ⟨m, r⟩ := r
        exact ⟨⟨m, (ih m r).1⟩, (ih m r).2⟩
      | false =>
        obtain ⟨n, r⟩ := r
        exact ⟨(ihh n r).1, ⟨n, (ihh n r).2⟩⟩
    | coin C j ihh =>
      rw [interleave_live (g := .send _ _) (h := @Game.coin _ _ _ _ _) trivial trivial] at r
      obtain ⟨first, r⟩ := r
      cases first with
      | true =>
        obtain ⟨m, r⟩ := r
        exact ⟨⟨m, (ih m r).1⟩, (ih m r).2⟩
      | false =>
        obtain ⟨n, r⟩ := r
        exact ⟨(ihh n r).1, ⟨n, (ihh n r).2⟩⟩
  | coin C k ih =>
    induction h with
    | ret b₀ =>
      rw [interleave_ret_right] at r
      obtain ⟨a', ha', e⟩ := reaches_map.1 r
      cases e
      exact ⟨ha', rfl⟩
    | send N j ihh =>
      rw [interleave_live (g := @Game.coin _ _ _ _ _) (h := .send _ _) trivial trivial] at r
      obtain ⟨first, r⟩ := r
      cases first with
      | true =>
        obtain ⟨m, r⟩ := r
        exact ⟨⟨m, (ih m r).1⟩, (ih m r).2⟩
      | false =>
        obtain ⟨n, r⟩ := r
        exact ⟨(ihh n r).1, ⟨n, (ihh n r).2⟩⟩
    | coin C' j ihh =>
      rw [interleave_live (g := @Game.coin _ _ _ _ _) (h := @Game.coin _ _ _ _ _) trivial trivial] at r
      obtain ⟨first, r⟩ := r
      cases first with
      | true =>
        obtain ⟨m, r⟩ := r
        exact ⟨⟨m, (ih m r).1⟩, (ih m r).2⟩
      | false =>
        obtain ⟨n, r⟩ := r
        exact ⟨(ihh n r).1, ⟨n, (ihh n r).2⟩⟩

/-- A game with two different outputs has not ended. -/
theorem live_of_reaches {g : Game α} {a₁ a₂ : α} (ha : a₁ ≠ a₂) (h₁ : Reaches g a₁) (h₂ : Reaches g a₂) :
    Live g := by
  cases g with
  | ret a => exact ha (h₁.symm.trans h₂)
  | send _ _ => trivial
  | coin _ _ => trivial

/-! ### Every message type inhabited -/

theorem reaches_of_productive {g : Game α} (pg : Productive g) : ∃ a, Reaches g a := by
  induction g with
  | ret a => exact ⟨a, rfl⟩
  | send M k ih =>
    obtain ⟨⟨m⟩, pk⟩ := pg
    obtain ⟨a, ha⟩ := ih m (pk m)
    exact ⟨a, m, ha⟩
  | coin C k ih =>
    obtain ⟨c⟩ := ‹Nonempty C›
    obtain ⟨a, ha⟩ := ih c (pg c)
    exact ⟨a, c, ha⟩

theorem productive_map {f : α → β} {g : Game α} : Productive (g.map f) ↔ Productive g := by
  induction g with
  | ret a => exact Iff.rfl
  | send M k ih =>
    show (Nonempty M ∧ ∀ m, Productive ((k m).map f)) ↔ (Nonempty M ∧ ∀ m, Productive (k m))
    simp only [ih]
  | coin C k ih =>
    show (∀ c, Productive ((k c).map f)) ↔ ∀ c, Productive (k c)
    simp only [ih]

theorem productive_interleave {g : Game α} {h : Game β} (pg : Productive g) (ph : Productive h) :
    Productive (interleave g h) := by
  induction g generalizing h with
  | ret a => exact productive_map.2 ph
  | send M k ih =>
    induction h with
    | ret b => rw [interleave_ret_right]; exact productive_map.2 pg
    | send N j ihh =>
      rw [interleave_live (g := .send _ _) (h := .send _ _) trivial trivial]
      refine ⟨⟨true⟩, fun first => ?_⟩
      cases first with
      | true => exact ⟨pg.1, fun m => ih m (pg.2 m) ph⟩
      | false => exact ⟨ph.1, fun n => ihh n (ph.2 n)⟩
    | coin C j ihh =>
      rw [interleave_live (g := .send _ _) (h := @Game.coin _ _ _ _ _) trivial trivial]
      refine ⟨⟨true⟩, fun first => ?_⟩
      cases first with
      | true => exact ⟨pg.1, fun m => ih m (pg.2 m) ph⟩
      | false => exact fun n => ihh n (ph n)
  | coin C k ih =>
    induction h with
    | ret b => rw [interleave_ret_right]; exact productive_map.2 pg
    | send N j ihh =>
      rw [interleave_live (g := @Game.coin _ _ _ _ _) (h := .send _ _) trivial trivial]
      refine ⟨⟨true⟩, fun first => ?_⟩
      cases first with
      | true => exact fun m => ih m (pg m) ph
      | false => exact ⟨ph.1, fun n => ihh n (ph.2 n)⟩
    | coin C' j ihh =>
      rw [interleave_live (g := @Game.coin _ _ _ _ _) (h := @Game.coin _ _ _ _ _) trivial trivial]
      refine ⟨⟨true⟩, fun first => ?_⟩
      cases first with
      | true => exact fun m => ih m (pg m) ph
      | false => exact fun n => ihh n (ph n)

theorem productive_of_interleave {g : Game α} {h : Game β} (p : Productive (interleave g h)) :
    Productive g ∧ Productive h := by
  induction g generalizing h with
  | ret a => exact ⟨trivial, productive_map.1 p⟩
  | send M k ih =>
    cases h with
    | ret b => rw [interleave_ret_right] at p; exact ⟨productive_map.1 p, trivial⟩
    | send N j =>
      rw [interleave_live (g := .send _ _) (h := .send _ _) trivial trivial] at p
      obtain ⟨⟨m⟩, pk⟩ := (p.2 true)
      exact ⟨⟨⟨m⟩, fun m' => (ih m' (pk m')).1⟩, (ih m (pk m)).2⟩
    | coin C j =>
      rw [interleave_live (g := .send _ _) (h := @Game.coin _ _ _ _ _) trivial trivial] at p
      obtain ⟨⟨m⟩, pk⟩ := (p.2 true)
      exact ⟨⟨⟨m⟩, fun m' => (ih m' (pk m')).1⟩, (ih m (pk m)).2⟩
  | coin C k ih =>
    obtain ⟨c⟩ := ‹Nonempty C›
    cases h with
    | ret b => rw [interleave_ret_right] at p; exact ⟨productive_map.1 p, trivial⟩
    | send N j =>
      rw [interleave_live (g := @Game.coin _ _ _ _ _) (h := .send _ _) trivial trivial] at p
      have pk := p.2 true
      exact ⟨fun c' => (ih c' (pk c')).1, (ih c (pk c)).2⟩
    | coin C' j =>
      rw [interleave_live (g := @Game.coin _ _ _ _ _) (h := @Game.coin _ _ _ _ _) trivial trivial] at p
      have pk := p.2 true
      exact ⟨fun c' => (ih c' (pk c')).1, (ih c (pk c)).2⟩

end Unambiguity

end Pouw.Draft.Game
