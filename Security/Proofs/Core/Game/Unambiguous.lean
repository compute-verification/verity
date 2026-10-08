import Definitions.Core.Decisive
import Mathlib.Order.ConditionallyCompleteLattice.Basic

/-!
# A batch of decisive games reads as no other list of games

`unambiguous_of_decisive`: every game of `L` decisive (`Decisive`: every message type inhabited, two runs ending
differently) makes `L` `Unambiguous`.

The proof follows the equation `interleave g h = interleave g' h'` down both readings at once. Where both readings have
both games live, the root is the scheduling choice in both, so the games advance alike (`interleave_live`). A reading
can only drift where one of its games ends and the other reading's does not; then the game that ended is a relabelled
copy of the other reading's interleaving, which is strictly deeper (`depth_interleave`: the least depth of
`interleave x y` is `d x + d y + min (d x) (d y)`). Following `h` first (`split_right`) and `g` first (`split_left`)
each either finds the readings equal or fixes how their least depths differ (`Split`), and the four ways of drifting on
both sides contradict the depth of the root (`interleave_inj`). The game that ends needs no non-constancy below its
root: only `g` and `h` themselves are asked to be non-constant. `batch` then peels one game at a time (`batch_inj`),
the rest of a batch of decisive games being non-constant itself, or the empty batch, whose leaves have length zero.

A constant game breaks it (`not_unambiguous_of_constant`): `[send Bool fun _ => ret true, send Bool fun b => ret b]` has
the batch of `[ret true, g₂']`, where `g₂'` replays the first game's move and the schedule's choice.
-/

namespace Verity.Game

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
theorem map_injective {f : α → β} (hf : Function.Injective f) {g g' : Game α} (e : g.map f = g'.map f) :
    g = g' := by
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

theorem reaches_map {f : α → β} {g : Game α} {c : β} :
    Reaches (g.map f) c ↔ ∃ a, Reaches g a ∧ f a = c := by
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
theorem live_of_reaches {g : Game α} {a₁ a₂ : α} (ha : a₁ ≠ a₂) (h₁ : Reaches g a₁)
    (h₂ : Reaches g a₂) : Live g := by
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

/-! ### Least depth -/

/-- The least number of nodes on a run of `g` to its end (counting a message type with no message as one node). -/
noncomputable def depth : Game α → ℕ
  | .ret _ => 0
  | .send _ k => (⨅ m, depth (k m)) + 1
  | @Game.coin _ _ _ _ k => (⨅ c, depth (k c)) + 1

theorem depth_ret (a : α) : depth (Game.ret a) = 0 := rfl

theorem depth_send (M : Type) (k : M → Game α) : depth (Game.send M k) = (⨅ m, depth (k m)) + 1 := rfl

theorem depth_coin (C : Type) [Fintype C] [Nonempty C] (k : C → Game α) :
    depth (Game.coin C k) = (⨅ c, depth (k c)) + 1 := rfl

theorem depth_pos {g : Game α} (hg : Live g) : 0 < depth g := by
  cases g with
  | ret _ => exact hg.elim
  | send _ _ => exact Nat.succ_pos _
  | coin _ _ => exact Nat.succ_pos _

/-- A monotone map commutes with the least value of a nonempty family. -/
theorem iInf_monotone {ι : Type} [Nonempty ι] {φ : ℕ → ℕ} (hφ : Monotone φ) (f : ι → ℕ) :
    ⨅ i, φ (f i) = φ (⨅ i, f i) := by
  obtain ⟨i₀, hi₀⟩ := ciInf_mem f
  apply le_antisymm
  · rw [← hi₀]
    exact ciInf_le (OrderBot.bddBelow _) i₀
  · exact le_ciInf fun i => hφ (ciInf_le (OrderBot.bddBelow _) i)

theorem iInf_add_min {ι : Type} [Nonempty ι] (f : ι → ℕ) (c : ℕ) :
    ⨅ i, (f i + c + min (f i) c) = (⨅ i, f i) + c + min (⨅ i, f i) c :=
  iInf_monotone (φ := fun x => x + c + min x c) (fun x y hxy => by dsimp only; omega) f

theorem iInf_add_min' {ι : Type} [Nonempty ι] (f : ι → ℕ) (c : ℕ) :
    ⨅ i, (c + f i + min c (f i)) = c + (⨅ i, f i) + min c (⨅ i, f i) :=
  iInf_monotone (φ := fun x => c + x + min c x) (fun x y hxy => by dsimp only; omega) f

theorem iInf_bool (f : Bool → ℕ) : ⨅ b, f b = min (f true) (f false) := by
  apply le_antisymm
  · exact le_min (ciInf_le (OrderBot.bddBelow _) true) (ciInf_le (OrderBot.bddBelow _) false)
  · exact le_ciInf fun b => by cases b <;> simp

theorem depth_map (f : α → β) (g : Game α) : depth (g.map f) = depth g := by
  induction g with
  | ret a => rfl
  | send M k ih => rw [map_send, depth_send, depth_send]; simp only [ih]
  | coin C k ih => rw [map_coin, depth_coin, depth_coin]; simp only [ih]

theorem depth_live {g : Game α} {h : Game β} (hg : Live g) (hh : Live h) :
    depth (interleave g h) = min (depth (stepL g h)) (depth (stepR g h)) + 1 := by
  rw [interleave_live hg hh, depth_send, iInf_bool]
  rfl

/-- **The least depth of an interleaving**: play the shallower game first, the schedule's choice before each of its
moves, then the other. -/
theorem depth_interleave {g : Game α} {h : Game β} (pg : Productive g) (ph : Productive h) :
    depth (interleave g h) = depth g + depth h + min (depth g) (depth h) := by
  induction g generalizing h with
  | ret a => rw [interleave_ret_left, depth_map, depth_ret]; omega
  | send M k ih =>
    have : Nonempty M := pg.1
    have e1 : ∀ {h : Game β}, Productive h → ∀ m,
        depth (interleave (k m) h) = depth (k m) + depth h + min (depth (k m)) (depth h) :=
      fun ph m => ih m (pg.2 m) ph
    induction h with
    | ret b => rw [interleave_ret_right, depth_map, depth_ret]; omega
    | send N j ihh =>
      have : Nonempty N := ph.1
      have e2 : ∀ n, depth (interleave (Game.send M k) (j n)) =
          depth (Game.send M k) + depth (j n) + min (depth (Game.send M k)) (depth (j n)) :=
        fun n => ihh n (ph.2 n)
      rw [depth_live (g := .send _ _) (h := .send _ _) trivial trivial]
      show min ((⨅ m, depth (interleave (k m) (Game.send N j))) + 1)
        ((⨅ n, depth (interleave (Game.send M k) (j n))) + 1) + 1 = _
      rw [congrArg iInf (funext (e1 ph)), congrArg iInf (funext e2)]
      rw [iInf_add_min, iInf_add_min', depth_send M k, depth_send N j]
      omega
    | coin C j ihh =>
      have e2 : ∀ n, depth (interleave (Game.send M k) (j n)) =
          depth (Game.send M k) + depth (j n) + min (depth (Game.send M k)) (depth (j n)) :=
        fun n => ihh n (ph n)
      rw [depth_live (g := .send _ _) (h := @Game.coin _ _ _ _ _) trivial trivial]
      show min ((⨅ m, depth (interleave (k m) (Game.coin C j))) + 1)
        ((⨅ n, depth (interleave (Game.send M k) (j n))) + 1) + 1 = _
      rw [congrArg iInf (funext (e1 ph)), congrArg iInf (funext e2)]
      rw [iInf_add_min, iInf_add_min', depth_send M k, depth_coin C j]
      omega
  | coin C k ih =>
    have e1 : ∀ {h : Game β}, Productive h → ∀ m,
        depth (interleave (k m) h) = depth (k m) + depth h + min (depth (k m)) (depth h) :=
      fun ph m => ih m (pg m) ph
    induction h with
    | ret b => rw [interleave_ret_right, depth_map, depth_ret]; omega
    | send N j ihh =>
      have : Nonempty N := ph.1
      have e2 : ∀ n, depth (interleave (Game.coin C k) (j n)) =
          depth (Game.coin C k) + depth (j n) + min (depth (Game.coin C k)) (depth (j n)) :=
        fun n => ihh n (ph.2 n)
      rw [depth_live (g := @Game.coin _ _ _ _ _) (h := .send _ _) trivial trivial]
      show min ((⨅ m, depth (interleave (k m) (Game.send N j))) + 1)
        ((⨅ n, depth (interleave (Game.coin C k) (j n))) + 1) + 1 = _
      rw [congrArg iInf (funext (e1 ph)), congrArg iInf (funext e2)]
      rw [iInf_add_min, iInf_add_min', depth_coin C k, depth_send N j]
      omega
    | coin C' j ihh =>
      have e2 : ∀ n, depth (interleave (Game.coin C k) (j n)) =
          depth (Game.coin C k) + depth (j n) + min (depth (Game.coin C k)) (depth (j n)) :=
        fun n => ihh n (ph n)
      rw [depth_live (g := @Game.coin _ _ _ _ _) (h := @Game.coin _ _ _ _ _) trivial trivial]
      show min ((⨅ m, depth (interleave (k m) (Game.coin C' j))) + 1)
        ((⨅ n, depth (interleave (Game.coin C k) (j n))) + 1) + 1 = _
      rw [congrArg iInf (funext (e1 ph)), congrArg iInf (funext e2)]
      rw [iInf_add_min, iInf_add_min', depth_coin C k, depth_coin C' j]
      omega

/-! ### Two readings of one interleaving -/

/-- How two readings of an interleaving relate, followed down one game: the other games are equal (`P`), or the other
game of one reading is a relabelled interleaving of the other reading's with the extra depth `Y` of the followed game,
so strictly deeper. `u`, `u'` are the other games' least depths, `v`, `v'` the followed games'. -/
def Split (P : Prop) (u u' v v' : ℕ) : Prop :=
  P ∨ (∃ Y, 0 < Y ∧ u = u' + Y + min u' Y ∧ v' = v + Y) ∨
    (∃ Z, 0 < Z ∧ u' = u + Z + min u Z ∧ v = v' + Z)

/-- Every child of the followed games splits alike, so the games split so. -/
theorem split_iInf {ι : Type} [Nonempty ι] {P : Prop} {u u' : ℕ} {a b : ι → ℕ}
    (h : ∀ i, Split P u u' (a i) (b i)) : Split P u u' ((⨅ i, a i) + 1) ((⨅ i, b i) + 1) := by
  by_cases hP : P
  · exact Or.inl hP
  obtain ⟨i₀⟩ := ‹Nonempty ι›
  rcases h i₀ with hP' | ⟨Y, hY, hu, -⟩ | ⟨Z, hZ, hu, -⟩
  · exact absurd hP' hP
  · have hb : ∀ i, b i = a i + Y := fun i => by
      rcases h i with hP' | ⟨Y', hY', hu', hb⟩ | ⟨Z, hZ, hu', -⟩
      · exact absurd hP' hP
      · have : Y' = Y := by omega
        omega
      · omega
    have e : ⨅ i, b i = (⨅ i, a i) + Y := by
      simp only [hb]
      exact iInf_monotone (φ := fun x => x + Y) (fun x y hxy => by dsimp only; omega) a
    exact Or.inr (Or.inl ⟨Y, hY, hu, by omega⟩)
  · have ha : ∀ i, a i = b i + Z := fun i => by
      rcases h i with hP' | ⟨Y, hY, hu', -⟩ | ⟨Z', hZ', hu', ha⟩
      · exact absurd hP' hP
      · omega
      · have : Z' = Z := by omega
        omega
    have e : ⨅ i, a i = (⨅ i, b i) + Z := by
      simp only [ha]
      exact iInf_monotone (φ := fun x => x + Z) (fun x y hxy => by dsimp only; omega) b
    exact Or.inr (Or.inr ⟨Z, hZ, hu, by omega⟩)

/-- **Following the second game.** While both readings have both games live they advance the second game alike; where
one reading's second game ends and the other's does not, the first game that stays is a relabelled interleaving of the
other reading's two. -/
theorem split_right {g g' : Game α} (hg : Live g) (hg' : Live g') (pg : Productive g) (pg' : Productive g')
    {h h' : Game β} (ph : Productive h) (ph' : Productive h') (e : interleave g h = interleave g' h') :
    Split (g = g') (depth g) (depth g') (depth h) (depth h') := by
  obtain ⟨a₀, ha₀⟩ := reaches_of_productive pg
  induction h generalizing h' with
  | ret b =>
    rw [interleave_ret_right] at e
    cases h' with
    | ret b' =>
      rw [interleave_ret_right] at e
      have r : Reaches (g.map fun a => (a, b)) (a₀, b) := reaches_map.2 ⟨a₀, ha₀, rfl⟩
      rw [e] at r
      obtain ⟨_, -, hab⟩ := reaches_map.1 r
      obtain rfl : b' = b := (Prod.mk.inj hab).2
      exact Or.inl (map_injective (fun x y hxy => (Prod.mk.inj hxy).1) e)
    | send N' j' =>
      have hd := congrArg depth e
      rw [depth_map, depth_interleave pg' ph'] at hd
      exact Or.inr (Or.inl ⟨depth (Game.send N' j'), depth_pos (g := Game.send N' j') trivial, hd,
        by rw [depth_ret, Nat.zero_add]⟩)
    | coin C' j' =>
      have hd := congrArg depth e
      rw [depth_map, depth_interleave pg' ph'] at hd
      exact Or.inr (Or.inl ⟨depth (Game.coin C' j'), depth_pos (g := Game.coin C' j') trivial, hd,
        by rw [depth_ret, Nat.zero_add]⟩)
  | send N j ih =>
    cases h' with
    | ret b' =>
      rw [interleave_ret_right] at e
      have hd := congrArg depth e
      rw [depth_map, depth_interleave pg ph] at hd
      exact Or.inr (Or.inr ⟨depth (Game.send N j), depth_pos (g := Game.send N j) trivial, hd.symm,
        by rw [depth_ret, Nat.zero_add]⟩)
    | send N' j' =>
      rw [interleave_live hg (h := .send _ _) trivial, interleave_live hg' (h := .send _ _) trivial] at e
      injection e with _ e
      have e := congrFun e false
      change Game.send N (fun n => interleave g (j n)) = Game.send N' (fun n => interleave g' (j' n)) at e
      injection e with hN hj
      subst hN
      have hj := eq_of_heq hj
      have : Nonempty N := ph.1
      exact split_iInf fun n => ih n (ph.2 n) (ph'.2 n) (congrFun hj n)
    | coin C' j' =>
      rw [interleave_live hg (h := .send _ _) trivial,
        interleave_live hg' (h := @Game.coin _ _ _ _ _) trivial] at e
      injection e with _ e
      have e := congrFun e false
      change Game.send N (fun n => interleave g (j n)) = Game.coin C' (fun n => interleave g' (j' n)) at e
      cases e
  | coin C j ih =>
    cases h' with
    | ret b' =>
      rw [interleave_ret_right] at e
      have hd := congrArg depth e
      rw [depth_map, depth_interleave pg ph] at hd
      exact Or.inr (Or.inr ⟨depth (Game.coin C j), depth_pos (g := Game.coin C j) trivial, hd.symm,
        by rw [depth_ret, Nat.zero_add]⟩)
    | send N' j' =>
      rw [interleave_live hg (h := @Game.coin _ _ _ _ _) trivial,
        interleave_live hg' (h := .send _ _) trivial] at e
      injection e with _ e
      have e := congrFun e false
      change Game.coin C (fun n => interleave g (j n)) = Game.send N' (fun n => interleave g' (j' n)) at e
      cases e
    | coin C' j' =>
      rw [interleave_live hg (h := @Game.coin _ _ _ _ _) trivial,
        interleave_live hg' (h := @Game.coin _ _ _ _ _) trivial] at e
      injection e with _ e
      have e := congrFun e false
      change Game.coin C (fun n => interleave g (j n)) = Game.coin C' (fun n => interleave g' (j' n)) at e
      injection e with hC hF hj
      subst hC
      cases hF
      have hj := eq_of_heq hj
      exact split_iInf fun n => ih n (ph n) (ph' n) (congrFun hj n)

/-- **Following the first game.** The mirror of `split_right`. -/
theorem split_left {h h' : Game β} (hh : Live h) (hh' : Live h') (ph : Productive h) (ph' : Productive h')
    {g g' : Game α} (pg : Productive g) (pg' : Productive g') (e : interleave g h = interleave g' h') :
    Split (h = h') (depth h) (depth h') (depth g) (depth g') := by
  obtain ⟨b₀, hb₀⟩ := reaches_of_productive ph
  induction g generalizing g' with
  | ret a =>
    rw [interleave_ret_left] at e
    cases g' with
    | ret a' =>
      rw [interleave_ret_left] at e
      have r : Reaches (h.map fun b => (a, b)) (a, b₀) := reaches_map.2 ⟨b₀, hb₀, rfl⟩
      rw [e] at r
      obtain ⟨_, -, hab⟩ := reaches_map.1 r
      obtain rfl : a' = a := (Prod.mk.inj hab).1
      exact Or.inl (map_injective (fun x y hxy => (Prod.mk.inj hxy).2) e)
    | send M' k' =>
      have hd := congrArg depth e
      rw [depth_map, depth_interleave pg' ph'] at hd
      exact Or.inr (Or.inl ⟨depth (Game.send M' k'), depth_pos (g := Game.send M' k') trivial, by omega,
        by rw [depth_ret, Nat.zero_add]⟩)
    | coin C' k' =>
      have hd := congrArg depth e
      rw [depth_map, depth_interleave pg' ph'] at hd
      exact Or.inr (Or.inl ⟨depth (Game.coin C' k'), depth_pos (g := Game.coin C' k') trivial, by omega,
        by rw [depth_ret, Nat.zero_add]⟩)
  | send M k ih =>
    cases g' with
    | ret a' =>
      rw [interleave_ret_left] at e
      have hd := congrArg depth e
      rw [depth_map, depth_interleave pg ph] at hd
      exact Or.inr (Or.inr ⟨depth (Game.send M k), depth_pos (g := Game.send M k) trivial, by omega,
        by rw [depth_ret, Nat.zero_add]⟩)
    | send M' k' =>
      rw [interleave_live (g := .send _ _) trivial hh, interleave_live (g := .send _ _) trivial hh'] at e
      injection e with _ e
      have e := congrFun e true
      change Game.send M (fun m => interleave (k m) h) = Game.send M' (fun m => interleave (k' m) h') at e
      injection e with hM hk
      subst hM
      have hk := eq_of_heq hk
      have : Nonempty M := pg.1
      exact split_iInf fun m => ih m (pg.2 m) (pg'.2 m) (congrFun hk m)
    | coin C' k' =>
      rw [interleave_live (g := .send _ _) trivial hh,
        interleave_live (g := @Game.coin _ _ _ _ _) trivial hh'] at e
      injection e with _ e
      have e := congrFun e true
      change Game.send M (fun m => interleave (k m) h) = Game.coin C' (fun m => interleave (k' m) h') at e
      cases e
  | coin C k ih =>
    cases g' with
    | ret a' =>
      rw [interleave_ret_left] at e
      have hd := congrArg depth e
      rw [depth_map, depth_interleave pg ph] at hd
      exact Or.inr (Or.inr ⟨depth (Game.coin C k), depth_pos (g := Game.coin C k) trivial, by omega,
        by rw [depth_ret, Nat.zero_add]⟩)
    | send M' k' =>
      rw [interleave_live (g := @Game.coin _ _ _ _ _) trivial hh,
        interleave_live (g := .send _ _) trivial hh'] at e
      injection e with _ e
      have e := congrFun e true
      change Game.coin C (fun m => interleave (k m) h) = Game.send M' (fun m => interleave (k' m) h') at e
      cases e
    | coin C' k' =>
      rw [interleave_live (g := @Game.coin _ _ _ _ _) trivial hh,
        interleave_live (g := @Game.coin _ _ _ _ _) trivial hh'] at e
      injection e with _ e
      have e := congrFun e true
      change Game.coin C (fun m => interleave (k m) h) = Game.coin C' (fun m => interleave (k' m) h') at e
      injection e with hC hF hk
      subst hC
      cases hF
      have hk := eq_of_heq hk
      exact split_iInf fun m => ih m (pg m) (pg' m) (congrFun hk m)

/-- **Interleaving two non-constant games reads only one way.** Both games have every message type inhabited and two
different outputs; the other reading's games are arbitrary. -/
theorem interleave_inj {g g' : Game α} {h h' : Game β} (pg : Productive g) (ph : Productive h)
    {a₁ a₂ : α} (ha : a₁ ≠ a₂) (ha₁ : Reaches g a₁) (ha₂ : Reaches g a₂)
    {b₁ b₂ : β} (hb : b₁ ≠ b₂) (hb₁ : Reaches h b₁) (hb₂ : Reaches h b₂)
    (e : interleave g h = interleave g' h') : g = g' ∧ h = h' := by
  have pT : Productive (interleave g' h') := e ▸ productive_interleave pg ph
  obtain ⟨pg', ph'⟩ := productive_of_interleave pT
  have r₁ : Reaches (interleave g' h') (a₁, b₁) := e ▸ reaches_interleave ha₁ hb₁
  have r₂ : Reaches (interleave g' h') (a₂, b₂) := e ▸ reaches_interleave ha₂ hb₂
  have hg := live_of_reaches ha ha₁ ha₂
  have hh := live_of_reaches hb hb₁ hb₂
  have hg' := live_of_reaches ha (reaches_of_interleave r₁).1 (reaches_of_interleave r₂).1
  have hh' := live_of_reaches hb (reaches_of_interleave r₁).2 (reaches_of_interleave r₂).2
  have sR := split_right hg hg' pg pg' ph ph' e
  have sL := split_left hh hh' ph ph' pg pg' e
  have hd := congrArg depth e
  rw [depth_interleave pg ph, depth_interleave pg' ph'] at hd
  have p₁ := depth_pos hg
  have p₂ := depth_pos hg'
  have p₃ := depth_pos hh
  have p₄ := depth_pos hh'
  rcases sR with rfl | ⟨Y, hY, hu, hv⟩ | ⟨Z, hZ, hu, hv⟩ <;>
    rcases sL with rfl | ⟨X, hX, hx, hw⟩ | ⟨W, hW, hx, hw⟩ <;>
    first | exact ⟨rfl, rfl⟩ | omega

/-! ### Batches -/

theorem batch_cons (g : Game α) (gs : List (Game α)) :
    batch (g :: gs) = (interleave g (batch gs)).map fun x => x.1 :: x.2 := rfl

theorem productive_batch : ∀ {L : List (Game α)}, (∀ g ∈ L, Productive g) → Productive (batch L)
  | [], _ => trivial
  | g :: gs, h => by
    rw [batch_cons]
    exact productive_map.2 (productive_interleave (h g (by simp)) (productive_batch fun x hx => h x (by simp [hx])))

/-- Every output of a batch has one entry per game. -/
theorem reaches_batch_length : ∀ {L : List (Game α)} {l : List α}, Reaches (batch L) l → l.length = L.length
  | [], l, r => by
    have r : [] = l := r
    subst r
    rfl
  | g :: gs, l, r => by
    rw [batch_cons] at r
    obtain ⟨⟨a, l'⟩, hx, rfl⟩ := reaches_map.1 r
    simp [reaches_batch_length (reaches_of_interleave hx).2]

/-- **A batch of decisive games is the batch of no other list.** -/
theorem batch_inj {L : List (Game α)} (hL : ∀ g ∈ L, Decisive g) {L' : List (Game α)} (e : batch L' = batch L) :
    L' = L := by
  induction L generalizing L' with
  | nil =>
    cases L' with
    | nil => rfl
    | cons g' gs' =>
      have r : Reaches (batch ([] : List (Game α))) ([] : List α) := rfl
      rw [← e, batch_cons] at r
      obtain ⟨x, -, hx⟩ := reaches_map.1 r
      exact absurd hx (List.cons_ne_nil _ _)
  | cons g gs ih =>
    have pg := (hL g (by simp)).1
    have hgs : ∀ x ∈ gs, Decisive x := fun x hx => hL x (by simp [hx])
    have pB : Productive (batch gs) := productive_batch fun x hx => (hgs x hx).1
    cases L' with
    | nil =>
      obtain ⟨a, ha⟩ := reaches_of_productive pg
      obtain ⟨l, hl⟩ := reaches_of_productive pB
      have r : Reaches (batch (g :: gs)) (a :: l) := reaches_map.2 ⟨(a, l), reaches_interleave ha hl, rfl⟩
      rw [← e] at r
      have r : ([] : List α) = a :: l := r
      exact absurd r.symm (List.cons_ne_nil _ _)
    | cons g' gs' =>
      rw [batch_cons, batch_cons] at e
      have e := map_injective (fun (x y : α × List α) hxy => by
        obtain ⟨x1, x2⟩ := x
        obtain ⟨y1, y2⟩ := y
        obtain ⟨rfl, rfl⟩ := List.cons.inj hxy
        rfl) e
      obtain ⟨a₁, a₂, ha, ha₁, ha₂⟩ := (hL g (by simp)).2
      cases gs with
      | nil =>
        rw [show batch ([] : List (Game α)) = .ret [] from rfl, interleave_ret_right] at e
        have p' := productive_of_interleave (e ▸ productive_map.2 pg : Productive (interleave g' (batch gs')))
        obtain ⟨l', hl'⟩ := reaches_of_productive p'.2
        obtain ⟨a', ha'⟩ := reaches_of_productive p'.1
        have r := reaches_interleave ha' hl'
        rw [e] at r
        obtain ⟨_, -, hx⟩ := reaches_map.1 r
        obtain rfl : [] = l' := (Prod.mk.inj hx).2
        have hlen := reaches_batch_length hl'
        cases gs' with
        | nil =>
          rw [show batch ([] : List (Game α)) = .ret [] from rfl, interleave_ret_right] at e
          rw [map_injective (fun x y hxy => (Prod.mk.inj hxy).1) e]
        | cons _ _ => simp at hlen
      | cons r rs =>
        obtain ⟨c₁, c₂, hc, hc₁, hc₂⟩ := (hgs r (by simp)).2
        obtain ⟨l, hl⟩ := reaches_of_productive
          (productive_batch fun x hx => (hgs x (by simp [hx])).1 : Productive (batch rs))
        have hB₁ : Reaches (batch (r :: rs)) (c₁ :: l) :=
          reaches_map.2 ⟨(c₁, l), reaches_interleave hc₁ hl, rfl⟩
        have hB₂ : Reaches (batch (r :: rs)) (c₂ :: l) :=
          reaches_map.2 ⟨(c₂, l), reaches_interleave hc₂ hl, rfl⟩
        obtain ⟨rfl, hB⟩ := interleave_inj pg pB ha ha₁ ha₂ (fun h => hc (List.cons.inj h).1) hB₁ hB₂ e.symm
        rw [ih hgs hB.symm]

end Unambiguity

variable {α : Type}

/-- **A list of decisive games is unambiguous**: when every game of `L` has every message type inhabited and two runs
ending differently, `batch L` is the batch of no other list. -/
theorem unambiguous_of_decisive (L : List (Game α)) (h : ∀ g ∈ L, Decisive g) : Unambiguous L :=
  fun _ _ _ e => Unambiguity.batch_inj h e

/-- **A constant game breaks it.** The first game always accepts; its move and the schedule's choice fold into the
second game of another reading. -/
theorem not_unambiguous_of_constant :
    ¬ Unambiguous [Game.send Bool (fun _ => Game.ret true), Game.send Bool (fun b => Game.ret b)] := by
  intro h
  have e : batch ([] ++ Game.ret true :: [Game.send Bool fun first =>
      if first then Game.send Bool fun _ => Game.send Bool fun b => Game.ret b
      else Game.send Bool fun b => Game.send Bool fun _ => Game.ret b]) =
      batch [Game.send Bool (fun _ => Game.ret true), Game.send Bool (fun b => Game.ret b)] :=
    congrArg (Game.send Bool) (funext fun first => by cases first <;> rfl)
  have := h [] _ _ e
  injection this with h1
  cases h1

end Verity.Game
