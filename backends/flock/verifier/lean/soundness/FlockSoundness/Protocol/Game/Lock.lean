import FlockSoundness.Protocol.Game.Expect
import FlockSoundness.Protocol.Game.Interleave

/-!
# Lockstep coupling of an oracle game with its compiled game

The compiled protocol replaces each committed table by a cap and opens the queried columns with
paths at the end. `Lock get R i g g'` exhibits the compiled game `g'` running in lockstep with the
oracle game `g`: the same coins and the same messages, except
at a **table** node, where `g` receives a table `t : T` and `g'` a cap `d : D`. The node carries a
slot, and the index `i : I` records the tables fixed so far (`get i slot`). At the leaves the two
outputs satisfy `R i a b`. The compiled game's last message, the openings, has no counterpart in
the oracle game; the oracle game gets a dummy send for it (`withDummy`), which changes no
probability (`expect_withDummy`).

For a compiled strategy `σ'` and a choice of table at every table node (`plur`, the plurality of
the slot's openings over the continuation), `oracleStrat` is the oracle strategy `P̃` that sends
the chosen tables and otherwise plays `σ'`. `expect_le_coupled`: the compiled event has
probability at most the oracle event's under `P̃`, plus the probability of `Bad`, the runs whose
leaf fails the coupling. `leafE_off_le`: per slot, the probability that a run opens a position
off the table chosen at that slot is at most `√(N · Q · Pr[fork conflict])`: the rewinding lemma
(`Rewinding.expect_bad_le`) at the slot's node, averaged over prefixes by Jensen.
-/

namespace FlockSoundness

namespace Game

universe u

variable {α β : Type}

/-- The compiled game `g'` in lockstep with the oracle game `g` (see the module doc). `get i s`
is the table fixed at slot `s`, `none` before the slot's node. -/
inductive Lock (T D : Type) {I : Type} (get : I → ℕ → Option (T × D)) (R : I → α → β → Prop) :
    I → Game α → Game β → Type 1
  | ret {i : I} {a : α} {b : β} : R i a b → Lock T D get R i (.ret a) (.ret b)
  | coin {i : I} (C : Type) [Fintype C] [Nonempty C] {k : C → Game α} {k' : C → Game β} :
      ((c : C) → Lock T D get R i (k c) (k' c)) → Lock T D get R i (.coin C k) (.coin C k')
  | send {i : I} (M : Type) {k : M → Game α} {k' : M → Game β} :
      ((m : M) → Lock T D get R i (k m) (k' m)) → Lock T D get R i (.send M k) (.send M k')
  | table {i : I} (slot : ℕ) (upd : T → D → I) {k : T → Game α} {k' : D → Game β}
      (hfresh : get i slot = none) (hset : ∀ t d, get (upd t d) slot = some (t, d))
      (hkeep : ∀ t d s, s ≠ slot → get (upd t d) s = get i s) :
      ((t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)) →
        Lock T D get R i (.send T k) (.send D k')

namespace Lock

variable {T D I : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}

/-- The table chosen at a slot, from the cap and the continuation's strategy. -/
abbrev Chooser (T D : Type) (β : Type) := ℕ → D → (g : Game β) → Strategy g → T

/-- **The oracle-model prover `P̃`**: the chosen table at every table node, `σ'` elsewhere. -/
def oracleStrat (plur : Chooser T D β) : {i : I} → {g : Game α} → {g' : Game β} →
    Lock T D get R i g g' → Strategy g' → Strategy g
  | _, _, _, .ret _, _ => PUnit.unit
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => fun c : C => oracleStrat plur (L c) (σ' c)
  | _, _, _, .send _ L, σ' => ⟨σ'.1, oracleStrat plur (L σ'.1) σ'.2⟩
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ slot _ _ k' _ _ _ L, σ' =>
      ⟨plur slot σ'.1 (k' σ'.1) σ'.2,
        oracleStrat plur (L (plur slot σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2⟩


/-! ## Building locksteps -/

/-- Weaken the leaf relation. -/
def mono {R' : I → α → β → Prop} (h : ∀ i a b, R i a b → R' i a b) : {i : I} → {g : Game α} →
    {g' : Game β} → Lock T D get R i g g' → Lock T D get R' i g g'
  | _, _, _, .ret hr => .ret (h _ _ _ hr)
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => mono h (L c)
  | _, _, _, .send M L => .send M fun m => mono h (L m)
  | _, _, _, .table slot upd hf hs hk L => .table slot upd hf hs hk fun t d => mono h (L t d)

/-- **Sequencing** locksteps. -/
def bind {α₂ β₂ : Type} {R₂ : I → α₂ → β₂ → Prop} {f : α → Game α₂} {f' : β → Game β₂}
    (h : ∀ i a b, R i a b → Lock T D get R₂ i (f a) (f' b)) : {i : I} → {g : Game α} →
    {g' : Game β} → Lock T D get R i g g' → Lock T D get R₂ i (g.bind f) (g'.bind f')
  | _, _, _, .ret hr => h _ _ _ hr
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => bind h (L c)
  | _, _, _, .send M L => .send M fun m => bind h (L m)
  | _, _, _, .table slot upd hf hs hk L => .table slot upd hf hs hk fun t d => bind h (L t d)

/-- Relabel both outputs. -/
def map {α₂ β₂ : Type} {R₂ : I → α₂ → β₂ → Prop} (f : α → α₂) (f' : β → β₂)
    (h : ∀ i a b, R i a b → R₂ i (f a) (f' b)) {i : I} {g : Game α} {g' : Game β}
    (L : Lock T D get R i g g') : Lock T D get R₂ i (g.map f) (g'.map f') :=
  bind (fun i a b hr => .ret (h i a b hr)) L

/-- The same game on both sides, with no table nodes. -/
def same {i : I} : (g : Game α) → Lock T D get (fun i' a b => i' = i ∧ a = b) i g g
  | .ret _ => .ret ⟨rfl, rfl⟩
  | @Game.coin _ C _ _ k => .coin C fun c => same (k c)
  | .send M k => .send M fun m => same (k m)


/-! ## Two locksteps interleaved -/

section Interleave

variable {α₁ β₁ α₂ β₂ I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (T × D)}
  {get₂ : I₂ → ℕ → Option (T × D)} {R₁ : I₁ → α₁ → β₁ → Prop} {R₂ : I₂ → α₂ → β₂ → Prop}

/-- The tables two locksteps have fixed. -/
def getPair (get₁ : I₁ → ℕ → Option (T × D)) (get₂ : I₂ → ℕ → Option (T × D)) (i : I₁ × I₂)
    (s : ℕ) : Option (T × D) :=
  (get₁ i.1 s).orElse fun _ => get₂ i.2 s

/-- Both leaf relations. -/
def RPair (R₁ : I₁ → α₁ → β₁ → Prop) (R₂ : I₂ → α₂ → β₂ → Prop) (i : I₁ × I₂) (a : α₁ × α₂)
    (b : β₁ × β₂) : Prop :=
  R₁ i.1 a.1 b.1 ∧ R₂ i.2 a.2 b.2

/-- The two locksteps' slots are disjoint. -/
structure Disjoint (get₁ : I₁ → ℕ → Option (T × D)) (get₂ : I₂ → ℕ → Option (T × D)) : Prop where
  left : ∀ i₁ i₂ s, (get₁ i₁ s).isSome → get₂ i₂ s = none
  right : ∀ i₁ i₂ s, (get₂ i₂ s).isSome → get₁ i₁ s = none

variable [Inhabited T] [Inhabited D]

/-- Rep 2 ended with related outputs; rep 1 continues alone. -/
def embedLeft (hd : Disjoint get₁ get₂) {i₂ : I₂} {a₂ : α₂} {b₂ : β₂} (h₂ : R₂ i₂ a₂ b₂) :
    {i₁ : I₁} → {g₁ : Game α₁} → {g₁' : Game β₁} → Lock T D get₁ R₁ i₁ g₁ g₁' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (g₁.map fun a => (a, a₂))
        (g₁'.map fun b => (b, b₂))
  | _, _, _, .ret h₁ => .ret ⟨h₁, h₂⟩
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => embedLeft hd h₂ (L c)
  | _, _, _, .send M L => .send M fun m => embedLeft hd h₂ (L m)
  | _, _, _, .table slot upd hf hs hk L =>
      .table slot (fun t d => (upd t d, i₂)) (getPair_fresh₁ hd hf hs i₂) (getPair_set₁ hs i₂)
        (fun t d s h => getPair_keep₁ hk i₂ t d s h) fun t d => embedLeft hd h₂ (L t d)

/-- Rep 1 ended with related outputs; rep 2 continues alone. -/
def embedRight (hd : Disjoint get₁ get₂) {i₁ : I₁} {a₁ : α₁} {b₁ : β₁} (h₁ : R₁ i₁ a₁ b₁) :
    {i₂ : I₂} → {g₂ : Game α₂} → {g₂' : Game β₂} → Lock T D get₂ R₂ i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (g₂.map fun a => (a₁, a))
        (g₂'.map fun b => (b₁, b))
  | _, _, _, .ret h₂ => .ret ⟨h₁, h₂⟩
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => embedRight hd h₁ (L c)
  | _, _, _, .send M L => .send M fun m => embedRight hd h₁ (L m)
  | _, _, _, .table slot upd hf hs hk L =>
      .table slot (fun t d => (i₁, upd t d)) (getPair_fresh₂ hd hf hs i₁) (getPair_set₂ hd hs i₁)
        (fun t d s h => getPair_keep₂ hk i₁ t d s h) fun t d => embedRight hd h₁ (L t d)

/-- `interleaveWith` in lockstep: rep 2 advances, or rep 1 takes a step (`hadv`), or rep 2 has
ended (`hfin`). -/
def lockWith (hd : Disjoint get₁ get₂) {i₁ : I₁} {adv₁ : Game α₂ → Game (α₁ × α₂)}
    {adv₁' : Game β₂ → Game (β₁ × β₂)} {fin₁ : α₂ → Game (α₁ × α₂)} {fin₁' : β₂ → Game (β₁ × β₂)}
    (hadv : ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂}, Lock T D get₂ R₂ i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (adv₁ g₂) (adv₁' g₂'))
    (hfin : ∀ {i₂ : I₂} {a₂ : α₂} {b₂ : β₂}, R₂ i₂ a₂ b₂ →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (fin₁ a₂) (fin₁' b₂)) :
    {i₂ : I₂} → {g₂ : Game α₂} → {g₂' : Game β₂} → Lock T D get₂ R₂ i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (interleaveWith adv₁ fin₁ g₂)
        (interleaveWith adv₁' fin₁' g₂')
  | _, _, _, .ret h₂ => hfin h₂
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      .send Bool fun first => match first with
        | true => hadv (@Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L)
        | false => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => lockWith hd hadv hfin (L c)
  | _, _, _, .send M L =>
      .send Bool fun first => match first with
        | true => hadv (.send M L)
        | false => .send M fun m => lockWith hd hadv hfin (L m)
  | _, _, _, .table slot upd hf hs hk L =>
      .send Bool fun first => match first with
        | true => hadv (.table slot upd hf hs hk L)
        | false => .table slot (fun t d => (i₁, upd t d)) (getPair_fresh₂ hd hf hs i₁)
            (getPair_set₂ hd hs i₁) (fun t d s h => getPair_keep₂ hk i₁ t d s h)
            fun t d => lockWith hd hadv hfin (L t d)

/-- **Two locksteps interleaved**, whatever schedule the prover chooses. -/
def interleaveLock (hd : Disjoint get₁ get₂) : {i₁ : I₁} → {g₁ : Game α₁} → {g₁' : Game β₁} →
    Lock T D get₁ R₁ i₁ g₁ g₁' → ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂},
      Lock T D get₂ R₂ i₂ g₂ g₂' →
        Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (interleave g₁ g₂) (interleave g₁' g₂')
  | _, _, _, .ret h₁, _, _, _, L₂ => embedRight hd h₁ L₂
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L₁, _, _, _, L₂ =>
      lockWith hd (fun L₂ => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => interleaveLock hd (L₁ c) L₂)
        (fun h₂ => embedLeft hd h₂ (@Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L₁)) L₂
  | _, _, _, .send M L₁, _, _, _, L₂ =>
      lockWith hd (fun L₂ => .send M fun m => interleaveLock hd (L₁ m) L₂)
        (fun h₂ => embedLeft hd h₂ (.send M L₁)) L₂
  | _, _, _, .table slot upd hf hs hk L₁, _, _, i₂, L₂ =>
      lockWith hd (fun {i₂ _ _} L₂ => .table slot (fun t d => (upd t d, i₂))
          (getPair_fresh₁ hd hf hs i₂) (getPair_set₁ hs i₂) (fun t d s h => getPair_keep₁ hk i₂ t d s h)
          fun t d => interleaveLock hd (L₁ t d) L₂)
        (fun h₂ => embedLeft hd h₂ (.table slot upd hf hs hk L₁)) L₂

end Interleave

/-! ## Rewinding at every table node -/

/-! ## Two locksteps of one compiled game with the same slots -/

section InterleaveAligned

variable {α₁ β₁ α₂ β₂ I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (T × D)}
  {get₂ : I₂ → ℕ → Option (T × D)} [Inhabited T] [Inhabited D]
  {R₁ R₁' : I₁ → α₁ → β₁ → Prop} {R₂ R₂' : I₂ → α₂ → β₂ → Prop}

end InterleaveAligned

end Lock

end Game

end FlockSoundness
