import Definitions.Core.Game.Lock

/-!
# One lockstep among identical games

A batched session runs many tables concurrently, the prover choosing the schedule (`batch`). To
compile one table while every other game stays as it is, that table's lockstep runs beside the
same game on both sides: `leftSame` and `rightSame` for `interleave`, and `batchLock` for a batch
with the table at position `pre.length`. The index stays the table's, so its slots and choosers
carry over. The lemmas (alignment, `Aligned.batchLock`, and `value_batch_getD_le`) are in
`Proofs.Core.Game.Batch`.
-/

namespace Verity

namespace Game

variable {α β γ : Type}

namespace Lock

variable {T D I : Type} {get : I → ℕ → Option (T × D)}

/-! ## A lockstep beside the same game -/

section Same

variable {R : I → α → β → Prop}

/-- Relabel the oracle side's output only. -/
def mapL {α₂ : Type} {R₂ : I → α₂ → β → Prop} (f : α → α₂) (h : ∀ i a b, R i a b → R₂ i (f a) b) :
    {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Lock T D get R₂ i (g.map f) g'
  | _, _, _, .ret hr => .ret (h _ _ _ hr)
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => mapL f h (L c)
  | _, _, _, .send M L => .send M fun m => mapL f h (L m)
  | _, _, _, .table slot upd hf hs hk L => .table slot upd hf hs hk fun t d => mapL f h (L t d)

/-- `interleaveWith` over the same game on both sides, the index fixed. -/
def lockWithSame {R₂ : I → α × γ → β × γ → Prop} {i : I}
    {adv : Game γ → Game (α × γ)} {adv' : Game γ → Game (β × γ)}
    {fin : γ → Game (α × γ)} {fin' : γ → Game (β × γ)}
    (hadv : ∀ h : Game γ, Lock T D get R₂ i (adv h) (adv' h))
    (hfin : ∀ c : γ, Lock T D get R₂ i (fin c) (fin' c)) :
    (h : Game γ) → Lock T D get R₂ i (interleaveWith adv fin h) (interleaveWith adv' fin' h)
  | .ret c => hfin c
  | .send M k => .send Bool fun first => match first with
      | true => hadv (.send M k)
      | false => .send M fun m => lockWithSame hadv hfin (k m)
  | @Game.coin _ C iF iN k => .send Bool fun first => match first with
      | true => hadv (@Game.coin _ C iF iN k)
      | false => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => lockWithSame hadv hfin (k c)

/-- **A lockstep beside the same game**, the lockstep first. -/
def leftSame : {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' → (h : Game γ) →
    Lock T D get (fun i a b => R i a.1 b.1 ∧ a.2 = b.2) i (interleave g h) (interleave g' h)
  | _, _, _, @Lock.ret _ _ _ _ _ _ _ i a b hr, h =>
      Lock.map (fun c => (a, c)) (fun c => (b, c)) (fun _ _ _ hc => ⟨hc.1 ▸ hr, hc.2⟩)
        (Lock.same (i := i) h)
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L, h =>
      lockWithSame (fun h' => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => leftSame (L c) h')
        (fun c => Lock.map (fun a => (a, c)) (fun b => (b, c)) (fun _ _ _ hr => ⟨hr, rfl⟩)
          (@Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L)) h
  | _, _, _, .send M L, h =>
      lockWithSame (fun h' => .send M fun m => leftSame (L m) h')
        (fun c => Lock.map (fun a => (a, c)) (fun b => (b, c)) (fun _ _ _ hr => ⟨hr, rfl⟩)
          (.send M L)) h
  | _, _, _, .table slot upd hf hs hk L, h =>
      lockWithSame (fun h' => .table slot upd hf hs hk fun t d => leftSame (L t d) h')
        (fun c => Lock.map (fun a => (a, c)) (fun b => (b, c)) (fun _ _ _ hr => ⟨hr, rfl⟩)
          (.table slot upd hf hs hk L)) h

/-- `interleaveWith` with a lockstep for the second interaction. -/
def lockWithL {R₂ : I → γ × α → γ × β → Prop}
    {adv : Game α → Game (γ × α)} {adv' : Game β → Game (γ × β)}
    {fin : α → Game (γ × α)} {fin' : β → Game (γ × β)}
    (hadv : ∀ {i : I} {g : Game α} {g' : Game β}, Lock T D get R i g g' →
      Lock T D get R₂ i (adv g) (adv' g'))
    (hfin : ∀ {i : I} {a : α} {b : β}, R i a b → Lock T D get R₂ i (fin a) (fin' b)) :
    {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' →
      Lock T D get R₂ i (interleaveWith adv fin g) (interleaveWith adv' fin' g')
  | _, _, _, .ret hr => hfin hr
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      .send Bool fun first => match first with
        | true => hadv (@Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L)
        | false => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => lockWithL hadv hfin (L c)
  | _, _, _, .send M L =>
      .send Bool fun first => match first with
        | true => hadv (.send M L)
        | false => .send M fun m => lockWithL hadv hfin (L m)
  | _, _, _, .table slot upd hf hs hk L =>
      .send Bool fun first => match first with
        | true => hadv (.table slot upd hf hs hk L)
        | false => .table slot upd hf hs hk fun t d => lockWithL hadv hfin (L t d)

/-- The same game on both sides, then related outputs `a`, `b` on the right. -/
def sameThen (h : Game γ) {i : I} {a : α} {b : β} (hr : R i a b) :
    Lock T D get (fun i x y => x.1 = y.1 ∧ R i x.2 y.2) i (h.map fun c => (c, a))
      (h.map fun c => (c, b)) :=
  Lock.map (fun c => (c, a)) (fun c => (c, b)) (fun _ _ _ hc => ⟨hc.2, hc.1 ▸ hr⟩)
    (Lock.same (i := i) h)

/-- **The same game beside a lockstep**, the lockstep second. -/
def rightSame : (h : Game γ) → {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' →
    Lock T D get (fun i a b => a.1 = b.1 ∧ R i a.2 b.2) i (interleave h g) (interleave h g')
  | .ret c, _, _, _, L => Lock.map (fun a => (c, a)) (fun b => (c, b)) (fun _ _ _ hr => ⟨rfl, hr⟩) L
  | .send M k, _, _, _, L =>
      lockWithL (fun L' => .send M fun m => rightSame (k m) L') (fun hr => sameThen (.send M k) hr) L
  | @Game.coin _ C iF iN k, _, _, _, L =>
      lockWithL (fun L' => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => rightSame (k c) L')
        (fun hr => sameThen (@Game.coin _ C iF iN k) hr) L

end Same

/-- **One position's lockstep in a batch**: the lockstep at position `pre.length`, the same game on
both sides everywhere else. -/
def batchLock {ζ : Type} {R : I → ζ → ζ → Prop} (d : ζ) {i : I} {g g' : Game ζ}
    (L : Lock T D get R i g g') (post : List (Game ζ)) :
    (pre : List (Game ζ)) →
      Lock T D get (fun i a b => R i (a.getD pre.length d) (b.getD pre.length d)) i
        (batch (pre ++ g :: post)) (batch (pre ++ g' :: post))
  | [] => Lock.map (fun x => x.1 :: x.2) (fun x => x.1 :: x.2) (fun _ _ _ h => h.1)
      (leftSame L (batch post))
  | h :: pre => Lock.map (fun x => x.1 :: x.2) (fun x => x.1 :: x.2) (fun _ _ _ h' => h'.2)
      (rightSame h (batchLock d L post pre))

end Lock

end Game

end Verity
