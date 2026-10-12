import Proofs.Flock.Soundness.Game.Lock

/-!
# One lockstep among identical games

A batched session runs many tables concurrently, the prover choosing the schedule (`batch`). To
compile one table while every other game stays as it is, that table's lockstep runs beside the
same game on both sides: `leftSame` and `rightSame` for `interleave`, and `batchLock` for a batch
with the table at position `pre.length`. The index stays the table's, so its slots and choosers
carry over, and alignment is preserved (`Aligned.batchLock`).

`value_batch_getD_le`: an event of one position's output is at most as likely, for an optimal
prover, as the same event in that position's game alone.
-/

namespace FlockSoundness

namespace Game

variable {α β γ : Type}

/-- Interactions run concurrently, the prover choosing the schedule; the outputs in order. -/
def batch : List (Game α) → Game (List α)
  | [] => .ret []
  | g :: gs => (g.interleave (batch gs)).map fun x => x.1 :: x.2

/-- **One position of a batch.** An event of one position's output is at most as likely, for an
optimal prover, as the same event in that position's game alone. -/
theorem value_batch_getD_le (Q : α → Prop) (d : α) (g : Game α) (post : List (Game α)) :
    ∀ pre : List (Game α),
      value (fun outs => Q (outs.getD pre.length d)) (batch (pre ++ g :: post)) ≤ value Q g
  | [] => by
      simp only [List.nil_append, batch, value_map]
      calc value (fun x : α × List α => Q ((x.1 :: x.2).getD [].length d))
            (g.interleave (batch post))
          = value (fun x : α × List α => Q x.1 ∧ (fun _ : List α => True) x.2)
              (g.interleave (batch post)) := value_congr (fun x => by simp) _
        _ ≤ value Q g * value (fun _ : List α => True) (batch post) :=
            value_interleave_le Q (fun _ : List α => True) g (batch post)
        _ ≤ value Q g * 1 := mul_le_mul' le_rfl (value_le_one _ _)
        _ = value Q g := mul_one _
  | h :: pre => by
      simp only [List.cons_append, batch, value_map]
      calc value (fun x : α × List α => Q ((x.1 :: x.2).getD (h :: pre).length d))
            (h.interleave (batch (pre ++ g :: post)))
          = value (fun x : α × List α => (fun _ : α => True) x.1 ∧ Q (x.2.getD pre.length d))
              (h.interleave (batch (pre ++ g :: post))) := value_congr (fun x => by simp) _
        _ ≤ value (fun _ : α => True) h *
              value (fun outs => Q (outs.getD pre.length d)) (batch (pre ++ g :: post)) :=
            value_interleave_le (fun _ : α => True) (fun outs => Q (outs.getD pre.length d)) h
              (batch (pre ++ g :: post))
        _ ≤ 1 * value Q g := mul_le_mul' (value_le_one _ _) (value_batch_getD_le Q d g post pre)
        _ = value Q g := one_mul _

namespace Lock

theorem value_withDummy (O : Type) [Nonempty O] (p : α → Prop) :
    ∀ g : Game α, value p (withDummy O g) = value p g
  | .ret a => by
      classical
      simp only [withDummy, bind_ret, value_send, value_ret, ciSup_const]
  | .send M k => by
      simp only [withDummy, bind_send, value_send]
      exact iSup_congr fun m => value_withDummy O p (k m)
  | @Game.coin _ C _ _ k => by
      simp only [withDummy, bind_coin, value_coin]
      exact Finset.sum_congr rfl fun c _ => by rw [← value_withDummy O p (k c)]; rfl

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

/-! ## Alignment -/

section Aligned

theorem Aligned.map {α₂ β₂ : Type} {R₁ R₂ : I → α → β → Prop} {R₁' R₂' : I → α₂ → β₂ → Prop}
    {f₁ f₂ : α → α₂} {f' : β → β₂} {h₁ : ∀ i a b, R₁ i a b → R₁' i (f₁ a) (f' b)}
    {h₂ : ∀ i a b, R₂ i a b → R₂' i (f₂ a) (f' b)} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : Aligned L₁ L₂) :
    Aligned (Lock.map f₁ f' h₁ L₁) (Lock.map f₂ f' h₂ L₂) := by
  unfold Lock.map
  exact Aligned.bind (fun i₁ a₁ i₂ a₂ b r₁ r₂ => .ret (h₁ i₁ a₁ b r₁) (h₂ i₂ a₂ b r₂)) h

theorem Aligned.mono {R₁ R₂ R₁' R₂' : I → α → β → Prop} {h₁ : ∀ i a b, R₁ i a b → R₁' i a b}
    {h₂ : ∀ i a b, R₂ i a b → R₂' i a b} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : Aligned L₁ L₂) :
    Aligned (Lock.mono h₁ L₁) (Lock.mono h₂ L₂) := by
  induction h with
  | ret r₁ r₂ => exact .ret _ _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s ih

theorem Aligned.mapL {α₂ : Type} {R₁ R₂ : I → α → β → Prop} {R₁' R₂' : I → α₂ → β → Prop}
    {f₁ f₂ : α → α₂} {h₁ : ∀ i a b, R₁ i a b → R₁' i (f₁ a) b}
    {h₂ : ∀ i a b, R₂ i a b → R₂' i (f₂ a) b} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : Aligned L₁ L₂) :
    Aligned (Lock.mapL f₁ h₁ L₁) (Lock.mapL f₂ h₂ L₂) := by
  induction h with
  | ret r₁ r₂ => exact .ret _ _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s ih

theorem Aligned.same {i₁ i₂ : I} : ∀ g : Game α,
    Aligned (Lock.same (T := T) (D := D) (get := get) (i := i₁) g) (Lock.same (i := i₂) g)
  | .ret _ => .ret _ _
  | @Game.coin _ C iF iN k => @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
      Aligned.same (k c)
  | .send M k => .send M fun m => Aligned.same (k m)

theorem Aligned.lockWithSame {R₂ R₂' : I → α × γ → β × γ → Prop} {i i' : I}
    {adv adv₂ : Game γ → Game (α × γ)} {advC : Game γ → Game (β × γ)}
    {fin fin₂ : γ → Game (α × γ)} {finC : γ → Game (β × γ)}
    {hadv : ∀ h : Game γ, Lock T D get R₂ i (adv h) (advC h)}
    {hadv' : ∀ h : Game γ, Lock T D get R₂' i' (adv₂ h) (advC h)}
    {hfin : ∀ c : γ, Lock T D get R₂ i (fin c) (finC c)}
    {hfin' : ∀ c : γ, Lock T D get R₂' i' (fin₂ c) (finC c)}
    (hA : ∀ h, Aligned (hadv h) (hadv' h)) (hF : ∀ c, Aligned (hfin c) (hfin' c)) :
    ∀ h : Game γ, Aligned (Lock.lockWithSame hadv hfin h) (Lock.lockWithSame hadv' hfin' h)
  | .ret c => hF c
  | .send M k => by
      simp only [Lock.lockWithSame]
      refine .send Bool fun first => ?_
      cases first
      · exact .send M fun m => Aligned.lockWithSame hA hF (k m)
      · exact hA _
  | @Game.coin _ C iF iN k => by
      simp only [Lock.lockWithSame]
      refine .send Bool fun first => ?_
      cases first
      · exact @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          Aligned.lockWithSame hA hF (k c)
      · exact hA _

theorem Aligned.leftSame {R₁ R₂ : I → α → β → Prop} :
    ∀ {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β} {L₁ : Lock T D get R₁ i₁ g₁ g'}
      {L₂ : Lock T D get R₂ i₂ g₂ g'}, Aligned L₁ L₂ → ∀ h : Game γ,
        Aligned (Lock.leftSame L₁ h) (Lock.leftSame L₂ h)
  | _, _, _, _, _, _, _, .ret r₁ r₂, h => Aligned.map (Aligned.same h)
  | _, _, _, _, _, _, _, @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ L₁ L₂ hL, h =>
      Aligned.lockWithSame
        (hadv := fun h' => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => Lock.leftSame (L₁ c) h')
        (hadv' := fun h' => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => Lock.leftSame (L₂ c) h')
        (fun h' => @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          Aligned.leftSame (hL c) h')
        (fun _ => Aligned.map (@Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL)) h
  | _, _, _, _, _, _, _, @Aligned.send _ _ _ _ _ _ _ _ _ _ M _ _ _ L₁ L₂ hL, h =>
      Aligned.lockWithSame
        (hadv := fun h' => Lock.send M fun m => Lock.leftSame (L₁ m) h')
        (hadv' := fun h' => Lock.send M fun m => Lock.leftSame (L₂ m) h')
        (fun h' => .send M fun m => Aligned.leftSame (hL m) h')
        (fun _ => Aligned.map (.send M hL)) h
  | _, _, _, _, _, _, _,
      @Aligned.table _ _ _ _ _ _ _ _ _ _ s upd₁ upd₂ _ _ _ hf₁ hs₁ hk₁ hf₂ hs₂ hk₂ L₁ L₂ hL, h =>
      Aligned.lockWithSame
        (hadv := fun h' => Lock.table s upd₁ hf₁ hs₁ hk₁ fun t d => Lock.leftSame (L₁ t d) h')
        (hadv' := fun h' => Lock.table s upd₂ hf₂ hs₂ hk₂ fun t d => Lock.leftSame (L₂ t d) h')
        (fun h' => .table s fun t d => Aligned.leftSame (hL t d) h')
        (fun _ => Aligned.map (.table s hL)) h

theorem Aligned.lockWithL {R₁ R₂ : I → α → β → Prop} {R₁' R₂' : I → γ × α → γ × β → Prop}
    {adv adv₂ : Game α → Game (γ × α)} {advC : Game β → Game (γ × β)}
    {fin fin₂ : α → Game (γ × α)} {finC : β → Game (γ × β)}
    {hadv : ∀ {i : I} {g : Game α} {g' : Game β}, Lock T D get R₁ i g g' →
      Lock T D get R₁' i (adv g) (advC g')}
    {hadv' : ∀ {i : I} {g : Game α} {g' : Game β}, Lock T D get R₂ i g g' →
      Lock T D get R₂' i (adv₂ g) (advC g')}
    {hfin : ∀ {i : I} {a : α} {b : β}, R₁ i a b → Lock T D get R₁' i (fin a) (finC b)}
    {hfin' : ∀ {i : I} {a : α} {b : β}, R₂ i a b → Lock T D get R₂' i (fin₂ a) (finC b)}
    (hA : ∀ {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β} {L : Lock T D get R₁ i₁ g₁ g'}
      {L' : Lock T D get R₂ i₂ g₂ g'}, Aligned L L' → Aligned (hadv L) (hadv' L'))
    (hF : ∀ {i₁ i₂ : I} {a₁ a₂ : α} {b : β} (r : R₁ i₁ a₁ b) (r' : R₂ i₂ a₂ b),
      Aligned (hfin r) (hfin' r')) :
    ∀ {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β} {L : Lock T D get R₁ i₁ g₁ g'}
      {L' : Lock T D get R₂ i₂ g₂ g'}, Aligned L L' →
        Aligned (Lock.lockWithL hadv hfin L) (Lock.lockWithL hadv' hfin' L')
  | _, _, _, _, _, _, _, .ret r r' => hF r r'
  | _, _, _, _, _, _, _, @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL => by
      simp only [Lock.lockWithL]
      refine .send Bool fun first => ?_
      cases first
      · exact @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          Aligned.lockWithL hA hF (hL c)
      · exact hA (@Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL)
  | _, _, _, _, _, _, _, .send M hL => by
      simp only [Lock.lockWithL]
      refine .send Bool fun first => ?_
      cases first
      · exact .send M fun m => Aligned.lockWithL hA hF (hL m)
      · exact hA (.send M hL)
  | _, _, _, _, _, _, _, .table s hL => by
      simp only [Lock.lockWithL]
      refine .send Bool fun first => ?_
      cases first
      · exact .table s fun t d => Aligned.lockWithL hA hF (hL t d)
      · exact hA (.table s hL)

theorem Aligned.rightSame {R₁ R₂ : I → α → β → Prop} :
    ∀ (h : Game γ) {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β} {L₁ : Lock T D get R₁ i₁ g₁ g'}
      {L₂ : Lock T D get R₂ i₂ g₂ g'}, Aligned L₁ L₂ →
        Aligned (Lock.rightSame h L₁) (Lock.rightSame h L₂)
  | .ret c, _, _, _, _, _, _, _, hL => by
      simp only [Lock.rightSame]
      exact Aligned.map hL
  | .send M k, _, _, _, _, _, _, _, hL => by
      simp only [Lock.rightSame]
      exact Aligned.lockWithL
        (hadv := fun L' => Lock.send M fun m => Lock.rightSame (k m) L')
        (hadv' := fun L' => Lock.send M fun m => Lock.rightSame (k m) L')
        (hfin := fun hr => Lock.sameThen (.send M k) hr)
        (hfin' := fun hr => Lock.sameThen (.send M k) hr)
        (fun hL' => .send M fun m => Aligned.rightSame (k m) hL')
        (fun _ _ => Aligned.map (Aligned.same _)) hL
  | @Game.coin _ C iF iN k, _, _, _, _, _, _, _, hL => by
      simp only [Lock.rightSame]
      exact Aligned.lockWithL
        (hadv := fun L' => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => Lock.rightSame (k c) L')
        (hadv' := fun L' => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => Lock.rightSame (k c) L')
        (hfin := fun hr => Lock.sameThen (@Game.coin _ C iF iN k) hr)
        (hfin' := fun hr => Lock.sameThen (@Game.coin _ C iF iN k) hr)
        (fun hL' => @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          Aligned.rightSame (k c) hL')
        (fun _ _ => Aligned.map (Aligned.same _)) hL

/-- **Batching preserves alignment.** -/
theorem Aligned.batchLock {ζ : Type} {R₁ R₂ : I → ζ → ζ → Prop} (d : ζ) {i₁ i₂ : I}
    {g₁ g₂ g' : Game ζ} {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'}
    (hL : Aligned L₁ L₂) (post : List (Game ζ)) :
    ∀ pre : List (Game ζ),
      Aligned (Lock.batchLock d L₁ post pre) (Lock.batchLock d L₂ post pre)
  | [] => Aligned.map (Aligned.leftSame hL (batch post))
  | h :: pre => Aligned.map (Aligned.rightSame h (Aligned.batchLock d hL post pre))

end Aligned

end Lock

end Game

end FlockSoundness
