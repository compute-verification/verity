import Proofs.Core.Game.Lock
import Definitions.Core.Game.Batch

/-!
# One lockstep among identical games

A batched session runs many tables concurrently, the prover choosing the schedule (`batch`). To
compile one table while every other game stays as it is, that table's lockstep runs beside the
same game on both sides: `leftSame` and `rightSame` for `interleave`, and `batchLock` for a batch
with the table at position `pre.length`. The index stays the table's, so its slots and choosers
carry over, and alignment is preserved (`Aligned.batchLock`).

`value_batch_getD_le`: an event of one position's output is at most as likely, for an optimal
prover, as the same event in that position's game alone.

The definitions are in `Definitions.Core.Game.Batch`.
-/

namespace Verity

namespace Game

variable {α β γ : Type}

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

end Verity
