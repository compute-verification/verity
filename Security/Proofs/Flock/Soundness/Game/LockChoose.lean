import Proofs.Flock.Soundness.Game.Batch

/-!
# Fork probabilities that do not read the chooser

`forkE` reads the compiled side and the slots; a chooser enters only through the lockstep it selects below each table
node. Two locksteps whose structure below a table node is the same whatever table is chosen there (`AlignedT`: `Aligned`
with its table case over any two tables) have the same fork probability under any two choosers (`forkE_choose`).

* `forkE_mono`: the fork probability is monotone in the event it counts;
* `AlignedT.bind`, `.map`, `.mono`, `.mapL`, `.same`, `.leftSame`, `.rightSame`, `.batchLock`, `.embedLeft`,
  `.embedRight`, `.lockWith`, `.interleaveLock`: the combinators that build the compiled sessions' locksteps preserve it.
-/

namespace FlockSoundness

namespace Game

variable {α β γ : Type}

namespace Lock

variable {T D : Type}

/-- **Two locksteps of one compiled game with the same slots, whatever tables are chosen**: `Aligned`, with the
locksteps below a table node aligned for any two tables. -/
inductive AlignedT {I : Type} {get : I → ℕ → Option (T × D)} {R₁ R₂ : I → α → β → Prop} :
    {i₁ i₂ : I} → {g₁ g₂ : Game α} → {g' : Game β} →
      Lock T D get R₁ i₁ g₁ g' → Lock T D get R₂ i₂ g₂ g' → Prop
  | ret {i₁ i₂ : I} {a₁ a₂ : α} {b : β} (h₁ : R₁ i₁ a₁ b) (h₂ : R₂ i₂ a₂ b) :
      AlignedT (Lock.ret h₁) (Lock.ret h₂)
  | coin {i₁ i₂ : I} (C : Type) [Fintype C] [Nonempty C] {k₁ k₂ : C → Game α} {k' : C → Game β}
      {L₁ : (c : C) → Lock T D get R₁ i₁ (k₁ c) (k' c)}
      {L₂ : (c : C) → Lock T D get R₂ i₂ (k₂ c) (k' c)} :
      (∀ c, AlignedT (L₁ c) (L₂ c)) → AlignedT (Lock.coin C L₁) (Lock.coin C L₂)
  | send {i₁ i₂ : I} (M : Type) {k₁ k₂ : M → Game α} {k' : M → Game β}
      {L₁ : (m : M) → Lock T D get R₁ i₁ (k₁ m) (k' m)}
      {L₂ : (m : M) → Lock T D get R₂ i₂ (k₂ m) (k' m)} :
      (∀ m, AlignedT (L₁ m) (L₂ m)) → AlignedT (Lock.send M L₁) (Lock.send M L₂)
  | table {i₁ i₂ : I} (slot : ℕ) {upd₁ upd₂ : T → D → I} {k₁ k₂ : T → Game α} {k' : D → Game β}
      {hf₁ : get i₁ slot = none} {hs₁ : ∀ t d, get (upd₁ t d) slot = some (t, d)}
      {hk₁ : ∀ t d s, s ≠ slot → get (upd₁ t d) s = get i₁ s}
      {hf₂ : get i₂ slot = none} {hs₂ : ∀ t d, get (upd₂ t d) slot = some (t, d)}
      {hk₂ : ∀ t d s, s ≠ slot → get (upd₂ t d) s = get i₂ s}
      {L₁ : (t : T) → (d : D) → Lock T D get R₁ (upd₁ t d) (k₁ t) (k' d)}
      {L₂ : (t : T) → (d : D) → Lock T D get R₂ (upd₂ t d) (k₂ t) (k' d)} :
      (∀ t₁ t₂ d, AlignedT (L₁ t₁ d) (L₂ t₂ d)) →
        AlignedT (Lock.table slot upd₁ hf₁ hs₁ hk₁ L₁) (Lock.table slot upd₂ hf₂ hs₂ hk₂ L₂)

variable {I : Type} {get : I → ℕ → Option (T × D)}

/-- **Locksteps aligned whatever the tables have the same fork probability under any two choosers.** -/
theorem forkE_choose (plur₁ plur₂ : Chooser T D β) (slot : ℕ) (Φ : D → β → β → ℝ)
    {R₁ R₂ : I → α → β → Prop} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : AlignedT L₁ L₂) :
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
      · exact ih _ _ _ σ'.2

/-- **The fork probability is monotone in the event it counts.** -/
theorem forkE_mono {R : I → α → β → Prop} (plur : Chooser T D β) (slot : ℕ) {Φ Φ' : D → β → β → ℝ}
    (hΦ : ∀ d b b', Φ d b b' ≤ Φ' d b b') :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      forkE plur slot Φ L σ' ≤ forkE plur slot Φ' L σ'
  | _, _, _, .ret _, _ => le_rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [forkE]
      exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => forkE_mono plur slot hΦ (L c) (σ' c))
        (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ' => forkE_mono plur slot hΦ (L σ'.1) σ'.2
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' _ _ _ L, σ' => by
      simp only [forkE]
      split_ifs
      · exact expect_mono (fun b => expect_mono (hΦ σ'.1 b) _ _) _ _
      · exact forkE_mono plur slot hΦ _ σ'.2

/-! ## The combinators -/

theorem AlignedT.bind {α₂ β₂ : Type} {R₁ R₂ : I → α → β → Prop} {R₁' R₂' : I → α₂ → β₂ → Prop}
    {f₁ f₂ : α → Game α₂} {f' : β → Game β₂}
    {h₁ : ∀ i a b, R₁ i a b → Lock T D get R₁' i (f₁ a) (f' b)}
    {h₂ : ∀ i a b, R₂ i a b → Lock T D get R₂' i (f₂ a) (f' b)}
    (hh : ∀ i₁ a₁ i₂ a₂ b (r₁ : R₁ i₁ a₁ b) (r₂ : R₂ i₂ a₂ b),
      AlignedT (h₁ i₁ a₁ b r₁) (h₂ i₂ a₂ b r₂))
    {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : AlignedT L₁ L₂) :
    AlignedT (Lock.bind h₁ L₁) (Lock.bind h₂ L₂) := by
  induction h with
  | ret r₁ r₂ => exact hh _ _ _ _ _ r₁ r₂
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s ih

theorem AlignedT.map {α₂ β₂ : Type} {R₁ R₂ : I → α → β → Prop} {R₁' R₂' : I → α₂ → β₂ → Prop}
    {f₁ f₂ : α → α₂} {f' : β → β₂} {h₁ : ∀ i a b, R₁ i a b → R₁' i (f₁ a) (f' b)}
    {h₂ : ∀ i a b, R₂ i a b → R₂' i (f₂ a) (f' b)} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : AlignedT L₁ L₂) :
    AlignedT (Lock.map f₁ f' h₁ L₁) (Lock.map f₂ f' h₂ L₂) := by
  unfold Lock.map
  exact AlignedT.bind (fun i₁ a₁ i₂ a₂ b r₁ r₂ => .ret (h₁ i₁ a₁ b r₁) (h₂ i₂ a₂ b r₂)) h

theorem AlignedT.mono {R₁ R₂ R₁' R₂' : I → α → β → Prop} {h₁ : ∀ i a b, R₁ i a b → R₁' i a b}
    {h₂ : ∀ i a b, R₂ i a b → R₂' i a b} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : AlignedT L₁ L₂) :
    AlignedT (Lock.mono h₁ L₁) (Lock.mono h₂ L₂) := by
  induction h with
  | ret r₁ r₂ => exact .ret _ _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s ih

theorem AlignedT.mapL {α₂ : Type} {R₁ R₂ : I → α → β → Prop} {R₁' R₂' : I → α₂ → β → Prop}
    {f₁ f₂ : α → α₂} {h₁ : ∀ i a b, R₁ i a b → R₁' i (f₁ a) b}
    {h₂ : ∀ i a b, R₂ i a b → R₂' i (f₂ a) b} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : AlignedT L₁ L₂) :
    AlignedT (Lock.mapL f₁ h₁ L₁) (Lock.mapL f₂ h₂ L₂) := by
  induction h with
  | ret r₁ r₂ => exact .ret _ _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s ih

theorem AlignedT.same {i₁ i₂ : I} : ∀ g : Game α,
    AlignedT (Lock.same (T := T) (D := D) (get := get) (i := i₁) g) (Lock.same (i := i₂) g)
  | .ret _ => .ret _ _
  | @Game.coin _ C iF iN k => @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
      AlignedT.same (k c)
  | .send M k => .send M fun m => AlignedT.same (k m)

theorem AlignedT.lockWithSame {R₂ R₂' : I → α × γ → β × γ → Prop} {i i' : I}
    {adv adv₂ : Game γ → Game (α × γ)} {advC : Game γ → Game (β × γ)}
    {fin fin₂ : γ → Game (α × γ)} {finC : γ → Game (β × γ)}
    {hadv : ∀ h : Game γ, Lock T D get R₂ i (adv h) (advC h)}
    {hadv' : ∀ h : Game γ, Lock T D get R₂' i' (adv₂ h) (advC h)}
    {hfin : ∀ c : γ, Lock T D get R₂ i (fin c) (finC c)}
    {hfin' : ∀ c : γ, Lock T D get R₂' i' (fin₂ c) (finC c)}
    (hA : ∀ h, AlignedT (hadv h) (hadv' h)) (hF : ∀ c, AlignedT (hfin c) (hfin' c)) :
    ∀ h : Game γ, AlignedT (Lock.lockWithSame hadv hfin h) (Lock.lockWithSame hadv' hfin' h)
  | .ret c => hF c
  | .send M k => by
      simp only [Lock.lockWithSame]
      refine .send Bool fun first => ?_
      cases first
      · exact .send M fun m => AlignedT.lockWithSame hA hF (k m)
      · exact hA _
  | @Game.coin _ C iF iN k => by
      simp only [Lock.lockWithSame]
      refine .send Bool fun first => ?_
      cases first
      · exact @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          AlignedT.lockWithSame hA hF (k c)
      · exact hA _

theorem AlignedT.leftSame {R₁ R₂ : I → α → β → Prop} :
    ∀ {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β} {L₁ : Lock T D get R₁ i₁ g₁ g'}
      {L₂ : Lock T D get R₂ i₂ g₂ g'}, AlignedT L₁ L₂ → ∀ h : Game γ,
        AlignedT (Lock.leftSame L₁ h) (Lock.leftSame L₂ h)
  | _, _, _, _, _, _, _, .ret r₁ r₂, h => AlignedT.map (AlignedT.same h)
  | _, _, _, _, _, _, _, @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ L₁ L₂ hL, h =>
      AlignedT.lockWithSame
        (hadv := fun h' => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => Lock.leftSame (L₁ c) h')
        (hadv' := fun h' => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => Lock.leftSame (L₂ c) h')
        (fun h' => @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          AlignedT.leftSame (hL c) h')
        (fun _ => AlignedT.map (@AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL)) h
  | _, _, _, _, _, _, _, @AlignedT.send _ _ _ _ _ _ _ _ _ _ M _ _ _ L₁ L₂ hL, h =>
      AlignedT.lockWithSame
        (hadv := fun h' => Lock.send M fun m => Lock.leftSame (L₁ m) h')
        (hadv' := fun h' => Lock.send M fun m => Lock.leftSame (L₂ m) h')
        (fun h' => .send M fun m => AlignedT.leftSame (hL m) h')
        (fun _ => AlignedT.map (.send M hL)) h
  | _, _, _, _, _, _, _,
      @AlignedT.table _ _ _ _ _ _ _ _ _ _ s upd₁ upd₂ _ _ _ hf₁ hs₁ hk₁ hf₂ hs₂ hk₂ L₁ L₂ hL, h =>
      AlignedT.lockWithSame
        (hadv := fun h' => Lock.table s upd₁ hf₁ hs₁ hk₁ fun t d => Lock.leftSame (L₁ t d) h')
        (hadv' := fun h' => Lock.table s upd₂ hf₂ hs₂ hk₂ fun t d => Lock.leftSame (L₂ t d) h')
        (fun h' => .table s fun t₁ t₂ d => AlignedT.leftSame (hL t₁ t₂ d) h')
        (fun _ => AlignedT.map (.table s hL)) h

theorem AlignedT.lockWithL {R₁ R₂ : I → α → β → Prop} {R₁' R₂' : I → γ × α → γ × β → Prop}
    {adv adv₂ : Game α → Game (γ × α)} {advC : Game β → Game (γ × β)}
    {fin fin₂ : α → Game (γ × α)} {finC : β → Game (γ × β)}
    {hadv : ∀ {i : I} {g : Game α} {g' : Game β}, Lock T D get R₁ i g g' →
      Lock T D get R₁' i (adv g) (advC g')}
    {hadv' : ∀ {i : I} {g : Game α} {g' : Game β}, Lock T D get R₂ i g g' →
      Lock T D get R₂' i (adv₂ g) (advC g')}
    {hfin : ∀ {i : I} {a : α} {b : β}, R₁ i a b → Lock T D get R₁' i (fin a) (finC b)}
    {hfin' : ∀ {i : I} {a : α} {b : β}, R₂ i a b → Lock T D get R₂' i (fin₂ a) (finC b)}
    (hA : ∀ {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β} {L : Lock T D get R₁ i₁ g₁ g'}
      {L' : Lock T D get R₂ i₂ g₂ g'}, AlignedT L L' → AlignedT (hadv L) (hadv' L'))
    (hF : ∀ {i₁ i₂ : I} {a₁ a₂ : α} {b : β} (r : R₁ i₁ a₁ b) (r' : R₂ i₂ a₂ b),
      AlignedT (hfin r) (hfin' r')) :
    ∀ {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β} {L : Lock T D get R₁ i₁ g₁ g'}
      {L' : Lock T D get R₂ i₂ g₂ g'}, AlignedT L L' →
        AlignedT (Lock.lockWithL hadv hfin L) (Lock.lockWithL hadv' hfin' L')
  | _, _, _, _, _, _, _, .ret r r' => hF r r'
  | _, _, _, _, _, _, _, @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL => by
      simp only [Lock.lockWithL]
      refine .send Bool fun first => ?_
      cases first
      · exact @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          AlignedT.lockWithL hA hF (hL c)
      · exact hA (@AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL)
  | _, _, _, _, _, _, _, .send M hL => by
      simp only [Lock.lockWithL]
      refine .send Bool fun first => ?_
      cases first
      · exact .send M fun m => AlignedT.lockWithL hA hF (hL m)
      · exact hA (.send M hL)
  | _, _, _, _, _, _, _, .table s hL => by
      simp only [Lock.lockWithL]
      refine .send Bool fun first => ?_
      cases first
      · exact .table s fun t₁ t₂ d => AlignedT.lockWithL hA hF (hL t₁ t₂ d)
      · exact hA (.table s hL)

theorem AlignedT.rightSame {R₁ R₂ : I → α → β → Prop} :
    ∀ (h : Game γ) {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β} {L₁ : Lock T D get R₁ i₁ g₁ g'}
      {L₂ : Lock T D get R₂ i₂ g₂ g'}, AlignedT L₁ L₂ →
        AlignedT (Lock.rightSame h L₁) (Lock.rightSame h L₂)
  | .ret c, _, _, _, _, _, _, _, hL => by
      simp only [Lock.rightSame]
      exact AlignedT.map hL
  | .send M k, _, _, _, _, _, _, _, hL => by
      simp only [Lock.rightSame]
      exact AlignedT.lockWithL
        (hadv := fun L' => Lock.send M fun m => Lock.rightSame (k m) L')
        (hadv' := fun L' => Lock.send M fun m => Lock.rightSame (k m) L')
        (hfin := fun hr => Lock.sameThen (.send M k) hr)
        (hfin' := fun hr => Lock.sameThen (.send M k) hr)
        (fun hL' => .send M fun m => AlignedT.rightSame (k m) hL')
        (fun _ _ => AlignedT.map (AlignedT.same _)) hL
  | @Game.coin _ C iF iN k, _, _, _, _, _, _, _, hL => by
      simp only [Lock.rightSame]
      exact AlignedT.lockWithL
        (hadv := fun L' => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => Lock.rightSame (k c) L')
        (hadv' := fun L' => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => Lock.rightSame (k c) L')
        (hfin := fun hr => Lock.sameThen (@Game.coin _ C iF iN k) hr)
        (hfin' := fun hr => Lock.sameThen (@Game.coin _ C iF iN k) hr)
        (fun hL' => @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          AlignedT.rightSame (k c) hL')
        (fun _ _ => AlignedT.map (AlignedT.same _)) hL

theorem AlignedT.batchLock {ζ : Type} {R₁ R₂ : I → ζ → ζ → Prop} (d : ζ) {i₁ i₂ : I}
    {g₁ g₂ g' : Game ζ} {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'}
    (hL : AlignedT L₁ L₂) (post : List (Game ζ)) :
    ∀ pre : List (Game ζ),
      AlignedT (Lock.batchLock d L₁ post pre) (Lock.batchLock d L₂ post pre)
  | [] => AlignedT.map (AlignedT.leftSame hL (batch post))
  | h :: pre => AlignedT.map (AlignedT.rightSame h (AlignedT.batchLock d hL post pre))

section InterleaveAlignedT

variable {α₁ β₁ α₂ β₂ I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (T × D)}
  {get₂ : I₂ → ℕ → Option (T × D)} [Inhabited T] [Inhabited D]
  {R₁ R₁' : I₁ → α₁ → β₁ → Prop} {R₂ R₂' : I₂ → α₂ → β₂ → Prop}

theorem AlignedT.embedLeft (hd : Disjoint get₁ get₂) {i₂ i₂' : I₂} {a₂ a₂' : α₂} {b₂ : β₂}
    (h₂ : R₂ i₂ a₂ b₂) (h₂' : R₂' i₂' a₂' b₂) {i₁ i₁' : I₁} {g₁ g₁'' : Game α₁} {g₁' : Game β₁}
    {L : Lock T D get₁ R₁ i₁ g₁ g₁'} {L' : Lock T D get₁ R₁' i₁' g₁'' g₁'} (h : AlignedT L L') :
    AlignedT (embedLeft hd h₂ L) (embedLeft hd h₂' L') := by
  induction h with
  | ret r r' => exact .ret ⟨r, h₂⟩ ⟨r', h₂'⟩
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s ih

theorem AlignedT.embedRight (hd : Disjoint get₁ get₂) {i₁ i₁' : I₁} {a₁ a₁' : α₁} {b₁ : β₁}
    (h₁ : R₁ i₁ a₁ b₁) (h₁' : R₁' i₁' a₁' b₁) {i₂ i₂' : I₂} {g₂ g₂'' : Game α₂} {g₂' : Game β₂}
    {M : Lock T D get₂ R₂ i₂ g₂ g₂'} {M' : Lock T D get₂ R₂' i₂' g₂'' g₂'} (h : AlignedT M M') :
    AlignedT (embedRight hd h₁ M) (embedRight hd h₁' M') := by
  induction h with
  | ret r r' => exact .ret ⟨h₁, r⟩ ⟨h₁', r'⟩
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s ih

theorem AlignedT.lockWith (hd : Disjoint get₁ get₂) {i₁ i₁' : I₁}
    {adv₁ adv₁' : Game α₂ → Game (α₁ × α₂)} {advC : Game β₂ → Game (β₁ × β₂)}
    {fin₁ fin₁' : α₂ → Game (α₁ × α₂)} {finC : β₂ → Game (β₁ × β₂)}
    {hadv : ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂}, Lock T D get₂ R₂ i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (adv₁ g₂) (advC g₂')}
    {hadv' : ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂}, Lock T D get₂ R₂' i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁' R₂') (i₁', i₂) (adv₁' g₂) (advC g₂')}
    {hfin : ∀ {i₂ : I₂} {a₂ : α₂} {b₂ : β₂}, R₂ i₂ a₂ b₂ →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (fin₁ a₂) (finC b₂)}
    {hfin' : ∀ {i₂ : I₂} {a₂ : α₂} {b₂ : β₂}, R₂' i₂ a₂ b₂ →
      Lock T D (getPair get₁ get₂) (RPair R₁' R₂') (i₁', i₂) (fin₁' a₂) (finC b₂)}
    (hA : ∀ {i₂ i₂' : I₂} {g₂ g₂'' : Game α₂} {g₂' : Game β₂} {M : Lock T D get₂ R₂ i₂ g₂ g₂'}
      {M' : Lock T D get₂ R₂' i₂' g₂'' g₂'}, AlignedT M M' → AlignedT (hadv M) (hadv' M'))
    (hF : ∀ {i₂ i₂' : I₂} {a₂ a₂' : α₂} {b₂ : β₂} (r : R₂ i₂ a₂ b₂) (r' : R₂' i₂' a₂' b₂),
      AlignedT (hfin r) (hfin' r')) :
    ∀ {i₂ i₂' : I₂} {g₂ g₂'' : Game α₂} {g₂' : Game β₂} {M : Lock T D get₂ R₂ i₂ g₂ g₂'}
      {M' : Lock T D get₂ R₂' i₂' g₂'' g₂'}, AlignedT M M' →
        AlignedT (lockWith hd hadv hfin M) (lockWith hd hadv' hfin' M')
  | _, _, _, _, _, _, _, .ret r r' => hF r r'
  | _, _, _, _, _, _, _, @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL => by
      simp only [Lock.lockWith]
      refine .send Bool fun first => ?_
      cases first
      · exact @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          AlignedT.lockWith hd hA hF (hL c)
      · exact hA (@AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL)
  | _, _, _, _, _, _, _, .send M hL => by
      simp only [Lock.lockWith]
      refine .send Bool fun first => ?_
      cases first
      · exact .send M fun m => AlignedT.lockWith hd hA hF (hL m)
      · exact hA (.send M hL)
  | _, _, _, _, _, _, _, .table s hL => by
      simp only [Lock.lockWith]
      refine .send Bool fun first => ?_
      cases first
      · exact .table s fun t₁ t₂ d => AlignedT.lockWith hd hA hF (hL t₁ t₂ d)
      · exact hA (.table s hL)

theorem AlignedT.interleaveLock (hd : Disjoint get₁ get₂) :
    ∀ {i₁ i₁' : I₁} {g₁ g₁'' : Game α₁} {g₁' : Game β₁} {L : Lock T D get₁ R₁ i₁ g₁ g₁'}
      {L' : Lock T D get₁ R₁' i₁' g₁'' g₁'}, AlignedT L L' →
    ∀ {i₂ i₂' : I₂} {g₂ g₂'' : Game α₂} {g₂' : Game β₂} {M : Lock T D get₂ R₂ i₂ g₂ g₂'}
      {M' : Lock T D get₂ R₂' i₂' g₂'' g₂'}, AlignedT M M' →
        AlignedT (interleaveLock hd L M) (interleaveLock hd L' M')
  | _, _, _, _, _, _, _, .ret r r', _, _, _, _, _, _, _, hM => AlignedT.embedRight hd r r' hM
  | _, _, _, _, _, _, _, @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ L₁ L₂ hL,
      _, _, _, _, _, _, _, hM =>
      AlignedT.lockWith hd
        (hadv := fun M => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c =>
          Lock.interleaveLock hd (L₁ c) M)
        (hadv' := fun M => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c =>
          Lock.interleaveLock hd (L₂ c) M)
        (hfin := fun r => Lock.embedLeft hd r (@Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L₁))
        (hfin' := fun r => Lock.embedLeft hd r (@Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L₂))
        (fun hM' => @AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          AlignedT.interleaveLock hd (hL c) hM')
        (fun r r' => AlignedT.embedLeft hd r r' (@AlignedT.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL))
        hM
  | _, _, _, _, _, _, _, @AlignedT.send _ _ _ _ _ _ _ _ _ _ M _ _ _ L₁ L₂ hL,
      _, _, _, _, _, _, _, hM =>
      AlignedT.lockWith hd
        (hadv := fun M' => Lock.send M fun m => Lock.interleaveLock hd (L₁ m) M')
        (hadv' := fun M' => Lock.send M fun m => Lock.interleaveLock hd (L₂ m) M')
        (hfin := fun r => Lock.embedLeft hd r (Lock.send M L₁))
        (hfin' := fun r => Lock.embedLeft hd r (Lock.send M L₂))
        (fun hM' => .send M fun m => AlignedT.interleaveLock hd (hL m) hM')
        (fun r r' => AlignedT.embedLeft hd r r' (.send M hL)) hM
  | _, _, _, _, _, _, _,
      @AlignedT.table _ _ _ _ _ _ _ _ _ _ s upd₁ upd₂ _ _ _ hf₁ hs₁ hk₁ hf₂ hs₂ hk₂ L₁ L₂ hL,
      _, _, _, _, _, _, _, hM =>
      AlignedT.lockWith hd
        (hadv := fun {i₂ _ _} M => Lock.table s (fun t d => (upd₁ t d, i₂))
          (getPair_fresh₁ hd hf₁ hs₁ i₂) (getPair_set₁ hs₁ i₂)
          (fun t d s' h => getPair_keep₁ hk₁ i₂ t d s' h) fun t d => Lock.interleaveLock hd (L₁ t d) M)
        (hadv' := fun {i₂ _ _} M => Lock.table s (fun t d => (upd₂ t d, i₂))
          (getPair_fresh₁ hd hf₂ hs₂ i₂) (getPair_set₁ hs₂ i₂)
          (fun t d s' h => getPair_keep₁ hk₂ i₂ t d s' h) fun t d => Lock.interleaveLock hd (L₂ t d) M)
        (hfin := fun r => Lock.embedLeft hd r (Lock.table s upd₁ hf₁ hs₁ hk₁ L₁))
        (hfin' := fun r => Lock.embedLeft hd r (Lock.table s upd₂ hf₂ hs₂ hk₂ L₂))
        (fun hM' => .table s fun t₁ t₂ d => AlignedT.interleaveLock hd (hL t₁ t₂ d) hM')
        (fun r r' => AlignedT.embedLeft hd r r' (.table s hL)) hM

end InterleaveAlignedT

end Lock

end Game

end FlockSoundness
