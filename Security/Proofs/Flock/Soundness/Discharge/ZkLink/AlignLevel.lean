import Proofs.Flock.Soundness.Discharge.ZkLink.LockLevel

/-!
# The masked level 0's locksteps for two level-0 tables are aligned

`LockLevel`'s locksteps take the level-0 table `C₀` into the oracle games and the leaf relations only, never the
compiled side or the slots, so two tables give `Lock.Aligned` locksteps (`CompiledAlign` for the masked level 0).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

theorem aligned_lockQueryZ (A : Arith F K) (r : ℕ) (C₁ C₂ : Oracle F) (w₀ : ℕ) (i₁ i₂ : IRep F D) (lv : Level)
    (ch : List K) (start₁ start₂ start' : ℕ) (ht₁ : start₁ = start') (ht₂ : start₂ = start')
    (st₁ st₂ : LigState F K) (stC : LigStateC F K D) (h₁ : StRelZ r C₁ w₀ i₁ st₁ stC)
    (h₂ : StRelZ r C₂ w₀ i₂ st₂ stC) (hl₁ hl₂ : stC.level ≠ 0) (wm : Bool) :
    Lock.Aligned (lockQueryZ A r C₁ w₀ i₁ lv ch start₁ start' ht₁ st₁ stC h₁ hl₁ wm)
      (lockQueryZ A r C₂ w₀ i₂ lv ch start₂ start' ht₂ st₂ stC h₂ hl₂ wm) := by
  subst ht₁
  subst ht₂
  cases wm
  · exact .coin _ fun _ => .coin _ fun _ => .coin _ fun _ => .ret _ _
  · exact .coin _ fun _ => .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _ _

theorem aligned_lockRecZ (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K)
    (ext : ℕ) (C₁ C₂ : Oracle F) (w₀ : ℕ) (sch : Schedule) :
    ∀ (levs : List Level) (i₁ i₂ : IRep F D) (st₁ st₂ : LigState F K) (stC : LigStateC F K D)
      (h₁ : RecInvZ r C₁ w₀ sch levs i₁ st₁ stC) (h₂ : RecInvZ r C₂ w₀ sch levs i₂ st₂ stC),
      Lock.Aligned (lockRecZ A r hr n b initial ext C₁ w₀ sch levs i₁ st₁ stC h₁)
        (lockRecZ A r hr n b initial ext C₂ w₀ sch levs i₂ st₂ stC h₂)
  | [], _, _, _, _, _, _, _ => .ret _ _
  | [lv], i₁, i₂, st₁, st₂, stC, h₁, h₂ => by
      refine Lock.Aligned.bind ?_ (aligned_lockFold A r lv.lanes i₁ i₂ _ _ _ _ _ _ _ _ _ _)
      intro i₁' x₁ i₂' x₂ y r₁ r₂
      rcases x₁ with ⟨ch₁, x₁⟩
      rcases x₁ with ⟨s₁, q₁⟩
      rcases x₂ with ⟨ch₂, x₂⟩
      rcases x₂ with ⟨s₂, q₂⟩
      rcases y with ⟨ch, y⟩
      rcases y with ⟨sF, qF⟩
      obtain ⟨hi₁, hch₁, hlen₁, hsq₁⟩ := r₁
      obtain ⟨hi₂, hch₂, hlen₂, hsq₂⟩ := r₂
      simp only at hch₁ hch₂
      subst i₁' i₂' hch₁ hch₂
      refine .send _ fun yr => Lock.Aligned.bind ?_
        (aligned_lockQueryZ A r C₁ C₂ w₀ i₁ i₂ lv _ _ _ _ _ _ _ _ _ _ _ _ _ false)
      intro i3₁ st3₁ i3₂ st3₂ stC3 hq₁ hq₂
      obtain ⟨hi3₁, _⟩ := hq₁
      obtain ⟨hi3₂, _⟩ := hq₂
      subst i3₁ i3₂
      exact .ret _ _
  | lv :: lv' :: rest, i₁, i₂, st₁, st₂, stC, h₁, h₂ => by
      refine Lock.Aligned.bind ?_ (aligned_lockFold A r lv.lanes i₁ i₂ _ _ _ _ _ _ _ _ _ _)
      intro i₁' x₁ i₂' x₂ y r₁ r₂
      rcases x₁ with ⟨ch₁, x₁⟩
      rcases x₁ with ⟨s₁, q₁⟩
      rcases x₂ with ⟨ch₂, x₂⟩
      rcases x₂ with ⟨s₂, q₂⟩
      rcases y with ⟨ch, y⟩
      rcases y with ⟨sF, qF⟩
      obtain ⟨hi₁, hch₁, hlen₁, hsq₁⟩ := r₁
      obtain ⟨hi₂, hch₂, hlen₂, hsq₂⟩ := r₂
      simp only at hch₁ hch₂
      subst i₁' i₂' hch₁ hch₂
      refine .table _ fun C' cap => ?_
      refine Lock.Aligned.bind ?_ (aligned_lockTwoOod A r _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)
      intro i₂₁ x₁ i₂₂ x₂ y r₁ r₂
      rcases x₁ with ⟨s₁', x₁⟩
      rcases x₁ with ⟨q₁', oods₁⟩
      rcases x₂ with ⟨s₂', x₂⟩
      rcases x₂ with ⟨q₂', oods₂⟩
      rcases y with ⟨sF', y⟩
      rcases y with ⟨qF', oodsC⟩
      obtain ⟨hi₁, hoods₁, _⟩ := r₁
      obtain ⟨hi₂, hoods₂, _⟩ := r₂
      simp only at hoods₁ hoods₂
      subst i₂₁ i₂₂ hoods₁ hoods₂
      refine Lock.Aligned.bind ?_
        (aligned_lockQueryZ A r C₁ C₂ w₀ _ _ lv _ _ _ _ _ _ _ _ _ _ _ _ _ true)
      intro i3₁ st3₁ i3₂ st3₂ stC3 hq₁ hq₂
      obtain ⟨hi3₁, _⟩ := hq₁
      obtain ⟨hi3₂, _⟩ := hq₂
      subst i3₁ i3₂
      exact aligned_lockRecZ A r hr n b initial ext C₁ C₂ w₀ sch (lv' :: rest) _ _ _ _ _ _ _

theorem aligned_lockQuery0Z (A : Arith F K) (r : ℕ) (C₁ C₂ : Oracle F) (w₀ : ℕ) (i₁ i₂ : IRep F D) (lv : Level)
    (v₁ v₂ : ℕ → K) (col : List F → ℕ → K)
    (hc₁ : ∀ (row : List F) (p : ℕ), (∀ lane < w₀, row.getD lane 0 = C₁ lane p) → col row p = v₁ p)
    (hc₂ : ∀ (row : List F) (p : ℕ), (∀ lane < w₀, row.getD lane 0 = C₂ lane p) → col row p = v₂ p)
    (st₁ st₂ : LigState F K) (stC : LigStateC F K D) (h₁ : StRelZ r C₁ w₀ i₁ st₁ stC)
    (h₂ : StRelZ r C₂ w₀ i₂ st₂ stC) (hl₁ hl₂ : stC.level = 0) :
    Lock.Aligned (lockQuery0Z A r C₁ w₀ i₁ lv v₁ col hc₁ st₁ stC h₁ hl₁)
      (lockQuery0Z A r C₂ w₀ i₂ lv v₂ col hc₂ st₂ stC h₂ hl₂) :=
  .coin _ fun _ => .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _ _

theorem aligned_lockLigTailZ (A : Arith F K) (r : ℕ) (hr : r < 2) (sch : Schedule) (t : ℕ) (c₀ : F) (e : ℕ)
    (b : ℕ → F) (C₁ C₂ : Oracle F) (lv₀ : Level) (z₀ : Fin (sch.m - 7) → F) (β₀ : F) (initial : List K)
    (hinit : initial.length = sch.k0) (ρ : Fin e → K) (ybar : Fin t → K) (s : K) :
    ∀ (rest : List Level) (hlevs : sch.levels = lv₀ :: rest),
      Lock.Aligned (lockLigTailZ (D := D) A r hr sch t c₀ e b C₁ lv₀ z₀ β₀ initial hinit ρ ybar s rest hlevs)
        (lockLigTailZ (D := D) A r hr sch t c₀ e b C₂ lv₀ z₀ β₀ initial hinit ρ ybar s rest hlevs)
  | [], _ => .send _ fun _ => .ret _ _
  | lv₁ :: rest, hlevs => by
      refine .send _ fun msg' => .table _ fun C' cap => Lock.Aligned.bind ?_
        (aligned_lockTwoOod A r _ _ _ _ _ _ _ _ rfl rfl _ _ _ _ _ _ _ _)
      intro i₂₁ x₁ i₂₂ x₂ y r₁ r₂
      rcases x₁ with ⟨s₁', x₁⟩
      rcases x₁ with ⟨q₁', oods₁⟩
      rcases x₂ with ⟨s₂', x₂⟩
      rcases x₂ with ⟨q₂', oods₂⟩
      rcases y with ⟨sF', y⟩
      rcases y with ⟨qF', oodsC⟩
      obtain ⟨hi₁, hoods₁, _⟩ := r₁
      obtain ⟨hi₂, hoods₂, _⟩ := r₂
      simp only at hoods₁ hoods₂
      subst i₂₁ i₂₂ hoods₁ hoods₂
      refine Lock.Aligned.bind ?_ (aligned_lockQuery0Z A r C₁ C₂ _ _ _ lv₀ _ _ _ _ _ _ _ _ _ _ rfl rfl)
      intro i3₁ st3₁ i3₂ st3₂ stC3 hq₁ hq₂
      obtain ⟨hi3₁, _⟩ := hq₁
      obtain ⟨hi3₂, _⟩ := hq₂
      subst i3₁ i3₂
      exact aligned_lockRecZ A r hr _ b initial _ C₁ C₂ _ sch (lv₁ :: rest) _ _ _ _ _ _ _

theorem aligned_lockLigPadZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P : Pads0 np e) (t : ℕ) (c₀ : F)
    (T : F) (b : ℕ → F) (C₁ C₂ : Oracle F) :
    ∀ sch : Schedule, Lock.Aligned (lockLigPadZ (D := D) A r hr P t c₀ T b C₁ sch)
      (lockLigPadZ (D := D) A r hr P t c₀ T b C₂ sch)
  | ⟨_, _, [], _⟩ => .ret _ _
  | ⟨m, k0, lv₀ :: rest, ext⟩ => by
      refine .coin _ fun z₀ => .send _ fun y₀ => .coin _ fun β₀ => .send _ fun msg => ?_
      refine Lock.Aligned.bind ?_ (Lock.Aligned.refl (lockFoldZK A P r (fun _ => none) k0 0 _ _))
      intro i₁ x₁ i₂ x₂ y r₁ r₂
      obtain ⟨hi₁, hx₁, hl₁⟩ := r₁
      obtain ⟨hi₂, hx₂, hl₂⟩ := r₂
      subst hi₁ hi₂ hx₁ hx₂
      rcases x₂ with ⟨initial, s⟩
      refine .send _ fun ev => ?_
      refine Lock.Aligned.bind ?_ (Lock.Aligned.refl (Lock.same (drawKs A e)))
      intro i₁' ρ₁ i₂' ρ₂ ρ h₁ h₂
      obtain ⟨hi₁', hρ₁⟩ := h₁
      obtain ⟨hi₂', hρ₂⟩ := h₂
      subst hi₁' hi₂' hρ₁ hρ₂
      refine .send _ fun T' => .send _ fun ybar => ?_
      refine Lock.Aligned.bind ?_
        (aligned_lockLigTailZ A r hr ⟨m, k0, lv₀ :: rest, ext⟩ t c₀ e b C₁ C₂ lv₀ z₀ β₀ initial hl₂ ρ₂ ybar T'
          rest rfl)
      intro _ _ _ _ _ _ _
      exact .ret _ _

end FlockSoundness.Discharge.ZkLink
