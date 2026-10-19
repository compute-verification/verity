import Proofs.Flock.Soundness.Discharge.ZkLink.LockRep
import Proofs.Flock.Soundness.Game.LockChoose

/-!
# The masked session's fork probabilities do not read the chooser

Below every table node of `lockSessZ` (a level's cap, a pads root, a `τ` commitment) the lockstep's structure is the
same whatever table is chosen there: the table enters the index, the oracle games and the leaf relations, never the
compiled side or the slots. So the locksteps are `Lock.AlignedT`, builder by builder as `aligned_lockSessZ` is
`Lock.Aligned`, and a fork probability in the session is the same under any two choosers and any two level-0 tables
(`forkE_lockSessZ_choose`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

section Reindex

variable {α β T Dd I J : Type} {getI : I → ℕ → Option (T × Dd)} {getJ : J → ℕ → Option (T × Dd)}

theorem AlignedT.reindex {R₁ R₂ : I → α → β → Prop} (e₁ e₂ : I → J) (sl : ℕ → ℕ)
    (hget₁ : ∀ i s, getJ (e₁ i) (sl s) = getI i s)
    (hoff₁ : ∀ i i' s, (∀ s', sl s' ≠ s) → getJ (e₁ i) s = getJ (e₁ i') s)
    (hget₂ : ∀ i s, getJ (e₂ i) (sl s) = getI i s)
    (hoff₂ : ∀ i i' s, (∀ s', sl s' ≠ s) → getJ (e₂ i) s = getJ (e₂ i') s)
    {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β} {L₁ : Lock T Dd getI R₁ i₁ g₁ g'} {L₂ : Lock T Dd getI R₂ i₂ g₂ g'}
    (h : Lock.AlignedT L₁ L₂) :
    Lock.AlignedT (lockReindex e₁ sl hget₁ hoff₁ L₁) (lockReindex e₂ sl hget₂ hoff₂ L₂) := by
  induction h with
  | ret r₁ r₂ => exact .ret _ _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table (sl s) ih

end Reindex

/-! ## Level `ℓ ≥ 1` and the masked level 0 -/

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem alignedT_lockFold (A : Arith F K) (r : ℕ) :
    ∀ (k : ℕ) (i₁ i₂ : IRep F D) (P₁ P₂ : Opens F D → Prop) (s₁ s₂ : K) (q₁ q₂ : Quad K)
      (sF : Opens F D → K) (qF : Opens F D → Quad K) (h₁ : ∀ o, P₁ o → sF o = s₁ ∧ qF o = q₁)
      (h₂ : ∀ o, P₂ o → sF o = s₂ ∧ qF o = q₂),
      Lock.AlignedT (lockFold A r i₁ P₁ k s₁ q₁ sF qF h₁) (lockFold A r i₂ P₂ k s₂ q₂ sF qF h₂)
  | 0, _, _, _, _, _, _, _, _, _, _, _, _ => .ret _ _
  | k + 1, i₁, i₂, P₁, P₂, s₁, s₂, q₁, q₂, sF, qF, h₁, h₂ => by
      refine .coin _ fun w => .send _ fun msg => Lock.AlignedT.bind ?_
        (alignedT_lockFold A r k i₁ i₂ P₁ P₂ _ _ _ _ _ _ _ _)
      intro _ x₁ _ x₂ y r₁ r₂
      rcases x₁ with ⟨rs₁, x₁⟩
      rcases x₁ with ⟨s₁', q₁'⟩
      rcases x₂ with ⟨rs₂, x₂⟩
      rcases x₂ with ⟨s₂', q₂'⟩
      rcases y with ⟨rs', y⟩
      rcases y with ⟨sF', qF'⟩
      obtain ⟨_, _, _, _⟩ := r₁
      obtain ⟨_, _, _, _⟩ := r₂
      exact .ret _ _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem alignedT_lockTwoOod (A : Arith F K) (r : ℕ) (i₁ i₂ : IRep F D) (P₁ P₂ : Opens F D → Prop)
    (dim split₁ split₂ split' : ℕ) (hs₁ : split₁ = split') (hs₂ : split₂ = split') (s₁ s₂ : K)
    (q₁ q₂ : Quad K) (sF : Opens F D → K) (qF : Opens F D → Quad K)
    (h₁ : ∀ o, P₁ o → sF o = s₁ ∧ qF o = q₁) (h₂ : ∀ o, P₂ o → sF o = s₂ ∧ qF o = q₂) :
    Lock.AlignedT (lockTwoOod A r i₁ P₁ dim split₁ split' hs₁ s₁ q₁ sF qF h₁)
      (lockTwoOod A r i₂ P₂ dim split₂ split' hs₂ s₂ q₂ sF qF h₂) := by
  subst hs₁
  subst hs₂
  exact .coin _ fun _ => .send _ fun _ => .send _ fun _ => .coin _ fun _ =>
    .coin _ fun _ => .send _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _ _

theorem alignedT_lockQueryZ (A : Arith F K) (r : ℕ) (C₁ C₂ : Oracle F) (w₀ : ℕ) (i₁ i₂ : IRep F D) (lv : Level)
    (ch : List K) (start₁ start₂ start' : ℕ) (ht₁ : start₁ = start') (ht₂ : start₂ = start')
    (st₁ st₂ : LigState F K) (stC : LigStateC F K D) (h₁ : StRelZ r C₁ w₀ i₁ st₁ stC)
    (h₂ : StRelZ r C₂ w₀ i₂ st₂ stC) (hl₁ hl₂ : stC.level ≠ 0) (wm : Bool) :
    Lock.AlignedT (lockQueryZ A r C₁ w₀ i₁ lv ch start₁ start' ht₁ st₁ stC h₁ hl₁ wm)
      (lockQueryZ A r C₂ w₀ i₂ lv ch start₂ start' ht₂ st₂ stC h₂ hl₂ wm) := by
  subst ht₁
  subst ht₂
  cases wm
  · exact .coin _ fun _ => .coin _ fun _ => .coin _ fun _ => .ret _ _
  · exact .coin _ fun _ => .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _ _

theorem alignedT_lockRecZ (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K)
    (ext : ℕ) (C₁ C₂ : Oracle F) (w₀ : ℕ) (sch : Schedule) :
    ∀ (levs : List Level) (i₁ i₂ : IRep F D) (st₁ st₂ : LigState F K) (stC : LigStateC F K D)
      (h₁ : RecInvZ r C₁ w₀ sch levs i₁ st₁ stC) (h₂ : RecInvZ r C₂ w₀ sch levs i₂ st₂ stC),
      Lock.AlignedT (lockRecZ A r hr n b initial ext C₁ w₀ sch levs i₁ st₁ stC h₁)
        (lockRecZ A r hr n b initial ext C₂ w₀ sch levs i₂ st₂ stC h₂)
  | [], _, _, _, _, _, _, _ => .ret _ _
  | [lv], i₁, i₂, st₁, st₂, stC, h₁, h₂ => by
      refine Lock.AlignedT.bind ?_ (alignedT_lockFold A r lv.lanes i₁ i₂ _ _ _ _ _ _ _ _ _ _)
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
      refine .send _ fun yr => Lock.AlignedT.bind ?_
        (alignedT_lockQueryZ A r C₁ C₂ w₀ i₁ i₂ lv _ _ _ _ _ _ _ _ _ _ _ _ _ false)
      intro i3₁ st3₁ i3₂ st3₂ stC3 hq₁ hq₂
      obtain ⟨hi3₁, _⟩ := hq₁
      obtain ⟨hi3₂, _⟩ := hq₂
      subst i3₁ i3₂
      exact .ret _ _
  | lv :: lv' :: rest, i₁, i₂, st₁, st₂, stC, h₁, h₂ => by
      refine Lock.AlignedT.bind ?_ (alignedT_lockFold A r lv.lanes i₁ i₂ _ _ _ _ _ _ _ _ _ _)
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
      refine .table _ fun C'₁ C'₂ cap => ?_
      refine Lock.AlignedT.bind ?_ (alignedT_lockTwoOod A r _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)
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
      refine Lock.AlignedT.bind ?_
        (alignedT_lockQueryZ A r C₁ C₂ w₀ _ _ lv _ _ _ _ _ _ _ _ _ _ _ _ _ true)
      intro i3₁ st3₁ i3₂ st3₂ stC3 hq₁ hq₂
      obtain ⟨hi3₁, _⟩ := hq₁
      obtain ⟨hi3₂, _⟩ := hq₂
      subst i3₁ i3₂
      exact alignedT_lockRecZ A r hr n b initial ext C₁ C₂ w₀ sch (lv' :: rest) _ _ _ _ _ _ _

theorem alignedT_lockQuery0Z (A : Arith F K) (r : ℕ) (C₁ C₂ : Oracle F) (w₀ : ℕ) (i₁ i₂ : IRep F D) (lv : Level)
    (v₁ v₂ : ℕ → K) (col : List F → ℕ → K)
    (hc₁ : ∀ (row : List F) (p : ℕ), (∀ lane < w₀, row.getD lane 0 = C₁ lane p) → col row p = v₁ p)
    (hc₂ : ∀ (row : List F) (p : ℕ), (∀ lane < w₀, row.getD lane 0 = C₂ lane p) → col row p = v₂ p)
    (st₁ st₂ : LigState F K) (stC : LigStateC F K D) (h₁ : StRelZ r C₁ w₀ i₁ st₁ stC)
    (h₂ : StRelZ r C₂ w₀ i₂ st₂ stC) (hl₁ hl₂ : stC.level = 0) :
    Lock.AlignedT (lockQuery0Z A r C₁ w₀ i₁ lv v₁ col hc₁ st₁ stC h₁ hl₁)
      (lockQuery0Z A r C₂ w₀ i₂ lv v₂ col hc₂ st₂ stC h₂ hl₂) :=
  .coin _ fun _ => .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _ _

theorem alignedT_lockLigTailZ (A : Arith F K) (r : ℕ) (hr : r < 2) (sch : Schedule) (t : ℕ) (c₀ : F) (e : ℕ)
    (b : ℕ → F) (C₁ C₂ : Oracle F) (lv₀ : Level) (z₀ : Fin (sch.m - 7) → F) (β₀ : F) (initial : List K)
    (hinit : initial.length = sch.k0) (ρ : Fin e → K) (ybar : Fin t → K) (s : K) :
    ∀ (rest : List Level) (hlevs : sch.levels = lv₀ :: rest),
      Lock.AlignedT (lockLigTailZ (D := D) A r hr sch t c₀ e b C₁ lv₀ z₀ β₀ initial hinit ρ ybar s rest hlevs)
        (lockLigTailZ (D := D) A r hr sch t c₀ e b C₂ lv₀ z₀ β₀ initial hinit ρ ybar s rest hlevs)
  | [], _ => .send _ fun _ => .ret _ _
  | lv₁ :: rest, hlevs => by
      refine .send _ fun msg' => .table _ fun C'₁ C'₂ cap => Lock.AlignedT.bind ?_
        (alignedT_lockTwoOod A r _ _ _ _ _ _ _ _ rfl rfl _ _ _ _ _ _ _ _)
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
      refine Lock.AlignedT.bind ?_ (alignedT_lockQuery0Z A r C₁ C₂ _ _ _ lv₀ _ _ _ _ _ _ _ _ _ _ rfl rfl)
      intro i3₁ st3₁ i3₂ st3₂ stC3 hq₁ hq₂
      obtain ⟨hi3₁, _⟩ := hq₁
      obtain ⟨hi3₂, _⟩ := hq₂
      subst i3₁ i3₂
      exact alignedT_lockRecZ A r hr _ b initial _ C₁ C₂ _ sch (lv₁ :: rest) _ _ _ _ _ _ _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem alignedT_lockFoldZK (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (r : ℕ) (i : IRep F D) :
    ∀ (k j : ℕ) (s : LinK K np) (q : QuadL K np),
      Lock.AlignedT (lockFoldZK A P r i k j s q) (lockFoldZK A P r i k j s q)
  | 0, _, _, _ => .ret _ _
  | 1, _, _, _ => .coin _ fun _ => .ret _ _
  | k + 2, j, _, q => .coin _ fun w => .send _ fun msg => by
      refine Lock.AlignedT.bind ?_ (alignedT_lockFoldZK A P r i (k + 1) (j + 1) _ _)
      intro i₁ x₁ i₂ x₂ y r₁ r₂
      obtain ⟨hi₁, hx₁, hl₁⟩ := r₁
      obtain ⟨hi₂, hx₂, hl₂⟩ := r₂
      subst hi₁ hi₂ hx₁ hx₂
      rcases x₂ with ⟨rs, s''⟩
      exact .ret _ _

theorem alignedT_lockLigPadZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P : Pads0 np e) (t : ℕ) (c₀ : F)
    (T : F) (b : ℕ → F) (C₁ C₂ : Oracle F) :
    ∀ sch : Schedule, Lock.AlignedT (lockLigPadZ (D := D) A r hr P t c₀ T b C₁ sch)
      (lockLigPadZ (D := D) A r hr P t c₀ T b C₂ sch)
  | ⟨_, _, [], _⟩ => .ret _ _
  | ⟨m, k0, lv₀ :: rest, ext⟩ => by
      refine .coin _ fun z₀ => .send _ fun y₀ => .coin _ fun β₀ => .send _ fun msg => ?_
      refine Lock.AlignedT.bind ?_ (alignedT_lockFoldZK A P r (fun _ => none) k0 0 _ _)
      intro i₁ x₁ i₂ x₂ y r₁ r₂
      obtain ⟨hi₁, hx₁, hl₁⟩ := r₁
      obtain ⟨hi₂, hx₂, hl₂⟩ := r₂
      subst hi₁ hi₂ hx₁ hx₂
      rcases x₂ with ⟨initial, s⟩
      refine .send _ fun ev => ?_
      refine Lock.AlignedT.bind ?_ (Lock.AlignedT.same (drawKs A e))
      intro i₁' ρ₁ i₂' ρ₂ ρ h₁ h₂
      obtain ⟨hi₁', hρ₁⟩ := h₁
      obtain ⟨hi₂', hρ₂⟩ := h₂
      subst hi₁' hi₂' hρ₁ hρ₂
      refine .send _ fun T' => .send _ fun ybar => ?_
      refine Lock.AlignedT.bind ?_
        (alignedT_lockLigTailZ A r hr ⟨m, k0, lv₀ :: rest, ext⟩ t c₀ e b C₁ C₂ lv₀ z₀ β₀ initial hl₂ ρ₂ ybar T'
          rest rfl)
      intro _ _ _ _ _ _ _
      exact .ret _ _

/-! ## A rep, the table and the session -/

theorem alignedT_lockRestZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₁ C₂ : Oracle F)
    (extra : List (Claim F S.m)) (zc : ZcZK F np S.m) (ab : ClaimZK F np S.m) :
    Lock.AlignedT (lockRestZ (D := D) A r hr P₀ PA S sch hm t c₀ C₁ extra zc ab)
      (lockRestZ (D := D) A r hr P₀ PA S sch hm t c₀ C₂ extra zc ab) := by
  refine Lock.AlignedT.bind ?_ (Lock.AlignedT.same _)
  intro i₁ a₁ i₂ a₂ b r₁ r₂
  obtain ⟨hi₁, ha₁⟩ := r₁
  obtain ⟨hi₂, ha₂⟩ := r₂
  subst hi₁ hi₂ ha₁ ha₂
  rcases a₂ with ⟨oo, x⟩
  cases oo with
  | none => exact .ret _ _
  | some op =>
    refine Lock.AlignedT.bind ?_ (alignedT_lockLigPadZ A r hr P₀ t c₀ op.T op.b C₁ C₂ sch)
    intro _ _ _ _ _ _ _
    exact .send _ fun _ => .ret _ _

theorem alignedT_lockMaskedZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (tr : Triple np) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₁ C₂ : Oracle F)
    (extra : List (Claim F S.m)) :
    Lock.AlignedT (lockMaskedZ (D := D) A r hr P₀ PA tr S sch hm t c₀ C₁ extra)
      (lockMaskedZ (D := D) A r hr P₀ PA tr S sch hm t c₀ C₂ extra) := by
  refine Lock.AlignedT.map (Lock.AlignedT.bind ?_ (Lock.AlignedT.same _))
  intro i₁ a₁ i₂ a₂ b r₁ r₂
  obtain ⟨hi₁, ha₁⟩ := r₁
  obtain ⟨hi₂, ha₂⟩ := r₂
  subst hi₁ hi₂ ha₁ ha₂
  refine Lock.AlignedT.map (Lock.AlignedT.bind ?_ (Lock.AlignedT.same _))
  intro i₁' a₁' i₂' a₂' b' r₁' r₂'
  obtain ⟨hi₁', ha₁'⟩ := r₁'
  obtain ⟨hi₂', ha₂'⟩ := r₂'
  subst hi₁' hi₂' ha₁' ha₂'
  exact Lock.AlignedT.map (alignedT_lockRestZ A r hr P₀ PA S sch hm t c₀ C₁ C₂ extra a₂.1 a₂'.1)

theorem alignedT_lockInnerZ (Hs : Hashes F D Sl) (r : ℕ) (hr : r < 2) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ)
    (pos : F → Fin N) {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (lev₁ lev₂ : IRep F D) (O₁ O₂ : Oracle F)
    (c : Cap D) :
    Lock.AlignedT (lockInnerZ Hs r hr dom s nP q pos rows tr lev₁ O₁ c)
      (lockInnerZ Hs r hr dom s nP q pos rows tr lev₂ O₂ c) :=
  .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .table _ fun _ _ _ => .coin _ fun _ => .send _ fun _ =>
    .coin _ fun _ => .ret _ _

theorem alignedT_lockRepZ (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ) (hr : r < 2)
    (C₁ C₂ : Oracle F) (extra : List (Claim F T.S.m)) :
    Lock.AlignedT (lockRepZ A Hs T r hr C₁ extra) (lockRepZ A Hs T r hr C₂ extra) := by
  refine .table r fun O₁ O₂ c => Lock.AlignedT.bind ?_ (Lock.AlignedT.mono (AlignedT.reindex _ _ _ _ _ _ _
    (alignedT_lockMaskedZ A r hr T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ C₁ C₂ extra)))
  intro i₁ a₁ i₂ a₂ b r₁ r₂
  rcases i₁ with ⟨lev₁, p₁, t₁⟩
  rcases i₂ with ⟨lev₂, p₂, t₂⟩
  obtain ⟨hp₁, ht₁, h₁⟩ := r₁
  obtain ⟨hp₂, ht₂, h₂⟩ := r₂
  simp only at hp₁ ht₁ h₁ hp₂ ht₂ h₂
  subst hp₁ ht₁ hp₂ ht₂
  obtain ⟨hrows₁, hrec₁⟩ := h₁
  obtain ⟨hrows₂, hrec₂⟩ := h₂
  rcases a₁ with ⟨rows₁, v₁⟩
  rcases a₂ with ⟨rows₂, v₂⟩
  rcases b with ⟨rows', p⟩
  simp only at hrows₁ hrec₁ hrows₂ hrec₂
  subst hrows₁ hrows₂
  refine Lock.AlignedT.bind ?_
    (alignedT_lockInnerZ Hs r hr T.dom T.np T.nP T.q T.pos rows₂ T.tr lev₁ lev₂ O₁ O₂ c)
  intro _ _ _ _ _ _ _
  exact .ret _ _

theorem alignedT_lockAfterZ [Inhabited D] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
    (C₁ C₂ : Oracle F) (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) :
    Lock.AlignedT (lockAfterZ A Hs T C₁ cap₀ pts) (lockAfterZ A Hs T C₂ cap₀ pts) := by
  refine Lock.AlignedT.bind ?_ (Lock.AlignedT.interleaveLock disjoint_getZ
    (alignedT_lockRepZ A Hs T 0 (by norm_num) C₁ C₂ _) (alignedT_lockRepZ A Hs T 1 (by norm_num) C₁ C₂ _))
  intro _ _ _ _ _ _ _
  exact .send _ fun _ => .ret _ _

theorem alignedT_lockSessZ [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (C₁ C₂ : Oracle F) (cap₀ : Cap D) :
    Lock.AlignedT (lockSessZ A Hs T pre post C₁ cap₀) (lockSessZ A Hs T pre post C₂ cap₀) :=
  .coin _ fun pts => Lock.AlignedT.mono (Lock.AlignedT.batchLock dfltOutZC
    (Lock.AlignedT.mapL (alignedT_lockAfterZ A Hs T C₁ C₂ cap₀ pts)) _ _)

/-- **A fork probability in the masked session reads neither the chooser nor the level-0 table.** -/
theorem forkE_lockSessZ_choose [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (C₁ C₂ : Oracle F) (cap₀ : Cap D)
    (plur₁ plur₂ : Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl))) (slot : ℕ)
    (Φ : Cap D → List (OutZC F D Sl) → List (OutZC F D Sl) → ℝ)
    (σ' : Strategy (sessAtZ pre post (tabAfterZCP A Hs T cap₀))) :
    Lock.forkE plur₁ slot Φ (lockSessZ A Hs T pre post C₁ cap₀) σ' =
      Lock.forkE plur₂ slot Φ (lockSessZ A Hs T pre post C₂ cap₀) σ' :=
  Lock.forkE_choose plur₁ plur₂ slot Φ (alignedT_lockSessZ A Hs T pre post C₁ C₂ cap₀) σ'

end FlockSoundness.Discharge.ZkLink
