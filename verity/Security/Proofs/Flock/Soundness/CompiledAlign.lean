import Proofs.Flock.Soundness.CompiledLock

/-!
# The locksteps for two level-0 tables are aligned

The lockstep of `CompiledLock` is built for one level-0 table `C₀`: it enters the oracle games and
the leaf relations, never the compiled side or the slots. So the locksteps for two tables are
`Lock.Aligned`, builder by builder (`aligned_lockTable`), and the reps' fork probabilities do not
depend on the level-0 table (`Lock.forkE_congr`).
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

theorem aligned_lockFold (A : Arith F K) (r : ℕ) :
    ∀ (k : ℕ) (i₁ i₂ : IRep F D) (P₁ P₂ : Opens F D → Prop) (s₁ s₂ : K) (q₁ q₂ : Quad K)
      (sF : Opens F D → K) (qF : Opens F D → Quad K) (h₁ : ∀ o, P₁ o → sF o = s₁ ∧ qF o = q₁)
      (h₂ : ∀ o, P₂ o → sF o = s₂ ∧ qF o = q₂),
      Lock.Aligned (lockFold A r i₁ P₁ k s₁ q₁ sF qF h₁) (lockFold A r i₂ P₂ k s₂ q₂ sF qF h₂)
  | 0, _, _, _, _, _, _, _, _, _, _, _, _ => .ret _ _
  | k + 1, i₁, i₂, P₁, P₂, s₁, s₂, q₁, q₂, sF, qF, h₁, h₂ => by
      refine .coin _ fun w => .send _ fun msg => Lock.Aligned.bind ?_
        (aligned_lockFold A r k i₁ i₂ P₁ P₂ _ _ _ _ _ _ _ _)
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

theorem aligned_lockTwoOod (A : Arith F K) (r : ℕ) (i₁ i₂ : IRep F D) (P₁ P₂ : Opens F D → Prop)
    (dim split₁ split₂ split' : ℕ) (hs₁ : split₁ = split') (hs₂ : split₂ = split') (s₁ s₂ : K)
    (q₁ q₂ : Quad K) (sF : Opens F D → K) (qF : Opens F D → Quad K)
    (h₁ : ∀ o, P₁ o → sF o = s₁ ∧ qF o = q₁) (h₂ : ∀ o, P₂ o → sF o = s₂ ∧ qF o = q₂) :
    Lock.Aligned (lockTwoOod A r i₁ P₁ dim split₁ split' hs₁ s₁ q₁ sF qF h₁)
      (lockTwoOod A r i₂ P₂ dim split₂ split' hs₂ s₂ q₂ sF qF h₂) := by
  subst hs₁
  subst hs₂
  exact .coin _ fun _ => .send _ fun _ => .send _ fun _ => .coin _ fun _ =>
    .coin _ fun _ => .send _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _ _

theorem aligned_lockQuery (A : Arith F K) (r : ℕ) (C₁ C₂ : Oracle F) (i₁ i₂ : IRep F D) (lv : Level)
    (ch : List K) (start₁ start₂ start' : ℕ) (ht₁ : start₁ = start') (ht₂ : start₂ = start')
    (st₁ st₂ : LigState F K) (stC : LigStateC F K D) (h₁ : StRel r C₁ i₁ st₁ stC)
    (h₂ : StRel r C₂ i₂ st₂ stC) (wm : Bool) :
    Lock.Aligned (lockQuery A r C₁ i₁ lv ch start₁ start' ht₁ st₁ stC h₁ wm)
      (lockQuery A r C₂ i₂ lv ch start₂ start' ht₂ st₂ stC h₂ wm) := by
  subst ht₁
  subst ht₂
  cases wm
  · exact .coin _ fun _ => .coin _ fun _ => .coin _ fun _ => .ret _ _
  · exact .coin _ fun _ => .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _ _

theorem aligned_lockRec (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K)
    (ext : ℕ) (C₁ C₂ : Oracle F) (sch : Schedule) :
    ∀ (levs : List Level) (i₁ i₂ : IRep F D) (st₁ st₂ : LigState F K) (stC : LigStateC F K D)
      (h₁ : RecInv r C₁ sch levs i₁ st₁ stC) (h₂ : RecInv r C₂ sch levs i₂ st₂ stC),
      Lock.Aligned (lockRec A r hr n b initial ext C₁ sch levs i₁ st₁ stC h₁)
        (lockRec A r hr n b initial ext C₂ sch levs i₂ st₂ stC h₂)
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
      refine .send _ fun yr => Lock.Aligned.bind ?_ (aligned_lockQuery A r C₁ C₂ i₁ i₂ lv _ _ _ _ _ _ _ _ _ _ _ false)
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
      refine Lock.Aligned.bind ?_ (aligned_lockQuery A r C₁ C₂ _ _ lv _ _ _ _ _ _ _ _ _ _ _ true)
      intro i3₁ st3₁ i3₂ st3₂ stC3 hq₁ hq₂
      obtain ⟨hi3₁, _⟩ := hq₁
      obtain ⟨hi3₂, _⟩ := hq₂
      subst i3₁ i3₂
      exact aligned_lockRec A r hr n b initial ext C₁ C₂ sch (lv' :: rest) _ _ _ _ _ _ _

theorem aligned_lockLig (A : Arith F K) (r : ℕ) (hr : r < 2) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (T : F) (b : ℕ → F) (C₁ C₂ : Oracle F) :
    Lock.Aligned (lockLig (D := D) A r hr m k0 l₀ l₁ rest ext hk0 T b C₁)
      (lockLig (D := D) A r hr m k0 l₀ l₁ rest ext hk0 T b C₂) := by
  refine .coin _ fun z₀ => .send _ fun y₀ => .coin _ fun β₀ => .send _ fun msg =>
    Lock.Aligned.bind ?_ (aligned_lockFold A r k0 _ _ _ _ _ _ _ _ _ _ _ _)
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
  refine .table _ fun C' cap => Lock.Aligned.bind ?_
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
  refine Lock.Aligned.bind ?_
    (aligned_lockQuery A r C₁ C₂ _ _ l₀ _ _ _ _ rfl rfl _ _ _ _ _ true)
  intro i3₁ st3₁ i3₂ st3₂ stC3 hq₁ hq₂
  obtain ⟨hi3₁, _⟩ := hq₁
  obtain ⟨hi3₂, _⟩ := hq₂
  subst i3₁ i3₂
  exact aligned_lockRec A r hr _ b _ ext C₁ C₂ _ (l₁ :: rest) _ _ _ _ _ _ _

theorem aligned_lockRep (A : Arith F K) (r : ℕ) (hr : r < 2) (S : Statement) (m k0 : ℕ)
    (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m)
    (C₁ C₂ : Oracle F) (extra : List (Claim F S.m)) :
    Lock.Aligned (lockRep (D := D) A r hr S m k0 l₀ l₁ rest ext hk0 hm C₁ extra)
      (lockRep (D := D) A r hr S m k0 l₀ l₁ rest ext hk0 hm C₂ extra) := by
  refine Lock.Aligned.bind ?_ (Lock.Aligned.refl _)
  intro i₁ zc₁ i₂ zc₂ zc h₁ h₂
  obtain ⟨hi₁, hz₁⟩ := h₁
  obtain ⟨hi₂, hz₂⟩ := h₂
  subst i₁ i₂ hz₁ hz₂
  cases zc₂ with
  | none => exact .ret _ _
  | some zc =>
    refine Lock.Aligned.bind ?_ (Lock.Aligned.refl _)
    intro i₁ ab₁ i₂ ab₂ ab h₁ h₂
    obtain ⟨hi₁, ha₁⟩ := h₁
    obtain ⟨hi₂, ha₂⟩ := h₂
    subst i₁ i₂ ha₁ ha₂
    cases ab₂ with
    | none => exact .ret _ _
    | some ab =>
      refine Lock.Aligned.bind ?_ (Lock.Aligned.refl _)
      intro i₁ op₁ i₂ op₂ op h₁ h₂
      obtain ⟨hi₁, ho₁⟩ := h₁
      obtain ⟨hi₂, ho₂⟩ := h₂
      subst i₁ i₂ ho₁ ho₂
      cases op₂ with
      | none => exact .ret _ _
      | some op => exact aligned_lockLig A r hr m k0 l₀ l₁ rest ext hk0 op.T op.b C₁ C₂

/-- **The table's locksteps for two level-0 tables are aligned.** -/
theorem aligned_lockTable [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₁ C₂ : Oracle F)
    (cap₀ : Cap D) :
    Lock.Aligned (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₁ cap₀)
      (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₂ cap₀) := by
  refine .coin _ fun pts => Lock.Aligned.bind ?_ (Lock.Aligned.interleaveLock disjoint_reps
    (aligned_lockRep A 0 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₁ C₂ _)
    (aligned_lockRep A 1 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₁ C₂ _))
  intro _ _ _ _ _ _ _
  exact .send _ fun _ => .ret _ _

end FlockSoundness
