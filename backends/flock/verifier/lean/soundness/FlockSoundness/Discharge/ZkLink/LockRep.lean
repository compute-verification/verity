import FlockSoundness.Discharge.ZkLink.AlignLevel
import FlockSoundness.Protocol.Discharge.ZkLink.LockRep

/-!
# The primed `--zk` table in lockstep with the compiled one, every committed object a table

A rep of the compiled `--zk` table commits three kinds of object, each a table node of the Lock against the primed
table (`tabOrcP`), where every committed object is an oracle:

* the **pads root** (slot `r`): the oracle carrying the pads words (`padWords`) against the root's cap;
* **`τ`'s commitment** (slot `2 + r`): the oracle carrying `τ` at `(0, 0)` against the commitment's cap;
* each **level's cap** (slot `slotOf r ℓ + 4`): `LockLevel`'s tables, moved four slots up (`lockReindex`).

The index (`IZ`) records the three; `getZ r` reads it by slot, and the two reps' getters are disjoint (`disjoint_getZ`).
On openings that agree with the fixed tables (`AgreeZ` on the rows, `PadsAgree` on the pads pairs, `TauAgree` on `τ`),
the compiled verdict implies the primed one (`RepRelZ`, `TopRelZ`). The locksteps for two level-0 tables are aligned
(`aligned_lockSessZ`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-! ## Moving a lockstep's slots -/

section Reindex

variable {α β T Dd I J : Type} {getI : I → ℕ → Option (T × Dd)} {getJ : J → ℕ → Option (T × Dd)}

theorem Aligned.reindex {R₁ R₂ : I → α → β → Prop} (e : I → J) (sl : ℕ → ℕ)
    (hget : ∀ i s, getJ (e i) (sl s) = getI i s) (hoff : ∀ i i' s, (∀ s', sl s' ≠ s) → getJ (e i) s = getJ (e i') s)
    {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β} {L₁ : Lock T Dd getI R₁ i₁ g₁ g'} {L₂ : Lock T Dd getI R₂ i₂ g₂ g'}
    (h : Lock.Aligned L₁ L₂) : Lock.Aligned (lockReindex e sl hget hoff L₁) (lockReindex e sl hget hoff L₂) := by
  induction h with
  | ret r₁ r₂ => exact .ret _ _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table (sl s) ih

end Reindex

/-! ## The index -/

omit [Field F] [Fintype F] [DecidableEq F] in
theorem getPair_empty_none (s : ℕ) :
    Lock.getPair (getZ 0) (getZ 1) ((IZ.empty, IZ.empty) : IZ F D × IZ F D) s = none := by
  simp [Lock.getPair, getZ, IZ.empty, getRep]

/-! ## The masked phase -/

/-! ## The inner proof -/

/-! ## A rep -/

/-! ## The table and the session -/

/-! ## Alignment across level-0 oracles -/

theorem aligned_lockRestZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₁ C₂ : Oracle F)
    (extra : List (Claim F S.m)) (zc : ZcZK F np S.m) (ab : ClaimZK F np S.m) :
    Lock.Aligned (lockRestZ (D := D) A r hr P₀ PA S sch hm t c₀ C₁ extra zc ab)
      (lockRestZ (D := D) A r hr P₀ PA S sch hm t c₀ C₂ extra zc ab) := by
  refine Lock.Aligned.bind ?_ (Lock.Aligned.same _)
  intro i₁ a₁ i₂ a₂ b r₁ r₂
  obtain ⟨hi₁, ha₁⟩ := r₁
  obtain ⟨hi₂, ha₂⟩ := r₂
  subst hi₁ hi₂ ha₁ ha₂
  rcases a₂ with ⟨oo, x⟩
  cases oo with
  | none => exact .ret _ _
  | some op =>
    refine Lock.Aligned.bind ?_ (aligned_lockLigPadZ A r hr P₀ t c₀ op.T op.b C₁ C₂ sch)
    intro _ _ _ _ _ _ _
    exact .send _ fun _ => .ret _ _

theorem aligned_lockMaskedZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (tr : Triple np) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₁ C₂ : Oracle F)
    (extra : List (Claim F S.m)) :
    Lock.Aligned (lockMaskedZ (D := D) A r hr P₀ PA tr S sch hm t c₀ C₁ extra)
      (lockMaskedZ (D := D) A r hr P₀ PA tr S sch hm t c₀ C₂ extra) := by
  refine Lock.Aligned.map (Lock.Aligned.bind ?_ (Lock.Aligned.same _))
  intro i₁ a₁ i₂ a₂ b r₁ r₂
  obtain ⟨hi₁, ha₁⟩ := r₁
  obtain ⟨hi₂, ha₂⟩ := r₂
  subst hi₁ hi₂ ha₁ ha₂
  refine Lock.Aligned.map (Lock.Aligned.bind ?_ (Lock.Aligned.same _))
  intro i₁' a₁' i₂' a₂' b' r₁' r₂'
  obtain ⟨hi₁', ha₁'⟩ := r₁'
  obtain ⟨hi₂', ha₂'⟩ := r₂'
  subst hi₁' hi₂' ha₁' ha₂'
  exact Lock.Aligned.map (aligned_lockRestZ A r hr P₀ PA S sch hm t c₀ C₁ C₂ extra a₂.1 a₂'.1)

theorem aligned_lockInnerZ (Hs : Hashes F D Sl) (r : ℕ) (hr : r < 2) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ)
    (pos : F → Fin N) {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (lev₁ lev₂ : IRep F D) (O : Oracle F)
    (c : Cap D) :
    Lock.Aligned (lockInnerZ Hs r hr dom s nP q pos rows tr lev₁ O c)
      (lockInnerZ Hs r hr dom s nP q pos rows tr lev₂ O c) :=
  .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .table _ fun _ _ => .coin _ fun _ => .send _ fun _ =>
    .coin _ fun _ => .ret _ _

theorem aligned_lockRepZ (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ) (hr : r < 2)
    (C₁ C₂ : Oracle F) (extra : List (Claim F T.S.m)) :
    Lock.Aligned (lockRepZ A Hs T r hr C₁ extra) (lockRepZ A Hs T r hr C₂ extra) := by
  refine .table r fun O c => Lock.Aligned.bind ?_ (Lock.Aligned.mono (Aligned.reindex _ _ _ _
    (aligned_lockMaskedZ A r hr T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ C₁ C₂ extra)))
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
  refine Lock.Aligned.bind ?_ (aligned_lockInnerZ Hs r hr T.dom T.np T.nP T.q T.pos rows₂ T.tr lev₁ lev₂ O c)
  intro _ _ _ _ _ _ _
  exact .ret _ _

theorem aligned_lockAfterZ [Inhabited D] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
    (C₁ C₂ : Oracle F) (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) :
    Lock.Aligned (lockAfterZ A Hs T C₁ cap₀ pts) (lockAfterZ A Hs T C₂ cap₀ pts) := by
  refine Lock.Aligned.bind ?_ (Lock.Aligned.interleaveLock disjoint_getZ
    (aligned_lockRepZ A Hs T 0 (by norm_num) C₁ C₂ _) (aligned_lockRepZ A Hs T 1 (by norm_num) C₁ C₂ _))
  intro _ _ _ _ _ _ _
  exact .send _ fun _ => .ret _ _

theorem aligned_lockSessZ [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (C₁ C₂ : Oracle F) (cap₀ : Cap D) :
    Lock.Aligned (lockSessZ A Hs T pre post C₁ cap₀) (lockSessZ A Hs T pre post C₂ cap₀) :=
  .coin _ fun pts => Lock.Aligned.mono (Lock.Aligned.batchLock dfltOutZC
    (Lock.Aligned.mapL (aligned_lockAfterZ A Hs T C₁ C₂ cap₀ pts)) _ _)

end FlockSoundness.Discharge.ZkLink
