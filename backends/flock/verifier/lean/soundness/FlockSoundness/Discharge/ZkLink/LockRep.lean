import FlockSoundness.Discharge.ZkLink.AlignLevel

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

/-- **A lockstep over a larger index**: the index embedded by `e`, every slot moved by `sl`; `getJ` reads the embedded
index at the moved slots as `getI` reads the index, and elsewhere not at all. -/
def lockReindex {R : I → α → β → Prop} (e : I → J) (sl : ℕ → ℕ) (hget : ∀ i s, getJ (e i) (sl s) = getI i s)
    (hoff : ∀ i i' s, (∀ s', sl s' ≠ s) → getJ (e i) s = getJ (e i') s) :
    {i : I} → {g : Game α} → {g' : Game β} → Lock T Dd getI R i g g' →
      Lock T Dd getJ (fun j a b => ∃ i, j = e i ∧ R i a b) (e i) g g'
  | _, _, _, .ret hr => .ret ⟨_, rfl, hr⟩
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => lockReindex e sl hget hoff (L c)
  | _, _, _, .send M L => .send M fun m => lockReindex e sl hget hoff (L m)
  | _, _, _, .table slot upd hf hs hk L => .table (sl slot) (fun t d => e (upd t d))
      (by rw [hget]; exact hf) (fun t d => by rw [hget]; exact hs t d)
      (fun t d s hs' => by
        by_cases h : ∃ s', sl s' = s
        · obtain ⟨s', rfl⟩ := h
          rw [hget, hget]
          exact hk t d s' fun h' => hs' (by rw [h'])
        · push Not at h
          exact hoff _ _ s h)
      fun t d => lockReindex e sl hget hoff (L t d)

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

/-- **A compiled `--zk` rep's fixed tables**: by level (`IRep`), the pads oracle with its root's cap, and `τ`'s oracle
with its commitment's cap. -/
structure IZ (F D : Type) where
  lev : IRep F D
  pads : Option (Oracle F × Cap D)
  tau : Option (Oracle F × Cap D)

/-- Rep `r`'s fixed tables by slot: the pads at `r`, `τ` at `2 + r`, level `ℓ` at `slotOf r ℓ + 4`. -/
def getZ (r : ℕ) (i : IZ F D) (s : ℕ) : Option (Oracle F × Cap D) :=
  if s < 4 then (if s = r then i.pads else if s = 2 + r then i.tau else none) else getRep r i.lev (s - 4)

/-- No table fixed. -/
def IZ.empty : IZ F D := ⟨fun _ => none, none, none⟩

omit [Field F] [Fintype F] [DecidableEq F] in
theorem getZ_lev (r : ℕ) (i : IZ F D) (s : ℕ) : getZ r i (s + 4) = getRep r i.lev s := by
  simp [getZ]

omit [Field F] [Fintype F] [DecidableEq F] in
theorem getZ_off (r : ℕ) (lev lev' : IRep F D) (p t : Option (Oracle F × Cap D)) (s : ℕ)
    (hs : ∀ s', s' + 4 ≠ s) : getZ r ⟨lev, p, t⟩ s = getZ r ⟨lev', p, t⟩ s := by
  have : s < 4 := by
    by_contra h
    exact hs (s - 4) (by omega)
  simp [getZ, this]

theorem disjoint_getZ : Lock.Disjoint (getZ 0 : IZ F D → ℕ → Option (Oracle F × Cap D)) (getZ 1) where
  left i₁ i₂ s h := by
    unfold getZ at h ⊢
    by_cases hs : s < 4
    · simp only [hs, ↓reduceIte] at h ⊢
      have : s = 0 ∨ s = 2 := by
        by_contra hn
        push Not at hn
        simp [show s ≠ 0 by omega, show s ≠ 2 + 0 by omega] at h
      simp [show s ≠ 1 by omega, show s ≠ 2 + 1 by omega]
    · simp only [hs, ↓reduceIte] at h ⊢
      exact disjoint_reps.left _ _ _ h
  right i₁ i₂ s h := by
    unfold getZ at h ⊢
    by_cases hs : s < 4
    · simp only [hs, ↓reduceIte] at h ⊢
      have : s = 1 ∨ s = 3 := by
        by_contra hn
        push Not at hn
        simp [show s ≠ 1 by omega, show s ≠ 2 + 1 by omega] at h
      simp [show s ≠ 0 by omega, show s ≠ 2 + 0 by omega]
    · simp only [hs, ↓reduceIte] at h ⊢
      exact disjoint_reps.right _ _ _ h

omit [Field F] [Fintype F] [DecidableEq F] in
theorem getPair_empty_none (s : ℕ) :
    Lock.getPair (getZ 0) (getZ 1) ((IZ.empty, IZ.empty) : IZ F D × IZ F D) s = none := by
  simp [Lock.getPair, getZ, IZ.empty, getRep]

/-! ## The masked phase -/

/-- **The masked phase's leaf relation**: equal rows, and `RecRelZ` between the verdicts. -/
def MRelZ (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (sch : Schedule) {n np : ℕ} (i : IRep F D)
    (a : (Fin n → Lin F np) × Bool) (b : (Fin n → Lin F np) × PendL F D) : Prop :=
  a.1 = b.1 ∧ RecRelZ r C₀ w₀ sch i a.2 b.2

/-- **The rest of a masked rep in lockstep** (`restZK` beside `restZC`). -/
noncomputable def lockRestZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₀ : Oracle F) (extra : List (Claim F S.m))
    (zc : ZcZK F np S.m) (ab : ClaimZK F np S.m) :
    Lock (Oracle F) (Cap D) (getRep r) (MRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch) (fun _ => none)
      (restZK A P₀ PA S sch hm t c₀ r C₀ extra zc ab) (restZC (D := D) A P₀ PA S sch hm t c₀ r extra zc ab) :=
  Lock.bind (R := fun i' (o o' : Option (OpenOut F) × F × F) => i' = (fun _ => none : IRep F D) ∧ o = o')
    (fun i' o o' h => by
      obtain ⟨rfl, rfl⟩ := h
      rcases o with ⟨oo, x⟩
      cases oo with
      | none => exact .ret ⟨rfl, RecRelZ.reject r C₀ _ sch _⟩
      | some op =>
        exact Lock.bind (fun i lv lv' h => .send F fun fc => .ret ⟨by rw [h.1], h.2⟩)
          (lockLigPadZ A r hr P₀ t c₀ op.T op.b C₀ sch))
    (Lock.same _)

/-- **A masked rep in lockstep** (`repMaskedZK` beside `repMaskedZC`). -/
noncomputable def lockMaskedZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (tr : Triple np) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) :
    Lock (Oracle F) (Cap D) (getRep r) (MRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch) (fun _ => none)
      (repMaskedZK A P₀ PA tr S sch hm t c₀ r C₀ extra) (repMaskedZC (D := D) A P₀ PA tr S sch hm t c₀ r extra) :=
  Lock.bind (R := fun i' (o o' : ZcZK F np S.m × LinF F np) => i' = (fun _ => none : IRep F D) ∧ o = o')
    (fun i' o o' h => by
      obtain ⟨rfl, rfl⟩ := h
      exact Lock.map (R₂ := MRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch) (consRow o.2.toLin) (consRowC o.2.toLin)
        (fun _ a b (h : MRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch _ a b) =>
          ⟨by simp [consRow, consRowC, h.1], h.2⟩)
        (Lock.bind (R := fun i' (o o' : ClaimZK F np S.m × LinF F np) => i' = (fun _ => none : IRep F D) ∧ o = o')
          (fun i' o' o'' h => by
            obtain ⟨rfl, rfl⟩ := h
            exact Lock.map (R₂ := MRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch) (consRow o'.2.toLin) (consRowC o'.2.toLin)
              (fun _ a b (h : MRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch _ a b) =>
                ⟨by simp [consRow, consRowC, h.1], h.2⟩)
              (lockRestZ A r hr P₀ PA S sch hm t c₀ C₀ extra o.1 o'.1))
          (Lock.same _)))
    (Lock.same _)

end FlockSoundness.Discharge.ZkLink
