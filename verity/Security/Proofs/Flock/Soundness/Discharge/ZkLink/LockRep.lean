import Proofs.Flock.Soundness.Discharge.ZkLink.AlignLevel

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

/-- **A masked rep in lockstep** (`repMaskedZK`, its rows in flock-verify's order, beside `repMaskedZC`). -/
noncomputable def lockMaskedZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (tr : Triple np) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) :
    Lock (Oracle F) (Cap D) (getRep r) (MRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch) (fun _ => none)
      ((repMaskedZK A P₀ PA tr S sch hm t c₀ r C₀ extra).map fun o => (o.1 ∘ rowOrder, o.2))
      (repMaskedZC (D := D) A P₀ PA tr S sch hm t c₀ r extra) :=
  Lock.map (R₂ := MRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch) (fun o => (o.1 ∘ rowOrder, o.2))
    (fun o => (o.1 ∘ ![3, 0, 1, 2, 4, 5, 6], o.2))
    (fun _ a b (h : MRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch _ a b) => ⟨by rw [h.1]; rfl, h.2⟩) <|
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

/-! ## The inner proof -/

/-- **The pads openings agree with the fixed pads oracle**: a query's pads pair that verifies at a position against
the root's cap is the oracle's two words there. -/
def PadsAgree (Hs : Hashes F D Sl) (N q : ℕ) (j : IZ F D) (io : InnerOpens F D Sl) : Prop :=
  ∀ O c, j.pads = some (O, c) → ∀ jj < q, ∀ p : Fin N,
    Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => c 0) (Nat.log 2 N) p.val (io.pads jj).1 (io.pads jj).2.1
      (io.pads jj).2.2 → (io.pads jj).1 = (O 0 p.val, O 1 p.val)

/-- **`τ`'s opening agrees with the fixed `τ` oracle**: if its leaf is the commitment's cap, it is the oracle's
`(0, 0)`. -/
def TauAgree (Hs : Hashes F D Sl) (j : IZ F D) (io : InnerOpens F D Sl) : Prop :=
  ∀ Oτ cτ, j.tau = some (Oτ, cτ) → Hs.Lτ.leaf io.tau.1 io.tau.2 = cτ 0 → io.tau.1 = Oτ 0 0

/-- **The inner proof's leaf relation**: on openings that agree with the fixed pads and `τ`, the compiled check
implies the oracle verdict. -/
def InnerRelZ (Hs : Hashes F D Sl) (N q : ℕ) (lev : IRep F D) (j : IZ F D) (a : Bool)
    (v : InnerOpens F D Sl → Prop) : Prop :=
  j.lev = lev ∧ ∀ io, PadsAgree Hs N q j io → TauAgree Hs j io → v io → a = true

/-- **The inner proof in lockstep** (`innerGameP` on the fixed pads oracle beside `innerGameCP` on its root's cap),
`τ`'s commitment a table at slot `2 + r`. -/
def lockInnerZ (Hs : Hashes F D Sl) (r : ℕ) (hr : r < 2) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N)
    {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (lev : IRep F D) (O : Oracle F) (c : Cap D) :
    Lock (Oracle F) (Cap D) (getZ r) (InnerRelZ Hs N q lev) ⟨lev, some (O, c), none⟩
      (innerGameP dom s nP q pos rows tr (padWords O).1 (padWords O).2)
      (innerGameCP Hs.H Hs.Ep Hs.Lp Hs.Lτ dom s nP q pos rows tr (c 0)) :=
  .coin F fun ε => .send (F × F) fun ρσ => .coin (Fin (nA + 3) → F) fun γ =>
    .table (2 + r) (fun Oτ cτ => ⟨lev, some (O, c), some (Oτ, cτ)⟩)
      (by simp [getZ, show 2 + r < 4 by omega])
      (fun _ _ => by simp [getZ, show 2 + r < 4 by omega])
      (fun _ _ s' hs' => by simp [getZ, hs'])
      fun Oτ cτ => .coin F fun β => .send (Fin (s + nP) → F) fun y => .coin (Fin q → F) fun w =>
        .ret ⟨rfl, fun io hpads htau hv => by
          obtain ⟨hτ, hbat, hq⟩ := hv
          have hτ' : io.tau.1 = Oτ 0 0 := htau Oτ cτ rfl hτ
          refine decide_eq_true ⟨by rw [← hτ']; exact hbat, fun jj => ?_⟩
          obtain ⟨henc, hver⟩ := hq jj
          have hp := hpads O c rfl jj jj.isLt (pos (w jj)) hver
          rw [henc, hp]
          rfl⟩

/-! ## A rep -/

/-- **A rep's leaf relation**: on rows agreeing with the fixed level tables and inner openings agreeing with the fixed
pads and `τ`, the compiled verdicts imply the oracle verdict; and the levels' caps are recorded. -/
def RepRelZ (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ) (C₀ : Oracle F) (j : IZ F D) (acc : Bool)
    (pend : PendZ F D Sl) : Prop :=
  (∀ rows io, AgreeZ r C₀ (width₀ T) j.lev pend.lig.reqs rows → PadsAgree Hs T.N T.q j io → TauAgree Hs j io →
      pend.lig.verdict rows = true → pend.inner io → acc = true) ∧
    RecStruct T.sch j.lev pend.lig

/-- **A rep in lockstep** (`repZPAt` beside `repZCP`): the pads root a table at slot `r`, the masked phase over the
level slots moved four up, and the inner proof. -/
noncomputable def lockRepZ (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ) (hr : r < 2)
    (C₀ : Oracle F) (extra : List (Claim F T.S.m)) :
    Lock (Oracle F) (Cap D) (getZ r) (RepRelZ Hs T r C₀) IZ.empty (repZPAt A T C₀ r extra)
      (repZCP A Hs T r extra) :=
  .table r (fun O c => ⟨fun _ => none, some (O, c), none⟩)
    (by simp [getZ, IZ.empty, show r < 4 by omega])
    (fun _ _ => by simp [getZ, show r < 4 by omega])
    (fun _ _ s' hs' => by
      unfold getZ
      by_cases h4 : s' < 4
      · simp [h4, hs', IZ.empty]
      · simp [h4, IZ.empty])
    fun O c => Lock.bind
      (R := fun j (a : (Fin 7 → Lin F T.np) × Bool) (b : (Fin 7 → Lin F T.np) × PendL F D) =>
        j.pads = some (O, c) ∧ j.tau = none ∧ MRelZ r C₀ (width₀ T) T.sch j.lev a b)
      (fun j o o' h => by
        rcases j with ⟨lev, pads, tau⟩
        obtain ⟨hp, ht, h⟩ := h
        simp only at hp ht h
        subst hp ht
        obtain ⟨hrows, hrec⟩ := h
        rcases o with ⟨rows, v⟩
        rcases o' with ⟨rows', p⟩
        simp only at hrows hrec
        subst hrows
        exact Lock.bind (R := InnerRelZ Hs T.N T.q lev)
          (fun j' b iv h' => .ret ⟨fun rows io hag hpa hta hver hin => by
            obtain ⟨hlev, hinner⟩ := h'
            subst hlev
            have hv : p.verdict rows = v := hrec.1 rows hag
            rw [hinner io hpa hta hin, ← hv, hver]
            rfl, by obtain ⟨rfl, -⟩ := h'; exact hrec.2⟩)
          (lockInnerZ Hs r hr T.dom T.np T.nP T.q T.pos rows T.tr lev O c))
      (Lock.mono (fun _ _ _ h => by obtain ⟨_, rfl, h⟩ := h; exact ⟨rfl, rfl, h⟩)
        (lockReindex (fun lev => (⟨lev, some (O, c), none⟩ : IZ F D)) (· + 4) (fun i s => getZ_lev r _ s)
          (fun i i' s hs => getZ_off r i i' _ _ s hs)
          (lockMaskedZ A r hr T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ C₀ extra)))

/-! ## The table and the session -/

/-- **The table's leaf relation**: the level-0 oracle and cap kept; on an accepted run whose rows, pads and `τ`
openings agree with the fixed tables, the oracle verdict; on an accepted run every requested row verifies; and the
levels' caps are recorded. -/
def TopRelZ (Hs : Hashes F D Sl) (T : TabZK F mPts) (C₀ : Oracle F) (cap₀ : Cap D) (i : IZ F D × IZ F D)
    (a : Bool × Oracle F) (out : OutZC F D Sl) : Prop :=
  a.2 = C₀ ∧ out.cap₀ = cap₀ ∧
    (out.accepted = true → AgreeZ 0 C₀ (width₀ T) i.1.lev out.pend.1.lig.reqs out.opens.rows →
      AgreeZ 1 C₀ (width₀ T) i.2.lev out.pend.2.lig.reqs out.opens.rows →
      PadsAgree Hs T.N T.q i.1 (out.opens.inner 0) → PadsAgree Hs T.N T.q i.2 (out.opens.inner 1) →
      TauAgree Hs i.1 (out.opens.inner 0) → TauAgree Hs i.2 (out.opens.inner 1) → a.1 = true) ∧
    (out.accepted = true →
      OpensOKZ Hs.H Hs.E Hs.Lf (width₀ T) cap₀ 0 out.pend.1.lig out.opens.rows out.opens.salts ∧
        OpensOKZ Hs.H Hs.E Hs.Lf (width₀ T) cap₀ 1 out.pend.2.lig out.opens.rows out.opens.salts) ∧
    RecStruct T.sch i.1.lev out.pend.1.lig ∧ RecStruct T.sch i.2.lev out.pend.2.lig

/-- **The table in lockstep after the link points** (`tabOrcP` beside `tabAfterZCP`): both reps interleaved, then the
openings. -/
noncomputable def lockAfterZ [Inhabited D] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (C₀ : Oracle F)
    (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) :
    Lock (Oracle F) (Cap D) (Lock.getPair (getZ 0) (getZ 1)) (TopRelZ Hs T C₀ cap₀) (IZ.empty, IZ.empty)
      (tabOrcP (D := D) (Sl := Sl) A T C₀ pts) (tabAfterZCP A Hs T cap₀ pts) :=
  Lock.bind (R := Lock.RPair (RepRelZ Hs T 0 C₀) (RepRelZ Hs T 1 C₀))
    (fun i acc pend h => .send (OpensZ F D Sl) fun os => .ret (by
      obtain ⟨⟨hv₀, hs₀⟩, ⟨hv₁, hs₁⟩⟩ := h
      refine ⟨rfl, rfl, fun hacc ha₀ ha₁ hp₀ hp₁ ht₀ ht₁ => ?_, fun hacc => ?_, hs₀, hs₁⟩
      · simp only [outZC, Bool.and_eq_true, decide_eq_true_eq] at hacc
        simp [hv₀ os.rows (os.inner 0) ha₀ hp₀ ht₀ hacc.1.1 hacc.2.1,
          hv₁ os.rows (os.inner 1) ha₁ hp₁ ht₁ hacc.1.2 hacc.2.2.1]
      · simp only [outZC, Bool.and_eq_true, decide_eq_true_eq] at hacc
        exact hacc.2.2.2))
    (Lock.interleaveLock disjoint_getZ
      (lockRepZ A Hs T 0 (by norm_num) C₀ (extraClaims T.S T.ptLocal mPts pts))
      (lockRepZ A Hs T 1 (by norm_num) C₀ (extraClaims T.S T.ptLocal mPts pts)))

/-- The session's leaf relation: the table's, at its position. -/
def SessRelZ [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (C₀ : Oracle F)
    (cap₀ : Cap D) (n : ℕ) (i : IZ F D × IZ F D) (a b : List (OutZC F D Sl)) : Prop :=
  ∃ x, a.getD n dfltOutZC = orcOutZ A T cap₀ x ∧ TopRelZ Hs T C₀ cap₀ i x (b.getD n dfltOutZC)

/-- **The table in lockstep inside the session** (`orcSessZ` beside `sessAtZ` with `tabAfterZCP`): its lockstep at
position `pre.length`, the same game on both sides everywhere else. -/
noncomputable def lockSessZ [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (C₀ : Oracle F) (cap₀ : Cap D) :
    Lock (Oracle F) (Cap D) (Lock.getPair (getZ 0) (getZ 1)) (SessRelZ A Hs T C₀ cap₀ pre.length)
      (IZ.empty, IZ.empty) (orcSessZ A T pre post C₀ cap₀) (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) :=
  .coin (Fin 2 → Fin mPts → F) fun pts =>
    Lock.mono (fun _ a b h => by simpa only [SessRelZ, List.length_map] using h)
      (Lock.batchLock dfltOutZC
        (Lock.mapL (R₂ := fun i o b => ∃ x, o = orcOutZ A T cap₀ x ∧ TopRelZ Hs T C₀ cap₀ i x b)
          (orcOutZ A T cap₀) (fun _ a _ hr => ⟨a, rfl, hr⟩) (lockAfterZ A Hs T C₀ cap₀ pts))
        (post.map (· pts)) (pre.map (· pts)))

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
