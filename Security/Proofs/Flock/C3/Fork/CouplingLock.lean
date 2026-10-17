import Proofs.Flock.Soundness.Model.HoloCompiled
import Proofs.Flock.C3.Fork.CouplingPull

/-!
# a1l, step 3: the holographic table in lockstep

The compiled holographic table below the level-0 cap (`holoTableCmpL`) in lockstep with an oracle game
(`holoOrcL`): both reps as `lockTableL`'s (`lockHoloRep`, on `holoRepO`, which is `holoRep` with Ligerito's verdict
beside its outputs, so that the sparse phase runs on both sides whatever Ligerito said), their sparse phases and the
registered opening as the same games (`lockRegPhase`), and R's Ligerito on the registered word `w` as rep 0's
pulled from `2` (`lockLigR`). R's tables sit beside the reps' (`liftL`, `liftR`): the reps' at even slots, R's at
odd ones. The leaf relation (`TopRelH`) is `TopRelL`'s with R's stream as a third rep.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}

noncomputable section

/-- **One rep with hidden units, the coupling's oracle side**: `holoRep` with Ligerito's verdict beside the
zerocheck's and lincheck's outputs, which it hands on at every verdict, as `holoRepC` does. -/
def holoRepO (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) : Game (Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6)) × Bool) := do
  match ← zerocheck A S.m with
  | none => pure (none, false)
  | some zc =>
    match ← holoLincheck A S zc with
    | none => pure (none, false)
    | some lc =>
      match ← opening A (by omega) (lc.ab :: cClaim A zc :: extra) with
      | none => pure (none, false)
      | some op => do
        let ok ← ligerito A sch op.T op.b C₀
        pure (some (zc, lc), ok)

/-- **A rep with hidden units in lockstep**: the PIOP phases are the same games, Ligerito is `lockLig`, and both
sides hand on the same outputs. -/
def lockHoloRep (A : Arith F K) (r : ℕ) (hr : r < 2) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) :
    Lock (Oracle F) (Cap D) (getRep r)
      (fun i a b => a.1 = b.1 ∧ RecRel r C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ i a.2 b.2) (fun _ => none)
      (holoRepO A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm C₀ extra)
      (holoRepC A r S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm extra) :=
  Lock.bind (R := fun i' a b => i' = (fun _ => none) ∧ a = b)
    (fun i₁ zc zc' hzc => by
      obtain ⟨hi, rfl⟩ := hzc
      subst i₁
      match zc with
      | none => exact .ret ⟨rfl, RecRel.reject r C₀ _ _⟩
      | some zc =>
        exact Lock.bind (R := fun i' a b => i' = (fun _ => none) ∧ a = b)
          (fun i₂ lc lc' hlc => by
            obtain ⟨hi, rfl⟩ := hlc
            subst i₂
            match lc with
            | none => exact .ret ⟨rfl, RecRel.reject r C₀ _ _⟩
            | some lc =>
              exact Lock.bind (R := fun i' a b => i' = (fun _ => none) ∧ a = b)
                (fun i₃ op op' hop => by
                  obtain ⟨hi, rfl⟩ := hop
                  subst i₃
                  match op with
                  | none => exact .ret ⟨rfl, RecRel.reject r C₀ _ _⟩
                  | some op =>
                    exact Lock.map (fun ok => (some (zc, lc), ok)) (fun p => (some (zc, lc), p))
                      (fun _ _ _ h => ⟨rfl, h⟩) (lockLig A r hr m k0 l₀ l₁ rest ext hk0 op.T op.b C₀))
                (Lock.same (opening A (by omega) (lc.ab :: cClaim A zc :: extra))))
          (Lock.same (holoLincheck A S zc)))
    (Lock.same (zerocheck A S.m))

/-- **R's Ligerito in lockstep**, as rep `2` on the registered word `w`, at a schedule of no level (both sides
reject) or of at least two whose level 0 folds `k0` lanes: rep 0's lockstep, its openings pulled from `2`. -/
def lockLigR (A : Arith F K) (rsch : Schedule)
    (hsh : rsch.levels = [] ∨ 2 ≤ rsch.levels.length ∧ (rsch.levels.headD default).lanes = rsch.k0)
    (T : F) (b : ℕ → F) (w : Oracle F) :
    Lock (Oracle F) (Cap D) (getRep 0) (RecRel 2 w rsch) (fun _ => none) (ligerito A rsch T b w)
      (ligeritoC A 2 rsch T b) :=
  match rsch, hsh with
  | ⟨_, _, [], _⟩, _ => .ret (RecRel.reject 2 w _ _)
  | ⟨_, _, [_], _⟩, h => False.elim (by simp at h)
  | ⟨m, k0, l₀ :: l₁ :: rest, ext⟩, h => by
      rw [ligeritoC_pull, ← map_id' (ligerito A _ T b w)]
      exact Lock.map id (pullP 2) (fun _ _ _ h => RecRel.pull h)
        (lockLig A 0 (by norm_num) m k0 l₀ l₁ rest ext (by simpa using h) T b w)

/-- **The reps' sparse phases and the registered opening in lockstep**: the same games on the reps' outputs, and
`fin`'s lockstep on the opening's. -/
def lockRegPhase {α β I T D' : Type} {get : I → ℕ → Option (T × D')} {R : I → α → β → Prop} {i : I}
    (A : Arith F K) (S : Statement) (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ)
    (τ : Fin (2 ^ sh.sLog) → ℕ) {no : α} {no' : β} (hno : R i no no') {fin : OpenOut F → Game α}
    {fin' : OpenOut F → Game β} (hfin : ∀ op, Lock T D' get R i (fin op) (fin' op)) :
    ∀ reps, Lock T D' get R i (regPhase A S sh units τ no fin reps) (regPhase A S sh units τ no' fin' reps)
  | (some (zc₁, lc₁), some (zc₂, lc₂)) =>
    Lock.bind (R := fun i' a b => i' = i ∧ a = b)
      (fun i₁ c₁ c₁' hc => by
        obtain ⟨rfl, rfl⟩ := hc
        match c₁ with
        | none => exact .ret hno
        | some c₁ =>
          exact Lock.bind (R := fun i' a b => i' = i₁ ∧ a = b)
            (fun i₂ c₂ c₂' hc => by
              obtain ⟨rfl, rfl⟩ := hc
              match c₂ with
              | none => exact .ret hno
              | some c₂ =>
                by_cases hreg : 7 ≤ sh.mReg
                · dsimp only
                  rw [dite_eq_left_of_eq_true (eq_true hreg), dite_eq_left_of_eq_true (eq_true hreg)]
                  exact Lock.bind (R := fun i' a b => i' = i₂ ∧ a = b)
                    (fun i₃ op op' hop => by
                      obtain ⟨rfl, rfl⟩ := hop
                      match op with
                      | none => exact .ret hno
                      | some op => exact hfin op)
                    (Lock.same (opening A hreg [c₁.toClaim, c₂.toClaim]))
                · dsimp only
                  rw [dite_eq_right_of_eq_false (eq_false hreg), dite_eq_right_of_eq_false (eq_false hreg)]
                  exact .ret hno)
            (Lock.same (sparsePhase A sh (holoPub S sh units zc₂ lc₂) τ lc₂.h)))
      (Lock.same (sparsePhase A sh (holoPub S sh units zc₁ lc₁) τ lc₁.h))
  | (none, _) => .ret hno
  | (some (_, _), none) => .ret hno

/-- **The coupling's oracle game** after the level-0 table `C₀`: the link points, both reps as `holoRepO`, their
sparse phases and the registered opening with Ligerito on the registered word `w`, and a dummy last message.
Accepted iff both reps' Ligerito and R's accept. -/
def holoOrcL {Sl : Type} (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule)
    (w C₀ : Oracle F) : Game Bool :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    (interleave (holoRepO A S sch hm C₀ (extraClaims S ptLocal mPts pts))
      (holoRepO A S sch hm C₀ (extraClaims S ptLocal mPts pts))).bind fun reps =>
    (regPhase A S sh units τ false (fun op => ligerito A rsch op.T op.b w) (reps.1.1, reps.2.1)).bind fun okR =>
    Game.send (Opens F D × Salts Sl) fun _ => Game.ret (reps.1.2 && reps.2.2 && okR)

open Classical in
/-- The compiled holographic table's continuation after the level-0 cap. -/
def holoTableCmpL {Sl : Type} (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule)
    (capReg cap₀ : Cap D) : Game (OutHL F D Sl) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    (interleave (holoRepC (D := D) A 0 S sch hm (extraClaims S ptLocal mPts pts))
      (holoRepC (D := D) A 1 S sch hm (extraClaims S ptLocal mPts pts))).bind fun reps =>
    (regPhase A S sh units τ PendL.reject (fun op => ligeritoC (D := D) A 2 rsch op.T op.b)
      (reps.1.1, reps.2.1)).bind fun pendR =>
    Game.send (Opens F D × Salts Sl) fun os => Game.ret
      ⟨reps.1.2.verdict os.1 && reps.2.2.verdict os.1 && pendR.verdict os.1 &&
        decide (OpensOKL H E Lf cap₀ 0 reps.1.2 os.1 os.2 ∧ OpensOKL H E Lf cap₀ 1 reps.2.2 os.1 os.2 ∧
          OpensOKL H E Lf capReg 2 pendR os.1 os.2), cap₀, (reps.1.2, reps.2.2), pendR, os.1, os.2⟩

theorem holoTableCL_send {Sl : Type} (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule)
    (capReg : Cap D) :
    holoTableCL A H E Lf S sch hm ptLocal mPts sh units τ rsch capReg =
      Game.send (Cap D) (holoTableCmpL A H E Lf S sch hm ptLocal mPts sh units τ rsch capReg) :=
  rfl

/-- **The holographic table's leaf relation** below the level-0 cap `cap₀`: `TopRelL`'s, with R's stream a third
rep whose level 0 is the registered word `w` under the registered cap `capReg`, and its tables `i.2`. -/
def TopRelH {Sl : Type} (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch rsch : Schedule) (C₀ w : Oracle F) (cap₀ capReg : Cap D) (i : (IRep F D × IRep F D) × IRep F D)
    (v : Bool) (out : OutHL F D Sl) : Prop :=
  out.cap₀ = cap₀ ∧
    (out.accepted = true → AgreeRep 0 C₀ i.1.1 out.pend.1.reqs out.opens →
      AgreeRep 1 C₀ i.1.2 out.pend.2.reqs out.opens → AgreeRep 2 w i.2 out.pendR.reqs out.opens → v = true) ∧
    (out.accepted = true → OpensOKL H E Lf cap₀ 0 out.pend.1 out.opens out.salts ∧
      OpensOKL H E Lf cap₀ 1 out.pend.2 out.opens out.salts ∧
      OpensOKL H E Lf capReg 2 out.pendR out.opens out.salts) ∧
    RecStruct sch i.1.1 out.pend.1 ∧ RecStruct sch i.1.2 out.pend.2 ∧ RecStruct rsch i.2 out.pendR

/-- **The holographic table in lockstep** below the level-0 cap: the link points, both reps interleaved
(`lockHoloRep`, at the even slots), their sparse phases and the registered opening with R's Ligerito (`lockLigR`, at
the odd slots), and the openings. -/
def lockHoloL [Inhabited D] {Sl : Type} (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (sh : SparseShape)
    (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule)
    (hsh : rsch.levels = [] ∨ 2 ≤ rsch.levels.length ∧ (rsch.levels.headD default).lanes = rsch.k0)
    (C₀ w : Oracle F) (cap₀ capReg : Cap D) :
    Lock (Oracle F) (Cap D) (getEO (Lock.getPair (getRep 0) (getRep 1)) (getRep 0))
      (TopRelH H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ rsch C₀ w cap₀ capReg)
      (((fun _ => none), (fun _ => none)), (fun _ => none))
      (holoOrcL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts sh units τ rsch w C₀)
      (holoTableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts sh units τ rsch capReg cap₀) :=
  .coin (Fin 2 → Fin mPts → F) fun pts =>
    Lock.bind
      (fun i reps pend h => by
        obtain ⟨⟨i₀, i₁⟩, iR⟩ := i
        obtain ⟨hiR, ⟨ha₁, hr₁⟩, ⟨ha₂, hr₂⟩⟩ := h
        simp only at hiR
        subst hiR
        rcases reps with ⟨⟨oa₁, v₁⟩, ⟨oa₂, v₂⟩⟩
        rcases pend with ⟨⟨ob₁, p₁⟩, ⟨ob₂, p₂⟩⟩
        simp only at ha₁ ha₂ hr₁ hr₂
        subst ha₁ ha₂
        exact Lock.bind
          (fun i okR pendR h' => .send (Opens F D × Salts Sl) fun os => .ret (by
            obtain ⟨⟨j₀, j₁⟩, jR⟩ := i
            obtain ⟨hj, hRv, hRs⟩ := h'
            simp only [Prod.mk.injEq] at hj
            obtain ⟨rfl, rfl⟩ := hj
            refine ⟨rfl, fun hacc ha₀ ha₁ ha₂ => ?_, fun hacc => ?_, hr₁.2, hr₂.2, hRs⟩
            · simp only [Bool.and_eq_true, decide_eq_true_eq] at hacc
              have e₁ := (hr₁.1 os.1 ha₀).symm.trans hacc.1.1.1
              have e₂ := (hr₂.1 os.1 ha₁).symm.trans hacc.1.1.2
              have e₃ := (hRv os.1 ha₂).symm.trans hacc.1.2
              simp [e₁, e₂, e₃]
            · simp only [Bool.and_eq_true, decide_eq_true_eq] at hacc
              exact hacc.2))
          (liftR (get₁ := Lock.getPair (getRep 0) (getRep 1)) (i₀, i₁)
            (lockRegPhase A S sh units τ (RecRel.reject 2 w rsch _)
              (fun op => lockLigR A rsch hsh op.T op.b w) (oa₁, oa₂))))
      (liftL (get₂ := getRep 0) (fun _ => none)
        (Lock.interleaveLock disjoint_reps
          (lockHoloRep A 0 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₀ (extraClaims S ptLocal mPts pts))
          (lockHoloRep A 1 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₀ (extraClaims S ptLocal mPts pts))))

end

end FlockC3.Fork
