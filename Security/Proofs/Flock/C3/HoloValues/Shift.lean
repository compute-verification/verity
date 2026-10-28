import Proofs.Flock.C3.HoloValues.Runs

/-!
# V*'s Ligerito run one round later (`LigShift`)

`ligShift : LigShift`. A computation in `CM` or `LM` shifts (`CShift`, `LShift`) when, on coins `B` that draw at
round `k + 1` what `A` draws at round `k` (`Later`), it returns from round `k + 1` what it returns on `A` from `k`, one
round later. `coinVec` shifts, since it alone reads the counter and the coins; so do `pure` and the operations on
Ligerito's state, and bind keeps it. `ligeritoRun` is built from these, its loops by induction. `CoinSource.ofRounds` with a
round of some coins in front is `Later` than without it.
-/

namespace FlockC3.HoloValues

open Flock Flock.RecAlgebra

/-- `B` draws at round `k + 1` what `A` draws at `k`. Only draws that succeed: `CoinSource.ofRounds`' errors name the
round. -/
def Later (A B : RecAlgebra.CoinSource) : Prop := ∀ k n r, A.round k n = .ok r → B.round (k + 1) n = .ok r

/-- `m` returns on `B` from round `k + 1` what it returns on `A` from `k`, one round later, when `B` is `Later`. -/
def CShift {β : Type} (m : CM β) : Prop :=
  ∀ A B, Later A B → ∀ k k' (b : β), (m.run k).run A = .ok (b, k') → (m.run (k + 1)).run B = .ok (b, k' + 1)

theorem CShift.pure' {β : Type} (b : β) : CShift (pure b : CM β) := by
  intro A B _ k k' b' h
  have h' : (Except.ok (b, k) : Except String (β × ℕ)) = .ok (b', k') := h
  cases h'
  rfl

theorem CShift.bind' {β γ : Type} {m : CM β} {f : β → CM γ} (hm : CShift m) (hf : ∀ a, CShift (f a)) :
    CShift (m >>= f) := by
  intro A B hAB k k' b h
  rw [StateT.run_bind, ReaderT.run_bind] at h ⊢
  cases hm₀ : (m.run k).run A with
  | error e => rw [hm₀] at h; cases h
  | ok p =>
    rw [hm₀] at h
    rw [hm A B hAB k p.2 p.1 hm₀]
    exact hf p.1 A B hAB p.2 k' b h

theorem CShift.coinVec' (n : ℕ) : CShift (coinVec n) := by
  intro A B hAB k k' b h
  have hr : ∀ (j : ℕ) (C : RecAlgebra.CoinSource), (StateT.run (read : CM RecAlgebra.CoinSource) j).run C = pure (C, j) :=
    fun _ _ => rfl
  unfold coinVec at h ⊢
  by_cases hn : n = 0
  · rw [ite_eq_left hn] at h ⊢
    exact CShift.pure' _ A B hAB k k' b h
  rw [ite_eq_right hn] at h ⊢
  simp only [StateT.run_bind, ReaderT.run_bind, hr, StateT.run_get, pure_bind] at h ⊢
  cases hA : A.round k n with
  | error e => rw [hA] at h; cases h
  | ok r =>
    rw [hA] at h
    rw [hAB k n r hA]
    simp only [StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure] at h ⊢
    cases h
    rfl

theorem CShift.coinOne' : CShift coinOne := CShift.bind' (CShift.coinVec' 1) fun _ => CShift.pure' _

theorem CShift.coinK2' : CShift coinK2 := CShift.bind' (CShift.coinVec' 2) fun _ => CShift.pure' _

variable {α : Type}

/-- `CShift` from every Ligerito state. -/
def LShift {β : Type} (m : LM α β) : Prop := ∀ s, CShift (m.run s)

theorem LShift.bind' {β γ : Type} {m : LM α β} {f : β → LM α γ} (hm : LShift m) (hf : ∀ a, LShift (f a)) :
    LShift (m >>= f) := fun s => by
  rw [StateT.run_bind]; exact CShift.bind' (hm s) fun p => hf p.1 p.2

theorem LShift.pure' {β : Type} (b : β) : LShift (pure b : LM α β) := fun s => by
  rw [StateT.run_pure]; exact CShift.pure' _

theorem LShift.lift' {β : Type} {m : CM β} (hm : CShift m) : LShift (liftM m : LM α β) := fun s => by
  rw [StateT.run_monadLift]; exact CShift.bind' hm fun _ => CShift.pure' _

theorem LShift.get' : LShift (get : LM α (LSt α)) := fun s => by
  rw [StateT.run_get]; exact CShift.pure' _

theorem LShift.set' (s' : LSt α) : LShift (set s' : LM α PUnit) := fun s => by
  rw [StateT.run_set]; exact CShift.pure' _

theorem LShift.modify' (f : LSt α → LSt α) : LShift (modify f : LM α PUnit) := fun s => by
  rw [StateT.run_modify]; exact CShift.pure' _

theorem LShift.throw' {β : Type} (e : String) : LShift (throw e : LM α β) := by
  intro s A B _ k k' b h
  have h' : (Except.error e : Except String ((β × LSt α) × ℕ)) = .ok (b, k') := h
  cases h'

theorem LShift.throw_bind {β γ : Type} (e : String) (f : β → LM α γ) : LShift (throw e >>= f) := by
  intro s A B _ k k' b h
  have h' : (Except.error e : Except String ((γ × LSt α) × ℕ)) = .ok (b, k') := h
  cases h'

/-! ## Ligerito's steps -/

theorem nextMsg_shift (M : Messages α) : LShift (RecAlgebra.nextMsg M) := by
  unfold RecAlgebra.nextMsg
  refine LShift.bind' LShift.get' fun st => ?_
  split
  · exact LShift.bind' (LShift.set' _) fun _ => LShift.pure' _
  · exact LShift.throw' _

theorem nextOod_shift (M : Messages α) : LShift (RecAlgebra.nextOod M) := by
  unfold RecAlgebra.nextOod
  refine LShift.bind' LShift.get' fun st => ?_
  split
  · exact LShift.bind' (LShift.set' _) fun _ => LShift.pure' _
  · exact LShift.throw' _

variable [Val α]

theorem challenge_shift (M : Messages α) : LShift (RecAlgebra.challenge M) := by
  unfold RecAlgebra.challenge
  refine LShift.bind' (LShift.lift' CShift.coinK2') fun rk => LShift.bind' LShift.get' fun st => ?_
  split
  exact LShift.bind' (LShift.set' _) fun _ => LShift.bind' (nextMsg_shift M) fun _ =>
    LShift.bind' (LShift.modify' _) fun _ => LShift.pure' _

theorem challenges_shift (M : Messages α) : ∀ n, LShift (RecAlgebra.challenges M n)
  | 0 => by rw [RecAlgebra.challenges]; exact LShift.pure' _
  | n + 1 => by
    rw [RecAlgebra.challenges]
    exact LShift.bind' (challenge_shift M) fun _ => LShift.bind' (challenges_shift M n) fun _ => LShift.pure' _

theorem oodIntro_shift (M : Messages α) (dim split : ℕ) : LShift (RecAlgebra.oodIntro M dim split) := by
  unfold RecAlgebra.oodIntro RecAlgebra.addScaled
  exact LShift.bind' (LShift.lift' (CShift.coinVec' _)) fun _ => LShift.bind' (nextOod_shift M) fun _ =>
    LShift.bind' (nextMsg_shift M) fun _ => LShift.bind' (LShift.lift' CShift.coinOne') fun _ =>
      LShift.bind' (LShift.modify' _) fun _ => LShift.modify' _

theorem query_shift (sch : Schedule) (M : Messages α) (i : ℕ) (fold : Array F256) :
    LShift (RecAlgebra.query sch M i fold) := by
  unfold RecAlgebra.query
  exact LShift.bind' (LShift.lift' (CShift.coinVec' _)) fun _ =>
    LShift.bind' (LShift.lift' (CShift.coinVec' _)) fun _ => LShift.pure' _

theorem queryFold_shift (sch : Schedule) (M : Messages α) (i cols : ℕ) (fold : Array F256) :
    LShift (RecAlgebra.queryFold sch M i cols fold) := by
  unfold RecAlgebra.queryFold RecAlgebra.addScaled
  refine LShift.bind' (query_shift sch M i fold) fun q => ?_
  obtain ⟨pos, al, e⟩ := q
  exact LShift.bind' (nextMsg_shift M) fun _ => LShift.bind' (LShift.lift' CShift.coinOne') fun _ =>
    LShift.bind' (LShift.modify' _) fun _ => LShift.modify' _

theorem levelsLoop_shift (sch : Schedule) (M : Messages α) (R : ℕ) :
    ∀ fuel i split levels ext, LShift (RecAlgebra.levelsLoop sch M R fuel i split levels ext)
  | 0, _, _, _, _ => by rw [RecAlgebra.levelsLoop]; exact LShift.pure' _
  | fuel + 1, i, split, levels, ext => by
    rw [RecAlgebra.levelsLoop]
    by_cases h4 : split < 4
    · rw [ite_eq_left h4]; exact LShift.throw_bind _ _
    rw [ite_eq_right h4]
    refine LShift.bind' (challenges_shift M 4) fun chs => ?_
    by_cases hR : i + 1 < R
    · rw [ite_eq_left hR]
      exact LShift.bind' (oodIntro_shift M _ _) fun _ => LShift.bind' (oodIntro_shift M _ _) fun _ =>
        LShift.bind' (queryFold_shift sch M _ _ chs) fun _ => levelsLoop_shift sch M R fuel _ _ _ _
    · rw [ite_eq_right hR]
      refine LShift.bind' (query_shift sch M R chs) fun q => ?_
      obtain ⟨pos, al, e⟩ := q
      exact LShift.bind' (LShift.lift' CShift.coinOne') fun _ =>
        LShift.bind' (LShift.modify' _) fun _ => levelsLoop_shift sch M R fuel _ _ _ _

/-- **V*'s Ligerito run shifts.** -/
theorem ligeritoRun_shift (sch : Schedule) (M : Messages α) : LShift (RecAlgebra.ligeritoRun sch M) := by
  unfold RecAlgebra.ligeritoRun
  refine LShift.bind' (LShift.lift' (CShift.coinVec' _)) fun _ => LShift.bind' (nextOod_shift M) fun _ =>
    LShift.bind' (LShift.lift' CShift.coinOne') fun _ => LShift.bind' (LShift.modify' _) fun _ =>
      LShift.bind' (nextMsg_shift M) fun _ => LShift.bind' (LShift.modify' _) fun _ =>
        LShift.bind' (challenges_shift M _) fun _ => LShift.bind' (oodIntro_shift M _ _) fun _ =>
          LShift.bind' (oodIntro_shift M _ _) fun _ => LShift.bind' (queryFold_shift sch M _ _ _) fun _ =>
            LShift.bind' (levelsLoop_shift sch M _ _ _ _ _ _) fun p => ?_
  obtain ⟨levels, ext⟩ := p
  exact LShift.pure' _

/-! ## The coins one round later -/

/-- A round of some coins in front: `CoinSource.ofRounds` keeps it, so round `k + 1` is the shorter coins' round `k`. -/
theorem later_ofRounds {c : Array (Array F128)} {x : Array F128} (hx : x.size ≠ 0) :
    Later (CoinSource.ofRounds c) (CoinSource.ofRounds (#[x] ++ c)) := by
  intro k n r h
  have hx' : x ≠ #[] := fun e => hx (by rw [e]; rfl)
  have h1 : (#[x].filter (·.size ≠ 0)).size = 1 := by simp [hx']
  have e : ((#[x] ++ c).filter (·.size ≠ 0))[k + 1]? = (c.filter (·.size ≠ 0))[k]? := by
    rw [Array.filter_append, Array.getElem?_append_right (by omega), h1, Nat.add_sub_cancel]
    exact Array.size_append ..
  unfold CoinSource.ofRounds at h ⊢
  dsimp only at h ⊢
  rw [e]
  revert h
  cases (c.filter (·.size ≠ 0))[k]? with
  | none => intro h; cases h
  | some r' => dsimp only; split <;> intro h <;> first | exact h | cases h

/-- **F6: V*'s Ligerito run behind one more round** (`ligeritoRun_shift` at `later_ofRounds`). -/
theorem ligShift : LigShift := fun sch M s₀ k k' _ _ out hx h =>
  ligeritoRun_shift sch M s₀ _ _ (later_ofRounds hx) k k' out h

end FlockC3.HoloValues
