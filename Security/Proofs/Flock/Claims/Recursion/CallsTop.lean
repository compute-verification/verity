import Proofs.Flock.Claims.Recursion.CallsRep

/-!
# A verdict's SHA-512 calls on every run

A rep's verification calls SHA-512 for its Ligerito queries (`ligC`) and for at most each recorded round of its stream,
the transcript's potential at the start (`cl_verifyRep`). The verdict at a parsed record adds the decoder's calls, S17's
proof digests and two reps per table (`cl_verifyJ`); a bind may assume what its first run returned (`cl_bind_q`), here
the decoded session's round counts and the fast100 schedules (`mapM_ok`). The decoder calls SHA-512 a fixed number of
times per table and per rep stream and at most twice per round of each of the record's streams (`cl_decodeJ`). Its walk
picks each step by the head of the program it bounds (`cl_bind_of`, `cl_head`): a failed `refine` against the wrong
program unfolds both and costs more than the step.
-/

namespace FlockDeploy.KS

open Flock

theorem cl_throw_bind {α β : Type} (e : String) (f : α → LogV β) (k : ℕ) : CL (throw e >>= f) k := fun l => by
  have : (throw e >>= f : LogV β).go l = (.error e, l) := rfl
  rw [this]
  exact Nat.le_add_right _ _

theorem potT_init (rounds : Array Round) (d : ByteArray) : potT (Transcript.init rounds d) = rounds.size := rfl

/-- One step of a walk over a logged program, at any budget. -/
macro "cl_step" : tactic => `(tactic| first
  | with_reducible exact cl_throw_bind _ _ _
  | with_reducible exact calcCL.pure' _ _
  | with_reducible exact calcCL.mono (cl_throw _) (Nat.zero_le _)
  | with_reducible exact calcCL.mono (cl_liftM _) (Nat.zero_le _)
  | with_reducible refine calcCL.bind0 ?_ fun _ => ?_
  | with_reducible refine calcCL.ite_k ?_ ?_
  | split
  | dsimp only)

/-- A schedule's Ligerito calls. -/
abbrev ligC (sch : Schedule) : ℕ := (1 + (sch.levels.size - 1)) * qAll sch

/-- **A rep's verification makes at most `ligC sch + B` calls**, `B` bounding its session's streams' rounds, whether it
accepts or throws. -/
theorem cl_verifyRep (h : ByteArray → ByteArray) (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : Session)
    (i r : ℕ) (bytes : ByteArray) (B : ℕ) (hB : ∀ s ∈ sess.streams, s.rounds.size ≤ B) :
    CL (verifyRepWith (Hashes.logged h) spec st sch sess i r bytes) (ligC sch + B) := by
  unfold verifyRepWith
  repeat' (first
    | with_reducible refine calcCL.bind_le (cl_run (pot := potT) (n := ligC sch) ?_ _) (fun _ => ?_) ?_
    | with_reducible exact cs_ligerito h _ _ _ _ _ _
    | with_reducible exact (calcCS potT).mono (cs_zerocheck h _ _ _ _) (Nat.zero_le _)
    | with_reducible exact (calcCS potT).mono (cs_lincheck h _ _ _) (Nat.zero_le _)
    | with_reducible exact (calcCS potT).mono (cs_ringSwitch h _ _ _) (Nat.zero_le _)
    | tw_step h
    | cl_step)
  all_goals
    rw [potT_init]
    have := hB _ (Array.mem_of_find?_eq_some (by assumption))
    omega

/-- A bind whose continuation may assume what the first run returned. -/
theorem cl_bind_q {α β : Type} {x : LogV α} {f : α → LogV β} {n k : ℕ} (Q : α → Prop) (hx : CL x n)
    (hq : ∀ l a l', x.go l = (.ok a, l') → Q a) (hf : ∀ a, Q a → CL (f a) k) : CL (x >>= f) (n + k) := fun l => by
  have h1 := hx l
  rw [go_bind]
  rcases e : x.go l with ⟨r, l₁⟩
  rw [e] at h1
  cases r with
  | error _ => dsimp only at h1 ⊢; omega
  | ok a => have h2 := hf a (hq l a l₁ e) l₁; dsimp only at h1 ⊢; omega

/-- What every value a list's `mapM` returns satisfies, from what each element's run returns. -/
theorem list_mapM_ok {α β : Type} {g : α → LogV β} {p : β → Prop}
    (hg : ∀ a l b l', (g a).go l = (.ok b, l') → p b) :
    ∀ (xs : List α) (l : Array ByteArray) (bs : List β) (l' : Array ByteArray),
      (xs.mapM g).go l = (.ok bs, l') → ∀ b ∈ bs, p b
  | [], l, bs, l', e => by
    rw [List.mapM_nil, go_pure] at e
    cases e
    intro b hb
    cases hb
  | a :: xs, l, bs, l', e => by
    rw [List.mapM_cons, go_bind] at e
    rcases ha : (g a).go l with ⟨r, l₁⟩
    rw [ha] at e
    cases r with
    | error _ => cases e
    | ok b =>
      dsimp only at e
      rw [go_bind] at e
      rcases hb : (xs.mapM g).go l₁ with ⟨r', l₂⟩
      rw [hb] at e
      cases r' with
      | error _ => cases e
      | ok bs' =>
        dsimp only at e
        rw [go_pure] at e
        cases e
        intro y hy
        rcases List.mem_cons.1 hy with rfl | hy
        · exact hg a l _ _ ha
        · exact list_mapM_ok hg xs l₁ bs' _ hb y hy

theorem mapM_ok {α β : Type} {g : α → LogV β} {p : β → Prop} (hg : ∀ a l b l', (g a).go l = (.ok b, l') → p b)
    (xs : Array α) (l : Array ByteArray) (bs : Array β) (l' : Array ByteArray) (e : (xs.mapM g).go l = (.ok bs, l')) :
    ∀ b ∈ bs, p b := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind, go_bind] at e
  rcases h1 : (xs.toList.mapM g).go l with ⟨r, l₁⟩
  rw [h1] at e
  cases r with
  | error _ => cases e
  | ok ys =>
    dsimp only at e
    rw [go_pure] at e
    cases e
    intro b hb
    exact list_mapM_ok hg _ _ _ _ h1 b (by simpa using hb)

theorem ligC_default : ligC default = 0 := rfl

theorem ligC_get! {schs : Array Schedule} {L : ℕ} (h : ∀ s ∈ schs, ligC s ≤ L) (i : ℕ) : ligC schs[i]! ≤ L := by
  by_cases hi : i < schs.size
  · rw [getElem!_pos schs i hi]; exact h _ (Array.getElem_mem hi)
  · rw [getElem!_neg schs i hi, ligC_default]; exact Nat.zero_le _

/-- **A verdict's SHA-512 calls on every run**: the decoder's `D`, one per rep stream for S17's proof digests, and two
reps per table at `L + B` each, where `L` bounds every fast100 schedule's Ligerito calls and `B` the rounds of every
stream a decoder run returns, whether the verdict accepts or throws. -/
theorem cl_verifyJ (h : ByteArray → ByteArray) (keep : Bool → Bool) (spec : SessionSpec) (tabs : Array Setup) (j : Lean.Json)
    (proofs : Array ByteArray) (D L B : ℕ)
    (hD : CL (Record.decodeJWithK keep (Hashes.logged h) spec j) D)
    (hB : ∀ l sess l', (Record.decodeJWithK keep (Hashes.logged h) spec j).go l = (.ok sess, l') →
      ∀ s ∈ sess.streams, s.rounds.size ≤ B)
    (hL : ∀ m sch, fast100 m = some sch → ligC sch ≤ L) :
    CL (verifyJWithMK keep (Hashes.logged h) spec tabs j proofs)
      (D + (spec.repStreams.size + tabs.size * (2 * (L + B)))) := by
  unfold verifyJWithMK
  with_reducible refine calcCL.ite_k ?_ ?_
  · exact calcCL.mono (cl_throw _) (Nat.zero_le _)
  refine cl_bind_q (fun sess => ∀ s ∈ sess.streams, s.rounds.size ≤ B) hD hB fun sess hsB => ?_
  refine calcCL.bind0 (calcCL.mono (calcCL.forIn (c := 0) (fun _ _ => ?_) _ _) (Nat.le_of_eq (Nat.mul_zero _)))
    fun _ => ?_
  · repeat' cl_step
  refine calcCL.mono (cl_bind_q (n := 0) (k := spec.repStreams.size + tabs.size * (2 * (L + B)))
    (fun schs => ∀ s ∈ schs, ligC s ≤ L) ?_ ?_ fun schs hsL => ?_) (Nat.le_of_eq (Nat.zero_add _))
  · exact calcCL.mono (calcCL.mapM_const (c := 0) (fun _ => by repeat' cl_step) tabs) (Nat.le_of_eq (Nat.mul_zero _))
  · refine fun l a l' e => mapM_ok (fun st l b l' e => ?_) tabs l a l' e
    revert e
    split
    · intro e
      rw [go_pure] at e
      cases e
      exact hL _ _ (by assumption)
    · intro e
      rw [go_throw] at e
      cases e
  dsimp only
  with_reducible refine calcCL.ite_k ?_ ?_
  · with_reducible exact cl_throw_bind _ _ _
  with_reducible refine calcCL.ite_k ?_ ?_
  · with_reducible exact cl_throw_bind _ _ _
  with_reducible refine calcCL.bind_le (calcCL.forIn (c := 1) (fun _ _ => ?_) _ _) (fun _ => ?_) ?_
  · with_reducible refine calcCL.bind_le (cl_dg h _ _) (fun _ => ?_) (Nat.le_refl _)
    repeat' cl_step
  · with_reducible refine calcCL.bind_le (calcCL.range_forIn (c := 2 * (L + B)) (fun i _ => ?_) _ _) (fun _ => ?_) ?_
    · with_reducible refine calcCL.bind_le (calcCL.mono (cl_verifyRep h _ _ _ _ _ _ _ B hsB)
        (Nat.add_le_add_right (ligC_get! hsL i) B)) (fun _ => ?_) (by omega)
      with_reducible refine calcCL.bind_le (calcCL.mono (cl_verifyRep h _ _ _ _ _ _ _ B hsB)
        (Nat.add_le_add_right (ligC_get! hsL i) B)) (fun _ => ?_) (by omega)
      repeat' cl_step
    · exact calcCL.pure' _ _
    · simp only [Std.Legacy.Range.size, Array.size_zip, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, Nat.mul_one]
      omega
  · simp only [Array.size_zip, Nat.mul_one]
    omega

/-! ## The decoder -/

theorem Calc.list_mapM_mem {m : Type → Type} [Monad m] [LawfulMonad m] {P : ∀ {α : Type}, m α → ℕ → Prop}
    (C : Calc m P) {α β : Type} {g : α → m β} {c : ℕ} :
    ∀ xs : List α, (∀ a ∈ xs, P (g a) c) → P (xs.mapM g) (xs.length * c)
  | [], _ => by rw [List.mapM_nil]; exact C.pure' _ _
  | a :: xs, hg => by
    rw [List.mapM_cons]
    exact C.mono (C.bind (hg a (List.mem_cons_self ..)) fun b =>
      C.bind (Calc.list_mapM_mem C xs fun y hy => hg y (List.mem_cons_of_mem _ hy)) fun bs => C.pure (b :: bs))
      (by rw [List.length_cons, Nat.succ_mul]; omega)

/-- `mapM` at one bound per element, each element's run bounded where it is an element. -/
theorem Calc.mapM_mem {m : Type → Type} [Monad m] [LawfulMonad m] {P : ∀ {α : Type}, m α → ℕ → Prop}
    (C : Calc m P) {α β : Type} {g : α → m β} {c : ℕ} {xs : Array α} (hg : ∀ a ∈ xs, P (g a) c) :
    P (xs.mapM g) (xs.size * c) := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind]
  exact C.mono (C.bind (C.list_mapM_mem xs.toList fun a ha => hg a (by simpa using ha)) fun _ => C.pure _)
    (by rw [Array.length_toList]; omega)

theorem cl_of_map {α β : Type} (f : α → β) {x : LogV α} {n : ℕ} (h : CL (f <$> x) n) : CL x n := fun l => by
  have := h l
  rw [map_eq_pure_bind, go_bind] at this
  rcases e : x.go l with ⟨r, l₁⟩
  rw [e] at this
  cases r <;> exact this

theorem cl_mapIdxM {α β : Type} {g : ℕ → α → LogV β} {c : ℕ} (hg : ∀ i a, CL (g i a) c) (xs : Array α) :
    CL (xs.mapIdxM g) (xs.size * c) := by
  apply cl_of_map Array.toList
  rw [Array.toList_mapIdxM]
  have := calcCL.mapIdxM_go hg xs.toList #[]
  rw [Array.length_toList] at this
  exact this

/-- A bind of a lifted step may assume what the step returned. -/
theorem cl_liftM_bind_q {α β : Type} {v : Flock.V α} {f : α → LogV β} {k : ℕ} (Q : α → Prop)
    (hq : ∀ a, v = .ok a → Q a) (hf : ∀ a, Q a → CL (f a) k) : CL (liftM v >>= f) k := fun l => by
  rw [go_bind, go_liftM]
  cases e : v with
  | error _ => exact Nat.le_add_right _ _
  | ok a => exact hf a (hq a e) l

theorem cl_liftM_ok_bind {α β : Type} {a : α} {f : α → LogV β} {k : ℕ} (hf : CL (f a) k) :
    CL (liftM (Except.ok a : Flock.V α) >>= f) k :=
  cl_liftM_bind_q (fun b => b = a) (fun _ e => (Except.ok.inj e).symm) fun _ e => e ▸ hf

theorem cl_rounds {β : Type} {s : Lean.Json} {R k : ℕ} {f : Array Lean.Json → LogV β}
    (hR : ∀ rj, Record.arr s "rounds" = .ok rj → rj.size ≤ R) (hf : ∀ rj, rj.size ≤ R → CL (f rj) k) :
    CL (liftM (Record.arr s "rounds") >>= f) k :=
  cl_liftM_bind_q _ hR hf

theorem cl_ite_bind {α β : Type} {c : Prop} [Decidable c] {t e : LogV α} {f : α → LogV β} {k : ℕ}
    (ht : CL (t >>= f) k) (he : CL (e >>= f) k) : CL ((if c then t else e) >>= f) k := by
  by_cases hc : c
  · rw [if_pos hc]; exact ht
  · rw [if_neg hc]; exact he

theorem cl_bind_lift0 {α β : Type} {v : Flock.V α} {f : α → LogV β} {k : ℕ} (hf : ∀ a, CL (f a) k) :
    CL (liftM v >>= f) k :=
  calcCL.bind0 (calcCL.mono (cl_liftM v) (Nat.zero_le _)) hf

theorem cl_bind_dg {β : Type} {h : ByteArray → ByteArray} {spec : SessionSpec} {b : ByteArray}
    {f : ByteArray → LogV β} {k : ℕ} (hf : ∀ d, CL (f d) (k - 1)) (hk : 1 ≤ k) :
    CL (spec.dgWith (Hashes.logged h) b >>= f) k :=
  calcCL.bind_le (cl_dg h spec b) hf hk

theorem cl_bind_mapM {α β γ : Type} {g : α → LogV β} {xs : Array α} {f : Array β → LogV γ} {c k : ℕ}
    (hg : ∀ a ∈ xs, CL (g a) c) (hf : ∀ bs, CL (f bs) (k - xs.size * c)) (hk : xs.size * c ≤ k) :
    CL (xs.mapM g >>= f) k :=
  calcCL.bind_le (calcCL.mapM_mem hg) hf hk

theorem cl_bind_mapM0 {α β γ : Type} {g : α → LogV β} {xs : Array α} {f : Array β → LogV γ} {k : ℕ}
    (hg : ∀ a, CL (g a) 0) (hf : ∀ bs, CL (f bs) k) : CL (xs.mapM g >>= f) k :=
  calcCL.bind0 (calcCL.mono (calcCL.mapM_const hg xs) (Nat.le_of_eq (Nat.mul_zero _))) hf

theorem cl_bind_mapIdxM {α β γ : Type} {g : ℕ → α → LogV β} {xs : Array α} {f : Array β → LogV γ} {c k : ℕ}
    (hg : ∀ i a, CL (g i a) c) (hf : ∀ bs, CL (f bs) (k - xs.size * c)) (hk : xs.size * c ≤ k) :
    CL (xs.mapIdxM g >>= f) k :=
  calcCL.bind_le (cl_mapIdxM hg xs) hf hk

theorem cl_bind_forIn {α β γ : Type} {xs : Array α} {b : β} {body : α → β → LogV (ForInStep β)}
    {f : β → LogV γ} {c k : ℕ} (hb : ∀ a b, CL (body a b) c) (hf : ∀ b', CL (f b') (k - xs.size * c))
    (hk : xs.size * c ≤ k) : CL (forIn xs b body >>= f) k :=
  calcCL.bind_le (calcCL.forIn hb xs b) hf hk

theorem cl_bind_forIn0 {α β γ : Type} {xs : Array α} {b : β} {body : α → β → LogV (ForInStep β)}
    {f : β → LogV γ} {k : ℕ} (hb : ∀ a b, CL (body a b) 0) (hf : ∀ b', CL (f b') k) :
    CL (forIn xs b body >>= f) k :=
  calcCL.bind0 (calcCL.mono (calcCL.forIn hb xs b) (Nat.le_of_eq (Nat.mul_zero _))) hf

open Lean Elab Tactic Meta in
/-- Fails unless the goal bounds a bind whose first program's head is the constant `c`: a failed `refine` against
the wrong bind unfolds both programs and costs more than the step. -/
elab "cl_bind_of " c:ident : tactic => do
  let n ← realizeGlobalConstNoOverload c
  let tgt := (← instantiateMVars (← getMainTarget)).cleanupAnnotations
  unless tgt.getAppNumArgs == 3 do throwError "cl_bind_of: not a CL or Ret goal"
  let x := (tgt.getArg! 1).headBeta
  unless x.isAppOfArity ``Bind.bind 6 && (x.getArg! 4).getAppFn.isConstOf n do
    throwError "cl_bind_of: not a bind of {n}"

open Lean Elab Tactic Meta in
/-- Fails unless the goal bounds a program whose head is the constant `c`. -/
elab "cl_head " c:ident : tactic => do
  let n ← realizeGlobalConstNoOverload c
  let tgt := (← instantiateMVars (← getMainTarget)).cleanupAnnotations
  unless tgt.getAppNumArgs == 3 && (tgt.getArg! 1).headBeta.getAppFn.isConstOf n do
    throwError "cl_head: not a program headed by {n}"

/-- One step of the decoder's walk. -/
macro "dec_step" : tactic => `(tactic| first
  | (cl_bind_of MonadExcept.throw; with_reducible exact cl_throw_bind _ _ _)
  | (cl_bind_of SessionSpec.dgWith; with_reducible refine cl_bind_dg (fun _ => ?_) ?_)
  | (cl_bind_of ite; with_reducible refine cl_ite_bind ?_ ?_)
  | (cl_bind_of liftM; with_reducible refine cl_bind_lift0 fun _ => ?_)
  | (cl_bind_of Array.mapM; with_reducible refine cl_bind_mapM0 (fun _ => ?_) fun _ => ?_)
  | (cl_bind_of ForIn.forIn; with_reducible refine cl_bind_forIn0 (fun _ _ => ?_) fun _ => ?_)
  | (cl_head Bind.bind; with_reducible refine calcCL.bind0 ?_ fun _ => ?_)
  | (cl_head Pure.pure; with_reducible exact calcCL.pure' _ _)
  | (cl_head MonadExcept.throw; with_reducible exact calcCL.mono (cl_throw _) (Nat.zero_le _))
  | (cl_head liftM; with_reducible exact calcCL.mono (cl_liftM _) (Nat.zero_le _))
  | (cl_head ite; with_reducible refine calcCL.ite_k ?_ ?_)
  | split
  | dsimp only)

/-- **The decoder's SHA-512 calls on every run**: per table one for `root_B` and at most one for its publics, one for
`root_F`, at most two per round of each of the record's streams, one per rep stream for S13 and one for the link, `R`
bounding the record's streams' rounds. -/
theorem cl_decodeJ (h : ByteArray → ByteArray) (keep : Bool → Bool) (spec : SessionSpec) (j : Lean.Json)
    (sj : Array Lean.Json) (R : ℕ) (hsj : Record.arr j "streams" = .ok sj)
    (hR : ∀ s ∈ sj, ∀ rj, Record.arr s "rounds" = .ok rj → rj.size ≤ R) :
    CL (Record.decodeJWithK keep (Hashes.logged h) spec j)
      (2 * spec.tables.size + 2 + sj.size * (2 * R) + spec.repStreams.size) := by
  unfold Record.decodeJWithK
  rw [hsj]
  repeat' (first
    | (cl_bind_of liftM; with_reducible refine cl_liftM_ok_bind ?_)
    | (cl_bind_of liftM; with_reducible refine cl_rounds (hR _ (by assumption)) fun _ _ => ?_)
    | (cl_bind_of Array.mapM; with_reducible refine cl_bind_mapM (xs := sj) (c := 2 * R) (fun _ _ => ?_) (fun _ => ?_) ?_)
    | (cl_bind_of Array.mapM; with_reducible refine cl_bind_mapM (xs := spec.tables) (c := 1) (fun _ _ => ?_) (fun _ => ?_) ?_)
    | (cl_bind_of Array.mapIdxM; with_reducible refine cl_bind_mapIdxM (c := 2) (fun _ _ => ?_) (fun _ => ?_) ?_)
    | (cl_bind_of ForIn.forIn; with_reducible refine cl_bind_forIn (xs := spec.repStreams) (c := 1) (fun _ _ => ?_) (fun _ => ?_) ?_)
    | dec_step)
  all_goals omega

/-! ## What the decoder returns -/

/-- Some run of `x` returns `a`. -/
def Ok {α : Type} (x : LogV α) (a : α) : Prop := ∃ l l', x.go l = (.ok a, l')

/-- Every value a run of `x` returns satisfies `Q`. -/
def Ret {α : Type} (x : LogV α) (Q : α → Prop) : Prop := ∀ a, Ok x a → Q a

theorem ret_bind {α β : Type} {x : LogV α} {f : α → LogV β} {Q : β → Prop} (hf : ∀ a, Ok x a → Ret (f a) Q) :
    Ret (x >>= f) Q := by
  rintro b ⟨l, l', e⟩
  rw [go_bind] at e
  rcases hx : x.go l with ⟨r, l₁⟩
  rw [hx] at e
  cases r with
  | error _ => cases e
  | ok a => exact hf a ⟨l, l₁, hx⟩ b ⟨l₁, l', e⟩

theorem ret_lift_bind {α β : Type} {v : Flock.V α} {f : α → LogV β} {Q : β → Prop}
    (hf : ∀ a, v = .ok a → Ret (f a) Q) : Ret (liftM v >>= f) Q :=
  ret_bind fun a ⟨_, _, e⟩ => hf a (by rw [go_liftM] at e; exact (Prod.mk.inj e).1)

theorem ret_lift_ok_bind {α β : Type} {a : α} {f : α → LogV β} {Q : β → Prop} (hf : Ret (f a) Q) :
    Ret (liftM (Except.ok a : Flock.V α) >>= f) Q :=
  ret_lift_bind fun _ e => (Except.ok.inj e) ▸ hf

theorem ret_pure {α : Type} {a : α} {Q : α → Prop} (h : Q a) : Ret (pure a : LogV α) Q := by
  rintro b ⟨l, l', e⟩
  rw [go_pure] at e
  cases e
  exact h

theorem ret_throw {α : Type} (e : String) (Q : α → Prop) : Ret (throw e : LogV α) Q := by
  rintro b ⟨l, l', e'⟩
  rw [go_throw] at e'
  cases e'

theorem ret_throw_bind {α β : Type} (e : String) (f : α → LogV β) (Q : β → Prop) : Ret (throw e >>= f) Q :=
  ret_bind fun _ ⟨_, _, e'⟩ => by rw [go_throw] at e'; cases e'

theorem ret_ite {α : Type} {c : Prop} [Decidable c] {t e : LogV α} {Q : α → Prop} (ht : Ret t Q) (he : Ret e Q) :
    Ret (if c then t else e) Q := by
  by_cases hc : c
  · rw [if_pos hc]; exact ht
  · rw [if_neg hc]; exact he

/-- One step of a walk to what a program returns: a bind keeps its first program's run (`Ok`), a lifted step its
value's equation. -/
macro "ret_step" : tactic => `(tactic| first
  | (cl_bind_of MonadExcept.throw; with_reducible exact ret_throw_bind _ _ _)
  | (cl_bind_of liftM; with_reducible refine ret_lift_ok_bind ?_)
  | (cl_bind_of liftM; with_reducible refine ret_lift_bind fun _ _ => ?_)
  | (cl_head Bind.bind; with_reducible refine ret_bind fun _ _ => ?_)
  | (cl_head MonadExcept.throw; with_reducible exact ret_throw _ _)
  | (cl_head ite; with_reducible refine ret_ite ?_ ?_)
  | (cl_head Pure.pure; with_reducible refine ret_pure ?_)
  | split
  | dsimp only)

theorem list_mapM_ret {α β : Type} {g : α → LogV β} {p : β → Prop} :
    ∀ (xs : List α), (∀ a ∈ xs, Ret (g a) p) → Ret (xs.mapM g) (∀ b ∈ ·, p b)
  | [], _ => by rw [List.mapM_nil]; exact ret_pure fun b hb => by cases hb
  | a :: xs, hg => by
    rw [List.mapM_cons]
    refine ret_bind fun b hb => ret_bind fun bs hbs => ret_pure fun y hy => ?_
    rcases List.mem_cons.1 hy with rfl | hy
    · exact hg a (List.mem_cons_self ..) _ hb
    · exact list_mapM_ret xs (fun y hy => hg y (List.mem_cons_of_mem _ hy)) bs hbs _ hy

theorem mapM_ret {α β : Type} {g : α → LogV β} {p : β → Prop} {xs : Array α} (hg : ∀ a ∈ xs, Ret (g a) p)
    {bs : Array β} (h : Ok (xs.mapM g) bs) : ∀ b ∈ bs, p b := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind] at h
  refine ret_bind (fun ys hys => ret_pure (Q := (∀ b ∈ ·, p b)) fun b hb => ?_) bs h
  exact list_mapM_ret xs.toList (fun a ha => hg a (by simpa using ha)) ys hys b (by simpa using hb)

theorem mapIdxM_go_ok_length {α β : Type} {g : ℕ → α → LogV β} :
    ∀ (xs : List α) (acc : Array β) (ys : List β), Ok (List.mapIdxM.go g xs acc) ys → ys.length = acc.size + xs.length
  | [], acc, ys, ⟨l, l', e⟩ => by
    simp only [List.mapIdxM.go] at e
    rw [go_pure] at e
    cases e
    simp
  | a :: xs, acc, ys, ⟨l, l', e⟩ => by
    simp only [List.mapIdxM.go] at e
    rw [go_bind] at e
    rcases h1 : (g acc.size a).go l with ⟨r, l₁⟩
    rw [h1] at e
    cases r with
    | error _ => cases e
    | ok b =>
      have := mapIdxM_go_ok_length xs (acc.push b) ys ⟨l₁, l', e⟩
      simp only [Array.size_push, List.length_cons] at this ⊢
      omega

theorem mapIdxM_ok_size {α β : Type} {g : ℕ → α → LogV β} {xs : Array α} {ys : Array β} (h : Ok (xs.mapIdxM g) ys) :
    ys.size = xs.size := by
  obtain ⟨l, l', e⟩ := h
  have e' : (Array.toList <$> xs.mapIdxM g).go l = (.ok ys.toList, l') := by
    rw [map_eq_pure_bind, go_bind, e]
    rfl
  rw [Array.toList_mapIdxM] at e'
  have := mapIdxM_go_ok_length xs.toList #[] ys.toList ⟨l, l', e'⟩
  simpa using this

/-- **Every session the decoder returns has at most `R` rounds a stream**, `R` bounding the record's streams' rounds. -/
theorem ret_decodeJ_rounds (h : ByteArray → ByteArray) (keep : Bool → Bool) (spec : SessionSpec) (j : Lean.Json)
    (sj : Array Lean.Json) (R : ℕ) (hsj : Record.arr j "streams" = .ok sj)
    (hR : ∀ s ∈ sj, ∀ rj, Record.arr s "rounds" = .ok rj → rj.size ≤ R) :
    Ret (Record.decodeJWithK keep (Hashes.logged h) spec j) fun sess => ∀ s ∈ sess.streams, s.rounds.size ≤ R := by
  unfold Record.decodeJWithK
  rw [hsj]
  repeat' ret_step
  all_goals
    refine mapM_ret (fun s hs => ?_) ‹Ok (Array.mapM _ sj) _›
    repeat' ret_step
    all_goals
      rw [mapIdxM_ok_size ‹Ok (Array.mapIdxM _ _) _›]
      exact hR s hs _ ‹Record.arr s "rounds" = _›

/-! ## The schedules and the verdict -/

theorem fast100_none_of_gt (m : ℕ) (h : 35 < m) : fast100 m = none := by
  have : (m < 22 || m > 35) = true := by rw [Bool.or_eq_true, decide_eq_true_iff, decide_eq_true_iff]; omega
  unfold fast100
  rw [ite_eq_left_of_eq_true _ _ (eq_true this)]
  rfl

/-- **Every fast100 schedule's Ligerito makes at most `2^16` calls** (64,568 at `m = 35`, the most). -/
theorem ligC_fast100 {m : ℕ} {sch : Schedule} (hs : fast100 m = some sch) : ligC sch ≤ 2 ^ 16 := by
  have key : ∀ m < 36, (match fast100 m with
      | some s => decide (ligC s ≤ 2 ^ 16)
      | none => true) = true := by
    decide +kernel
  by_cases hm : m < 36
  · have h := key m hm
    rw [hs] at h
    simpa using h
  · rw [fast100_none_of_gt m (by omega)] at hs
    cases hs

/-- **A verdict's SHA-512 calls on every run, at a record of known shape**: the record's streams `sj`, each of at most
`R` rounds, whether the verdict accepts or throws. -/
theorem cl_verifyJ_rec (h : ByteArray → ByteArray) (keep : Bool → Bool) (spec : SessionSpec) (tabs : Array Setup)
    (j : Lean.Json) (proofs : Array ByteArray) (sj : Array Lean.Json) (R : ℕ) (hsj : Record.arr j "streams" = .ok sj)
    (hR : ∀ s ∈ sj, ∀ rj, Record.arr s "rounds" = .ok rj → rj.size ≤ R) :
    CL (verifyJWithMK keep (Hashes.logged h) spec tabs j proofs)
      (2 * spec.tables.size + 2 + sj.size * (2 * R) + spec.repStreams.size +
        (spec.repStreams.size + tabs.size * (2 * (2 ^ 16 + R)))) :=
  cl_verifyJ h keep spec tabs j proofs _ _ R (cl_decodeJ h keep spec j sj R hsj hR)
    (fun l sess l' e => ret_decodeJ_rounds h keep spec j sj R hsj hR sess ⟨l, l', e⟩)
    (fun _ _ hs => ligC_fast100 hs)

end FlockDeploy.KS
