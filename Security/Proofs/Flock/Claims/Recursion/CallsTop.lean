import Proofs.Flock.Claims.Recursion.CallsRep

/-!
# A verdict's SHA-512 calls on every run

A rep's verification calls SHA-512 for its Ligerito queries (`ligC`) and for at most each recorded round of its stream,
the transcript's potential at the start (`cl_verifyRep`). The verdict at a parsed record adds the decoder's calls, S17's
proof digests and two reps per table (`cl_verifyJ`); a bind may assume what its first run returned (`cl_bind_q`), here
the decoded session's round counts and the fast100 schedules (`mapM_ok`).
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

end FlockDeploy.KS
