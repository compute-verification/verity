module

public import Proofs.Properties.Flock.Nets.Cert

/-!
# A compression's run from its chunks

A run reads only the state's next bit and puts its rows in front of the state's (`Ext`), so what a chunk does from a state
with no rows it does from any (`Seg.ofExt`), and chunks compose (`Seg.bind'`). `seg_compress`: a compression whose
certificate's chunks all check runs from its first bit to `fin`, adding rows that each read at most 128 bits.
-/

@[expose] public section

open Flock Flock.HmNets FlockSoundness.Discharge.Hm

namespace FlockDeploy.Concrete.Nets

/-- `m` reads only the state's next bit, and puts its rows in front of the state's. -/
def Ext {α : Type} (m : M α) : Prop :=
  ∀ s : St, m.run s = ((m.run ⟨s.next, []⟩).1, ⟨(m.run ⟨s.next, []⟩).2.next, (m.run ⟨s.next, []⟩).2.rows ++ s.rows⟩)

namespace Ext

variable {α β : Type}

theorem pure' (a : α) : Ext (pure a : M α) := fun _ => rfl

theorem bind' {m : M α} {f : α → M β} (hm : Ext m) (hf : ∀ a, Ext (f a)) : Ext (m >>= f) := by
  intro s
  rw [run_bind, run_bind, hm s]
  generalize m.run ⟨s.next, []⟩ = p
  obtain ⟨a, ⟨n, rs⟩⟩ := p
  dsimp only
  rw [hf a ⟨n, rs ++ s.rows⟩, hf a ⟨n, rs⟩]
  simp only [List.append_assoc]

theorem ite' {c : Prop} [Decidable c] {m₁ m₂ : M α} (h₁ : Ext m₁) (h₂ : Ext m₂) : Ext (if c then m₁ else m₂) := by
  split
  · exact h₁
  · exact h₂

theorem mapM' (f : β → M α) (hf : ∀ b, Ext (f b)) : ∀ l : List β, Ext (l.mapM f)
  | [] => by rw [List.mapM_nil]; exact pure' _
  | b :: l => by
      rw [List.mapM_cons]
      exact bind' (hf b) fun _ => bind' (mapM' f hf l) fun _ => pure' _

end Ext

theorem ext_fresh (a b : Lin) : Ext (fresh a b) := fun _ => rfl

theorem ext_and (a b : Lin) : Ext (Flock.HmNets.and a b) := by
  unfold Flock.HmNets.and
  exact Ext.ite' (Ext.pure' _) (Ext.ite' (Ext.pure' _) (Ext.ite' (Ext.pure' _) (ext_fresh a b)))

theorem ext_maj (x y w : Lin) : Ext (maj x y w) := by
  unfold maj
  dsimp only
  exact Ext.ite' (Ext.ite' (Ext.pure' _) (Ext.ite' (Ext.pure' _) (Ext.pure' _)))
    (Ext.bind' (ext_and _ _) fun _ => Ext.pure' _)

theorem ext_commit (f : Lin) : Ext (commit f) := by
  unfold commit
  exact Ext.ite' (Ext.pure' _) (ext_fresh _ _)

theorem ext_chW (e f g : Word) : Ext (chW e f g) :=
  Ext.mapM' _ (fun _ => Ext.bind' (ext_and _ _) fun _ => Ext.pure' _) _

theorem ext_majW (a b c : Word) : Ext (majW a b c) := Ext.mapM' _ (fun _ => ext_maj _ _ _) _

theorem ext_commitW (w : Word) : Ext (commitW w) := Ext.mapM' _ ext_commit _

theorem ext_carryCommit (i : Nat) (c : Lin) : Ext (carryCommit i c) := by
  unfold carryCommit
  exact Ext.ite' (ext_commit c) (Ext.pure' _)

theorem ext_add2Go : ∀ (a b : Word) (c : Lin) (i : Nat), Ext (add2Go a b c i)
  | x :: a, y :: b, c, i => by
      unfold add2Go
      exact Ext.ite' (Ext.pure' _) (Ext.bind' (ext_maj x y c) fun c₁ =>
        Ext.bind' (ext_carryCommit i c₁) fun c₂ =>
          Ext.bind' (ext_add2Go a b c₂ (i + 1)) fun _ => Ext.pure' _)
  | [], _, _, _ => by unfold add2Go; exact Ext.pure' _
  | _ :: _, [], _, _ => by unfold add2Go; exact Ext.pure' _

theorem ext_add2 (a b : Word) : Ext (add2 a b) := ext_add2Go a b _ 0

theorem ext_addN : ∀ (acc : Word) (xs : List Word), Ext (addN acc xs)
  | _, [] => Ext.pure' _
  | acc, x :: xs => Ext.bind' (ext_add2 acc x) fun acc' => ext_addN acc' xs

theorem ext_schedStep (w : List Word) : Ext (schedStep w) :=
  Ext.bind' (ext_addN _ _) fun _ => Ext.bind' (ext_commitW _) fun _ => Ext.pure' _

theorem ext_schedule : ∀ (n : Nat) (w : List Word), Ext (schedule n w)
  | 0, _ => Ext.pure' _
  | n + 1, w => Ext.bind' (ext_schedStep w) fun w' => ext_schedule n w'

theorem ext_roundW (k : BitVec 64) (wt : Word) (r : Regs Word) : Ext (roundW k wt r) :=
  Ext.bind' (ext_chW _ _ _) fun _ => Ext.bind' (ext_addN _ _) fun _ =>
    Ext.bind' (ext_majW _ _ _) fun _ => Ext.bind' (ext_add2 _ _) fun _ =>
      Ext.bind' (ext_add2 _ _) fun _ => Ext.bind' (ext_commitW _) fun _ =>
        Ext.bind' (ext_add2 _ _) fun _ => Ext.bind' (ext_commitW _) fun _ => Ext.pure' _

theorem ext_rounds (w : List Word) : ∀ (n t : Nat) (r : Regs Word), Ext (rounds w n t r)
  | 0, _, _ => Ext.pure' _
  | n + 1, t, r => Ext.bind' (ext_roundW _ _ r) fun r' => ext_rounds w n (t + 1) r'

theorem ext_ff (cv : List Word) (r : Regs Word) : Ext (ff cv r) := Ext.mapM' _ (fun _ => ext_add2 _ _) _

/-- Run from next bit `n`, whatever the rows before, `m` returns `a` and moves to `n'`, putting in front of the rows some
that each read at most `W` bits. -/
def Seg {α : Type} (W : Nat) (m : M α) (n : Nat) (a : α) (n' : Nat) : Prop :=
  ∀ rows, ∃ new : List Row, m.run ⟨n, rows⟩ = (a, ⟨n', new ++ rows⟩) ∧ ∀ r ∈ new, r.a.s.length + r.b.s.length ≤ W

namespace Seg

variable {α β : Type} {W : Nat}

theorem pure' (a : α) (n : Nat) : Seg W (pure a : M α) n a n := fun _ => ⟨[], rfl, by simp⟩

theorem bind' {m : M α} {f : α → M β} {n n₁ n₂ : Nat} {a : α} {b : β} (hm : Seg W m n a n₁) (hf : Seg W (f a) n₁ b n₂) :
    Seg W (m >>= f) n b n₂ := by
  intro rows
  obtain ⟨new₁, h₁, w₁⟩ := hm rows
  obtain ⟨new₂, h₂, w₂⟩ := hf (new₁ ++ rows)
  refine ⟨new₂ ++ new₁, ?_, fun r hr => ?_⟩
  · rw [run_bind, h₁]
    simpa only [List.append_assoc] using h₂
  · rcases List.mem_append.1 hr with hr | hr
    · exact w₂ r hr
    · exact w₁ r hr

theorem ofExt {m : M α} {n n' : Nat} {a : α} (hm : Ext m) (h₁ : (m.run ⟨n, []⟩).1 = a)
    (h₂ : (m.run ⟨n, []⟩).2.next = n') (hw : widthOk W (m.run ⟨n, []⟩).2.rows = true) : Seg W m n a n' := by
  intro rows
  refine ⟨(m.run ⟨n, []⟩).2.rows, ?_, fun r hr => ?_⟩
  · rw [hm ⟨n, rows⟩]
    dsimp only
    rw [h₁, h₂]
  · simp only [widthOk, List.all_eq_true, Nat.ble_eq] at hw
    exact hw r hr

end Seg

theorem schedule_add : ∀ (a b : Nat) (w : List Word), schedule (a + b) w = schedule a w >>= schedule b
  | 0, b, w => by rw [Nat.zero_add]; exact (pure_bind w (schedule b)).symm
  | a + 1, b, w => by
    rw [Nat.add_right_comm]
    show (schedStep w >>= fun w' => schedule (a + b) w') = (schedStep w >>= fun w' => schedule a w') >>= schedule b
    rw [bind_assoc]
    congr
    funext w'
    exact schedule_add a b w'

theorem rounds_add (w : List Word) : ∀ (a b t : Nat) (r : Regs Word),
    rounds w (a + b) t r = rounds w a t r >>= fun r' => rounds w b (t + a) r'
  | 0, b, t, r => by rw [Nat.zero_add, Nat.add_zero]; exact (pure_bind r _).symm
  | a + 1, b, t, r => by
    rw [Nat.add_right_comm]
    show (roundW (kAt t) (w.getD t []) r >>= fun r' => rounds w (a + b) (t + 1) r') =
      (roundW (kAt t) (w.getD t []) r >>= fun r' => rounds w a (t + 1) r') >>= fun r' => rounds w b (t + (a + 1)) r'
    rw [bind_assoc]
    congr
    funext r'
    rw [rounds_add w a b (t + 1) r', show t + (a + 1) = t + 1 + a by omega]

theorem Regs.toList_inj {α : Type} {r r' : Regs α} (h : r.toList = r'.toList) : r = r' := by
  cases r
  cases r'
  simp only [Regs.toList, List.cons.injEq] at h
  obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, -⟩ := h
  rfl

theorem seg_schedOk {msg : List Word} {c : Cert} {j : Nat} (h : schedOk msg c j = true) :
    Seg 128 (schedule 16 (schedAt msg c j)) (schedNext c j) (schedAt msg c (j + 1)) (schedNext c (j + 1)) := by
  simp only [schedOk, scheduleK_eq, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  exact Seg.ofExt (ext_schedule _ _) h.1.1 h.1.2 h.2

theorem seg_roundOk {cv msg : List Word} {c : Cert} {j : Nat} (h : roundOk cv msg c j = true) :
    Seg 128 (rounds (schedAt msg c 4) 8 (8 * j) (regsAt cv c (8 * j))) (roundNext c (8 * j)) (regsAt cv c (8 * j + 8))
      (roundNext c (8 * j + 8)) := by
  simp only [roundOk, roundsK_eq, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  exact Seg.ofExt (ext_rounds _ _ _ _) (Regs.toList_inj h.1.1) h.1.2 h.2

theorem forall_lt4 {p : Nat → Prop} (h0 : p 0) (h1 : p 1) (h2 : p 2) (h3 : p 3) : ∀ j < 4, p j
  | 0, _ => h0 | 1, _ => h1 | 2, _ => h2 | 3, _ => h3 | _ + 4, h => absurd h (by omega)

theorem forall_lt10 {p : Nat → Prop} (h0 : p 0) (h1 : p 1) (h2 : p 2) (h3 : p 3) (h4 : p 4) (h5 : p 5) (h6 : p 6)
    (h7 : p 7) (h8 : p 8) (h9 : p 9) : ∀ j < 10, p j
  | 0, _ => h0 | 1, _ => h1 | 2, _ => h2 | 3, _ => h3 | 4, _ => h4 | 5, _ => h5 | 6, _ => h6 | 7, _ => h7 | 8, _ => h8
  | 9, _ => h9 | _ + 10, h => absurd h (by omega)

/-- A certified compression's output. -/
def ffOut (cv : List Word) (c : Cert) : List Word := ((ff cv (regsAt cv c 80)).run ⟨roundNext c 80, []⟩).1

theorem seg_sched (msg : List Word) (c : Cert) (hs : ∀ j < 4, schedOk msg c j = true) :
    ∀ j ≤ 4, Seg 128 (schedule (16 * j) msg) c.n0 (schedAt msg c j) (schedNext c j)
  | 0, _ => by
    have e : schedAt msg c 0 = msg := by simp [schedAt]
    rw [e]
    exact Seg.pure' _ _
  | j + 1, hj => by
    rw [show 16 * (j + 1) = 16 * j + 16 by omega, schedule_add]
    exact Seg.bind' (seg_sched msg c hs j (by omega)) (seg_schedOk (hs j (by omega)))

theorem seg_rounds (cv msg : List Word) (c : Cert) (hr : ∀ j < 10, roundOk cv msg c j = true) :
    ∀ j ≤ 10, Seg 128 (rounds (schedAt msg c 4) (8 * j) 0 (Regs.ofList cv)) (schedNext c 4) (regsAt cv c (8 * j))
      (roundNext c (8 * j))
  | 0, _ => Seg.pure' _ _
  | j + 1, hj => by
    rw [show 8 * (j + 1) = 8 * j + 8 by omega, rounds_add, Nat.zero_add]
    exact Seg.bind' (seg_rounds cv msg c hr j (by omega)) (seg_roundOk (hr j (by omega)))

/-- **A certified compression**: when every chunk of `c` checks, `compress cv msg` runs from bit `c.n0` to `c.fin` with
output `ffOut cv c`, adding rows that each read at most 128 bits. -/
theorem seg_compress (cv msg : List Word) (c : Cert) (hs : ∀ j < 4, schedOk msg c j = true)
    (hr : ∀ j < 10, roundOk cv msg c j = true) (hf : ffOk cv c = true) :
    Seg 128 (compress cv msg) c.n0 (ffOut cv c) c.fin := by
  simp only [ffOk, ffK_eq, Bool.and_eq_true, beq_iff_eq] at hf
  exact Seg.bind' (seg_sched msg c hs 4 (by omega)) (Seg.bind' (seg_rounds cv msg c hr 10 (by omega))
    (Seg.ofExt (ext_ff _ _) rfl hf.1.1 hf.1.2))

/-- `ffOut` as the kernel runs it. -/
noncomputable def ffOutK (cv : List Word) (c : Cert) : List Word := ((ffK cv (regsAt cv c 80)).run ⟨roundNext c 80, []⟩).1

theorem ffOutK_eq (cv : List Word) (c : Cert) : ffOutK cv c = ffOut cv c := by
  simp only [ffOutK, ffOut, ffK_eq]

/-- A certified compression's output bits each read at most 16 bits. -/
theorem ffOut_width {cv : List Word} {c : Cert} (hf : ffOk cv c = true) : ∀ w ∈ ffOut cv c, ∀ f ∈ w, f.s.length ≤ 16 := by
  simp only [ffOk, ffK_eq, Bool.and_eq_true, List.all_eq_true, Nat.ble_eq] at hf
  exact hf.2

end FlockDeploy.Concrete.Nets
