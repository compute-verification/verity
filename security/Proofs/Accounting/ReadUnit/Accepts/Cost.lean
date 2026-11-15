module

public import Flock.ReadNet.Template
public import Proofs.Flock.Soundness.Discharge.Hm.Fresh

/-!
# How many variables a builder program adds

`Cost k m Q`: from any state, `m` advances the next variable by at most `k`, and its result satisfies `Q`. It composes
through `bind`, `if` and `mapM`, so a bound on a generator's rows follows its code. SHA-512's compression adds at most
`COMPRESS` variables on words of at most 64 bits (`cost_compress`), and `shaW n` at most `n` times that (`cost_shaW`).
-/

@[expose] public section

namespace Accounting.ReadUnit.Accepts

open Flock.HmNets hiding and

/-- From any state, `m` advances the next variable by at most `k`, and its result satisfies `Q`. -/
def Cost {α : Type} (k : ℕ) (m : M α) (Q : α → Prop) : Prop :=
  ∀ s : St, (m.run s).2.next ≤ s.next + k ∧ Q (m.run s).1

namespace Cost

variable {α β : Type}

theorem pure' {a : α} {Q : α → Prop} (k : ℕ) (h : Q a) : Cost k (pure a : M α) Q :=
  fun s => ⟨Nat.le_add_right s.next k, h⟩

theorem bind' {k₁ k₂ : ℕ} {m : M α} {f : α → M β} {P : α → Prop} {Q : β → Prop} (hm : Cost k₁ m P)
    (hf : ∀ a, P a → Cost k₂ (f a) Q) : Cost (k₁ + k₂) (m >>= f) Q := by
  intro s
  obtain ⟨h1, hP⟩ := hm s
  obtain ⟨h2, hQ⟩ := hf _ hP (m.run s).2
  have e : (m >>= f).run s = (f (m.run s).1).run (m.run s).2 := rfl
  rw [e]
  exact ⟨by omega, hQ⟩

theorem mono {k k' : ℕ} {m : M α} {P Q : α → Prop} (hm : Cost k m P) (hk : k ≤ k') (h : ∀ a, P a → Q a) :
    Cost k' m Q :=
  fun s => ⟨(hm s).1.trans (by omega), h _ (hm s).2⟩

theorem ite' {c : Prop} [Decidable c] {k : ℕ} {m₁ m₂ : M α} {Q : α → Prop} (h₁ : c → Cost k m₁ Q)
    (h₂ : ¬ c → Cost k m₂ Q) : Cost k (if c then m₁ else m₂) Q := by
  split_ifs with hc
  · exact h₁ hc
  · exact h₂ hc

theorem mapM' {k : ℕ} (f : β → M α) {P : β → Prop} {Q : α → Prop} (hf : ∀ b, P b → Cost k (f b) Q) :
    ∀ l : List β, (∀ b ∈ l, P b) →
      Cost (l.length * k) (l.mapM f) fun r => r.length = l.length ∧ ∀ x ∈ r, Q x
  | [], _ => by
      rw [List.mapM_nil]
      exact pure' _ ⟨rfl, by simp⟩
  | b :: l, hl => by
      rw [List.mapM_cons]
      have hb := hf b (hl b List.mem_cons_self)
      have hrest := mapM' f hf l fun b' hb' => hl b' (List.mem_cons_of_mem _ hb')
      refine mono (bind' hb fun a ha => bind' hrest fun r hr => pure' 0 (a := a :: r) ?_) ?_ fun _ h => h
      · refine ⟨by simp [hr.1], ?_⟩
        intro x hx
        rcases List.mem_cons.1 hx with rfl | hx
        · exact ha
        · exact hr.2 x hx
      · rw [List.length_cons, Nat.add_mul, Nat.one_mul]
        omega

end Cost

/-! ## The builder's gadgets -/

theorem cost_fresh (a b : Lin) : Cost 1 (fresh a b) fun _ => True := fun _ => ⟨Nat.le_refl _, trivial⟩

theorem cost_and (a b : Lin) : Cost 1 (Flock.HmNets.and a b) fun _ => True := by
  unfold Flock.HmNets.and
  exact Cost.ite' (fun _ => Cost.pure' _ trivial) fun _ => Cost.ite' (fun _ => Cost.pure' _ trivial) fun _ =>
    Cost.ite' (fun _ => Cost.pure' _ trivial) fun _ => cost_fresh a b

theorem cost_maj (x y w : Lin) : Cost 1 (maj x y w) fun _ => True := by
  unfold maj
  dsimp only
  exact Cost.ite' (fun _ => Cost.ite' (fun _ => Cost.pure' _ trivial) fun _ =>
      Cost.ite' (fun _ => Cost.pure' _ trivial) fun _ => Cost.pure' _ trivial)
    fun _ => Cost.mono (Cost.bind' (cost_and _ _) fun _ _ => Cost.pure' 0 (Q := fun _ => True) trivial) le_rfl fun _ h => h

theorem cost_commit (f : Lin) : Cost 1 (commit f) fun _ => True := by
  unfold commit
  exact Cost.ite' (fun _ => Cost.pure' _ trivial) fun _ => cost_fresh _ _

theorem cost_carryCommit (i : ℕ) (c : Lin) : Cost 1 (carryCommit i c) fun _ => True := by
  unfold carryCommit
  exact Cost.ite' (fun _ => cost_commit c) fun _ => Cost.pure' _ trivial

/-! ## Words -/

/-- Every word at most 64 bits. -/
def W64 (l : List Word) : Prop := ∀ x ∈ l, x.length ≤ 64

/-- At most eight words, each at most 64 bits: a chaining value or a digest. -/
def W8 (l : List Word) : Prop := l.length ≤ 8 ∧ W64 l

theorem sum_map_le {γ : Type} (l : List γ) (f : γ → ℕ) (c : ℕ) (h : ∀ x ∈ l, f x ≤ c) :
    (l.map f).sum ≤ l.length * c := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.add_mul, Nat.one_mul]
    have h1 := h a List.mem_cons_self
    have h2 := ih fun x hx => h x (List.mem_cons_of_mem _ hx)
    omega

theorem length_flatten_le {l : List Word} {n c : ℕ} (hl : l.length ≤ n) (h : ∀ x ∈ l, x.length ≤ c) :
    l.flatten.length ≤ n * c := by
  rw [List.length_flatten]
  exact (sum_map_le l List.length c h).trans (Nat.mul_le_mul_right c hl)

theorem flatten_le_of_W8 {l : List Word} (h : W8 l) : l.flatten.length ≤ 512 :=
  length_flatten_le (n := 8) (c := 64) h.1 h.2

theorem cost_chW (e f g : Word) : Cost e.length (chW e f g) fun r => r.length ≤ e.length :=
  Cost.mono (Cost.mapM' _ (P := fun _ => True) (k := 1)
      (fun _ _ => Cost.mono (Cost.bind' (cost_and _ _) fun _ _ => Cost.pure' 0 (Q := fun _ => True) trivial) le_rfl fun _ h => h) _
      fun _ _ => trivial)
    (by simp only [List.length_zip, Nat.mul_one]; omega) fun r h => by
      rw [h.1]
      simp only [List.length_zip]
      omega

theorem cost_majW (a b c : Word) : Cost a.length (majW a b c) fun r => r.length ≤ a.length :=
  Cost.mono (Cost.mapM' _ (P := fun _ => True) (k := 1) (fun _ _ => cost_maj _ _ _) _ fun _ _ => trivial)
    (by simp only [List.length_zip, Nat.mul_one]; omega) fun r h => by
      rw [h.1]
      simp only [List.length_zip]
      omega

theorem cost_commitW (w : Word) : Cost w.length (commitW w) fun r => r.length = w.length :=
  Cost.mono (Cost.mapM' _ (P := fun _ => True) (k := 1) (fun _ _ => cost_commit _) _ fun _ _ => trivial)
    (by omega) fun _ h => h.1

theorem cost_add2Go : ∀ (a b : Word) (c : Lin) (i : ℕ),
    Cost (2 * a.length) (add2Go a b c i) fun r => r.length ≤ a.length
  | x :: a, y :: b, c, i => by
      unfold add2Go
      refine Cost.ite' (fun _ => Cost.pure' _ (by simp)) fun _ => ?_
      refine Cost.mono (Cost.bind' (cost_maj x y c) fun c₁ _ => Cost.bind' (cost_carryCommit i c₁) fun c₂ _ =>
        Cost.bind' (cost_add2Go a b c₂ (i + 1)) fun r hr =>
          Cost.pure' 0 (a := (x ^^^ y ^^^ c) :: r) (Q := fun r => r.length ≤ (x :: a).length) ?_) ?_ fun _ h => h
      · simp only [List.length_cons]
        omega
      · simp only [List.length_cons]
        omega
  | [], _, _, _ => by
      unfold add2Go
      exact Cost.pure' _ (by simp)
  | _ :: _, [], _, _ => by
      unfold add2Go
      exact Cost.pure' _ (by simp)

theorem cost_add2 (a b : Word) : Cost (2 * a.length) (add2 a b) fun r => r.length ≤ a.length := cost_add2Go a b _ 0

theorem cost_addN : ∀ (acc : Word) (xs : List Word), acc.length ≤ 64 →
    Cost (128 * xs.length) (addN acc xs) fun r => r.length ≤ 64
  | _, [], h => Cost.pure' _ h
  | acc, x :: xs, h => Cost.mono
      (Cost.bind' (Cost.mono (cost_add2 acc x) (k' := 128) (by omega) fun r hr => hr.trans h) fun acc' h' =>
        cost_addN acc' xs h')
      (by simp only [List.length_cons]; omega) fun _ h => h

theorem length_bsig0W_le (a : Word) : (bsig0W a).length ≤ a.length := by
  simp only [bsig0W, xor3, xorW, rot, List.length_zipWith, List.length_append, List.length_drop, List.length_take]
  omega

theorem cost_add2_64 (a b : Word) (ha : a.length ≤ 64) : Cost 128 (add2 a b) fun c => c.length ≤ 64 :=
  Cost.mono (cost_add2 a b) (by omega) fun c hc => (by omega : c.length ≤ 64)

theorem cost_commitW_64 (w : Word) (hw : w.length ≤ 64) : Cost 64 (commitW w) fun c => c.length ≤ 64 :=
  Cost.mono (cost_commitW w) hw fun c hc => (by omega : c.length ≤ 64)

/-! ## SHA-512's compression -/

theorem getD_le {w : List Word} (hw : W64 w) (i : ℕ) : (w.getD i []).length ≤ 64 := by
  rw [List.getD_eq_getElem?_getD]
  cases h : w[i]? with
  | none => exact Nat.zero_le _
  | some x => exact hw x (List.mem_of_getElem? h)

theorem getElem!_le {w : List Word} (hw : W64 w) (i : ℕ) : (w[i]!).length ≤ 64 := by
  rw [List.getElem!_eq_getElem?_getD]
  cases h : w[i]? with
  | none => exact Nat.zero_le _
  | some x => exact hw x (List.mem_of_getElem? h)

theorem cost_schedStep (w : List Word) (hw : W64 w) : Cost 448 (schedStep w) W64 :=
  Cost.mono (Cost.bind' (cost_addN _ [_, _, _] (getD_le hw _)) fun x hx =>
      Cost.bind' (cost_commitW_64 x hx) fun y hy =>
        Cost.pure' 0 (a := w ++ [y]) (Q := W64) fun z hz => by
          rcases List.mem_append.1 hz with hz | hz
          · exact hw z hz
          · rw [List.mem_singleton.1 hz]
            exact hy)
    (by simp) fun _ h => h

theorem cost_schedule : ∀ (n : ℕ) (w : List Word), W64 w → Cost (448 * n) (schedule n w) W64
  | 0, _, hw => Cost.pure' _ hw
  | n + 1, w, hw => Cost.mono (Cost.bind' (cost_schedStep w hw) fun w' hw' => cost_schedule n w' hw') (by omega)
      fun _ h => h

/-- SHA-512's eight working variables, each at most 64 bits. -/
def RegsOk (r : Regs Word) : Prop := ∀ x ∈ r.toList, x.length ≤ 64

theorem cost_roundW (k : BitVec 64) (wt : Word) (r : Regs Word) (hr : RegsOk r) : Cost 1152 (roundW k wt r) RegsOk := by
  have ha : r.a.length ≤ 64 := hr _ (by simp [Regs.toList])
  have hd : r.d.length ≤ 64 := hr _ (by simp [Regs.toList])
  have he : r.e.length ≤ 64 := hr _ (by simp [Regs.toList])
  have hh : r.h.length ≤ 64 := hr _ (by simp [Regs.toList])
  have hs := length_bsig0W_le r.a
  refine Cost.mono (Cost.bind' (Cost.mono (cost_chW r.e r.f r.g) he (Q := fun _ : Word => True) fun _ _ => trivial)
    fun ch _ => Cost.bind' (Cost.mono (cost_addN r.h [bsig1W r.e, ch, constW k, wt] hh) (k' := 512) (by simp)
      (Q := fun c : Word => c.length ≤ 64) fun _ h => h) fun t1 ht1 =>
    Cost.bind' (Cost.mono (cost_majW r.a r.b r.c) ha (Q := fun _ : Word => True) fun _ _ => trivial) fun mj _ =>
    Cost.bind' (cost_add2_64 _ mj (hs.trans ha)) fun t2 _ =>
    Cost.bind' (cost_add2_64 r.d t1 hd) fun e he' =>
    Cost.bind' (cost_commitW_64 e he') fun e' he'' =>
    Cost.bind' (cost_add2_64 t1 t2 ht1) fun a ha' =>
    Cost.bind' (cost_commitW_64 a ha') fun a' ha'' =>
    Cost.pure' 0 (a := (⟨a', r.a, r.b, r.c, e', r.e, r.f, r.g⟩ : Regs Word)) (Q := RegsOk) ?_) (by simp)
      fun _ h => h
  intro x hx
  simp only [Regs.toList, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact ha''
  · exact ha
  · exact hr _ (by simp [Regs.toList])
  · exact hr _ (by simp [Regs.toList])
  · exact he''
  · exact he
  · exact hr _ (by simp [Regs.toList])
  · exact hr _ (by simp [Regs.toList])

theorem cost_rounds (w : List Word) : ∀ (n t : ℕ) (r : Regs Word), RegsOk r → Cost (1152 * n) (rounds w n t r) RegsOk
  | 0, _, _, hr => Cost.pure' _ hr
  | n + 1, t, r, hr => Cost.mono (Cost.bind' (cost_roundW _ (w.getD t []) r hr) fun r' hr' => cost_rounds w n (t + 1) r' hr')
      (by omega) fun _ h => h

theorem regsOk_ofList {cv : List Word} (hcv : W64 cv) : RegsOk (Regs.ofList cv) := by
  intro x hx
  simp only [Regs.ofList, Regs.toList, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> exact getElem!_le hcv _

/-- The variables one SHA-512 compression adds, at most: the schedule's, the rounds' and the feed-forward's. -/
def COMPRESS : ℕ := 448 * 64 + 1152 * 80 + 1024

theorem cost_compress (cv msg : List Word) (hcv : W64 cv) (hmsg : W64 msg) : Cost COMPRESS (compress cv msg) W8 := by
  refine Cost.mono (Cost.bind' (cost_schedule 64 msg hmsg) fun w _ =>
    Cost.bind' (cost_rounds w 80 0 (Regs.ofList cv) (regsOk_ofList hcv)) fun r _ =>
    Cost.mono (Cost.mapM' (fun x : Word × Word => add2 x.1 x.2) (P := fun x => x.1.length ≤ 64) (k := 128)
        (fun x hx => Cost.mono (cost_add2 x.1 x.2) (by have h : x.1.length ≤ 64 := hx; omega) fun c hc => hc.trans hx) (cv.zip r.toList)
        fun x hx => hcv _ (List.of_mem_zip (show (x.1, x.2) ∈ cv.zip r.toList from hx)).1)
      (k' := 1024) ?_ ?_) ?_ fun _ h => h
  · simp only [List.length_zip, Regs.toList, List.length_cons, List.length_nil]
    omega
  · intro y hy
    refine ⟨?_, hy.2⟩
    rw [hy.1]
    simp only [List.length_zip, Regs.toList, List.length_cons, List.length_nil]
    omega
  · simp [COMPRESS]

theorem cost_shaW : ∀ (n : ℕ) (cv msg : List Word), W8 cv → W64 msg →
    Cost (n * COMPRESS) (Flock.ReadNet.shaW n cv msg) W8
  | 0, _, _, hcv, _ => Cost.pure' _ hcv
  | n + 1, cv, msg, hcv, hmsg => Cost.mono
      (Cost.bind' (cost_compress cv (msg.take 16) hcv.2 fun x hx => hmsg x (List.mem_of_mem_take hx)) fun st hst =>
        cost_shaW n st (msg.drop 16) hst fun x hx => hmsg x (List.mem_of_mem_drop hx))
      (by rw [Nat.add_mul, Nat.one_mul]; omega) fun _ h => h

end Accounting.ReadUnit.Accepts
