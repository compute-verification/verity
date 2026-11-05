/-
The verifier's builder is total: every row it adds reads only bits committed before it (`Scoped`), so forward evaluation
from any inputs satisfies every row (`fwd_holds`). A Hoare-style judgement (`Tot L m o`: from a scoped state whose `next`
bounds the forms `L`, `m` reaches a scoped state whose `next` bounds the forms `o` of its result) carries that through
the gadgets, the adder, the schedule and the rounds to both slot circuits (`tot_compression`, `tot_hm96Row`). The
converse of `laid_rows` (`laid_total`) then lays the values out: from any inputs `x`, `laidVal` satisfies every row of
the laid-out net, and both of the verifier's nets hold from any inputs (`sha_total`, `hm_total`). No SHA-512 value is
evaluated: the step from the builder's rows to the net's is the layout's, read off `layout_rows` and `layout_gaps`.
-/
module

public import Proofs.Flock.Soundness.Discharge.Hm.Nets

@[expose] public section

namespace FlockDeploy.Concrete.Total

open Flock
open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-- Form `f` reads only bits below `n`. -/
def Below (n : Nat) (f : Lin) : Prop := ∀ v ∈ f.s, v < n

/-- Every form of `l` reads only bits below `n`. -/
def BelowL (n : Nat) (l : List Lin) : Prop := ∀ f ∈ l, Below n f

theorem Below.mono {n n' : Nat} {f : Lin} (h : n ≤ n') (hf : Below n f) : Below n' f :=
  fun v hv => Nat.lt_of_lt_of_le (hf v hv) h

theorem BelowL.mono {n n' : Nat} {l : List Lin} (h : n ≤ n') (hl : BelowL n l) : BelowL n' l :=
  fun f hf => (hl f hf).mono h

theorem mem_symm : ∀ (a b : List Var) (v : Var), v ∈ HmNets.symm a b → v ∈ a ∨ v ∈ b
  | [], ys, v, h => by simp only [HmNets.symm] at h; exact Or.inr h
  | x :: xs, [], v, h => by simp only [HmNets.symm] at h; exact Or.inl h
  | x :: xs, y :: ys, v, h => by
    unfold HmNets.symm at h
    split at h
    · rcases List.mem_cons.1 h with rfl | h
      · exact Or.inl List.mem_cons_self
      · rcases mem_symm xs (y :: ys) v h with h | h
        · exact Or.inl (List.mem_cons_of_mem _ h)
        · exact Or.inr h
    · split at h
      · rcases List.mem_cons.1 h with rfl | h
        · exact Or.inr List.mem_cons_self
        · rcases mem_symm (x :: xs) ys v h with h | h
          · exact Or.inl h
          · exact Or.inr (List.mem_cons_of_mem _ h)
      · rcases mem_symm xs ys v h with h | h
        · exact Or.inl (List.mem_cons_of_mem _ h)
        · exact Or.inr (List.mem_cons_of_mem _ h)

theorem Below.xor {n : Nat} {f g : Lin} (hf : Below n f) (hg : Below n g) : Below n (f ^^^ g) := fun v hv =>
  (mem_symm f.s g.s v hv).elim (hf v) (hg v)

theorem below_var {n v : Nat} (h : v < n) : Below n (Lin.var v) := fun u hu => by
  simp only [Lin.var, List.mem_singleton] at hu
  exact hu ▸ h

theorem below_const (n : Nat) (b : Bool) : Below n (Lin.const b) := fun _ h => by simp [Lin.const] at h
theorem below_zero (n : Nat) : Below n Lin.zero := fun _ h => by simp [Lin.zero] at h
theorem below_one (n : Nat) : Below n Lin.one := fun _ h => by simp [Lin.one] at h

theorem belowL_nil (n : Nat) : BelowL n [] := fun _ h => by cases h

theorem belowL_cons {n : Nat} {f : Lin} {l : List Lin} : BelowL n (f :: l) ↔ Below n f ∧ BelowL n l :=
  ⟨fun h => ⟨h f List.mem_cons_self, fun g hg => h g (List.mem_cons_of_mem _ hg)⟩,
   fun ⟨hf, hl⟩ g hg => (List.mem_cons.1 hg).elim (fun e => e ▸ hf) (hl g)⟩

theorem belowL_append {n : Nat} {a b : List Lin} : BelowL n (a ++ b) ↔ BelowL n a ∧ BelowL n b :=
  ⟨fun h => ⟨fun f hf => h f (List.mem_append_left _ hf), fun f hf => h f (List.mem_append_right _ hf)⟩,
   fun ⟨ha, hb⟩ f hf => (List.mem_append.1 hf).elim (ha f) (hb f)⟩

theorem BelowL.sub {n : Nat} {a b : List Lin} (h : BelowL n b) (hs : ∀ f ∈ a, f ∈ b) : BelowL n a :=
  fun f hf => h f (hs f hf)

/-- Rows, newest first: each reads only bits below its own output, which is above every older row's. -/
def Chain : List Row → Prop
  | [] => True
  | r :: rs => Chain rs ∧ Below r.out r.a ∧ Below r.out r.b ∧ ∀ r' ∈ rs, r'.out < r.out

/-- The builder's state is scoped: its rows chain, each output below `next`. -/
def Scoped (s : St) : Prop := Chain s.rows ∧ ∀ r ∈ s.rows, r.out < s.next

theorem scoped_empty : Scoped St.empty := ⟨trivial, fun _ h => by cases h⟩

/-- Forward evaluation of rows `rs` (newest first) from the bits `x`: each row's output set to its product, oldest
first. -/
def fwd (x : Var → Bool) : List Row → Var → Bool
  | [] => x
  | r :: rs => fun v => if v = r.out then (ev (fwd x rs) r.a && ev (fwd x rs) r.b) else fwd x rs v

theorem ev_congr {z z' : Var → Bool} (f : Lin) (h : ∀ v ∈ f.s, z v = z' v) : ev z f = ev z' f := by
  obtain ⟨s, c⟩ := f
  unfold ev
  dsimp only at h ⊢
  induction s with
  | nil => rfl
  | cons v s ih =>
    simp only [List.foldr_cons]
    rw [h v List.mem_cons_self, ih (fun u hu => h u (List.mem_cons_of_mem _ hu))]

theorem chain_below : ∀ {rs : List Row}, Chain rs → ∀ r ∈ rs, Below r.out r.a ∧ Below r.out r.b
  | [], _, _, h => by cases h
  | r :: rs, ⟨hc, ha, hb, _⟩, r', h => by
    rcases List.mem_cons.1 h with rfl | h
    · exact ⟨ha, hb⟩
    · exact chain_below hc r' h

/-- **Forward evaluation satisfies chained rows.** -/
theorem fwd_holds (x : Var → Bool) : ∀ {rs : List Row}, Chain rs → ∀ r ∈ rs, rowHolds (fwd x rs) r
  | [], _, _, h => by cases h
  | r :: rs, ⟨hc, ha, hb, hlt⟩, r', h => by
    have same : ∀ f : Lin, Below r.out f → ev (fwd x (r :: rs)) f = ev (fwd x rs) f := fun f hf =>
      ev_congr f fun v hv => by
        have : v ≠ r.out := Nat.ne_of_lt (hf v hv)
        simp only [fwd, this, ite_false]
    rcases List.mem_cons.1 h with rfl | h
    · unfold rowHolds
      rw [same _ ha, same _ hb]
      show (if r'.out = r'.out then _ else _) = _
      rw [ite_eq_left rfl]
    · have ih := fwd_holds x hc r' h
      have hr' := hlt r' h
      obtain ⟨ha', hb'⟩ := chain_below hc r' h
      unfold rowHolds at ih ⊢
      rw [same _ (ha'.mono (Nat.le_of_lt hr')), same _ (hb'.mono (Nat.le_of_lt hr'))]
      show (if r'.out = r.out then _ else _) = _
      rw [ite_eq_right (Nat.ne_of_lt hr')]
      exact ih

theorem fwd_low (x : Var → Bool) {n : Nat} : ∀ {rs : List Row}, (∀ r ∈ rs, n ≤ r.out) → ∀ v, v < n → fwd x rs v = x v
  | [], _, _, _ => rfl
  | r :: rs, h, v, hv => by
    have hne : v ≠ r.out := by have := h r List.mem_cons_self; omega
    show (if v = r.out then _ else _) = _
    rw [ite_eq_right hne]
    exact fwd_low x (fun r' h' => h r' (List.mem_cons_of_mem _ h')) v hv

theorem stHolds_fwd (x : Var → Bool) {s : St} (h : Scoped s) : stHolds (fwd x s.rows) s := fwd_holds x h.1

/-! ## The judgement -/

/-- **Running `m` keeps the builder scoped**: from a scoped state whose `next` bounds every form of `L`, it reaches a
scoped state, `next` no smaller, whose `next` bounds every form of `o` of its result. -/
def Tot {α : Type} (L : List Lin) (m : M α) (o : α → List Lin) : Prop :=
  ∀ s, Scoped s → BelowL s.next L →
    Scoped (m.run s).2 ∧ s.next ≤ (m.run s).2.next ∧ BelowL (m.run s).2.next (o (m.run s).1)

namespace Tot

variable {α β : Type}

theorem pure' {L : List Lin} {a : α} {o : α → List Lin} (h : ∀ n, BelowL n L → BelowL n (o a)) :
    Tot L (pure a : M α) o :=
  fun _ hs hl => ⟨hs, Nat.le_refl _, h _ hl⟩

theorem bind' {L : List Lin} {m : M α} {f : α → M β} {o : α → List Lin} {o' : β → List Lin} (hm : Tot L m o)
    (hf : ∀ a, Tot (L ++ o a) (f a) o') : Tot L (m >>= f) o' := fun s hs hl => by
  obtain ⟨h1, h2, h3⟩ := hm s hs hl
  obtain ⟨h4, h5, h6⟩ := hf _ _ h1 (belowL_append.2 ⟨hl.mono h2, h3⟩)
  exact ⟨h4, Nat.le_trans h2 h5, h6⟩

theorem weaken {L L' : List Lin} {m : M α} {o o' : α → List Lin} (h : Tot L m o)
    (hL : ∀ n, BelowL n L' → BelowL n L) (ho : ∀ n a, BelowL n (o a) → BelowL n (o' a)) : Tot L' m o' :=
  fun s hs hl => let ⟨a, b, c⟩ := h s hs (hL _ hl); ⟨a, b, ho _ _ c⟩

theorem ite' {L : List Lin} {o : α → List Lin} {c : Prop} [Decidable c] {m₁ m₂ : M α} (h₁ : Tot L m₁ o)
    (h₂ : Tot L m₂ o) : Tot L (if c then m₁ else m₂) o := by
  split
  · exact h₁
  · exact h₂

theorem mapM' {L : List Lin} (f : β → M α) (o : α → List Lin) :
    ∀ l : List β, (∀ b ∈ l, Tot L (f b) o) → Tot L (l.mapM f) (fun r => (r.map o).flatten)
  | [], _ => by
    rw [List.mapM_nil]
    exact pure' fun n _ => by simpa using belowL_nil n
  | b :: l, h => by
    rw [List.mapM_cons]
    refine bind' (h b List.mem_cons_self) fun a => ?_
    refine bind' (weaken (mapM' f o l fun b' hb => h b' (List.mem_cons_of_mem _ hb))
      (fun n hn => (belowL_append.1 hn).1) fun _ _ h => h) fun r => ?_
    refine pure' fun n hn => ?_
    rw [List.map_cons, List.flatten_cons]
    obtain ⟨hn, hr⟩ := belowL_append.1 hn
    exact belowL_append.2 ⟨(belowL_append.1 hn).2, hr⟩

end Tot

/-! ## The gadgets -/

theorem tot_fresh (a b : Lin) : Tot [a, b] (fresh a b) fun r => [r] := by
  intro s ⟨hc, ho⟩ hl
  have ha := hl a (by simp)
  have hb := hl b (by simp)
  refine ⟨⟨⟨hc, ha, hb, ho⟩, fun r hr => ?_⟩, Nat.le_succ _, belowL_cons.2 ⟨below_var (Nat.lt_succ_self _), belowL_nil _⟩⟩
  rcases List.mem_cons.1 hr with rfl | hr
  · exact Nat.lt_succ_self _
  · exact Nat.lt_succ_of_lt (ho r hr)

theorem tot_and (a b : Lin) : Tot [a, b] (Flock.HmNets.and a b) fun r => [r] := by
  unfold Flock.HmNets.and
  refine Tot.ite' (Tot.pure' fun n hn => ?_) (Tot.ite' (Tot.pure' fun n hn => ?_) (Tot.ite' (Tot.pure' fun n hn => ?_)
    (tot_fresh a b)))
  all_goals
    refine belowL_cons.2 ⟨?_, belowL_nil _⟩
    split
    · first | exact hn a (by simp) | exact hn b (by simp)
    · exact below_zero n



theorem tot_maj (x y z : Lin) : Tot [x, y, z] (maj x y z) fun r => [r] := by
  unfold maj
  dsimp only
  split
  · refine Tot.ite' (Tot.pure' fun n _ => belowL_cons.2 ⟨below_zero n, belowL_nil _⟩)
      (Tot.ite' (Tot.pure' fun n _ => belowL_cons.2 ⟨below_one n, belowL_nil _⟩)
        (Tot.pure' fun n hn => belowL_cons.2 ⟨?_, belowL_nil _⟩))
    split
    · rename_i v t hv
      exact hn v (List.mem_filter.1 (hv ▸ List.mem_cons_self)).1
    · exact below_const n _
  · refine Tot.bind' (Tot.weaken (tot_and (x ^^^ z) (y ^^^ z)) (fun n hn => ?_) fun _ _ h => h)
      fun r => Tot.pure' fun n hn => ?_
    · have hx := hn x (by simp)
      have hy := hn y (by simp)
      have hz := hn z (by simp)
      exact belowL_cons.2 ⟨hx.xor hz, belowL_cons.2 ⟨hy.xor hz, belowL_nil _⟩⟩
    · exact belowL_cons.2 ⟨(hn z (by simp)).xor (hn r (by simp)), belowL_nil _⟩

theorem tot_commit (f : Lin) : Tot [f] (commit f) fun r => [r] := by
  unfold commit
  exact Tot.ite' (Tot.pure' fun _ h => h) (Tot.weaken (tot_fresh f Lin.one)
    (fun n hn => belowL_cons.2 ⟨hn f (by simp), belowL_cons.2 ⟨below_one n, belowL_nil _⟩⟩) fun _ _ h => h)

/-! ## Words -/

theorem BelowL.drop' {n : Nat} {w : List Lin} (h : BelowL n w) (k : Nat) : BelowL n (w.drop k) :=
  h.sub fun _ hf => List.mem_of_mem_drop hf

theorem BelowL.take' {n : Nat} {w : List Lin} (h : BelowL n w) (k : Nat) : BelowL n (w.take k) :=
  h.sub fun _ hf => List.mem_of_mem_take hf

theorem belowL_replicate_zero (n k : Nat) : BelowL n (List.replicate k Lin.zero) := fun _ hf =>
  (List.eq_of_mem_replicate hf) ▸ below_zero n

theorem belowL_rot {n : Nat} {w : Word} (h : BelowL n w) (k : Nat) : BelowL n (rot w k) :=
  belowL_append.2 ⟨h.drop' k, h.take' k⟩

theorem belowL_shr {n : Nat} {w : Word} (h : BelowL n w) (k : Nat) : BelowL n (shr w k) :=
  belowL_append.2 ⟨h.drop' k, belowL_replicate_zero n k⟩

theorem belowL_xorW {n : Nat} : ∀ {a b : Word}, BelowL n a → BelowL n b → BelowL n (xorW a b)
  | [], _, _, _ => belowL_nil n
  | _ :: _, [], _, _ => belowL_nil n
  | x :: a, y :: b, ha, hb => by
    obtain ⟨hx, ha⟩ := belowL_cons.1 ha
    obtain ⟨hy, hb⟩ := belowL_cons.1 hb
    exact belowL_cons.2 ⟨hx.xor hy, belowL_xorW ha hb⟩

theorem belowL_xor3 {n : Nat} {a b c : Word} (ha : BelowL n a) (hb : BelowL n b) (hc : BelowL n c) :
    BelowL n (xor3 a b c) :=
  belowL_xorW (belowL_xorW ha hb) hc

theorem belowL_constW (n : Nat) (k : BitVec 64) : BelowL n (constW k) := fun f hf => by
  unfold constW at hf
  obtain ⟨i, _, rfl⟩ := List.mem_map.1 hf
  exact below_const n _

theorem belowL_bsig0W {n : Nat} {w : Word} (h : BelowL n w) : BelowL n (bsig0W w) :=
  belowL_xor3 (belowL_rot h _) (belowL_rot h _) (belowL_rot h _)
theorem belowL_bsig1W {n : Nat} {w : Word} (h : BelowL n w) : BelowL n (bsig1W w) :=
  belowL_xor3 (belowL_rot h _) (belowL_rot h _) (belowL_rot h _)
theorem belowL_ssig0W {n : Nat} {w : Word} (h : BelowL n w) : BelowL n (ssig0W w) :=
  belowL_xor3 (belowL_rot h _) (belowL_rot h _) (belowL_shr h _)
theorem belowL_ssig1W {n : Nat} {w : Word} (h : BelowL n w) : BelowL n (ssig1W w) :=
  belowL_xor3 (belowL_rot h _) (belowL_rot h _) (belowL_shr h _)

theorem belowL_flatten {n : Nat} {l : List (List Lin)} : BelowL n l.flatten ↔ ∀ w ∈ l, BelowL n w :=
  ⟨fun h w hw f hf => h f (List.mem_flatten.2 ⟨w, hw, hf⟩),
   fun h f hf => let ⟨w, hw, hf⟩ := List.mem_flatten.1 hf; h w hw f hf⟩

theorem flatten_singletons (r : List Lin) : (r.map fun a => [a]).flatten = r := by
  induction r with
  | nil => rfl
  | cons a r ih => simp [ih]

theorem belowL_getD {n : Nat} {l : List (List Lin)} (h : BelowL n l.flatten) (i : Nat) : BelowL n (l.getD i []) := by
  rw [List.getD_eq_getElem?_getD]
  cases hi : l[i]? with
  | none => exact belowL_nil n
  | some w => exact belowL_flatten.1 h w (List.mem_of_getElem? hi)

theorem tot_chW (e f g : Word) : Tot (e ++ f ++ g) (chW e f g) id := by
  unfold chW
  refine Tot.weaken (Tot.mapM' _ (fun r => [r]) _ fun x hx => ?_) (fun _ h => h) fun n r h => ?_
  · obtain ⟨h1, h23⟩ := List.of_mem_zip hx
    obtain ⟨h2, h3⟩ := List.of_mem_zip h23
    refine Tot.bind' (Tot.weaken (tot_and _ _) (fun n hn => ?_) fun _ _ h => h) fun r => Tot.pure' fun n hn => ?_
    · exact belowL_cons.2 ⟨hn _ (by simp [h1]),
        belowL_cons.2 ⟨(hn _ (by simp [h2])).xor (hn _ (by simp [h3])), belowL_nil _⟩⟩
    · exact belowL_cons.2 ⟨(hn _ (by simp [h3])).xor (hn r (by simp)), belowL_nil _⟩
  · rw [flatten_singletons] at h
    exact h

theorem tot_majW (a b c : Word) : Tot (a ++ b ++ c) (majW a b c) id := by
  unfold majW
  refine Tot.weaken (Tot.mapM' _ (fun r => [r]) _ fun x hx => ?_) (fun _ h => h) fun n r h => ?_
  · obtain ⟨h1, h23⟩ := List.of_mem_zip hx
    obtain ⟨h2, h3⟩ := List.of_mem_zip h23
    exact Tot.weaken (tot_maj _ _ _) (fun n hn => hn.sub fun f hf => by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl <;> simp [h1, h2, h3]) fun _ _ h => h
  · rw [flatten_singletons] at h
    exact h

theorem tot_commitW (w : Word) : Tot w (commitW w) id := by
  unfold commitW
  refine Tot.weaken (Tot.mapM' _ (fun r => [r]) _ fun x hx => ?_) (fun _ h => h) fun n r h => ?_
  · exact Tot.weaken (tot_commit x) (fun n hn => hn.sub fun f hf => by simp at hf; simp [hf, hx]) fun _ _ h => h
  · rw [flatten_singletons] at h
    exact h

theorem tot_carryCommit (i : Nat) (c : Lin) : Tot [c] (carryCommit i c) fun r => [r] := by
  unfold carryCommit
  exact Tot.ite' (tot_commit c) (Tot.pure' fun _ h => h)

theorem tot_add2Go : ∀ (a b : Word) (c : Lin) (i : Nat), Tot (a ++ b ++ [c]) (add2Go a b c i) id
  | x :: a, y :: b, c, i => by
    rw [add2Go]
    refine Tot.ite' (Tot.pure' fun n hn => ?_) ?_
    · exact belowL_cons.2 ⟨((hn x (by simp)).xor (hn y (by simp))).xor (hn c (by simp)), belowL_nil _⟩
    · refine Tot.bind' (Tot.weaken (tot_maj x y c) (fun n hn => hn.sub fun _ _ => by grind) fun _ _ h => h)
        fun c₁ => ?_
      refine Tot.bind' (Tot.weaken (tot_carryCommit i c₁) (fun n hn => hn.sub fun _ _ => by grind) fun _ _ h => h)
        fun c₂ => ?_
      refine Tot.bind' (Tot.weaken (tot_add2Go a b c₂ (i + 1)) (fun n hn => hn.sub fun _ _ => by grind)
        fun _ _ h => h) fun r => ?_
      exact Tot.pure' fun n hn =>
        belowL_cons.2 ⟨((hn x (by simp)).xor (hn y (by simp))).xor (hn c (by simp)), hn.sub fun _ _ => by grind⟩
  | [], _, _, _ => Tot.pure' fun n _ => belowL_nil n
  | _ :: _, [], _, _ => Tot.pure' fun n _ => belowL_nil n

theorem tot_add2 (a b : Word) : Tot (a ++ b) (add2 a b) id :=
  Tot.weaken (tot_add2Go a b Lin.zero 0)
    (fun n hn => belowL_append.2 ⟨hn, belowL_cons.2 ⟨below_zero n, belowL_nil n⟩⟩) fun _ _ h => h

theorem tot_addN : ∀ (acc : Word) (xs : List Word), Tot (acc ++ xs.flatten) (addN acc xs) id
  | acc, [] => by
    rw [addN]
    exact Tot.pure' fun n hn => hn.sub fun _ _ => by grind
  | acc, x :: xs => by
    rw [addN]
    refine Tot.bind' (Tot.weaken (tot_add2 acc x) (fun n hn => hn.sub fun _ _ => by grind) fun _ _ h => h)
      fun acc' => ?_
    exact Tot.weaken (tot_addN acc' xs) (fun n hn => hn.sub fun _ _ => by grind) fun _ _ h => h


theorem tot_schedStep (w : List Word) : Tot w.flatten (schedStep w) List.flatten := by
  unfold schedStep
  have hg : ∀ n, BelowL n w.flatten → ∀ i, BelowL n (w.getD i []) := fun n h i => belowL_getD h i
  refine Tot.bind' (Tot.weaken (tot_addN _ _) (fun n hn => ?_) fun _ _ h => h) fun x => ?_
  · simp only [List.flatten_cons, List.flatten_nil, List.append_nil]
    exact belowL_append.2 ⟨hg n hn _, belowL_append.2 ⟨belowL_ssig0W (hg n hn _),
      belowL_append.2 ⟨hg n hn _, belowL_ssig1W (hg n hn _)⟩⟩⟩
  refine Tot.bind' (Tot.weaken (tot_commitW x) (fun n hn => (belowL_append.1 hn).2) fun _ _ h => h) fun x' => ?_
  exact Tot.pure' fun n hn => by
    rw [List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil]
    obtain ⟨hn, hx'⟩ := belowL_append.1 hn
    exact belowL_append.2 ⟨(belowL_append.1 hn).1, hx'⟩

theorem tot_schedule : ∀ (k : Nat) (w : List Word), Tot w.flatten (schedule k w) List.flatten
  | 0, w => Tot.pure' fun _ h => h
  | k + 1, w => by
    rw [schedule]
    exact Tot.bind' (tot_schedStep w) fun w' =>
      Tot.weaken (tot_schedule k w') (fun n hn => (belowL_append.1 hn).2) fun _ _ h => h

/-- The forms of the eight registers. -/
abbrev regsL (r : Regs Word) : List Lin := r.toList.flatten

theorem regsL_iff {n : Nat} {r : Regs Word} : BelowL n (regsL r) ↔
    BelowL n r.a ∧ BelowL n r.b ∧ BelowL n r.c ∧ BelowL n r.d ∧ BelowL n r.e ∧ BelowL n r.f ∧ BelowL n r.g ∧
      BelowL n r.h := by
  simp only [regsL, Regs.toList, List.flatten_cons, List.flatten_nil, List.append_nil, belowL_append]

theorem tot_roundW (k : BitVec 64) (wt : Word) (r : Regs Word) : Tot (wt ++ regsL r) (roundW k wt r) regsL := by
  unfold roundW
  have hr : ∀ n, BelowL n (wt ++ regsL r) → BelowL n wt ∧ BelowL n r.a ∧ BelowL n r.b ∧ BelowL n r.c ∧
      BelowL n r.d ∧ BelowL n r.e ∧ BelowL n r.f ∧ BelowL n r.g ∧ BelowL n r.h := fun n hn => by
    obtain ⟨h1, h2⟩ := belowL_append.1 hn
    exact ⟨h1, regsL_iff.1 h2⟩
  refine Tot.bind' (Tot.weaken (tot_chW r.e r.f r.g) (fun n hn => ?_) fun _ _ h => h) fun ch => ?_
  · obtain ⟨-, -, -, -, -, he, hf, hg, -⟩ := hr n hn
    exact belowL_append.2 ⟨belowL_append.2 ⟨he, hf⟩, hg⟩
  refine Tot.bind' (Tot.weaken (tot_addN r.h [bsig1W r.e, ch, constW k, wt]) (fun n hn => ?_) fun _ _ h => h)
    fun t1 => ?_
  · obtain ⟨hn, hch⟩ := belowL_append.1 hn
    obtain ⟨hwt, -, -, -, -, he, -, -, hh⟩ := hr n hn
    simp only [List.flatten_cons, List.flatten_nil, List.append_nil]
    exact belowL_append.2 ⟨hh, belowL_append.2 ⟨belowL_bsig1W he, belowL_append.2 ⟨hch,
      belowL_append.2 ⟨belowL_constW n k, hwt⟩⟩⟩⟩
  refine Tot.bind' (Tot.weaken (tot_majW r.a r.b r.c) (fun n hn => ?_) fun _ _ h => h) fun mj => ?_
  · obtain ⟨⟨hn, -⟩, -⟩ := belowL_append.1 hn |>.imp belowL_append.1 id
    obtain ⟨-, ha, hb, hc, -⟩ := hr n hn
    exact belowL_append.2 ⟨belowL_append.2 ⟨ha, hb⟩, hc⟩
  refine Tot.bind' (Tot.weaken (tot_add2 (bsig0W r.a) mj) (fun n hn => ?_) fun _ _ h => h) fun t2 => ?_
  · obtain ⟨hn, hmj⟩ := belowL_append.1 hn
    obtain ⟨⟨hn, -⟩, -⟩ := belowL_append.1 hn |>.imp belowL_append.1 id
    obtain ⟨-, ha, -⟩ := hr n hn
    exact belowL_append.2 ⟨belowL_bsig0W ha, hmj⟩
  refine Tot.bind' (Tot.weaken (tot_add2 r.d t1) (fun n hn => ?_) fun _ _ h => h) fun e => ?_
  · obtain ⟨⟨⟨hn, ht1⟩, -⟩, -⟩ := (belowL_append.1 hn).imp (fun h => (belowL_append.1 h).imp belowL_append.1 id) id
    obtain ⟨-, -, -, -, hd, -⟩ := hr n (belowL_append.1 hn).1
    exact belowL_append.2 ⟨hd, ht1⟩
  refine Tot.bind' (Tot.weaken (tot_commitW e) (fun n hn => (belowL_append.1 hn).2) fun _ _ h => h) fun e' => ?_
  refine Tot.bind' (Tot.weaken (tot_add2 t1 t2) (fun n hn => hn.sub fun _ _ => by grind) fun _ _ h => h) fun a => ?_
  refine Tot.bind' (Tot.weaken (tot_commitW a) (fun n hn => (belowL_append.1 hn).2) fun _ _ h => h) fun a' => ?_
  refine Tot.pure' fun n hn => ?_
  have h0 : BelowL n (wt ++ regsL r) := hn.sub fun _ _ => by grind
  obtain ⟨-, ha, hb, hc, -, he, hf, hg, -⟩ := hr n h0
  refine regsL_iff.2 ⟨hn.sub fun _ _ => by grind, ha, hb, hc, hn.sub fun _ _ => by grind, he, hf, hg⟩

theorem tot_rounds (w : List Word) : ∀ (k t : Nat) (r : Regs Word), Tot (w.flatten ++ regsL r) (rounds w k t r) regsL
  | 0, _, r => Tot.pure' fun _ h => (belowL_append.1 h).2
  | k + 1, t, r => by
    rw [rounds]
    refine Tot.bind' (Tot.weaken (tot_roundW (kAt t) (w.getD t []) r) (fun n hn => ?_) fun _ _ h => h) fun r' => ?_
    · obtain ⟨hw, hr⟩ := belowL_append.1 hn
      exact belowL_append.2 ⟨belowL_getD hw t, hr⟩
    exact Tot.weaken (tot_rounds w k (t + 1) r') (fun n hn => by
      obtain ⟨hn, hr'⟩ := belowL_append.1 hn
      exact belowL_append.2 ⟨(belowL_append.1 hn).1, hr'⟩) fun _ _ h => h


theorem belowL_getElem! {n : Nat} {l : List Word} (h : BelowL n l.flatten) (i : Nat) : BelowL n l[i]! := by
  rw [List.getElem!_eq_getElem?_getD]
  cases hi : l[i]? with
  | none => exact belowL_nil n
  | some w => exact belowL_flatten.1 h w (List.mem_of_getElem? hi)

theorem regsL_ofList {n : Nat} {cv : List Word} (h : BelowL n cv.flatten) : BelowL n (regsL (Regs.ofList cv)) :=
  regsL_iff.2 ⟨belowL_getElem! h 0, belowL_getElem! h 1, belowL_getElem! h 2, belowL_getElem! h 3,
    belowL_getElem! h 4, belowL_getElem! h 5, belowL_getElem! h 6, belowL_getElem! h 7⟩

theorem tot_compress (cv msg : List Word) : Tot (cv.flatten ++ msg.flatten) (compress cv msg) List.flatten := by
  unfold compress
  refine Tot.bind' (Tot.weaken (tot_schedule 64 msg) (fun n hn => (belowL_append.1 hn).2) fun _ _ h => h)
    fun w => ?_
  refine Tot.bind' (Tot.weaken (tot_rounds w 80 0 (Regs.ofList cv)) (fun n hn => ?_) fun _ _ h => h) fun r => ?_
  · obtain ⟨hn, hw⟩ := belowL_append.1 hn
    exact belowL_append.2 ⟨hw, regsL_ofList (belowL_append.1 hn).1⟩
  refine Tot.weaken (Tot.mapM' _ id _ fun x hx => ?_) (fun _ h => h) fun n r h => ?_
  · obtain ⟨h1, h2⟩ := List.of_mem_zip hx
    refine Tot.weaken (tot_add2 x.1 x.2) (fun n hn => ?_) fun _ _ h => h
    obtain ⟨⟨hcv, -⟩, hr⟩ := (belowL_append.1 hn).imp (fun h => (belowL_append.1 h).imp belowL_append.1 id) id
    exact belowL_append.2 ⟨belowL_flatten.1 hcv.1 _ h1, belowL_flatten.1 hr _ h2⟩
  · simpa using h

theorem belowL_words {n : Nat} {bits : List Lin} (h : BelowL n bits) (k : Nat) : BelowL n (words bits k).flatten :=
  belowL_flatten.2 fun w hw => by
    unfold words at hw
    obtain ⟨i, _, rfl⟩ := List.mem_map.1 hw
    exact (h.drop' _).take' _

theorem tot_input (k : Nat) : Tot [] (input k) id := by
  intro s ⟨hc, ho⟩ _
  refine ⟨⟨hc, fun r hr => Nat.lt_of_lt_of_le (ho r hr) (Nat.le_add_right _ _)⟩, Nat.le_add_right _ _, fun f hf => ?_⟩
  obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hf
  exact below_var (by simp at hi; show s.next + i < s.next + k; omega)

/-- Pairs of forms: both. -/
abbrev pairL (x : List Lin × List Lin) : List Lin := x.1 ++ x.2

theorem tot_compression (k : Nat) :
    Tot [] (compression k) fun p => (p.1.map pairL).flatten ++ p.2.flatten := by
  unfold compression
  refine Tot.bind' (Tot.mapM' _ pairL _ fun _ _ => ?_) fun ins => ?_
  · refine Tot.bind' (tot_input 512) fun cv => Tot.bind' (Tot.weaken (tot_input 1024) (fun _ _ => belowL_nil _)
      fun _ _ h => h) fun msg => Tot.pure' fun n hn => ?_
    exact hn.sub fun _ _ => by grind
  refine Tot.bind' (Tot.mapM' _ id _ fun x hx => ?_) fun outs => Tot.pure' fun n hn => ?_
  · refine Tot.bind' (Tot.weaken (tot_compress (words x.1 8) (words x.2 16)) (fun n hn => ?_) fun _ _ h => h)
      fun out => Tot.pure' fun n hn => (belowL_append.1 hn).2
    have hx' : BelowL n (pairL x) := belowL_flatten.1 hn _ (List.mem_map.2 ⟨x, hx, rfl⟩)
    exact belowL_append.2 ⟨belowL_words (belowL_append.1 hx').1 8, belowL_words (belowL_append.1 hx').2 16⟩
  · obtain ⟨⟨-, hins⟩, houts⟩ := (belowL_append.1 hn).imp belowL_append.1 id
    exact belowL_append.2 ⟨hins, by simpa using houts⟩


theorem below_getD {n : Nat} {l : List Lin} (h : BelowL n l) (i : Nat) : Below n (l.getD i Lin.zero) := by
  rw [List.getD_eq_getElem?_getD]
  cases hi : l[i]? with
  | none => exact below_zero n
  | some f => exact h f (List.mem_of_getElem? hi)

theorem belowL_saltWords {n : Nat} {y : List Lin} (h : BelowL n y) : BelowL n (saltWords y).flatten :=
  belowL_flatten.2 fun w hw f hf => by
    unfold saltWords at hw
    obtain ⟨i, _, rfl⟩ := List.mem_map.1 hw
    obtain ⟨j, _, rfl⟩ := List.mem_map.1 hf
    exact below_getD h _

theorem below_maskBit {n : Nat} (key : ByteArray) {y : List Lin} (hy : BelowL n y) (i : Nat) {x : Lin}
    (hx : Below n x) : Below n (maskBit key y i x) := by
  have hget : ∀ j, y.toArray.getD j Lin.zero = y.getD j Lin.zero := fun j => by
    rw [List.getD_eq_getElem?_getD]
    by_cases hj : j < y.length
    · simp [Array.getD, hj]
    · simp [Array.getD, hj]
  unfold maskBit
  dsimp only
  generalize List.range 1536 = l
  induction l with
  | nil => exact hx
  | cons j l ih =>
    rw [List.foldr_cons]
    split
    · rw [hget]
      exact (below_getD hy j).xor ih
    · exact ih

theorem tot_hm96Row (key : ByteArray) :
    Tot [] (hm96Row key) fun p => p.1 ++ p.2.1 ++ p.2.2.1 ++ p.2.2.2 := by
  unfold hm96Row
  refine Tot.bind' (tot_input 512) fun cv => ?_
  refine Tot.bind' (Tot.weaken (tot_input 1024) (fun _ _ => belowL_nil _) fun _ _ h => h) fun pad => ?_
  refine Tot.bind' (Tot.weaken (tot_input 1536) (fun _ _ => belowL_nil _) fun _ _ h => h) fun y => ?_
  refine Tot.bind' (Tot.weaken (tot_compress (words cv 8) (words pad 16)) (fun n hn => ?_) fun _ _ h => h)
    fun x => ?_
  · have hcv : BelowL n cv := hn.sub fun _ _ => by grind
    have hpad : BelowL n pad := hn.sub fun _ _ => by grind
    exact belowL_append.2 ⟨belowL_words hcv 8, belowL_words hpad 16⟩
  refine Tot.bind' (Tot.weaken (tot_compress (saltMid.map constW) ((saltWords y).take 16)) (fun n hn => ?_)
    fun _ _ h => h) fun st => ?_
  · have hyw := belowL_saltWords (n := n) (y := y) (hn.sub fun _ _ => by grind)
    refine belowL_append.2 ⟨belowL_flatten.2 fun w hw => ?_, belowL_flatten.2 fun w hw => ?_⟩
    · obtain ⟨k, _, rfl⟩ := List.mem_map.1 hw
      exact belowL_constW n k
    · exact belowL_flatten.1 hyw w (List.mem_of_mem_take hw)
  refine Tot.bind' (Tot.weaken (tot_compress st ((saltWords y).drop 16 ++ saltTail.map constW)) (fun n hn => ?_)
    fun _ _ h => h) fun st' => ?_
  · have hyw := belowL_saltWords (n := n) (y := y) (hn.sub fun _ _ => by grind)
    refine belowL_append.2 ⟨(belowL_append.1 hn).2, belowL_flatten.2 fun w hw => ?_⟩
    rcases List.mem_append.1 hw with hw | hw
    · exact belowL_flatten.1 hyw w (List.mem_of_mem_drop hw)
    · obtain ⟨k, _, rfl⟩ := List.mem_map.1 hw
      exact belowL_constW n k
  refine Tot.pure' fun n hn => ?_
  have hy : BelowL n y := hn.sub fun _ _ => by grind
  have hx : BelowL n x.flatten := hn.sub fun _ _ => by grind
  have hst : BelowL n st'.flatten := (belowL_append.1 hn).2
  refine belowL_append.2 ⟨hn.sub fun _ _ => by grind, belowL_append.2 ⟨fun f hf => ?_, fun f hf => ?_⟩⟩
  · obtain ⟨i, _, rfl⟩ := List.mem_map.1 hf
    exact below_maskBit key hy i (below_getD hx _)
  · obtain ⟨t, _, rfl⟩ := List.mem_map.1 hf
    exact below_getD hst _


/-! ## The layout's padding rows, and the converse of `laid_rows` -/

/-- **The layout's padding rows are empty in A**: those between the ANDs and the output group, and those of the output
group past the outputs. -/
theorem layout_gaps (st : St) (inBits outBits : Array Nat) (outs : Array Lin) :
    let l := layout st inBits outBits outs
    let inRows := groupWords (inBits.foldl (· + ·) 0) * WORD
    let span := groupWords (outBits.foldl (· + ·) 0) * WORD
    let start := (inRows + st.rows.length + span - 1) / span * span
    outs.size ≤ span →
      (∀ i, inRows + st.rows.length ≤ i → i < start → l.ra.getD i #[] = #[]) ∧
        ∀ i, start + outs.size ≤ i → i < start + span → l.ra.getD i #[] = #[] := by
  intro l inRows span start hsz
  let nIn := inBits.foldl (· + ·) 0
  have hIn : nIn ≤ inRows := le_groupWords nIn
  have hspan : 0 < span := groupWords_pos _
  have hst : inRows + st.rows.length ≤ start := le_roundUp _ _ hspan
  let rows := st.rows.reverse.toArray
  let e := fun (n : Nat) => Array.replicate n (#[] : Array Nat)
  let ins := (Array.range nIn).map fun i => #[i]
  have hra : l.ra = ins ++ e (inRows - nIn) ++ rows.map (cols nIn inRows (start + span) ·.a) ++
      e (start - inRows - st.rows.length) ++ outs.map (cols nIn inRows (start + span)) ++ e (span - outs.size) ++
      #[#[start + span]] := by
    simp only [l, layout, rows, e, ins, start, span, inRows, nIn, List.size_toArray, List.length_reverse]
  have hs : ∀ n, (e n).size = n := fun n => Array.size_replicate
  have hr : rows.size = st.rows.length := by simp [rows]
  have hi : ins.size = nIn := by simp [ins]
  refine ⟨fun i h1 h2 => ?_, fun i h1 h2 => ?_⟩
  · rw [hra]
    simp (disch := (simp only [Array.size_append, Array.size_map, hs, hr, hi, List.size_toArray, Array.size_singleton]; omega))
      only [getD_append_left, getD_append_right]
    simp only [Array.size_append, Array.size_map, hs, hi, hr]
    simp only [e, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
    split <;> rfl
  · rw [hra]
    simp (disch := (simp only [Array.size_append, Array.size_map, hs, hr, hi, List.size_toArray, Array.size_singleton]; omega))
      only [getD_append_left, getD_append_right]
    simp only [Array.size_append, Array.size_map, hs, hi, hr]
    simp only [e, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
    split <;> rfl


/-- **The column values that satisfy a laid-out net from inputs `x`**: `x` on the input group's rows, forward evaluation
on the ANDs' rows, each output's form on its row, 1 at the constant and 0 on the padding. -/
def laidVal (st : St) (inBits outBits : Array Nat) (outs : Array Lin) (x : Nat → Bool) (i : Nat) : Bool :=
  if i < inRowsOf inBits then x i
  else if i < inRowsOf inBits + st.rows.length then fwd x st.rows (inBits.foldl (· + ·) 0 + (i - inRowsOf inBits))
  else if outStart st inBits outBits ≤ i ∧ i < outStart st inBits outBits + outs.size then
    ev (fwd x st.rows) (outs.getD (i - outStart st inBits outBits) Lin.zero)
  else decide (i = outStart st inBits outBits + spanOf outBits)

theorem good_le {nIn : Nat} {st : St} (hg : Good nIn st) : ∀ r ∈ st.rows, nIn ≤ r.out := fun r hr => by
  obtain ⟨k, hk, he⟩ := List.getElem_of_mem (List.mem_reverse.2 hr)
  rw [hg.2 k r (he ▸ List.getElem?_eq_getElem hk)]
  exact Nat.le_add_right _ _

/-- **A laid-out net's rows hold from any inputs** (the converse of `laid_rows`): if the builder's state is scoped
(`Scoped`), its rows output `nIn, nIn + 1, …` (`Good`), and every output reads only committed bits and fits the output
group, `laidVal` satisfies every row from the end of the input group to the constant. -/
theorem laid_total (st : St) (inBits outBits : Array Nat) (outs : Array Lin) (x : Nat → Bool)
    (hg : Good (inBits.foldl (· + ·) 0) st) (hi : Scoped st) (ho : BelowL st.next outs.toList)
    (hsz : outs.size ≤ outBits.foldl (· + ·) 0) :
    LaidRows (layout st inBits outBits outs) (inRowsOf inBits) (laidVal st inBits outBits outs x) := by
  have hsz' : outs.size ≤ spanOf outBits := Nat.le_trans hsz (le_groupWords _)
  obtain ⟨hc, -, h3, h4⟩ := layout_rows st inBits outBits outs
  obtain ⟨g1, g2⟩ := layout_gaps st inBits outBits outs hsz'
  have hS : (inRowsOf inBits + st.rows.length + spanOf outBits - 1) / spanOf outBits * spanOf outBits =
      outStart st inBits outBits := rfl
  simp only [hS] at hc h3 h4 g1 g2
  generalize hz : fwd x st.rows = z
  have hv : laidVal st inBits outBits outs x = fun i =>
      if i < inRowsOf inBits then x i
      else if i < inRowsOf inBits + st.rows.length then z (inBits.foldl (· + ·) 0 + (i - inRowsOf inBits))
      else if outStart st inBits outBits ≤ i ∧ i < outStart st inBits outBits + outs.size then
        ev z (outs.getD (i - outStart st inBits outBits) Lin.zero)
      else decide (i = outStart st inBits outBits + spanOf outBits) := by
    rw [← hz]; rfl
  rw [hv]
  clear hv
  have hIn : inBits.foldl (· + ·) 0 ≤ inRowsOf inBits := le_groupWords _
  have hspan : 0 < spanOf outBits := groupWords_pos _
  have hst : inRowsOf inBits + st.rows.length ≤ outStart st inBits outBits := le_roundUp _ _ hspan
  clear hS
  generalize outStart st inBits outBits = S at *
  simp only [inRowsOf, spanOf] at *
  generalize groupWords (inBits.foldl (· + ·) 0) * WORD = R at *
  generalize inBits.foldl (· + ·) 0 = nIn at *
  have hzl : ∀ u, u < nIn → z u = x u := fun u hu => hz ▸ fwd_low x (good_le hg) u hu
  have hnext : st.next = nIn + st.rows.length := hg.1
  generalize hv : (fun i => if i < R then x i
      else if i < R + st.rows.length then z (nIn + (i - R))
      else if S ≤ i ∧ i < S + outs.size then ev z (outs.getD (i - S) Lin.zero)
      else decide (i = S + groupWords (outBits.foldl (· + ·) 0) * WORD)) = v
  have hvc : ∀ u, u < st.next → v (col nIn R u) = z u := fun u hu => by
    rw [← hv]
    by_cases h : u < nIn
    · simp only [ite_true, col, h, show u < R by omega]
      exact (hzl u h).symm
    · simp only [ite_true, ite_false, col, h, show ¬ R + (u - nIn) < R by omega, show R + (u - nIn) < R + st.rows.length by omega]
      rw [Nat.add_sub_cancel_left, Nat.add_sub_of_le (Nat.le_of_not_lt h)]
  have hev : ∀ f, Below st.next f → ev (fun u => v (col nIn R u)) f = ev z f := fun f hf =>
    ev_congr f fun u hu => hvc u (hf u hu)
  have hcv : v (layout st inBits outBits outs).constPos = true := by
    rw [hc, ← hv]
    simp only [ite_false, show ¬ S + groupWords (outBits.foldl (· + ·) 0) * WORD < R by omega,
      show ¬ S + groupWords (outBits.foldl (· + ·) 0) * WORD < R + st.rows.length by omega,
      show ¬ (S ≤ S + groupWords (outBits.foldl (· + ·) 0) * WORD ∧
        S + groupWords (outBits.foldl (· + ·) 0) * WORD < S + outs.size) by omega, ite_false, decide_true]
  refine ⟨hcv, fun i h1 h2 => ?_⟩
  rw [hc] at h2
  by_cases ha : i < R + st.rows.length
  · obtain ⟨k, rfl⟩ : ∃ k, i = R + k := ⟨i - R, by omega⟩
    have hk : k < st.rows.reverse.length := by simp; omega
    obtain ⟨hA, hB⟩ := h3 k hk
    have hr : st.rows.reverse[k] ∈ st.rows := List.mem_reverse.1 (List.getElem_mem hk)
    have hout : st.rows.reverse[k].out = nIn + k := hg.2 k _ (List.getElem?_eq_getElem hk)
    obtain ⟨hba, hbb⟩ := chain_below hi.1 _ hr
    have hlt := hi.2 _ hr
    rw [hA, hB, xorRow_cols v hcv, xorRow_cols v hcv, hev _ (hba.mono (Nat.le_of_lt hlt)),
      hev _ (hbb.mono (Nat.le_of_lt hlt))]
    have hrow := hz ▸ fwd_holds x hi.1 _ hr
    unfold rowHolds at hrow
    rw [← hrow, hout, ← hv]
    simp only [ite_true, ite_false, show ¬ R + k < R by omega, ha]
    rw [Nat.add_sub_cancel_left]
  · by_cases hb : i < S
    · rw [g1 i (by omega) hb]
      rw [← hv]
      simp only [ite_false, xorRow, List.foldr_nil, Bool.false_and, show ¬ i < R by omega, ha,
        show ¬ (S ≤ i ∧ i < S + outs.size) by omega, ite_false]
      exact (decide_eq_false (by omega)).symm
    · by_cases hc' : i < S + outs.size
      · obtain ⟨j, rfl⟩ : ∃ j, i = S + j := ⟨i - S, by omega⟩
        have hj : j < outs.size := by omega
        obtain ⟨hA, hB⟩ := h4 j hj
        rw [hA, hB, xorRow_cols v hcv, xorRow_single, hcv, Bool.and_true,
          hev _ (ho _ (Array.getElem_mem_toList hj))]
        rw [← hv]
        simp only [ite_true, ite_false, and_self, show ¬ S + j < R by omega, ha, show S ≤ S + j ∧ S + j < S + outs.size by omega, ite_true,
          ite_false, Nat.add_sub_cancel_left]
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some]
      · rw [g2 i (by omega) h2]
        rw [← hv]
        simp only [ite_false, xorRow, List.foldr_nil, Bool.false_and, show ¬ i < R by omega, ha,
          show ¬ (S ≤ i ∧ i < S + outs.size) by omega, ite_false]
        exact (decide_eq_false (by omega)).symm


/-! ## The two nets -/

theorem laidVal_input {st : St} {inBits outBits : Array Nat} {outs : Array Lin} {x : Nat → Bool} {i : Nat}
    (h : i < inRowsOf inBits) : laidVal st inBits outBits outs x i = x i := by
  unfold laidVal
  rw [ite_eq_left h]

/-- **The verifier's `sha512x3` net holds from any inputs**: `laidVal` of its build satisfies every row of its layout. -/
theorem sha_total (x : Nat → Bool) :
    LaidRows (shaLaid ()) (inRowsOf SHA_IN)
      (laidVal (compression3.run St.empty).2 SHA_IN SHA_OUT (compression3.run St.empty).1.2.flatten.toArray x) := by
  have h : Tot [] compression3 _ := tot_compression 3
  have h := h St.empty scoped_empty (belowL_nil _)
  have hz := stHolds_fwd x h.1
  obtain ⟨-, hl3, -⟩ := compression3_spec _ hz
  have hlens := compression3_lens _ hz
  have hflat : (compression3.run St.empty).1.2.flatten.length = 1536 := by
    rw [List.length_flatten, List.map_congr_left (g := fun _ => 512) hlens, List.map_const', List.sum_replicate, hl3]
    rfl
  rw [shaLaid_eq]
  refine laid_total _ _ _ _ x (by rw [sum_SHA_IN]; exact good_compression3) h.1 ?_ ?_
  · rw [List.toList_toArray]
    exact (belowL_append.1 h.2.2).2
  · rw [List.size_toArray, hflat]
    decide

/-- **The verifier's `hm96` net holds from any inputs**: `laidVal` of its build satisfies every row of its layout. -/
theorem hm_total (x : Nat → Bool) :
    LaidRows (hmLaid ()) (inRowsOf HM_IN)
      (laidVal ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT
        ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.toArray x) := by
  have h := tot_hm96Row Hm96.Default512.key St.empty scoped_empty (belowL_nil _)
  obtain ⟨-, -, -, hl, -⟩ := hm96Row_spec _ _ (stHolds_fwd x h.1)
  rw [hmLaid_eq]
  refine laid_total _ _ _ _ x (by rw [sum_HM_IN]; exact good_hm96Row _) h.1 ?_ ?_
  · rw [List.toList_toArray]
    exact (belowL_append.1 h.2.2).2
  · rw [List.size_toArray, hl]
    decide

end FlockDeploy.Concrete.Total
