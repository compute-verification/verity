import Proofs.Properties.Nci.Resolve

/-!
# Lemmas for `BodiesOk`: types, loops and streams

What the decoder's spec (`Proofs.Properties.Nci.BodiesOk.Decode`) reads off `Flock.Program`'s code: type equality from
`Ty.beq`, the leaves of the types a node computes, the leaves of a batch member mapped over an axis (`axis_spec`), and
invariants of `for` loops in `Id`, `Except` and the decoder's `StateT` monad, index by index.
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk

open Flock.Program

/-! ### Types -/

theorem zip_all {α β : Type} (as : Array α) (bs : Array β) (p : α → β → Bool) (hs : as.size = bs.size)
    (h : (as.zip bs).all (fun x => p x.1 x.2) = true) :
    ∀ i (h1 : i < as.size), p as[i] (bs[i]'(hs ▸ h1)) = true := by
  intro i h1
  rw [Array.all_eq_true] at h
  have := h i (by simp [Array.size_zip]; omega)
  simpa [Array.getElem_zip] using this

theorem beq_eq : ∀ (n : ℕ) (a b : Ty), sizeOf a < n → a.beq b = true → a = b := by
  intro n
  induction n with
  | zero => intro a b h; omega
  | succ n ih =>
    intro a b hs h
    unfold Ty.beq at h
    cases a <;> cases b <;> simp at h
    · rw [h]
    · rename_i n1 e1 n2 e2
      obtain ⟨rfl, h⟩ := h
      rw [ih e1 e2 (by simp at hs; omega) h]
    · rename_i as bs
      obtain ⟨hsz, h⟩ := h
      have := zip_all as.attach bs (fun x y => x.1.beq y) (by simpa using hsz) (by simpa using h)
      congr 1
      apply Array.ext hsz
      intro i h1 h2
      have hi := this i (by simpa using h1)
      simp only [Array.getElem_attach] at hi
      exact ih _ _ (by
        have := Array.sizeOf_lt_of_mem (Array.getElem_mem h1)
        simp at hs; omega) hi
    · rename_i as bs
      obtain ⟨hsz, h⟩ := h
      have := zip_all as.attach bs (fun x y => x.1.1 == y.1 && x.1.2.beq y.2) (by simpa using hsz) (by simpa using h)
      congr 1
      apply Array.ext hsz
      intro i h1 h2
      have hi := this i (by simpa using h1)
      simp only [Array.getElem_attach, Bool.and_eq_true, beq_iff_eq] at hi
      have hlt : sizeOf as[i].2 < n := by
        have := Array.sizeOf_lt_of_mem (Array.getElem_mem h1)
        have h3 : sizeOf as[i].2 < sizeOf as[i] := by
          rcases as[i] with ⟨x, y⟩; simp
        simp at hs; omega
      exact Prod.ext hi.1 (ih _ _ hlt hi.2)

/-- `Ty`'s `==` is equality. -/
theorem ty_beq {a b : Ty} (h : (a == b) = true) : a = b := beq_eq _ a b (Nat.lt_succ_self _) h

theorem ty_not_beq {a b : Ty} (h : (!(a == b)) = false) : a = b := ty_beq (by simpa using h)

theorem leaves_tuple2 (a b : Ty) : (Ty.tuple #[a, b]).leaves = a.leaves + b.leaves := by
  rw [Ty.leaves]; simp

theorem leaves_array (n : ℕ) (e : Ty) : (Ty.array n e).leaves = n * e.leaves := by
  rw [Ty.leaves]

theorem leaves_value (w : ℕ) : (Ty.value w).leaves = 1 := by
  rw [Ty.leaves]

/-- A batch argument mapped over axis `ax` of length `ln`: member `m`'s leaf `j` is in range of the whole. -/
theorem axis_spec : ∀ (ax : ℕ) (t : Ty) (ln : ℕ) (mt : Ty), t.removeAxis ax = some (ln, mt) →
    ∃ L, t.axisDims ax = some (ln, L) ∧ t.leaves = ln * mt.leaves ∧ L ∣ mt.leaves ∧
      ∀ j < mt.leaves, ∀ m < ln, (j / L * ln + m) * L + j % L < t.leaves := by
  intro ax
  induction ax with
  | zero =>
    intro t ln mt h
    cases t <;> simp [Ty.removeAxis] at h
    rename_i n e
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨e.leaves, by simp [Ty.axisDims], leaves_array _ _, dvd_refl _, fun j hj m hm => ?_⟩
    rw [leaves_array, Nat.div_eq_of_lt hj, Nat.mod_eq_of_lt hj]
    have : (m + 1) * e.leaves ≤ n * e.leaves := Nat.mul_le_mul_right _ hm
    simp only [Nat.zero_mul, Nat.zero_add]
    rw [Nat.add_mul, Nat.one_mul] at this
    omega
  | succ ax ih =>
    intro t ln mt h
    cases t <;> simp [Ty.removeAxis] at h
    rename_i n e
    obtain ⟨inner, hr, rfl⟩ := h
    obtain ⟨L, hax, hlv, hdvd, hb⟩ := ih e ln inner hr
    refine ⟨L, by simpa [Ty.axisDims] using hax, ?_, ?_, fun j hj m hm => ?_⟩
    · rw [leaves_array, leaves_array, hlv]; ring
    · rw [leaves_array]; exact Dvd.dvd.mul_left hdvd n
    · rw [leaves_array] at hj ⊢
      have hIpos : 0 < inner.leaves := by
        rcases Nat.eq_zero_or_pos inner.leaves with h0 | h0
        · rw [h0] at hj; simp at hj
        · exact h0
      obtain ⟨s, hs⟩ := hdvd
      have hLpos : 0 < L := by
        rcases Nat.eq_zero_or_pos L with h0 | h0
        · rw [h0] at hs; omega
        · exact h0
      have hq : j / inner.leaves < n := (Nat.div_lt_iff_lt_mul hIpos).2 hj
      have hr : j % inner.leaves < inner.leaves := Nat.mod_lt _ hIpos
      have hjr : j = j % inner.leaves + (j / inner.leaves * s) * L := by
        calc j = inner.leaves * (j / inner.leaves) + j % inner.leaves := (Nat.div_add_mod j _).symm
          _ = _ := by rw [hs]; ring
      have hdiv : j / L = j % inner.leaves / L + j / inner.leaves * s := by
        conv => lhs; rw [hjr]
        rw [Nat.add_mul_div_right _ _ hLpos]
      have hmod : j % L = j % inner.leaves % L := by
        conv => lhs; rw [hjr]
        rw [Nat.add_mul_mod_self_right]
      have hA := hb _ hr m hm
      rw [hlv] at hA
      rw [hdiv, hmod]
      have e1 : ((j % inner.leaves / L + j / inner.leaves * s) * ln + m) * L + j % inner.leaves % L =
          ((j % inner.leaves / L * ln + m) * L + j % inner.leaves % L) +
            j / inner.leaves * (ln * (L * s)) := by ring
      rw [e1, ← hs, hlv]
      have hq1 : (j / inner.leaves + 1) * (ln * inner.leaves) ≤ n * (ln * inner.leaves) :=
        Nat.mul_le_mul_right _ hq
      rw [Nat.add_mul, Nat.one_mul] at hq1
      generalize j / inner.leaves * (ln * inner.leaves) = Y at hq1 ⊢
      generalize n * (ln * inner.leaves) = Z at hq1 ⊢
      omega

/-! ### Loops -/

/-- An invariant of a `for` loop in `Id`, index by index: it holds after every element unless the loop stops early,
when `D` holds. -/
theorem forIn_id_inv {α β : Type} (f : α → β → Id (ForInStep β)) (D : β → Prop) :
    ∀ (l : List α) (I : ℕ → β → Prop) (b : β), I 0 b →
    (∀ i (hi : i < l.length) b, I i b → match (f l[i] b).run with
      | .yield b' => I (i + 1) b' | .done b' => D b') →
    I l.length (forIn l b f).run ∨ D (forIn l b f).run
  | [], _, _, h0, _ => Or.inl h0
  | x :: xs, I, b, h0, hs => by
    have h := hs 0 (by simp) b h0
    simp only [List.getElem_cons_zero] at h
    rw [List.forIn_cons]
    generalize f x b = r at h
    rcases r with b' | b'
    · exact Or.inr h
    · have := forIn_id_inv f D xs (fun i b => I (i + 1) b) b' h
        (fun i hi b hb => hs (i + 1) (by simp; omega) b hb)
      simp only [List.length_cons]
      exact this

/-- The same in `Except`: an invariant of a `for` loop that finished. -/
theorem forIn_except_inv {α β ε : Type} (f : α → β → Except ε (ForInStep β)) (D : β → Prop) :
    ∀ (l : List α) (I : ℕ → β → Prop) (b r : β), I 0 b →
    (∀ i (hi : i < l.length) b s, I i b → f l[i] b = .ok s → match s with
      | .yield b' => I (i + 1) b' | .done b' => D b') →
    forIn l b f = .ok r → I l.length r ∨ D r
  | [], _, b, r, h0, _, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h; subst h; exact Or.inl h0
  | x :: xs, I, b, r, h0, hs, h => by
    rw [List.forIn_cons] at h
    obtain ⟨s, hfs, h⟩ := Walk.except_bind_ok h
    have h1 := hs 0 (by simp) b s h0 (by simpa using hfs)
    rcases s with b' | b'
    · simp only [pure, Except.pure, Except.ok.injEq] at h; subst h; exact Or.inr h1
    · have := forIn_except_inv f D xs (fun i b => I (i + 1) b) b' r h1
        (fun i hi b s hb hf => hs (i + 1) (by simp; omega) b s hb (by simpa using hf)) h
      simpa using this

/-- The same in a state monad over `Except` (the decoder's `D`). -/
theorem forIn_stateT_inv {α β σ ε : Type} (f : α → β → StateT σ (Except ε) (ForInStep β)) (D : β → σ → Prop) :
    ∀ (l : List α) (I : ℕ → β → σ → Prop) (b r : β) (s s' : σ), I 0 b s →
    (∀ i (hi : i < l.length) b s t s', I i b s → f l[i] b s = .ok (t, s') → match t with
      | .yield b' => I (i + 1) b' s' | .done b' => D b' s') →
    forIn l b f s = .ok (r, s') → I l.length r s' ∨ D r s'
  | [], _, b, r, s, s', h0, _, h => by
    simp only [List.forIn_nil, pure, StateT.pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h; exact Or.inl h0
  | x :: xs, I, b, r, s, s', h0, hs, h => by
    rw [List.forIn_cons] at h
    simp only [bind, StateT.bind] at h
    obtain ⟨⟨t, s1⟩, hfs, h⟩ := Walk.except_bind_ok h
    have h1 := hs 0 (by simp) b s t s1 h0 (by simpa using hfs)
    rcases t with b' | b'
    · simp only [pure, StateT.pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h; exact Or.inr h1
    · have := forIn_stateT_inv f D xs (fun i b s => I (i + 1) b s) b' r s1 s' h1
        (fun i hi b s t s' hb hf => hs (i + 1) (by simp; omega) b s t s' hb (by simpa using hf)) h
      simpa using this

theorem range_length (a b : ℕ) : (List.range' a (([a:b] : Std.Legacy.Range).size) 1).length = b - a := by
  simp [Std.Legacy.Range.size]

/-! ### Arrays -/

theorem mapM_size {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α) (ys : List β), l.mapM f = .ok ys → ys.length = l.length
  | [], ys, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h; subst h; rfl
  | a :: l, ys, h => by
    rw [List.mapM_cons] at h
    obtain ⟨c, -, h⟩ := Walk.except_bind_ok h
    obtain ⟨cs, hl, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
    simp [mapM_size f l cs hl]

theorem array_mapM_size {α β ε : Type} (f : α → Except ε β) (xs : Array α) (ys : Array β)
    (h : xs.mapM f = .ok ys) : ys.size = xs.size := by
  rw [Array.mapM_eq_mapM_toList] at h
  rcases hl : xs.toList.mapM f with e | l <;> simp only [hl, Functor.map, Except.map] at h
  · exact absurd h (by intro h; injection h)
  injection h with h; subst h
  simp [mapM_size f _ l hl]

/-- `Array.mapM` in `Except`, element by element. -/
theorem list_mapM_get {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α) (ys : List β), l.mapM f = .ok ys → ∀ i (h1 : i < l.length) (h2 : i < ys.length),
      f l[i] = .ok ys[i]
  | [], ys, _, i, h1, _ => by simp at h1
  | a :: l, ys, h, i, h1, h2 => by
    rw [List.mapM_cons] at h
    obtain ⟨c, hc, h⟩ := Walk.except_bind_ok h
    obtain ⟨cs, hl, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
    rcases i with _ | i
    · simpa using hc
    · simpa using list_mapM_get f l cs hl i (by simpa using h1) (by simpa using h2)

theorem array_mapM_get {α β ε : Type} (f : α → Except ε β) (xs : Array α) (ys : Array β)
    (h : xs.mapM f = .ok ys) : ∀ i (h1 : i < xs.size) (h2 : i < ys.size), f xs[i] = .ok ys[i] := by
  rw [Array.mapM_eq_mapM_toList] at h
  rcases hl : xs.toList.mapM f with e | l <;> simp only [hl, Functor.map, Except.map] at h
  · exact absurd h (by intro h; injection h)
  injection h with h; subst h
  intro i h1 h2
  have := list_mapM_get f _ l hl i (by simpa using h1) (by simpa using h2)
  simpa using this

theorem getElem!_push_lt {α : Type} [Inhabited α] (a : Array α) (x : α) {i : ℕ} (h : i < a.size) :
    (a.push x)[i]! = a[i]! := by
  rw [getElem!_pos (a.push x) i (by simp; omega), getElem!_pos a i h, Array.getElem_push_lt]

theorem getElem!_push_eq {α : Type} [Inhabited α] (a : Array α) (x : α) : (a.push x)[a.size]! = x := by
  rw [getElem!_pos (a.push x) a.size (by simp), Array.getElem_push_eq]

/-! ### Streams -/

/-- One step of the stream a zipped `for` loop reads an array through. -/
theorem next_some {α : Type} (s : Subarray α) (h : s.start < s.stop) :
    ∃ s', Std.Stream.next? s = some (s.array[s.start]'(Nat.lt_of_lt_of_le h s.stop_le_array_size), s') ∧
      s'.array = s.array ∧ s'.start = s.start + 1 ∧ s'.stop = s.stop := by
  refine ⟨⟨{ s.internalRepresentation with start := s.start + 1, start_le_stop := Nat.succ_le_of_lt h }⟩, ?_, rfl, rfl, rfl⟩
  show (if h : s.start < s.stop then _ else _) = _
  split
  · rfl
  · contradiction

theorem next_at {α : Type} [Inhabited α] (s : Subarray α) (a : Array α) (i : ℕ) (ha : s.array = a)
    (hst : s.start = i) (h : i < s.stop) :
    ∃ s', Std.Stream.next? s = some (a[i]!, s') ∧ s'.array = a ∧ s'.start = i + 1 ∧ s'.stop = s.stop := by
  subst ha hst
  obtain ⟨s', hn, h1, h2, h3⟩ := next_some s h
  refine ⟨s', ?_, h1, h2, h3⟩
  rw [hn, getElem!_pos s.array s.start (Nat.lt_of_lt_of_le h s.stop_le_array_size)]

theorem next_none {α : Type} (s : Subarray α) (h : ¬ s.start < s.stop) : Std.Stream.next? s = none := by
  show (if h : s.start < s.stop then _ else _) = _
  split
  · contradiction
  · rfl

theorem toStream_array {α : Type} (a : Array α) :
    (Std.toStream a).array = a ∧ (Std.toStream a).start = 0 ∧ (Std.toStream a).stop = a.size := by
  show (a.toSubarray 0 a.size).array = a ∧ (a.toSubarray 0 a.size).start = 0 ∧ (a.toSubarray 0 a.size).stop = a.size
  simp [Array.start_toSubarray, Array.stop_toSubarray, Array.array_toSubarray]

end NciCheck.Properties.SecurityProofs.BodiesOk
