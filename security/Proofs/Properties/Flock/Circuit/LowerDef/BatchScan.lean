module

public import Proofs.Properties.Flock.Circuit.LowerDef

/-! # `lowerDef`'s batch case past its bindings' parts, and its scan case, over `LowerDef`'s forms and node values -/

@[expose] public section

namespace Flock.Chain.LowerDef

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue
open Flock.Program hiding Form

/-! ## The batch case: where a piece of members stops -/

/-- What `pieceEnd`'s end `m1` says of input `p` from member `m0`: a mapped input's reference for member `m0` lies in an
affine part, which holds its references up to member `m1`. -/
def EndOk (m0 m1 : ℕ) (p : Arg × ℕ) : Prop :=
  ∀ d L, p.1.dims = some (d, L) → L ≠ 0 →
    memberIx p.1 m0 p.2 < p.1.seq.count ∧
    (∃ p0 i0 b0 c0, p.1.seq.parts[partAt p.1.seq (memberIx p.1 m0 p.2)]! = .affine p0 i0 b0 c0) ∧
    m1 ≤ m0 + (p.1.seq.starts[partAt p.1.seq (memberIx p.1 m0 p.2)]! +
      p.1.seq.parts[partAt p.1.seq (memberIx p.1 m0 p.2)]!.count - memberIx p.1 m0 p.2 + L - 1) / L

theorem endOk_mono {m0 m1 m1' : ℕ} {p : Arg × ℕ} (h : m1' ≤ m1) (hp : EndOk m0 m1 p) : EndOk m0 m1' p :=
  fun d L hd hL => let ⟨a, b, c⟩ := hp d L hd hL; ⟨a, b, by omega⟩

theorem forIn_end {m0 : ℕ}
    {f : Arg × ℕ → Option (Option ℕ) × ℕ → Except String (ForInStep (Option (Option ℕ) × ℕ))}
    (hf : ∀ p m s, f p (none, m) = .ok s →
      s = .done (some none, m) ∨ ∃ m', s = .yield (none, m') ∧ m' ≤ m ∧ EndOk m0 m' p) :
    ∀ (l : List (Arg × ℕ)) (m : ℕ) (s : Option (Option ℕ) × ℕ), forIn l (none, m) f = .ok s →
      (s.1 = none ∨ s.1 = some none) ∧ (s.1 = none → s.2 ≤ m ∧ ∀ p ∈ l, EndOk m0 s.2 p) := by
  intro l
  induction l with
  | nil =>
    intro m s h
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨Or.inl rfl, fun _ => ⟨le_refl _, fun p hp => absurd hp List.not_mem_nil⟩⟩
  | cons p l ih =>
    intro m s h
    rw [List.forIn_cons] at h
    cases hfp : f p (none, m) with
    | error e => rw [hfp] at h; simp [bind, Except.bind] at h
    | ok st =>
      rw [hfp] at h
      simp only [bind, Except.bind] at h
      rcases hf p m st hfp with hd | ⟨m', hy, hm', hE⟩
      · subst hd
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨Or.inr rfl, fun hs => by simp at hs⟩
      · subst hy
        obtain ⟨h0, h12⟩ := ih m' s h
        refine ⟨h0, fun hs => ?_⟩
        obtain ⟨h1, h2⟩ := h12 hs
        refine ⟨by omega, fun q hq => ?_⟩
        rcases List.mem_cons.mp hq with rfl | hq
        · exact endOk_mono h1 hE
        · exact h2 q hq

/-- **`pieceEnd`'s end**: at most `n`, and each mapped input's references for members `[m0, m1)` in one affine part. -/
theorem pieceEnd_spec {ins : Array (Arg × ℕ)} {n m0 m1 : ℕ} (h : pieceEnd ins n m0 = .ok (some m1)) :
    m1 ≤ n ∧ ∀ p ∈ ins.toList, EndOk m0 m1 p := by
  unfold pieceEnd at h
  dsimp only at h
  simp only [bind, Except.bind] at h
  split at h
  · simp at h
  · rename_i s hfor
    rw [← Array.forIn_toList] at hfor
    obtain ⟨s1, s2⟩ := s
    have key := forIn_end (m0 := m0) ?_ ins.toList n _ hfor
    · cases s1 with
      | some r =>
        rcases key.1 with h0 | h0
        · simp at h0
        · simp only [Option.some.injEq] at h0
          subst h0
          simp [pure, Except.pure] at h
      | none =>
        simp only [pure, Except.pure, Except.ok.injEq, Option.some.injEq] at h
        subst h
        exact key.2 rfl
    · intro p m s hs
      obtain ⟨a, t⟩ := p
      simp only at hs
      rcases hd : a.dims with _ | ⟨d, L⟩
      · simp only [hd, pure, Except.pure, Except.ok.injEq] at hs
        subst hs
        exact Or.inr ⟨m, rfl, le_refl _, fun d' L' hd' _ => by rw [hd] at hd'; simp at hd'⟩
      · simp only [hd] at hs
        by_cases hL : L = 0
        · simp only [hL, beq_self_eq_true, ite_true, pure, Except.pure, Except.ok.injEq] at hs
          subst hs
          exact Or.inr ⟨m, rfl, le_refl _, fun d' L' hd' hL' => by
            rw [hd] at hd'; simp only [Option.some.injEq, Prod.mk.injEq] at hd'; omega⟩
        · have hL' : (L == 0) = false := by simpa using hL
          simp only [hL', Bool.false_eq_true, ite_false] at hs
          by_cases hk : a.seq.count ≤ memberIx a m0 t
          · simp [hk, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
          · simp only [hk, ite_false] at hs
            cases hpart : a.seq.parts[partAt a.seq (memberIx a m0 t)]! with
            | affine p0 i0 b0 c0 =>
              simp only [hpart, pure, Except.pure, Except.ok.injEq] at hs
              subst hs
              refine Or.inr ⟨_, rfl, Nat.min_le_left _ _, fun d' L' hd' _ => ?_⟩
              rw [hd] at hd'
              simp only [Option.some.injEq, Prod.mk.injEq] at hd'
              obtain ⟨-, rfl⟩ := hd'
              refine ⟨by simp only; omega, ⟨p0, i0, b0, c0, hpart⟩, ?_⟩
              simp only [hpart]
              exact Nat.min_le_right _ _
            | explicit items =>
              simp only [hpart, pure, Except.pure, Except.ok.injEq] at hs
              exact Or.inl hs.symm
            | strided => simp [hpart, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
            | grid => simp [hpart, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs


/-! ## The batch case: a piece's bindings -/

/-- A fold that pushes one element a step, each satisfying `P` of its step's element. -/
theorem foldlM_pushes {α β : Type} {F : Array β → α → Except String (Array β)} {P : α → β → Prop}
    {S : List α} (hF : ∀ bs j bs', j ∈ S → F bs j = .ok bs' → ∃ b, bs' = bs.push b ∧ P j b) :
    ∀ (l : List α) (acc bs : Array β), (∀ j ∈ l, j ∈ S) → l.foldlM F acc = .ok bs →
      ∃ ls, bs.toList = acc.toList ++ ls ∧ List.Forall₂ P l ls := by
  intro l
  induction l with
  | nil =>
    intro acc bs _ h
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨[], by simp, List.Forall₂.nil⟩
  | cons j l ih =>
    intro acc bs hS h
    rw [List.foldlM_cons] at h
    cases hj : F acc j with
    | error e => rw [hj] at h; simp [bind, Except.bind] at h
    | ok bs1 =>
      rw [hj] at h
      simp only [bind, Except.bind] at h
      obtain ⟨b, rfl, hb⟩ := hF acc j bs1 (hS j (by simp)) hj
      obtain ⟨ls, h1, h2⟩ := ih _ _ (fun i hi => hS i (by simp [hi])) h
      exact ⟨b :: ls, by rw [h1]; simp, List.Forall₂.cons hb h2⟩

/-- A fold over a work that pushes one element a step, each satisfying `P` of its step's element at every work `R` of
the last. -/
theorem foldlM_pushesW {α β σ : Type} {F : σ × Array β → α → Except String (σ × Array β)} {R : σ → σ → Prop}
    (hR : ∀ a, R a a) (hT : ∀ a b c, R a b → R b c → R a c) {P : α → β → σ → Prop}
    (hP : ∀ j b w w', P j b w → R w w' → P j b w')
    {S : List α} {w₀ : σ}
    (hF : ∀ w bs j r, j ∈ S → R w₀ w → F (w, bs) j = .ok r → ∃ b, r.2 = bs.push b ∧ R w r.1 ∧ P j b r.1) :
    ∀ (l : List α) (w : σ) (acc : Array β) (r : σ × Array β), (∀ j ∈ l, j ∈ S) → R w₀ w →
      l.foldlM F (w, acc) = .ok r →
      R w r.1 ∧ ∃ ls, r.2.toList = acc.toList ++ ls ∧ List.Forall₂ (fun j b => P j b r.1) l ls := by
  intro l
  induction l with
  | nil =>
    intro w acc r _ _ h
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨hR _, [], by simp, List.Forall₂.nil⟩
  | cons j l ih =>
    intro w acc r hS hw0 h
    rw [List.foldlM_cons] at h
    cases hj : F (w, acc) j with
    | error e => rw [hj] at h; simp [bind, Except.bind] at h
    | ok r1 =>
      rw [hj] at h
      simp only [bind, Except.bind] at h
      obtain ⟨b, h1, hw, hb⟩ := hF w acc j r1 (hS j (by simp)) hw0 hj
      obtain ⟨r11, r12⟩ := r1
      simp only at h1 hw hb
      subst h1
      obtain ⟨hw', ls, h2, h3⟩ := ih _ _ _ (fun i hi => hS i (by simp [hi])) (hT _ _ _ hw0 hw) h
      exact ⟨hT _ _ _ hw hw', b :: ls, by rw [h2]; simp, List.Forall₂.cons (hP _ _ _ _ hb hw') h3⟩

theorem forall₂_map {α β γ : Type} {P : α → β → Prop} {g : β → γ} {f : α → γ} (h : ∀ a b, P a b → g b = f a) :
    ∀ {l : List α} {ls : List β}, List.Forall₂ P l ls → ls.map g = l.map f
  | _, _, .nil => rfl
  | _, _, .cons hab hl => by simp only [List.map_cons, h _ _ hab, forall₂_map h hl]

theorem toList_eq_range {α : Type} [Inhabited α] (xs : Array α) :
    xs.toList = (List.range xs.size).map fun j => xs[j]! := by
  apply List.ext_getElem
  · simp
  · intro i h1 h2
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
    rw [getElem!_pos xs i (by simpa using h1)]

/-- Member `m`'s value of batch input `p`: its argument's reference for `m` (`memberIx`) in the walk. -/
def memberVal (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (m : ℕ) (p : Arg × ℕ) : Bool :=
  refVal xs st vals (p.1.seq.toArray.getD (memberIx p.1 m p.2) default)

theorem memberIx_add {a : Arg} {d L : ℕ} (hd : a.dims = some (d, L)) (m j t : ℕ) :
    memberIx a (m + j) t = memberIx a m t + j * L := by
  simp only [memberIx, hd]
  ring

theorem memberIx_none {a : Arg} (hd : a.dims = none) (m t : ℕ) : memberIx a m t = t := by
  simp only [memberIx, hd]

/-- **A mapped input's stepping binding is each member's reference**: within `pieceEnd`'s end, member `m0 + j`'s. -/
theorem step_input {V : ℕ → Bool} {w : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) {a : Arg} {t d L m0 m1 : ℕ} (hq : seqOk a.seq = true)
    (hd : a.dims = some (d, L)) (hE : EndOk m0 m1 (a, t)) {r : Ref} {b : Bind}
    (hr : getRef a.seq (memberIx a m0 t) = .ok r) (hb : w.stepOf st r L (m1 - m0) = .ok b) :
    b.step % 2 = 0 ∧ ∀ j < m1 - m0, V (b.a + j * b.step) = memberVal xs st vals (m0 + j) (a, t) := by
  obtain ⟨hk, rfl⟩ := getRef_ok hr
  obtain ⟨hs, hv⟩ := stepOf_val hF hP hb
  refine ⟨hs, fun j hj => ?_⟩
  rw [hv j hj]
  unfold memberVal
  simp only
  rw [memberIx_add hd]
  by_cases hL : L = 0
  · subst hL
    simp only [Nat.mul_zero, Nat.add_zero]
    rw [refAt_eq hq hk]
  · obtain ⟨-, ⟨p0, i0, b0, c0, hpart⟩, hle⟩ := hE d L hd hL
    simp only at hpart hle
    obtain ⟨hi, h1, h2⟩ := partAt_spec hq hk
    have hjL : memberIx a m0 t + j * L <
        a.seq.starts[partAt a.seq (memberIx a m0 t)]! + a.seq.parts[partAt a.seq (memberIx a m0 t)]!.count := by
      have h3 : j + 1 ≤ (a.seq.starts[partAt a.seq (memberIx a m0 t)]! +
          a.seq.parts[partAt a.seq (memberIx a m0 t)]!.count - memberIx a m0 t + L - 1) / L := by omega
      rw [Nat.le_div_iff_mul_le (Nat.pos_of_ne_zero hL), Nat.add_mul, Nat.one_mul] at h3
      generalize j * L = q at h3 ⊢
      omega
    rw [refAt_eq hq hk, toArray_getD_part hq hi h1 h2, toArray_getD_part hq hi (le_trans h1 (Nat.le_add_right _ _)) hjL, hpart]
    change refVal xs st vals ⟨p0, i0, b0 + (memberIx a m0 t - a.seq.starts[partAt a.seq (memberIx a m0 t)]!) + j * L⟩ =
      refVal xs st vals ⟨p0, i0, b0 + (memberIx a m0 t + j * L - a.seq.starts[partAt a.seq (memberIx a m0 t)]!)⟩
    have e : b0 + (memberIx a m0 t - a.seq.starts[partAt a.seq (memberIx a m0 t)]!) + j * L =
        b0 + (memberIx a m0 t + j * L - a.seq.starts[partAt a.seq (memberIx a m0 t)]!) := by
      generalize j * L = q
      omega
    rw [e]

/-- **An input committed for one member is its reference**: `Work.commit` of member `m0`'s reference's form. -/
theorem commit_input {V : ℕ → Bool} {w : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) {a : Arg} {k : ℕ} (hq : seqOk a.seq = true) {r : Ref}
    {f : Form} (hr : getRef a.seq k = .ok r) (hf : w.formOf st r = .ok f) (hV : RowsHold V (w.commit f).1) :
    V (w.commit f).2 = refVal xs st vals (a.seq.toArray.getD k default) := by
  obtain ⟨hk, rfl⟩ := getRef_ok hr
  rw [commit_val w f hV, formOf_val hF hP hf, refAt_eq hq hk]

theorem forall₂_mem {α β : Type} {P : α → β → Prop} :
    ∀ {l : List α} {ls : List β}, List.Forall₂ P l ls → ∀ b ∈ ls, ∃ a ∈ l, P a b
  | _, _, .nil => fun _ hb => absurd hb List.not_mem_nil
  | _, _, .cons hab hl => fun b hb => by
    rcases List.mem_cons.mp hb with rfl | hb
    · exact ⟨_, List.mem_cons_self .., hab⟩
    · obtain ⟨a, ha, h⟩ := forall₂_mem hl b hb
      exact ⟨a, List.mem_cons_of_mem _ ha, h⟩


/-! ## The batch case: its pieces -/

/-- Work `b` extends work `a`: the same inputs, outputs and calls, more rows and columns. -/
def WR (a b : Work) : Prop :=
  b.ins = a.ins ∧ b.outs = a.outs ∧ b.calls = a.calls ∧ a.rows.toList <+: b.rows.toList ∧ a.next ≤ b.next

theorem wr_refl (a : Work) : WR a a := ⟨rfl, rfl, rfl, List.prefix_refl _, le_refl _⟩

theorem wr_trans {a b c : Work} (h1 : WR a b) (h2 : WR b c) : WR a c :=
  ⟨h2.1.trans h1.1, h2.2.1.trans h1.2.1, h2.2.2.1.trans h1.2.2.1, h1.2.2.2.1.trans h2.2.2.2.1,
    le_trans h1.2.2.2.2 h2.2.2.2.2⟩

theorem commit_next (w : Work) (f : Form) : w.next ≤ (w.commit f).1.next := by
  unfold Work.commit
  split
  · exact le_refl _
  · simp [Work.row]

theorem commit_wr (w : Work) (f : Form) : WR w (w.commit f).1 :=
  let ⟨a, b, c, d⟩ := commit_frame w f
  ⟨a, b, c, d, commit_next w f⟩

theorem formsOk_wr {V : ℕ → Bool} {w w' : Work} {vals : Array (List Bool)} (hF : FormsOk V w vals) (h : WR w w') :
    FormsOk V w' vals := by
  unfold FormsOk
  rw [h.2.1]
  exact hF

theorem paramsOk_wr {V : ℕ → Bool} {w w' : Work} {xs : List Bool} (hP : ParamsOk V w.ins xs) (h : WR w w') :
    ParamsOk V w'.ins xs := by
  rw [h.1]
  exact hP

theorem memberVal_none {xs : List Bool} {st : Array ℕ} {vals : Array (List Bool)} {p : Arg × ℕ}
    (hd : p.1.dims = none) (m : ℕ) : memberVal xs st vals m p = memberVal xs st vals 0 p := by
  simp only [memberVal, memberIx_none hd]

/-- What `Work.batch` binds input `p` to for every member: nothing when mapped, else a binding stepping 0 whose atom is
its reference's value once the rows past `W` hold. -/
def SharedOk (V : ℕ → Bool) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (W : Work) (p : Arg × ℕ)
    (ob : Option Bind) : Prop :=
  (p.1.dims.isSome = true → ob = none) ∧
  (p.1.dims = none → ∃ b, ob = some b ∧ b.step = 0 ∧
    ∀ W' : Work, W.rows.toList <+: W'.rows.toList → RowsHold V W' → V b.a = memberVal xs st vals 0 p)

/-- A piece's bindings: each stepping an even number of atoms, copy `j`'s the batch's inputs for member `m0 + j`. -/
def PieceOk (V : ℕ → Bool) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (ins : Array (Arg × ℕ))
    (pc : ℕ × ℕ × Array Bind) : Prop :=
  (∀ b ∈ pc.2.2.toList, b.step % 2 = 0) ∧
    ∀ j < pc.2.1, pc.2.2.toList.map (fun b => V (b.a + j * b.step)) = ins.toList.map (memberVal xs st vals (pc.1 + j))

theorem getElem!_mem {α : Type} [Inhabited α] {xs : Array α} {j : ℕ} (hj : j < xs.size) : xs[j]! ∈ xs.toList := by
  rw [getElem!_pos xs j hj]
  exact Array.getElem_mem_toList hj

/-- **`Work.pieces` covers members `[m0, n)`**, each by a piece whose bindings are its members' inputs. -/
theorem pieces_spec {V : ℕ → Bool} {w0 : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w0 vals) (hP : ParamsOk V w0.ins xs) {ins : Array (Arg × ℕ)} {shared : Array (Option Bind)}
    (hq : ∀ p ∈ ins.toList, seqOk p.1.seq = true) {Wsh : Work}
    (hsh : ∀ j < ins.size, SharedOk V xs st vals Wsh ins[j]! shared[j]!) (hWsh : WR Wsh w0) {n : ℕ} :
    ∀ k m0 (w : Work) acc r, n - m0 = k → WR w0 w → w.pieces st ins shared n m0 acc = .ok r →
      WR w r.1 ∧ acc.toList <+: r.2.toList ∧
      ∀ W : Work, r.1.rows.toList <+: W.rows.toList → RowsHold V W → ∀ m, m0 ≤ m → m < n →
        ∃ pc ∈ r.2.toList, pc.1 ≤ m ∧ m < pc.1 + pc.2.1 ∧ PieceOk V xs st vals ins pc := by
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
  intro m0 w acc r hk hw h
  have hFw := formsOk_wr hF hw
  have hPw := paramsOk_wr hP hw
  rw [Work.pieces.eq_1] at h
  by_cases hn : n ≤ m0
  · rw [dif_pos hn] at h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨wr_refl _, List.prefix_refl _, fun _ _ _ m h1 h2 => absurd (lt_of_le_of_lt h1 h2) (by omega)⟩
  · rw [dif_neg hn] at h
    simp only [bind, Except.bind] at h
    cases hpe : pieceEnd ins n m0 with
    | error e => rw [hpe] at h; simp at h
    | ok o =>
      rw [hpe] at h
      simp only at h
      by_cases h1 : m0 + 1 < o.getD (m0 + 1)
      · rw [dif_pos h1] at h
        cases o with
        | none => simp at h1
        | some m1 =>
          simp only [Option.getD_some] at h1 h
          obtain ⟨hm1n, hE⟩ := pieceEnd_spec hpe
          split at h
          · simp at h
          rename_i bs hbs
          obtain ⟨ih1, ih2, ih3⟩ := ih (n - m1) (by omega) m1 w _ r rfl hw h
          refine ⟨ih1, (List.prefix_append _ _).trans (by simpa [Array.toList_push] using ih2), ?_⟩
          intro W hW hR m hm0 hmn
          by_cases hm : m < m1
          · refine ⟨(m0, m1 - m0, bs), ih2.subset (by simp [Array.toList_push]), hm0,
              by simp only; omega, ?_⟩
            have hWw : w.rows.toList <+: W.rows.toList := ih1.2.2.2.1.trans hW
            obtain ⟨ls, hl1, hl2⟩ := foldlM_pushes
              (P := fun (j : ℕ) (b : Bind) => b.step % 2 = 0 ∧
                ∀ jj < m1 - m0, V (b.a + jj * b.step) = memberVal xs st vals (m0 + jj) ins[j]!)
              (S := List.range ins.size) (fun bs j bs' hj hF' => by
                have hj' : j < ins.size := List.mem_range.mp hj
                cases hs : shared[j]! with
                | some b =>
                  rw [hs] at hF'
                  simp only [pure, Except.pure, Except.ok.injEq] at hF'
                  subst hF'
                  refine ⟨b, rfl, ?_⟩
                  obtain ⟨hs1, hs2⟩ := hsh j hj'
                  have hd : ins[j]!.1.dims = none := by
                    cases hd' : ins[j]!.1.dims with
                    | none => rfl
                    | some _ => rw [hs1 (by simp [hd'])] at hs; simp at hs
                  obtain ⟨b', hb', hst, hv⟩ := hs2 hd
                  rw [hs] at hb'
                  simp only [Option.some.injEq] at hb'
                  subst hb'
                  refine ⟨by omega, fun jj _ => ?_⟩
                  rw [hst, Nat.mul_zero, Nat.add_zero, memberVal_none hd,
                    hv W ((hWsh.2.2.2.1.trans hw.2.2.2.1).trans hWw) hR]
                | none =>
                  rw [hs] at hF'
                  rcases hp : ins[j]! with ⟨a, t⟩
                  try simp only [hp] at hF'
                  try simp only [bind, Except.bind] at hF'
                  split at hF'
                  · simp at hF'
                  rename_i rf hr
                  split at hF'
                  · simp at hF'
                  rename_i b hb
                  simp only [pure, Except.pure, Except.ok.injEq] at hF'
                  subst hF'
                  refine ⟨b, rfl, ?_⟩
                  have hmem : (a, t) ∈ ins.toList := hp ▸ getElem!_mem hj'
                  cases hd : a.dims with
                  | none =>
                    obtain ⟨b', hb', -⟩ := (hsh j hj').2 (by rw [hp]; exact hd)
                    rw [hs] at hb'
                    simp at hb'
                  | some dL =>
                    obtain ⟨d, L⟩ := dL
                    simp only [hd, Option.map_some, Option.getD_some] at hb
                    obtain ⟨he, hv⟩ := step_input hFw hPw (hq _ hmem) hd (hE _ hmem) hr hb
                    exact ⟨he, fun jj hjj => by rw [hv jj hjj]⟩)
              (List.range ins.size) #[] bs (fun _ h => h) hbs
            simp only [List.nil_append] at hl1
            refine ⟨fun b hb => ?_, fun jj hjj => ?_⟩
            · simp only at hb
              rw [hl1] at hb
              obtain ⟨j, -, hj⟩ := forall₂_mem hl2 b hb
              exact hj.1
            · simp only at hjj ⊢
              rw [hl1, forall₂_map (f := fun j => memberVal xs st vals (m0 + jj) ins[j]!)
                (fun j b hb => hb.2 jj hjj) hl2, toList_eq_range ins, List.map_map]
              rfl
          · exact ih3 W hW hR m (by omega) hmn
      · rw [dif_neg h1] at h
        split at h
        · simp at h
        rename_i r1 hr1
        obtain ⟨w', bs⟩ := r1
        simp only at h
        obtain ⟨hww', ls, hl1, hl2⟩ := foldlM_pushesW (R := WR) wr_refl (fun _ _ _ => wr_trans)
          (P := fun (j : ℕ) (b : Bind) (W' : Work) => b.step = 0 ∧ ∀ W : Work, W'.rows.toList <+: W.rows.toList → RowsHold V W →
            V b.a = memberVal xs st vals m0 ins[j]!)
          (fun j b W1 W2 hb h12 => ⟨hb.1, fun W hW hR => hb.2 W (h12.2.2.2.1.trans hW) hR⟩)
          (S := List.range ins.size) (w₀ := w0) (fun w1 bs j r1 hj hw1 hF' => by
            have hj' : j < ins.size := List.mem_range.mp hj
            cases hs : shared[j]! with
            | some b =>
              rw [hs] at hF'
              simp only [pure, Except.pure, Except.ok.injEq] at hF'
              subst hF'
              refine ⟨b, rfl, wr_refl _, ?_⟩
              obtain ⟨hs1, hs2⟩ := hsh j hj'
              have hd : ins[j]!.1.dims = none := by
                cases hd' : ins[j]!.1.dims with
                | none => rfl
                | some _ => rw [hs1 (by simp [hd'])] at hs; simp at hs
              obtain ⟨b', hb', hst, hv⟩ := hs2 hd
              rw [hs] at hb'
              simp only [Option.some.injEq] at hb'
              subst hb'
              exact ⟨hst, fun W hW hR => by
                rw [memberVal_none hd, hv W ((hWsh.2.2.2.1.trans hw1.2.2.2.1).trans hW) hR]⟩
            | none =>
              rw [hs] at hF'
              rcases hp : ins[j]! with ⟨a, t⟩
              try simp only [hp] at hF'
              try simp only [bind, Except.bind] at hF'
              split at hF'
              · simp at hF'
              rename_i rf hr
              split at hF'
              · simp at hF'
              rename_i f hf
              simp only [pure, Except.pure, Except.ok.injEq] at hF'
              subst hF'
              have hmem : (a, t) ∈ ins.toList := hp ▸ getElem!_mem hj'
              refine ⟨⟨(w1.commit f).2, 0⟩, rfl, commit_wr w1 f, rfl, fun W hW hR => ?_⟩
              exact commit_input (formsOk_wr hF hw1) (paramsOk_wr hP hw1) (hq _ hmem) hr hf
                (rowsHold_of_prefix hW hR))
          (List.range ins.size) w #[] (w', bs) (fun _ h => h) hw hr1
        simp only [List.nil_append] at hl1
        obtain ⟨ih1, ih2, ih3⟩ := ih (n - (m0 + 1)) (by omega) (m0 + 1) w' _ r rfl (wr_trans hw hww') h
        refine ⟨wr_trans hww' ih1, (List.prefix_append _ _).trans (by simpa [Array.toList_push] using ih2), ?_⟩
        intro W hW hR m hm0 hmn
        by_cases hm : m = m0
        · refine ⟨(m0, 1, bs), ih2.subset (by simp [Array.toList_push]), by simp only; omega, by simp only; omega, ?_⟩
          have hWw : w'.rows.toList <+: W.rows.toList := ih1.2.2.2.1.trans hW
          refine ⟨fun b hb => ?_, fun jj hjj => ?_⟩
          · simp only at hb
            rw [hl1] at hb
            obtain ⟨j, -, hj⟩ := forall₂_mem hl2 b hb
            rw [hj.1]
          · simp only at hjj ⊢
            have hjj0 : jj = 0 := by omega
            subst hjj0
            rw [hl1, forall₂_map (f := fun j => memberVal xs st vals (m0 + 0) ins[j]!)
              (fun j b hb => by simp only [Nat.zero_mul, Nat.add_zero]; exact hb.2 W hWw hR) hl2,
              toList_eq_range ins, List.map_map]
            rfl
        · exact ih3 W hW hR m (by omega) hmn


/-! ## The batch case: its inputs, its shared bindings, a batch of none -/

/-- A fold over a work whose every step extends it, `R` of the first. -/
theorem foldlM_frame {α β : Type} {F : Work × β → α → Except String (Work × β)} {S : List α} {w₀ : Work}
    (hF : ∀ s a r, a ∈ S → WR w₀ s.1 → F s a = .ok r → WR s.1 r.1) :
    ∀ (l : List α) (s r : Work × β), (∀ a ∈ l, a ∈ S) → WR w₀ s.1 → l.foldlM F s = .ok r → WR s.1 r.1 := by
  intro l
  induction l with
  | nil =>
    intro s r _ _ h
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact wr_refl _
  | cons a l ih =>
    intro s r hS hw h
    rw [List.foldlM_cons] at h
    cases ha : F s a with
    | error e => rw [ha] at h; simp [bind, Except.bind] at h
    | ok s1 =>
      rw [ha] at h
      simp only [bind, Except.bind] at h
      have h1 := hF s a s1 (hS a (by simp)) hw ha
      exact wr_trans h1 (ih s1 r (fun b hb => hS b (by simp [hb])) (wr_trans hw h1) h)

/-- `Work.binds` only extends the work. -/
theorem binds_wr {w w' : Work} {st : Array ℕ} {callee : Defn} {x : Node} {bs : Array Bind}
    (hb : w.binds st callee x = .ok (w', bs)) : WR w w' := by
  unfold Work.binds at hb
  refine foldlM_frame (S := List.range x.args.size) (w₀ := w) (fun s i r _ _ h => ?_) _ (w, #[]) (w', bs)
    (fun _ h => h) (wr_refl _) hb
  refine foldlM_frame (S := List.range _) (w₀ := s.1) (fun s' t r' _ _ h' => ?_) _ s r (fun _ h => h) (wr_refl _) h
  obtain ⟨w1, out⟩ := s'
  simp only [bind, Except.bind, pure, Except.pure] at h'
  repeat' (first | (simp at h'; done) | split at h')
  all_goals
    simp only [Except.ok.injEq] at h'
    subst h'
    first | exact wr_refl _ | exact commit_wr _ _

theorem foldl_push_flat {α β : Type} (g : α → List β) :
    ∀ (l : List α) (acc : Array β), (l.foldl (fun acc i => (g i).foldl (fun acc t => acc.push t) acc) acc).toList =
      acc.toList ++ l.flatMap g
  | [], acc => by simp
  | i :: l, acc => by
    simp only [List.foldl_cons, List.flatMap_cons]
    rw [foldl_push_flat g l]
    have : ((g i).foldl (fun acc t => acc.push t) acc).toList = acc.toList ++ g i := by
      have := foldl_push_map id (g i) acc
      simp only [id] at this
      rw [this]
      simp
    rw [this, List.append_assoc]

/-- `Work.batch`'s inputs: each argument's references, as many as its parameter's leaves when mapped. -/
theorem batchIns_toList (callee : Defn) (x : Node) :
    ((List.range x.args.size).foldl (fun acc i =>
      let a := x.args[i]!
      let cnt := if a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
      (List.range cnt).foldl (fun acc t => acc.push (a, t)) acc) #[]).toList =
    (List.range x.args.size).flatMap fun i =>
      (List.range (if (x.args[i]!).dims.isSome then (callee.params[i]!.2).leaves else (x.args[i]!).seq.count)).map
        fun t => (x.args[i]!, t) := by
  have := foldl_push_flat (fun i => (List.range (if (x.args[i]!).dims.isSome then (callee.params[i]!.2).leaves
    else (x.args[i]!).seq.count)).map fun t => (x.args[i]!, t)) (List.range x.args.size) #[]
  simp only [List.foldl_map] at this
  simpa using this

/-- Member `m`'s inputs as the walk reads them (`walkStep`'s batch). -/
def batchArgs (v : Ref → Bool) (callee : Defn) (x : Node) (m : ℕ) : List Bool :=
  (List.range x.args.size).flatMap fun i =>
    match (x.args.getD i default).dims with
    | none => argVals v (x.args.getD i default) (x.args.getD i default).seq.count id
    | some (dd, L) => argVals v (x.args.getD i default) ((callee.params.getD i default).2.leaves)
        fun t => (t / L * dd) * L + t % L + m * L

theorem getElem!_eq_getD {α : Type} [Inhabited α] (xs : Array α) (i : ℕ) : xs[i]! = xs.getD i default := by
  by_cases h : i < xs.size
  · rw [getElem!_pos xs i h, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h, Option.getD_some]
  · rw [getElem!_neg xs i h, Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega), Option.getD_none]

theorem batchArgs_eq (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node) (m : ℕ) :
    ((List.range x.args.size).flatMap fun i =>
      (List.range (if (x.args[i]!).dims.isSome then (callee.params[i]!.2).leaves else (x.args[i]!).seq.count)).map
        fun t => (x.args[i]!, t)).map (memberVal xs st vals m) = batchArgs (refVal xs st vals) callee x m := by
  rw [List.map_flatMap]
  unfold batchArgs
  congr 1
  funext i
  rw [getElem!_eq_getD x.args i, getElem!_eq_getD callee.params i]
  cases hd : (x.args.getD i default).dims with
  | none =>
    simp only [hd, Option.isSome_none, Bool.false_eq_true, ite_false, List.map_map, argVals]
    apply List.map_congr_left
    intro t _
    simp only [Function.comp, memberVal, memberIx_none hd, id]
  | some dL =>
    obtain ⟨dd, L⟩ := dL
    simp only [hd, Option.isSome_some, ite_true, List.map_map, argVals]
    apply List.map_congr_left
    intro t _
    simp only [Function.comp, memberVal, memberIx, hd]
    congr 2
    ring

theorem ins_seq {callee : Defn} {x : Node} (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true) :
    ∀ p ∈ ((List.range x.args.size).flatMap fun i =>
      (List.range (if (x.args[i]!).dims.isSome then (callee.params[i]!.2).leaves else (x.args[i]!).seq.count)).map
        fun t => (x.args[i]!, t)), seqOk p.1.seq = true := by
  intro p hp
  obtain ⟨i, hi, hp⟩ := List.mem_flatMap.mp hp
  obtain ⟨t, -, rfl⟩ := List.mem_map.mp hp
  exact hseq _ (getElem!_mem (List.mem_range.mp hi))

theorem formsOk_outs {V : ℕ → Bool} {w w' : Work} {vals : Array (List Bool)} (h : w.outs = w'.outs)
    (hF : FormsOk V w vals) : FormsOk V w' vals := by
  unfold FormsOk at hF ⊢
  rw [← h]
  exact hF

theorem forall₂_getElem! {α β : Type} [Inhabited α] [Inhabited β] {R : α → β → Prop} {xs : Array α} {ys : Array β}
    (h : List.Forall₂ R xs.toList ys.toList) {j : ℕ} (hj : j < xs.size) : R xs[j]! ys[j]! := by
  have hl := h.length_eq
  simp only [Array.length_toList] at hl
  rw [getElem!_pos xs j hj, getElem!_pos ys j (by omega)]
  have := (List.forall₂_iff_get.mp h).2 j (by simpa using hj) (by simp; omega)
  simpa using this

/-- **A batch node**: its members' copies one region after the work, member `m`'s copy a copy of a CALL binding the
walk's inputs for member `m`, and its leaves the members' returns one after another. -/
theorem batch_node {V : ℕ → Bool} {w w' : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    {callee : Defn} {x : Node} {e cols : ℕ} {rets : Array ℕ}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true)
    (hb : w.batch st callee x e cols rets = .ok w') :
    ∃ base, w'.ins = w.ins ∧ w.rows.toList <+: w'.rows.toList ∧ w.calls.toList <+: w'.calls.toList ∧
      w.next ≤ base ∧ w'.next = base + x.n * cols ∧ w'.outs = w.outs.push (.copies base cols x.n rets) ∧
      (RowsHold V w' → ∀ m < x.n, ∃ p ∈ w'.calls.toList, p.def_ = e ∧ p.stride = cols ∧ ∃ j < p.count,
        p.base + j * cols = base + m * cols ∧ (∀ b ∈ p.binds.toList, b.step % 2 = 0) ∧
        p.binds.toList.map (fun b => V (b.a + j * b.step)) = batchArgs (refVal xs st vals) callee x m) := by
  unfold Work.batch at hb
  by_cases hn0 : (x.n == 0) = true
  · rw [ite_eq_left hn0] at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · simp at hb
    rename_i wb hwb
    obtain ⟨w1, bs⟩ := wb
    simp only [pure, Except.pure, Except.ok.injEq] at hb
    subst hb
    have hW := binds_wr hwb
    have hn : x.n = 0 := by simpa using hn0
    refine ⟨w1.next, hW.1, hW.2.2.2.1, ?_, hW.2.2.2.2, by simp [Work.place, hn], by simp [Work.place, hn, hW.2.1],
      fun _ m hm => absurd hm (by omega)⟩
    simp only [Work.place, Array.toList_push, hW.2.2.1]
    exact List.prefix_append _ _
  · rw [ite_eq_right hn0] at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · simp at hb
    rename_i ws hws
    obtain ⟨w1, shared⟩ := ws
    simp only at hb
    split at hb
    · simp at hb
    rename_i pr hpr
    obtain ⟨w2, ps⟩ := pr
    simp only [pure, Except.pure, Except.ok.injEq] at hb
    subst hb
    have hq := ins_seq (callee := callee) hseq
    rw [← batchIns_toList] at hq
    rw [← Array.foldlM_toList] at hws
    obtain ⟨hW1, ls, hl1, hl2⟩ := foldlM_pushesW (R := WR) wr_refl (fun _ _ _ => wr_trans)
      (P := fun p ob W' => SharedOk V xs st vals W' p ob)
      (fun p ob W1 W2 h h12 => ⟨h.1, fun hd => by
        obtain ⟨b, h1, h2, h3⟩ := h.2 hd
        exact ⟨b, h1, h2, fun W' hW hR => h3 W' (h12.2.2.2.1.trans hW) hR⟩⟩)
      (S := _) (w₀ := w) (fun w0' acc p r hp hw0 hstep => by
        obtain ⟨a, t⟩ := p
        simp only at hstep
        by_cases hs : a.dims.isSome = true
        · rw [ite_eq_left hs] at hstep
          cases hstep
          exact ⟨none, rfl, wr_refl _, fun _ => rfl, fun hd => by simp only at hd; simp [hd] at hs⟩
        · rw [ite_eq_right hs] at hstep
          try simp only [bind, Except.bind] at hstep
          split at hstep
          · simp at hstep
          rename_i rf hr
          split at hstep
          · simp at hstep
          rename_i f hf
          cases hstep
          refine ⟨some ⟨(w0'.commit f).2, 0⟩, rfl, commit_wr _ _, fun h => absurd h hs, fun hd =>
            ⟨_, rfl, rfl, fun W' hW hR => ?_⟩⟩
          simp only [memberVal, memberIx_none hd]
          exact commit_input (formsOk_wr hF hw0) (paramsOk_wr hP hw0) (hq _ hp) hr hf (rowsHold_of_prefix hW hR))
      _ w #[] (w1, shared) (fun _ h => h) (wr_refl _) hws
    simp only [List.nil_append] at hl1
    rw [← hl1] at hl2
    have hsh : ∀ j < _, SharedOk V xs st vals w1 _ _ := fun j hj => forall₂_getElem! hl2 hj
    obtain ⟨hW2, -, hcov⟩ := pieces_spec (formsOk_wr hF hW1) (paramsOk_wr hP hW1) hq hsh (wr_refl _) _ 0 w1 #[]
      (w2, ps) rfl (wr_refl _) hpr
    simp only at hW2 hcov
    have hW := wr_trans hW1 hW2
    refine ⟨w2.next, hW.1, hW.2.2.2.1, ?_, hW.2.2.2.2, rfl, by simp [hW.2.1], fun hR m hm => ?_⟩
    · simp only [Array.toList_append, hW.2.2.1]
      exact List.prefix_append _ _
    obtain ⟨pc, hpc, hle, hlt, hok⟩ := hcov _ (List.prefix_refl _) hR m (Nat.zero_le _) hm
    obtain ⟨m0, cnt, bs⟩ := pc
    have hle' : m0 ≤ m := hle
    have hlt' : m < m0 + cnt := hlt
    refine ⟨⟨e, cnt, w2.next + m0 * cols, cols, bs⟩, ?_, rfl, rfl, m - m0, by show m - m0 < cnt; omega, ?_, hok.1, ?_⟩
    · simp only [Array.toList_append, Array.toList_map, List.mem_append, List.mem_map]
      exact Or.inr ⟨(m0, cnt, bs), hpc, rfl⟩
    · simp only
      rw [Nat.add_assoc, ← Nat.add_mul, Nat.add_sub_cancel' hle']
    · simp only
      rw [hok.2 (m - m0) (by show m - m0 < cnt; omega), Nat.add_sub_cancel' hle', batchIns_toList, batchArgs_eq]

/-- **A batch node's leaves are its members' returns** one after another, `rets.size` each. -/
theorem batch_forms {V : ℕ → Bool} {w w' : Work} {vals : Array (List Bool)} {base n cols : ℕ} {rets : Array ℕ}
    (hF : FormsOk V w vals) (ho : w'.outs = w.outs.push (.copies base cols n rets)) {l : ℕ → List Bool}
    (hlen : ∀ m < n, (l m).length = rets.size)
    (hl : ∀ m < n, ∀ k < rets.size, V (2 * (base + m * cols + rets[k]!)) = (l m).getD k false) :
    FormsOk V w' (vals.push ((List.range n).map l).flatten) := by
  refine formsOk_outs (w := { w with outs := w.outs.push (.copies base cols n rets) }) ho.symm
    (formsOk_push hF rfl ?_)
  intro j hj
  have hL : 0 < rets.size := Nat.pos_of_ne_zero (fun h0 => by rw [h0] at hj; simp at hj)
  have hjn : j / rets.size < n := by rw [Nat.div_lt_iff_lt_mul hL]; exact hj
  rw [getD_flatten_eq (L := rets.size) (fun l' hl' => by
      obtain ⟨m, hm, rfl⟩ := List.mem_map.mp hl'
      exact hlen m (List.mem_range.mp hm)) (by simpa using hj),
    List.getElem?_map, List.getElem?_range hjn]
  simp only [Option.map_some, Option.getD_some]
  rw [← hl _ hjn _ (Nat.mod_lt _ hL)]


/-- **A primitive's template Definition computes its gate** (`gateDef`): its one return column, once its rows hold, is
the gate on its inputs. `hcol` is `Def.wellFormed`'s rows inside the columns, which `selfCheck` holds it to. -/
theorem gateTmpl {g : Gate} {ins : ℕ} {w : Work} {o : Form}
    (hop : ({} : Work).op g ((Array.range ins).map Form.input) = .ok (w, o)) {v : ℕ → Bool} {inp : List Bool}
    (hlen : inp.length = ins)
    (hcol : ∀ r ∈ ((w.commitOwn o).1.finish ins).rows.toList, r.1 < ((w.commitOwn o).1.finish ins).cols)
    (hH : ∀ r ∈ ((w.commitOwn o).1.finish ins).rows.toList,
      v r.1 = rowVal (((w.commitOwn o).1.finish ins).refV v inp true) r) :
    (w.commitOwn o).2 < ((w.commitOwn o).1.finish ins).cols ∧ v (w.commitOwn o).2 = gateVal g inp := by
  have hB := op_bound hop (fun o' ho => by
    rw [Array.toList_map, List.mem_map] at ho
    obtain ⟨k, -, rfl⟩ := ho
    exact formB_input _ _)
  have hR := rowsHold_finish hcol hH
  obtain ⟨-, -, -, hpre, -, hc, hval⟩ := commitOwn_spec (V := atomV ((w.commitOwn o).1.finish ins) v inp) w o hB.2
  refine ⟨hc, ?_⟩
  have h1 := hval hR
  rw [atomV_own hc] at h1
  rw [h1]
  obtain ⟨-, h2⟩ := op_val hop (rowsHold_of_prefix hpre hR)
  rw [h2]
  congr 1
  rw [Array.toList_map, List.map_map, Array.toList_range]
  apply List.ext_getElem
  · simp [hlen]
  · intro i hi1 hi2
    have hi : i < ins := by simpa using hi1
    simp only [List.getElem_map, List.getElem_range, Function.comp, val_input]
    rw [atomV_input (D := (w.commitOwn o).1.finish ins) (show i < ins from hi), List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem hi2, Option.getD_some]


/-! ## The scan case: its loops, a stepping `xs` argument -/

/-- A loop every step of which yields, keeping `Inv` of the elements it has run through. -/
theorem forIn_inv {α σ : Type} {f : α → σ → Except String (ForInStep σ)} {S : List α} {Inv : List α → σ → Prop}
    (hf : ∀ pre a s r, pre ++ [a] <+: S → Inv pre s → f a s = .ok r → ∃ s', r = .yield s' ∧ Inv (pre ++ [a]) s') :
    ∀ (l pre : List α) (s r : σ), pre ++ l = S → Inv pre s → forIn l s f = .ok r → Inv S r := by
  intro l
  induction l with
  | nil =>
    intro pre s r hS hI h
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    rw [List.append_nil] at hS
    subst hS
    exact hI
  | cons a l ih =>
    intro pre s r hS hI h
    rw [List.forIn_cons] at h
    cases ha : f a s with
    | error e => rw [ha] at h; simp [bind, Except.bind] at h
    | ok r1 =>
      rw [ha] at h
      obtain ⟨s', rfl, hI'⟩ := hf pre a s r1 (by rw [← hS]; exact ⟨l, by simp⟩) hI ha
      simp only [bind, Except.bind] at h
      exact ih (pre ++ [a]) s' r (by rw [← hS]; simp) hI' h

theorem forIn_range {σ : Type} (n : ℕ) (s : σ) (f : ℕ → σ → Except String (ForInStep σ)) :
    forIn [:n] s f = forIn (List.range n) s f := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', List.range_eq_range']
  simp [Std.Legacy.Range.size]

theorem prefix_range {l : List ℕ} {n : ℕ} (h : l <+: List.range n) : l = List.range l.length := by
  rw [List.prefix_iff_eq_take.mp h, List.take_range, List.length_range]

theorem prefix_range_last {pre : List ℕ} {i n : ℕ} (h : pre ++ [i] <+: List.range n) :
    pre = List.range pre.length ∧ i = pre.length := by
  have h1 := prefix_range h
  simp only [List.length_append, List.length_singleton, List.range_succ] at h1
  have h2 := List.append_inj h1 (by simp)
  simp only [List.cons.injEq, and_true] at h2
  exact ⟨h2.1, h2.2⟩

theorem forall₂_append {α β : Type} {R : α → β → Prop} :
    ∀ {l1 : List α} {k1 : List β} {l2 : List α} {k2 : List β}, List.Forall₂ R l1 k1 → List.Forall₂ R l2 k2 →
      List.Forall₂ R (l1 ++ l2) (k1 ++ k2)
  | _, _, _, _, .nil, h => h
  | _, _, _, _, .cons hab hl, h => .cons hab (forall₂_append hl h)

/-- The part of `q` that `partIx` names holds reference `k`. -/
theorem partIx_mem {q : Seq} (hq : seqOk q = true) {k : ℕ} (hk : k < q.count) :
    partIx q k < q.parts.size ∧ q.starts[partIx q k]! ≤ k ∧ k < q.starts[partIx q k]! + q.parts[partIx q k]!.count := by
  have he := seqOk_eq hq
  obtain ⟨_, hsz, hst, hcnt⟩ := InferenceOnlyVerifier.Properties.SecurityProofs.Walk.ofParts_spec q.parts
  rw [he] at hsz hst hcnt
  have hpos : 0 < q.parts.size := by
    rcases Nat.eq_zero_or_pos q.parts.size with h | h
    · have : q.parts.toList = [] := by simpa [← Array.length_toList] using h
      rw [hcnt, this] at hk
      simp at hk
    · exact h
  have h0 : q.starts[0]! = 0 := by rw [hst 0 hpos]; simp
  obtain ⟨r1, r2, r3⟩ := partIx_spec q k (by omega) q.parts.size hpos
  have hlen : ∀ i < q.parts.size, q.starts[i]! + q.parts[i]!.count = ((q.parts.toList.take (i + 1)).map Part.count).sum :=
    fun i hi => by rw [hst i hi, InferenceOnlyVerifier.Properties.SecurityProofs.Walk.sum_take_succ_map q.parts hi]
  have hpi : partIx q k = (List.range q.parts.size).foldl (fun i x => if q.starts[x]! ≤ k then x else i) 0 := rfl
  rw [hpi]
  refine ⟨r1, r2, ?_⟩
  set r := (List.range q.parts.size).foldl (fun i x => if q.starts[x]! ≤ k then x else i) 0
  rcases Nat.lt_or_ge (r + 1) q.parts.size with hl | hg
  · have := r3 (r + 1) (by omega) hl
    rw [hst (r + 1) hl] at this
    rw [hlen r r1]
    exact this
  · have hre : r + 1 = q.parts.toList.length := by simp; omega
    rw [hlen r r1, hre, List.take_length, ← hcnt]
    exact hk

/-- **A scan's `xs` reference stepping its element's leaves an iteration**: within one affine part, iteration `j`'s
reference is the walk's `k0 + j · L`. -/
theorem stepping_spec {q : Seq} (hq : seqOk q = true) {k0 n L : ℕ} {what : String} {r : Ref} {s : ℕ}
    (h : stepping q k0 n L what = .ok (r, s)) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) :
    (n ≤ 1 → s = 0 ∧ k0 < q.count ∧ r = refAt q k0) ∧
    (1 < n → s = L ∧ ∀ j < n, refVal xs st vals ⟨r.p, r.i, r.j + j * L⟩ =
      refVal xs st vals (q.toArray.getD (k0 + j * L) default)) := by
  unfold stepping at h
  by_cases hn : n ≤ 1
  · rw [ite_eq_left hn] at h
    refine ⟨fun _ => ?_, fun h1 => absurd h1 (by omega)⟩
    cases hg : getRef q k0 with
    | error e => rw [hg] at h; simp [Functor.map, Except.map] at h
    | ok r0 =>
      rw [hg] at h
      simp only [Functor.map, Except.map, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨hk, rfl⟩ := getRef_ok hg
      exact ⟨rfl, hk, rfl⟩
  · rw [ite_eq_right hn] at h
    refine ⟨fun h1 => absurd h1 hn, fun _ => ?_⟩
    by_cases hc : q.count ≤ k0 + (n - 1) * L
    · rw [ite_eq_left hc] at h
      simp [throw, throwThe, MonadExceptOf.throw] at h
    · rw [ite_eq_right hc] at h
      by_cases hi : (partIx q (k0 + (n - 1) * L) != partIx q k0) = true
      · rw [ite_eq_left hi] at h
        simp [throw, throwThe, MonadExceptOf.throw] at h
      · rw [ite_eq_right hi] at h
        simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hi
        cases hpart : q.parts[partIx q k0]! with
        | affine p0 i0 b0 c0 =>
          simp only [hpart, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          refine ⟨rfl, fun j hj => ?_⟩
          have hjL : j * L ≤ (n - 1) * L := Nat.mul_le_mul_right L (by omega)
          obtain ⟨a1, a2, a3⟩ := partIx_mem hq (k := k0) (by omega)
          obtain ⟨-, -, b3⟩ := partIx_mem hq (k := k0 + (n - 1) * L) (by omega)
          rw [hi] at b3
          rw [toArray_getD_part hq a1 (le_trans a2 (Nat.le_add_right _ _)) (by omega), hpart]
          change refVal xs st vals ⟨p0, i0, b0 + (k0 - q.starts[partIx q k0]!) + j * L⟩ =
            refVal xs st vals ⟨p0, i0, b0 + (k0 + j * L - q.starts[partIx q k0]!)⟩
          have e : b0 + (k0 - q.starts[partIx q k0]!) + j * L = b0 + (k0 + j * L - q.starts[partIx q k0]!) := by
            generalize j * L = u at hjL ⊢
            omega
          rw [e]
        | strided => simp [hpart, throw, throwThe, MonadExceptOf.throw] at h
        | explicit => simp [hpart, throw, throwThe, MonadExceptOf.throw] at h
        | grid => simp [hpart, throw, throwThe, MonadExceptOf.throw] at h

/-- **`Work.bindOne`**: a reference committed when it steps 0, else bound stepping (`stepOf_val`). -/
theorem bindOne_spec {V : ℕ → Bool} {w w1 : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) {r : Ref} {s n : ℕ} {b : Bind}
    (h : w.bindOne st r s n = .ok (w1, b)) :
    WR w w1 ∧ b.step % 2 = 0 ∧
      (s = 0 → b.step = 0 ∧ ∀ W : Work, w1.rows.toList <+: W.rows.toList → RowsHold V W → V b.a = refVal xs st vals r) ∧
      (s ≠ 0 → ∀ j < n, V (b.a + j * b.step) = refVal xs st vals ⟨r.p, r.i, r.j + j * s⟩) := by
  unfold Work.bindOne at h
  by_cases hs : s = 0
  · subst hs
    simp only [beq_self_eq_true, ite_true, bind, Except.bind] at h
    split at h
    · simp at h
    rename_i f hf
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨commit_wr w f, rfl, fun _ => ⟨rfl, fun W hW hR => ?_⟩, fun h => absurd rfl h⟩
    rw [commit_val w f (rowsHold_of_prefix hW hR), formOf_val hF hP hf]
  · have hs' : (s == 0) = false := by simpa using hs
    simp only [hs', Bool.false_eq_true, ite_false, bind, Except.bind] at h
    split at h
    · simp at h
    rename_i b' hb
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨he, hv⟩ := stepOf_val hF hP hb
    exact ⟨wr_refl _, he, fun h => absurd h hs, fun _ j hj => hv j hj⟩


/-! ## The scan case: its bindings -/

/-- Iteration `m`'s value of leaf `t` of a scan's argument `i`, as the walk reads it: an `xs` argument's reference
stepped `m` elements, any other's reference `t`. -/
def scanArg (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node) (i t m : ℕ) : Bool :=
  refVal xs st vals ((x.args[i]!).seq.toArray.getD
    (if 0 < i ∧ i ≤ x.nxs then t + m * (callee.params[i]!.2).leaves else t) default)

/-- What `Work.scan` binds leaf `it.2` of argument `it.1` to: in the first CALL (`fr.1`) its value at iteration 0, in
the rest (`fr.2`, an argument past the carry) its value at iteration `j + 1` in copy `j`. -/
def ScanQ (V : ℕ → Bool) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node)
    (it : ℕ × ℕ) (fr : Bind × Bind) (W' : Work) : Prop :=
  fr.1.step % 2 = 0 ∧ (0 < it.1 → fr.2.step % 2 = 0) ∧
    ∀ W : Work, W'.rows.toList <+: W.rows.toList → RowsHold V W →
      V fr.1.a = scanArg xs st vals callee x it.1 it.2 0 ∧
      (0 < it.1 → ∀ j, j + 1 < x.n → V (fr.2.a + j * fr.2.step) = scanArg xs st vals callee x it.1 it.2 (j + 1))

theorem scanQ_mono {V : ℕ → Bool} {xs : List Bool} {st : Array ℕ} {vals : Array (List Bool)} {callee : Defn}
    {x : Node} {it : ℕ × ℕ} {fr : Bind × Bind} {W1 W2 : Work} (h : ScanQ V xs st vals callee x it fr W1)
    (h12 : WR W1 W2) : ScanQ V xs st vals callee x it fr W2 :=
  ⟨h.1, h.2.1, fun W hW hR => h.2.2 W (h12.2.2.2.1.trans hW) hR⟩

/-- A scan's leaves bound, argument by argument. -/
def scanIdx (callee : Defn) (I : List ℕ) : List (ℕ × ℕ) :=
  I.flatMap fun i => (List.range (callee.params[i]!.2).leaves).map (i, ·)

/-- `Work.scan`'s loop over its arguments, through `pre`: the work extended, the first CALL's and the rest's bindings
`ScanQ` of each leaf so far, and the carry's leaves once argument 0 is through. -/
def ScanInv (V : ℕ → Bool) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node)
    (w : Work) (pre : List ℕ) (s : Work × Array Bind × Array Bind × Array ℕ) : Prop :=
  WR w s.1 ∧ ∃ L : List (Bind × Bind), List.Forall₂ (fun it fr => ScanQ V xs st vals callee x it fr s.1)
    (scanIdx callee pre) L ∧ s.2.1.toList = L.map Prod.fst ∧ s.2.2.1.toList = L.map Prod.snd ∧
    s.2.2.2.toList = (if pre = [] then [] else List.range (callee.params[0]!.2).leaves)

/-- Its loop over argument `i`'s leaves, through `pre`, from state `s0`. -/
def ScanInvT (V : ℕ → Bool) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node)
    (s0 : Work × Array Bind × Array Bind × Array ℕ) (i : ℕ) (pre : List ℕ)
    (s : Work × Array Bind × Array Bind × Array ℕ) : Prop :=
  WR s0.1 s.1 ∧ ∃ L : List (Bind × Bind), List.Forall₂ (fun it fr => ScanQ V xs st vals callee x it fr s.1)
    (pre.map (i, ·)) L ∧ s.2.1.toList = s0.2.1.toList ++ L.map Prod.fst ∧
    s.2.2.1.toList = s0.2.2.1.toList ++ L.map Prod.snd ∧ s.2.2.2.toList = s0.2.2.2.toList ++ (if i = 0 then pre else [])

theorem forall₂_mono {α β : Type} {R S : α → β → Prop} (h : ∀ a b, R a b → S a b) :
    ∀ {l : List α} {k : List β}, List.Forall₂ R l k → List.Forall₂ S l k
  | _, _, .nil => .nil
  | _, _, .cons hab hl => .cons (h _ _ hab) (forall₂_mono h hl)

theorem forall₂_map' {α β γ : Type} {P : α → β → Prop} {g : β → γ} {f : α → γ} :
    ∀ {l : List α} {ls : List β}, List.Forall₂ P l ls → (∀ a ∈ l, ∀ b, P a b → g b = f a) → ls.map g = l.map f
  | _, _, .nil, _ => rfl
  | _, _, .cons hab hl, h => by
    simp only [List.map_cons, h _ List.mem_cons_self _ hab,
      forall₂_map' hl (fun a ha => h a (List.mem_cons_of_mem _ ha))]

theorem forall₂_drop' {α β : Type} {R : α → β → Prop} :
    ∀ (n : ℕ) {l : List α} {k : List β}, List.Forall₂ R l k → List.Forall₂ R (l.drop n) (k.drop n)
  | 0, _, _, h => by simpa using h
  | _ + 1, _, _, .nil => .nil
  | n + 1, _, _, .cons _ h => by simpa using forall₂_drop' n h

theorem scanIdx_append (callee : Defn) (I J : List ℕ) :
    scanIdx callee (I ++ J) = scanIdx callee I ++ scanIdx callee J := by
  simp [scanIdx, List.flatMap_append]

theorem scanIdx_cons (callee : Defn) (i : ℕ) (I : List ℕ) :
    scanIdx callee (i :: I) = (List.range (callee.params[i]!.2).leaves).map (i, ·) ++ scanIdx callee I := by
  simp [scanIdx]

theorem scanIdx_single (callee : Defn) (i : ℕ) :
    scanIdx callee [i] = (List.range (callee.params[i]!.2).leaves).map (i, ·) := by
  simp [scanIdx]

/-- A scan's leaves: argument 0's, the carry's `init`, then the others', each of an argument past 0. -/
theorem scanIdx_range (callee : Defn) {k : ℕ} (hk : 0 < k) :
    ∃ T, scanIdx callee (List.range k) = (List.range (callee.params[0]!.2).leaves).map (0, ·) ++ T ∧
      ∀ it ∈ T, 0 < it.1 := by
  obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  rw [List.range_succ_eq_map, scanIdx_cons]
  refine ⟨_, rfl, fun it hit => ?_⟩
  simp only [scanIdx, List.mem_flatMap, List.mem_map] at hit
  obtain ⟨i, ⟨j, -, rfl⟩, t, -, rfl⟩ := hit
  simp

theorem restMap_toList {rest : Array Bind} {carries : Array ℕ} {L : List (Bind × Bind)} {c base cols : ℕ}
    {rets : Array ℕ} (hc : c ≤ rest.size) (hcar : carries.toList = List.range c) (hr : rest.toList = L.map Prod.snd) :
    ((Array.range rest.size).map fun k =>
      if k < c then ({ a := 2 * (base + rets[carries[k]!]!), step := 2 * cols } : Bind) else rest[k]!).toList =
      (List.range c).map (fun k => ({ a := 2 * (base + rets[k]!), step := 2 * cols } : Bind)) ++
        (L.drop c).map Prod.snd := by
  have hlen : rest.size = L.length := by rw [← Array.length_toList, hr, List.length_map]
  apply List.ext_getElem
  · simp; omega
  · intro k h1 h2
    simp only [Array.toList_map, Array.toList_range, List.getElem_map, List.getElem_range]
    by_cases hk : k < c
    · rw [ite_eq_left hk, List.getElem_append_left (by simpa using hk)]
      simp only [List.getElem_map, List.getElem_range]
      have hcs : k < carries.size := by rw [← Array.length_toList, hcar, List.length_range]; exact hk
      have : carries[k]! = k := by
        rw [getElem!_pos carries k hcs, ← Array.getElem_toList]
        simp [hcar]
      rw [this]
    · rw [ite_eq_right hk, List.getElem_append_right (by simp; omega)]
      simp only [List.getElem_map, List.getElem_drop, List.length_map, List.length_range]
      have hks : k < rest.size := by simpa using h1
      rw [getElem!_pos rest k hks, ← Array.getElem_toList]
      simp only [hr, List.getElem_map]
      congr 2
      omega

/-- **A scan node**: its `n` iterations one region after the work, iteration 0 the first CALL's copy, binding the
walk's `init` and the other arguments' iteration-0 leaves, and iteration `m ≥ 1` copy `m − 1` of the second, binding
iteration `m − 1`'s carry returns and the other arguments' iteration-`m` leaves. -/
theorem scan_node {V : ℕ → Bool} {w w' : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    {callee : Defn} {x : Node} {e cols : ℕ} {rets : Array ℕ}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true)
    (hs : w.scan st callee x e cols rets = .ok w') :
    ∃ base, w'.ins = w.ins ∧ w.rows.toList <+: w'.rows.toList ∧ w.calls.toList <+: w'.calls.toList ∧
      w.next ≤ base ∧ w'.next = base + x.n * cols ∧
      w'.outs = w.outs.push (.scan base cols rets (callee.params[0]!.2).leaves x.n) ∧
      0 < x.n ∧ 0 < x.args.size ∧ (callee.params[0]!.2).leaves ≤ rets.size ∧
      (RowsHold V w' → ∀ m < x.n, ∃ p ∈ w'.calls.toList, p.def_ = e ∧ p.stride = cols ∧ ∃ j < p.count,
        p.base + j * cols = base + m * cols ∧ (∀ b ∈ p.binds.toList, b.step % 2 = 0) ∧
        p.binds.toList.map (fun b => V (b.a + j * b.step)) =
          if m = 0 then (scanIdx callee (List.range x.args.size)).map (fun it => scanArg xs st vals callee x it.1 it.2 0)
          else (List.range (callee.params[0]!.2).leaves).map (fun k => V (2 * (base + (m - 1) * cols + rets[k]!))) ++
            ((scanIdx callee (List.range x.args.size)).drop (callee.params[0]!.2).leaves).map
              (fun it => scanArg xs st vals callee x it.1 it.2 m)) := by
  unfold Work.scan at hs
  dsimp only at hs
  split at hs
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
  rename_i hn
  split at hs
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
  rename_i ha
  split at hs
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
  rename_i hc
  simp only [bind, Except.bind] at hs
  split at hs
  · simp at hs
  rename_i s hfor
  cases hs
  rw [forIn_range] at hfor
  have hn' : 0 < x.n := Nat.pos_of_ne_zero (by simpa using hn)
  have ha' : 0 < x.args.size := Nat.pos_of_ne_zero (by simpa using ha)
  have hc' : (callee.params[0]!.2).leaves ≤ rets.size := by omega
  have key := forIn_inv (Inv := ScanInv V xs st vals callee x w) (S := List.range x.args.size) ?step
    (List.range x.args.size) [] _ s (by simp) ⟨wr_refl _, [], .nil, rfl, rfl, by simp⟩ hfor
  case step =>
    intro pre i s0 r hpre hI h'
    obtain ⟨-, hpi⟩ := prefix_range_last hpre
    have hi : i < x.args.size := by
      have := hpre.length_le
      simp at this
      omega
    have hsi := hseq _ (getElem!_mem hi)
    try dsimp only at h'
    try simp only [bind, Except.bind] at h'
    split at h'
    · simp at h'
    rename_i s1 hin
    cases h'
    refine ⟨_, rfl, ?_⟩
    rw [forIn_range] at hin
    have hF0 := formsOk_wr hF hI.1
    have hP0 := paramsOk_wr hP hI.1
    have key2 := forIn_inv (Inv := ScanInvT V xs st vals callee x (s0.1, s0.2.1, s0.2.2.1, s0.2.2.2) i)
      (S := List.range (callee.params[i]!.2).leaves) ?tstep _ [] _ s1 (by simp)
      ⟨wr_refl _, [], .nil, by simp, by simp, by simp⟩ hin
    case tstep =>
      intro preT t s r hpreT hJ h''
      obtain ⟨-, hpt⟩ := prefix_range_last hpreT
      have ht : t < (callee.params[i]!.2).leaves := by
        have := hpreT.length_le
        simp at this
        omega
      have hFs := formsOk_wr hF0 hJ.1
      have hPs := paramsOk_wr hP0 hJ.1
      obtain ⟨hJw, L', hL', hf', hr', hc''⟩ := hJ
      try dsimp only at h''
      split at h''
      · rename_i hi0
        simp only [beq_iff_eq] at hi0
        try simp only [bind, Except.bind] at h''
        split at h''
        · simp at h''
        rename_i rf hrf
        split at h''
        · simp at h''
        rename_i pr hpr
        obtain ⟨w', b⟩ := pr
        cases h''
        obtain ⟨hk, rfl⟩ := getRef_ok hrf
        obtain ⟨hwb, hsb, h0, -⟩ := bindOne_spec hFs hPs hpr
        refine ⟨_, rfl, wr_trans hJw hwb, L' ++ [(b, b)], ?_, ?_, ?_, ?_⟩
        · rw [List.map_append]
          refine forall₂_append (forall₂_mono (fun it fr h => scanQ_mono h hwb) hL') (.cons ?_ .nil)
          refine ⟨hsb, fun h => absurd h (by simp [hi0]), fun W hW hR => ⟨?_, fun h => absurd h (by simp [hi0])⟩⟩
          rw [(h0 rfl).2 W hW hR, refAt_eq hsi hk]
          simp [scanArg, hi0]
        · simp [hf']
        · simp [hr']
        · simp [hc'', hi0]
      · rename_i hi0
        split at h''
        · rename_i hix
          try simp only [bind, Except.bind] at h''
          split at h''
          · simp at h''
          rename_i rs hrs
          obtain ⟨rf, sf⟩ := rs
          simp only at h''
          split at h''
          · simp at h''
          rename_i pr hpr
          obtain ⟨w', b⟩ := pr
          cases h''
          have hip : 0 < i := by simp only [beq_iff_eq] at hi0; omega
          obtain ⟨hsp1, hsp2⟩ := stepping_spec hsi hrs xs st vals
          obtain ⟨hwb, hsb, h0, h1⟩ := bindOne_spec hFs hPs hpr
          refine ⟨_, rfl, wr_trans hJw hwb, L' ++ [(b, ⟨b.a + b.step, b.step⟩)], ?_, ?_, ?_, ?_⟩
          · rw [List.map_append]
            refine forall₂_append (forall₂_mono (fun it fr h => scanQ_mono h hwb) hL') (.cons ?_ .nil)
            refine ⟨hsb, fun _ => hsb, fun W hW hR => ?_⟩
            simp only [scanArg, hip, hix, and_self, ite_true]
            by_cases hn1 : x.n ≤ 1
            · obtain ⟨hs0, hk, rfl⟩ := hsp1 hn1
              refine ⟨?_, fun _ j hj => absurd hj (by omega)⟩
              rw [(h0 hs0).2 W hW hR, refAt_eq hsi hk, Nat.zero_mul, Nat.add_zero]
            · obtain ⟨hsL, hv⟩ := hsp2 (by omega)
              have hs0 : sf ≠ 0 := by rw [hsL]; omega
              refine ⟨?_, fun _ j hj => ?_⟩
              · have := h1 hs0 0 hn'
                rw [Nat.zero_mul, Nat.add_zero] at this
                rw [this, hsL]
                have := hv 0 hn'
                simp only [Nat.zero_mul, Nat.add_zero] at this ⊢
                exact this
              · have := h1 hs0 (j + 1) hj
                rw [show b.a + b.step + j * b.step = b.a + (j + 1) * b.step by ring, this, hsL, hv (j + 1) hj]
          · simp [hf']
          · simp [hr']
          · simp [hc'', show ¬i = 0 by simpa using hi0]
        · rename_i hix
          try simp only [bind, Except.bind] at h''
          split at h''
          · simp at h''
          rename_i rf hrf
          split at h''
          · simp at h''
          rename_i pr hpr
          obtain ⟨w', b⟩ := pr
          cases h''
          obtain ⟨hk, rfl⟩ := getRef_ok hrf
          obtain ⟨hwb, hsb, h0, -⟩ := bindOne_spec hFs hPs hpr
          refine ⟨_, rfl, wr_trans hJw hwb, L' ++ [(b, b)], ?_, ?_, ?_, ?_⟩
          · rw [List.map_append]
            refine forall₂_append (forall₂_mono (fun it fr h => scanQ_mono h hwb) hL') (.cons ?_ .nil)
            have hv : ∀ W : Work, w'.rows.toList <+: W.rows.toList → RowsHold V W →
                V b.a = scanArg xs st vals callee x i t 0 := fun W hW hR => by
              rw [(h0 rfl).2 W hW hR, refAt_eq hsi hk]
              simp [scanArg, hix]
            refine ⟨hsb, fun _ => hsb, fun W hW hR => ⟨hv W hW hR, fun _ j _ => ?_⟩⟩
            rw [(h0 rfl).1, Nat.mul_zero, Nat.add_zero, hv W hW hR]
            simp [scanArg, hix]
          · simp [hf']
          · simp [hr']
          · simp [hc'', show ¬i = 0 by simpa using hi0]
    obtain ⟨hW1, L', hL', hf', hr', hc''⟩ := key2
    obtain ⟨hW0, L, hL, hf, hr, hc0⟩ := hI
    refine ⟨wr_trans hW0 hW1, L ++ L', ?_, ?_, ?_, ?_⟩
    · rw [scanIdx_append, scanIdx_single]
      exact forall₂_append (forall₂_mono (fun it fr h => scanQ_mono h hW1) hL) hL'
    · simp only at hf' ⊢
      rw [hf', hf, List.map_append]
    · simp only at hr' ⊢
      rw [hr', hr, List.map_append]
    · simp only at hc'' ⊢
      rw [hc'', hc0]
      by_cases hp : pre = []
      · subst hp
        simp at hpi
        subst hpi
        simp
      · have : i ≠ 0 := by
          intro h0
          rw [h0] at hpi
          exact hp (List.length_eq_zero_iff.mp hpi.symm)
        simp [hp, this]
  obtain ⟨hW, L, hL, hf, hr, hcar⟩ := key
  have hne : List.range x.args.size ≠ [] := by rw [Ne, List.range_eq_nil]; omega
  simp only [hne, ite_false] at hcar
  obtain ⟨T, hT, hTpos⟩ := scanIdx_range callee ha'
  have hLlen := hL.length_eq
  have hcL : (callee.params[0]!.2).leaves ≤ L.length := by rw [← hLlen, hT]; simp
  have hrsz : (callee.params[0]!.2).leaves ≤ s.2.2.1.size := by
    rw [← Array.length_toList, hr, List.length_map]; exact hcL
  have hdrop : (scanIdx callee (List.range x.args.size)).drop (callee.params[0]!.2).leaves = T := by
    rw [hT, List.drop_left' (by simp)]
  have hLd := forall₂_drop' (callee.params[0]!.2).leaves hL
  rw [hdrop] at hLd
  refine ⟨s.1.next, hW.1, hW.2.2.2.1, ?_, hW.2.2.2.2, rfl, by simp [hW.2.1], hn', ha', hc', fun hR m hm => ?_⟩
  · rw [← hW.2.2.1]
    simp only
    split
    · simp only [Array.toList_push, List.append_assoc]
      exact List.prefix_append _ _
    · simp only [Array.toList_push]
      exact List.prefix_append _ _
  have hR' := fun it fr (h : ScanQ V xs st vals callee x it fr s.1) => h.2.2 _ (List.prefix_refl _) hR
  by_cases hm0 : m = 0
  · refine ⟨⟨e, 1, s.1.next, cols, s.2.1⟩, ?_, rfl, rfl, 0, by simp, by simp [hm0], ?_, ?_⟩
    · simp only
      split <;> simp
    · intro b hb
      simp only at hb
      rw [hf] at hb
      obtain ⟨fr, hfr, rfl⟩ := List.mem_map.mp hb
      obtain ⟨it, -, hq⟩ := forall₂_mem hL fr hfr
      exact hq.1
    · rw [if_pos hm0]
      simp only
      rw [hf, List.map_map]
      exact forall₂_map (fun it fr h => by
        simp only [Function.comp, Nat.zero_mul, Nat.add_zero]
        exact (hR' it fr h).1) hL
  · have hn1 : 1 < x.n := by omega
    obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
    refine ⟨⟨e, x.n - 1, s.1.next + cols, cols, Array.map (fun k => if k < (callee.params[0]!.2).leaves then
      { a := 2 * (s.1.next + rets[s.2.2.2[k]!]!), step := 2 * cols } else s.2.2.1[k]!) (Array.range s.2.2.1.size)⟩, ?_, rfl, rfl, m', by simp; omega, by simp only; ring, ?_, ?_⟩
    · simp [hn1]
    · intro b hb
      simp only at hb
      rw [restMap_toList hrsz hcar hr, List.mem_append] at hb
      rcases hb with hb | hb
      · obtain ⟨k, -, rfl⟩ := List.mem_map.mp hb
        simp
      · obtain ⟨fr, hfr, rfl⟩ := List.mem_map.mp hb
        obtain ⟨it, hit, hq⟩ := forall₂_mem hLd fr hfr
        exact hq.2.1 (hTpos it hit)
    · rw [if_neg hm0]
      simp only
      rw [restMap_toList hrsz hcar hr, List.map_append, List.map_map, List.map_map, hdrop]
      congr 1
      · apply List.map_congr_left
        intro k _
        simp only [Function.comp, Nat.add_sub_cancel]
        congr 1
        ring
      · exact forall₂_map' hLd (fun it hit fr h => by
          simp only [Function.comp]
          exact (hR' it fr h).2 (hTpos it hit) m' (by omega))


/-- **A scan node's leaves are the last carry, then each iteration's `y`**, from its iterations' returns. -/
theorem scan_forms {V : ℕ → Bool} {w w' : Work} {vals : Array (List Bool)} {base n c cols : ℕ} {rets : Array ℕ}
    (hF : FormsOk V w vals) (ho : w'.outs = w.outs.push (.scan base cols rets c n)) (hn : 0 < n)
    (hc : c ≤ rets.size) {l : ℕ → List Bool} (hlen : ∀ m < n, (l m).length = rets.size)
    (hl : ∀ m < n, ∀ k < rets.size, V (2 * (base + m * cols + rets[k]!)) = (l m).getD k false) :
    FormsOk V w' (vals.push ((l (n - 1)).take c ++ ((List.range n).map fun m => (l m).drop c).flatten)) := by
  refine formsOk_outs (w := { w with outs := w.outs.push (.scan base cols rets c n) }) ho.symm
    (formsOk_push hF rfl ?_)
  have htake : ((l (n - 1)).take c).length = c := by
    rw [List.length_take, hlen _ (by omega)]
    omega
  intro j hj
  refine ⟨fun hjc => ?_, fun hjc => ?_⟩
  · rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by omega), List.getElem?_take_of_lt hjc,
      ← List.getD_eq_getElem?_getD, hl _ (by omega) _ (by omega)]
  · have hy : 0 < rets.size - c := Nat.pos_of_ne_zero (fun h0 => by rw [h0] at hj; omega)
    have hjn : (j - c) / (rets.size - c) < n := by rw [Nat.div_lt_iff_lt_mul hy]; omega
    have hr := Nat.mod_lt (j - c) hy
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), htake, ← List.getD_eq_getElem?_getD,
      getD_flatten_eq (L := rets.size - c) (fun l' hl' => by
        obtain ⟨m, hm, rfl⟩ := List.mem_map.mp hl'
        rw [List.length_drop, hlen m (List.mem_range.mp hm)]) (by simp; omega),
      List.getElem?_map, List.getElem?_range hjn]
    simp only [Option.map_some, Option.getD_some]
    rw [List.getD_eq_getElem?_getD, List.getElem?_drop, ← List.getD_eq_getElem?_getD, ← hl _ hjn _ (by omega)]

/-- Iteration `m`'s inputs past the carry, as the walk reads them (`walkStep`'s scan). -/
def scanWalkTail (v : Ref → Bool) (callee : Defn) (x : Node) (m : ℕ) : List Bool :=
  ((List.range x.args.size).drop 1).flatMap fun i =>
    if i ≤ x.nxs then argVals v (x.args.getD i default) ((callee.params.getD i default).2.leaves)
      (fun t => t + m * (callee.params.getD i default).2.leaves)
    else argVals v (x.args.getD i default) ((callee.params.getD i default).2.leaves) id

theorem scanTail_eq (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node)
    (ha : 0 < x.args.size) (m : ℕ) :
    ((scanIdx callee (List.range x.args.size)).drop (callee.params[0]!.2).leaves).map
      (fun it => scanArg xs st vals callee x it.1 it.2 m) = scanWalkTail (refVal xs st vals) callee x m := by
  obtain ⟨k, hk⟩ : ∃ k, x.args.size = k + 1 := ⟨x.args.size - 1, by omega⟩
  unfold scanWalkTail
  rw [hk, List.range_succ_eq_map, scanIdx_cons, List.drop_left' (by simp), List.drop_succ_cons, List.drop_zero]
  simp only [scanIdx, List.flatMap_map, List.map_flatMap, List.map_map]
  congr 1
  funext j
  by_cases hj : j + 1 ≤ x.nxs
  · simp [hj, Function.comp, scanArg, argVals, getElem!_eq_getD]
  · simp [hj, Function.comp, scanArg, argVals, getElem!_eq_getD]

theorem scanInit_eq (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node) :
    ((List.range (callee.params[0]!.2).leaves).map (0, ·)).map (fun it => scanArg xs st vals callee x it.1 it.2 0) =
      argVals (refVal xs st vals) (x.args.getD 0 default) ((callee.params.getD 0 default).2.leaves) id := by
  simp [scanArg, argVals, getElem!_eq_getD, Function.comp]

/-- Iteration 0's inputs: `init`, then the rest's. -/
theorem scanIns0_eq (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node)
    (ha : 0 < x.args.size) :
    (scanIdx callee (List.range x.args.size)).map (fun it => scanArg xs st vals callee x it.1 it.2 0) =
      argVals (refVal xs st vals) (x.args.getD 0 default) ((callee.params.getD 0 default).2.leaves) id ++
        scanWalkTail (refVal xs st vals) callee x 0 := by
  obtain ⟨T, hT, -⟩ := scanIdx_range callee ha
  have hd : (scanIdx callee (List.range x.args.size)).drop (callee.params[0]!.2).leaves = T := by
    rw [hT, List.drop_left' (by simp)]
  rw [← scanTail_eq xs st vals callee x ha 0, ← scanInit_eq, hd, hT, List.map_append]

end Flock.Chain.LowerDef
