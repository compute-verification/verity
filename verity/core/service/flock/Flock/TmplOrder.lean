/-
`Tmpl.ordered`, the order a templated unit's description is checked for, gives the order `Net.checkOrder`
checks on the rows it unrolls to (`rows_ordered`): every row from the input rows to the constant reads, in A and in B,
the constant or an earlier column that is not input padding (`FlockSoundness.RowOrder`, stated here as it reads), given
the input port bits' self rows `Net.ofRows` checks (`Net.inputsOk`). So a reader that checks the description need not walk
the unrolled rows for their order. The proof is an induction over the copies `place` writes: a copy past the input rows
of an ordered Definition, its inputs bound to the constant or to earlier columns (port bits if input columns), writes only
rows that read such columns (`place_good`); a copy in the input rows writes only there (`place_local`).
-/
import Flock.Tmpl

namespace Flock.TmplOrder

open Flock.Tmpl

/-! ## `canon`, `write`, and folds -/

theorem mem_dropPairs : ∀ {l : List Nat} {x : Nat}, x ∈ dropPairs l → x ∈ l
  | [], _, h => by simp [dropPairs] at h
  | [_], _, h => by simpa [dropPairs] using h
  | a :: b :: rest, x, h => by
    simp only [dropPairs] at h
    split at h
    · exact List.mem_cons.2 (Or.inr (List.mem_cons.2 (Or.inr (mem_dropPairs h))))
    · rcases List.mem_cons.1 h with h | h
      · exact List.mem_cons.2 (Or.inl h)
      · exact List.mem_cons.2 (Or.inr (mem_dropPairs h))

/-- Every column of a canonical row is one of the row's. -/
theorem mem_canon {xs : Array Nat} {x : Nat} (h : x ∈ (Flock.Tmpl.canon xs).toList) : x ∈ xs.toList := by
  have h' : x ∈ dropPairs xs.toList.mergeSort := h
  exact List.mem_mergeSort.1 (mem_dropPairs h')

theorem bind_ok {α β : Type} {x : Except String α} {f : α → Except String β} {v : β}
    (h : (x >>= f) = .ok v) : ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- A property every step of a fold keeps holds at its end. -/
theorem foldlM_inv {α σ : Type} (P : σ → Prop) {f : σ → α → Except String σ} :
    ∀ {l : List α} {s s' : σ}, (∀ a ∈ l, ∀ u u', P u → f u a = .ok u' → P u') → P s → l.foldlM f s = .ok s' → P s'
  | [], _, _, _, hs, h => by cases h; exact hs
  | a :: l, s, _, hf, hs, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨s1, h1, h2⟩ := bind_ok h
    exact foldlM_inv P (fun b hb => hf b (List.mem_cons.2 (Or.inr hb)))
      (hf a (List.mem_cons.2 (Or.inl rfl)) s s1 hs h1) h2

theorem getD_set! (xs : Array (Array Nat)) (c x : Nat) (v : Array Nat) :
    (xs.set! c v).getD x #[] = xs.getD x #[] ∨ (x = c ∧ (xs.set! c v).getD x #[] = v) := by
  simp only [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
  by_cases hx : c = x
  · subst hx
    by_cases hs : c < xs.size
    · exact Or.inr ⟨rfl, by simp [hs]⟩
    · exact Or.inl (by simp [hs])
  · exact Or.inl (by simp [hx])

/-- `write` changes column `c`'s rows only, to `a` and `b`. -/
theorem write_ok {acc acc' : Acc} {c : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc') (x : Nat) :
    (acc'.ra.getD x #[] = acc.ra.getD x #[] ∨ (x = c ∧ acc'.ra.getD x #[] = a)) ∧
      (acc'.rb.getD x #[] = acc.rb.getD x #[] ∨ (x = c ∧ acc'.rb.getD x #[] = b)) := by
  unfold write at h
  split at h
  · cases h
  · cases h
    exact ⟨getD_set! acc.ra c x a, getD_set! acc.rb c x b⟩

theorem getD_of_some {defs : Array Def} {d : Nat} {df : Def} (h : defs[d]? = some df) :
    d < defs.size ∧ defs.getD d default = df := by
  refine ⟨?_, by simp [h]⟩
  rcases Nat.lt_or_ge d defs.size with hd | hd
  · exact hd
  · rw [Array.getElem?_eq_none hd] at h; cases h

theorem imp_of_or {x y : Bool} (h : (!x || y) = true) (hx : x = true) : y = true := by
  subst hx; simpa using h

/-- Copy `j < n` of a run: a reference linear in `j`, below a bound linear in `j` at copies `0` and `n`, is below it at
copy `j`. -/
theorem interp {A B s S j n : Nat} (hj : j ≤ n) (h0 : A ≤ B) (hn : A + n * s ≤ B + n * S) : A + j * s ≤ B + j * S := by
  rcases Nat.le_total s S with h | h
  · have := Nat.mul_le_mul (Nat.le_refl j) h
    omega
  · obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le hj
    rw [Nat.add_mul, Nat.add_mul] at hn
    have := Nat.mul_le_mul (Nat.le_refl m) h
    omega

/-! ## References -/

/-- A column a row at `w` may read: the constant, or an earlier column, a port bit (`P`) if an input column. -/
abbrev Reads (ic k : Nat) (P : Nat → Prop) (w c : Nat) : Prop := c = k ∨ (c < w ∧ (c < ic → P c))

/-- Column `x`'s rows are the same in both. -/
abbrev Same (x : Nat) (a b : Acc) : Prop := a.ra.getD x #[] = b.ra.getD x #[] ∧ a.rb.getD x #[] = b.rb.getD x #[]

/-- Every row from the input rows `ic` to the constant `k` reads only columns it may. -/
def Good (ic k : Nat) (P : Nat → Prop) (acc : Acc) : Prop :=
  ∀ r, ic ≤ r → r < k →
    (∀ c ∈ (acc.ra.getD r #[]).toList, Reads ic k P r c) ∧ (∀ c ∈ (acc.rb.getD r #[]).toList, Reads ic k P r c)

theorem good_empty {ic k : Nat} {P : Nat → Prop} {n : Nat} : Good ic k P (Acc.empty n) := by
  intro r _ _
  have e : (Array.replicate n (#[] : Array Nat)).getD r #[] = #[] := by
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
    split <;> rfl
  refine ⟨fun c hc => ?_, fun c hc => ?_⟩
  all_goals
    change c ∈ ((Array.replicate n (#[] : Array Nat)).getD r #[]).toList at hc
    rw [e] at hc
    simp at hc

theorem good_write {ic k : Nat} {P : Nat → Prop} {acc acc' : Acc} {w : Nat} {a b : Array Nat}
    (hg : Good ic k P acc) (h : write acc w a b = .ok acc')
    (ha : ic ≤ w → w < k → (∀ c ∈ a.toList, Reads ic k P w c) ∧ ∀ c ∈ b.toList, Reads ic k P w c) :
    Good ic k P acc' := by
  intro r h1 h2
  obtain ⟨ea, eb⟩ := write_ok h r
  constructor
  · rcases ea with e | ⟨e, e'⟩
    · rw [e]; exact (hg r h1 h2).1
    · subst e; rw [e']; exact (ha h1 h2).1
  · rcases eb with e | ⟨e, e'⟩
    · rw [e]; exact (hg r h1 h2).2
    · subst e; rw [e']; exact (ha h1 h2).2

theorem good_same {ic k : Nat} {P : Nat → Prop} {acc acc' : Acc} (hg : Good ic k P acc)
    (hs : ∀ x, ic ≤ x → Same x acc' acc) : Good ic k P acc' := by
  intro r h1 h2
  rw [(hs r h1).1, (hs r h1).2]
  exact hg r h1 h2

theorem getD_cases (l : List Nat) (i d : Nat) : l.getD i d = d ∨ l.getD i d ∈ l := by
  rw [List.getD_eq_getElem?_getD]
  cases h : l[i]? with
  | none => exact Or.inl rfl
  | some v => exact Or.inr (List.mem_of_getElem? h)

/-- A reference of a copy at `o` past the input rows, its inputs bound to columns it may read, is one a row at `w ≥ o` may
read, when an own column `v` it names lands before `w`. -/
theorem ref_reads {df : Def} {o w ic k : Nat} {P : Nat → Prop} {inp : List Nat} {v : Nat}
    (hinp : ∀ x ∈ inp, Reads ic k P o x) (hic : ic ≤ o) (how : o ≤ w) (hv : v < df.cols → o + v < w) :
    Reads ic k P w (df.ref o inp k v) := by
  unfold Def.ref
  by_cases h1 : v < df.cols
  · rw [ite_eq_left h1]
    exact Or.inr ⟨hv h1, fun h => absurd h (by omega)⟩
  · rw [ite_eq_right h1]
    split
    · rcases getD_cases inp (v - df.cols) k with e | e
      · rw [e]; exact Or.inl rfl
      · rcases hinp _ e with e' | ⟨e1, e2⟩
        · exact Or.inl e'
        · exact Or.inr ⟨by omega, e2⟩
    · exact Or.inl rfl

/-- The root's references: its own columns, or the constant. -/
theorem root_ref {df : Def} {k v : Nat} : (v < df.cols ∧ df.ref 0 [] k v = v) ∨ df.ref 0 [] k v = k := by
  unfold Def.ref
  by_cases h1 : v < df.cols
  · exact Or.inl ⟨h1, by rw [ite_eq_left h1]; omega⟩
  · right
    rw [ite_eq_right h1]
    split <;> simp

theorem mem_bound {c : Call} {g : Nat → Nat} {j x : Nat} (h : x ∈ c.bound g j) :
    ∃ s ∈ c.segs.toList, ∃ r < s.reps, ∃ t < s.len, x = g (s.at_ + r * s.rstride + j * s.step + t) := by
  unfold Call.bound at h
  simp only [List.mem_flatMap, List.mem_map, List.mem_range] at h
  obtain ⟨s, hs, r, hr, t, ht, rfl⟩ := h
  exact ⟨s, hs, r, hr, t, ht, rfl⟩

/-- An ordered segment's reference for copy `j`, run `r`: past the Definition's own columns, or an own column before copy
`j`'s first column. -/
theorem seg_ordered {cols base stride count : Nat} {s : Seg} (h : Seg.ordered cols base stride count s = true)
    {j r t : Nat} (hj : j < count) (hr : r < s.reps) (ht : t < s.len) :
    cols ≤ s.at_ + r * s.rstride + j * s.step + t ∨
      (s.at_ + r * s.rstride + j * s.step + t < cols ∧ s.at_ + r * s.rstride + j * s.step + t < base + j * stride ∧
        s.at_ + r * s.rstride + j * s.step + t < s.hi count) := by
  unfold Seg.ordered Seg.hi at h
  unfold Seg.hi
  simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.and_eq_true] at h
  have hm : j * s.step ≤ (count - 1) * s.step := Nat.mul_le_mul (by omega) (Nat.le_refl _)
  have hrm : r * s.rstride ≤ (s.reps - 1) * s.rstride := Nat.mul_le_mul (by omega) (Nat.le_refl _)
  rcases h with h | ⟨⟨h1, h2⟩, h3⟩
  · rcases h with h | h
    · omega
    · left; omega
  · right
    have := interp (A := s.at_ + (s.reps - 1) * s.rstride + s.len) (B := base) (s := s.step) (S := stride) (j := j)
      (n := count - 1) (by omega) h2 (by omega)
    omega

theorem portSpan_mono {t : Tmpl} {lo hi lo' hi' : Nat} (h : t.portSpan lo hi = true) (h1 : lo ≤ lo')
    (h2 : hi' ≤ hi) : t.portSpan lo' hi' = true := by
  unfold Tmpl.portSpan at h ⊢
  rw [List.any_eq_true] at h ⊢
  obtain ⟨g, hg, hp⟩ := h
  refine ⟨g, hg, ?_⟩
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hp ⊢
  omega

/-! ## Copies -/

/-- **A copy writes only its own columns.** -/
theorem place_local {defs : Array Def} {k : Nat}
    (hfits : ∀ (d : Nat) (df : Def), defs[d]? = some df → df.fits defs = true) (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' →
      ∀ x, (x < o ∨ o + (defs.getD d default).cols ≤ x) → Same x acc' acc := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h x hx
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      obtain ⟨-, hdd⟩ := getD_of_some hdf
      rw [hdd] at hx
      have hf := hfits d df hdf
      unfold Def.fits at hf
      simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hf
      obtain ⟨hrows, hcalls⟩ := hf
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      have s1 : Same x acc1 acc := foldlM_inv (fun a => Same x a acc) (fun r hr u u' hu hw => by
          obtain ⟨ea, eb⟩ := write_ok hw x
          have := hrows r hr
          refine ⟨?_, ?_⟩
          · rcases ea with e | ⟨e, -⟩
            · exact e.trans hu.1
            · omega
          · rcases eb with e | ⟨e, -⟩
            · exact e.trans hu.2
            · omega) ⟨rfl, rfl⟩ h1
      have s2 : Same x acc' acc1 := foldlM_inv (fun a => Same x a acc1) (fun c hc u u' hu hw => by
          have hs : Same x u' u := foldlM_inv (fun a => Same x a u) (fun j hj w w' hw0 hp => by
              have hj' := List.mem_range.1 hj
              have hm : j * c.stride ≤ (c.count - 1) * c.stride := Nat.mul_le_mul (by omega) (Nat.le_refl _)
              have hc' := hcalls c hc
              have := ih hp x (by omega)
              exact ⟨this.1.trans hw0.1, this.2.trans hw0.2⟩) ⟨rfl, rfl⟩ hw
          exact ⟨hs.1.trans hu.1, hs.2.trans hu.2⟩) ⟨rfl, rfl⟩ h2
      exact ⟨s2.1.trans s1.1, s2.2.trans s1.2⟩

/-- A row of a copy at `o` past the input rows, of a Definition whose rows read only past its own columns or earlier
ones, reads only columns a row at its column may. -/
theorem row_reads {df : Def} {o ic k : Nat} {P : Nat → Prop} {inp : List Nat} {col : Nat} {a : Array Nat}
    (hinp : ∀ x ∈ inp, Reads ic k P o x) (hic : ic ≤ o) (ha : ∀ v ∈ a.toList, df.cols ≤ v ∨ v < col) :
    ∀ c ∈ (Flock.Tmpl.canon (a.map (df.ref o inp k))).toList, Reads ic k P (o + col) c := by
  intro c hc
  have hc' := mem_canon hc
  rw [Array.toList_map] at hc'
  obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hc'
  exact ref_reads hinp hic (Nat.le_add_right o col) fun hvc => by
    rcases ha v hv with h | h
    · omega
    · omega

/-- **An ordered copy past the input rows keeps every row reading only columns it may.** -/
theorem place_good {defs : Array Def} {k ic : Nat} {P : Nat → Prop} {F : Array Bool}
    (hF : ∀ (d : Nat) (df : Def), defs[d]? = some df → F.getD d false = true → df.ordered F = true) (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → F.getD d false = true → ic ≤ o →
      (∀ x ∈ inp, Reads ic k P o x) → Good ic k P acc → Good ic k P acc' := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h hd hic hinp hg
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      have ho := hF d df hdf hd
      unfold Def.ordered at ho
      simp only [Bool.and_eq_true, Bool.or_eq_true, List.all_eq_true, decide_eq_true_eq, List.mem_append] at ho
      obtain ⟨hrows, hcalls⟩ := ho
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      have g1 : Good ic k P acc1 := foldlM_inv (Good ic k P) (fun r hr u u' hu hw =>
          good_write hu hw fun _ _ =>
            ⟨row_reads hinp hic fun v hv => hrows r hr v (Or.inl hv),
              row_reads hinp hic fun v hv => hrows r hr v (Or.inr hv)⟩) hg h1
      exact foldlM_inv (Good ic k P) (fun c hc u u' hu hw =>
        foldlM_inv (Good ic k P) (fun j hj w w' hw0 hp => by
          have hj' := List.mem_range.1 hj
          obtain ⟨hFc, hsegs⟩ := hcalls c hc
          refine ih hp hFc (by omega) (fun x hx => ?_) hw0
          obtain ⟨s, hs, r, hr, t, ht, rfl⟩ := mem_bound hx
          rcases seg_ordered (hsegs s hs) hj' hr ht with hv | ⟨-, hv2, -⟩
          · exact ref_reads hinp hic (by omega) fun h => absurd h (by omega)
          · exact ref_reads hinp hic (by omega) fun _ => by omega) hu hw) g1 h2

/-- A root row past the input rows, reading only the constant or earlier columns, port bits if input columns. -/
theorem root_row_reads {df : Def} {ic k : Nat} {P : Nat → Prop} {col : Nat} {a : Array Nat} (hk : k = df.cols - 1)
    (ha : ∀ v ∈ a.toList, df.cols ≤ v + 1 ∨ (v < col ∧ (ic ≤ v ∨ P v))) :
    ∀ c ∈ (Flock.Tmpl.canon (a.map (df.ref 0 [] k))).toList, Reads ic k P (0 + col) c := by
  intro c hc
  have hc' := mem_canon hc
  rw [Array.toList_map] at hc'
  obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hc'
  rcases root_ref (df := df) (k := k) (v := v) with ⟨h1, e⟩ | e
  · rw [e]
    rcases ha v hv with h | ⟨h2, h3⟩
    · exact Or.inl (by omega)
    · exact Or.inr ⟨by omega, fun h => h3.resolve_left (by omega)⟩
  · rw [e]; exact Or.inl rfl

/-! ## The unit -/

/-- **An ordered template's rows read only columns they may.** -/
theorem rows_good {t : Tmpl} {acc : Acc} (ho : t.ordered = true) (hr : t.rows = .ok acc) :
    Good t.inCols t.constPos (fun c => t.portSpan c (c + 1) = true) acc := by
  unfold Tmpl.ordered at ho
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at ho
  obtain ⟨hdefs, hroot⟩ := ho
  have hfits : ∀ (d : Nat) (df : Def), t.defs[d]? = some df → df.fits t.defs = true := fun d df h => by
    obtain ⟨hd, e⟩ := getD_of_some h
    rw [← e]; exact (hdefs d hd).1
  have hF : ∀ (d : Nat) (df : Def), t.defs[d]? = some df → (flags t.defs).getD d false = true →
      df.ordered (flags t.defs) = true := fun d df h hf => by
    obtain ⟨hd, e⟩ := getD_of_some h
    rw [← e]; exact imp_of_or (hdefs d hd).2 hf
  unfold Tmpl.rows at hr
  obtain ⟨acc1, h1, h2⟩ := bind_ok hr
  refine good_write ?_ h2 fun _ h => absurd h (Nat.lt_irrefl _)
  simp only [place] at h1
  cases hdf : t.defs[t.root]? with
  | none => simp only [hdf] at h1; cases h1
  | some df =>
    simp only [hdf] at h1
    obtain ⟨-, hdd⟩ := getD_of_some hdf
    have hk : t.constPos = df.cols - 1 := by unfold Tmpl.constPos; rw [hdd]
    unfold Tmpl.rootOrdered at hroot
    simp only [hdd, Bool.and_eq_true, Bool.or_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq,
      List.mem_append] at hroot
    obtain ⟨hrows, hcalls⟩ := hroot
    obtain ⟨acc2, h3, h4⟩ := bind_ok h1
    have g2 : Good t.inCols t.constPos (fun c => t.portSpan c (c + 1) = true) acc2 :=
      foldlM_inv (Good t.inCols t.constPos (fun c => t.portSpan c (c + 1) = true)) (fun r hr u u' hu hw =>
        good_write hu hw fun hw1 _ => by
          rcases hrows r hr with h | h
          · exact absurd hw1 (by omega)
          · exact ⟨root_row_reads hk fun v hv => h v (Or.inl hv), root_row_reads hk fun v hv => h v (Or.inr hv)⟩)
        good_empty h3
    exact foldlM_inv (Good t.inCols t.constPos (fun c => t.portSpan c (c + 1) = true)) (fun c hc u u' hu hw =>
      foldlM_inv (Good t.inCols t.constPos (fun c => t.portSpan c (c + 1) = true)) (fun j hj w w' hw0 hp => by
        have hj' := List.mem_range.1 hj
        have hm : j * c.stride ≤ (c.count - 1) * c.stride := Nat.mul_le_mul (by omega) (Nat.le_refl _)
        rcases hcalls c hc with hin | ⟨⟨hb, hFc⟩, hsegs⟩
        · exact good_same hw0 fun x hx => place_local hfits _ hp x (Or.inr (by omega))
        · refine place_good hF _ hp hFc (by omega) (fun x hx => ?_) hw0
          obtain ⟨s, hs, r, hr, tt, ht, rfl⟩ := mem_bound hx
          have hjs : j * s.step ≤ (c.count - 1) * s.step := Nat.mul_le_mul (by omega) (Nat.le_refl _)
          have hrs : r * s.rstride ≤ (s.reps - 1) * s.rstride := Nat.mul_le_mul (by omega) (Nat.le_refl _)
          rcases hsegs s hs with ⟨⟨hl, ha⟩, hhi⟩ | ⟨hso, hport⟩
          · unfold Seg.hi at hhi
            rcases root_ref (df := df) (k := t.constPos) (v := s.at_ + r * s.rstride + j * s.step + tt) with ⟨-, e⟩ | e
            · rw [e]; exact Or.inl (by omega)
            · rw [e]; exact Or.inl rfl
          · rcases root_ref (df := df) (k := t.constPos) (v := s.at_ + r * s.rstride + j * s.step + tt) with ⟨hv', e⟩ | e
            · rw [e]
              rcases seg_ordered hso hj' hr ht with hv | ⟨-, hv2, hv3⟩
              · exact absurd hv' (by omega)
              · refine Or.inr ⟨by omega, fun hlt => ?_⟩
                rcases hport with hp' | hp'
                · exact absurd hp' (by omega)
                · exact portSpan_mono hp' (by omega) (by omega)
            · rw [e]; exact Or.inl rfl) hu hw) g2 h4

/-- **An ordered template unrolls to rows in the order `Net.checkOrder` checks** (`FlockSoundness.RowOrder`): every
row from the input rows to the constant reads, in A and in B, the constant or an earlier column that is not input
padding, given the input port bits' self rows (`Net.inputsOk`, which `Net.ofRows` checks). -/
theorem rows_ordered {t : Tmpl} {acc : Acc} (ho : t.ordered = true) (hr : t.rows = .ok acc)
    (hin : Net.inputsOk t.inGroups acc.ra acc.rb = true) :
    ∀ r, t.inCols ≤ r → r < t.constPos → ∀ c ∈ (acc.ra.getD r #[]).toList ++ (acc.rb.getD r #[]).toList,
      c = t.constPos ∨
        (c < r ∧ ¬ (c < t.inCols ∧ (acc.ra.getD c #[]).isEmpty = true ∧ (acc.rb.getD c #[]).isEmpty = true)) := by
  have hg := rows_good ho hr
  unfold Net.inputsOk at hin
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq, Array.all_eq_true',
    beq_iff_eq] at hin
  intro r h1 h2 c hc
  have hrc : Reads t.inCols t.constPos (fun c => t.portSpan c (c + 1) = true) r c := by
    rcases List.mem_append.1 hc with hc | hc
    · exact (hg r h1 h2).1 c hc
    · exact (hg r h1 h2).2 c hc
  rcases hrc with e | ⟨hlt, hp⟩
  · exact Or.inl e
  · refine Or.inr ⟨hlt, fun hpad => ?_⟩
    obtain ⟨hci, hea, -⟩ := hpad
    have hP : t.portSpan c (c + 1) = true := hp hci
    unfold Tmpl.portSpan at hP
    rw [List.any_eq_true] at hP
    obtain ⟨g, hg', hq⟩ := hP
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hq
    have := (hin.2 g (Array.mem_toList_iff.1 hg') (c - g.col * WORD) (by omega)).1
    rw [show g.col * WORD + (c - g.col * WORD) = c by omega] at this
    rw [this] at hea
    simp at hea

end Flock.TmplOrder
