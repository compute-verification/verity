import Proofs.Flock.Soundness.Discharge.Layout.InRegion
import Batteries.Tactic.OpenPrivate

/-!
# Every public leaf is inside an in region

`in_pinned` reads a public leaf's bits from the in region holding its word. `in_cover`: every public leaf `jj` of a
unit's public leaf map (`leaves_public`) has one, region `x`, `(w, s)`, from a word of the leaf group (`inCol c ≤ w`)
with the leaf's word `inCol c + jj / 8` among its `2^s`. The leaf's word is one of `HmIn.words` (`words_mem`, through
`Array.qsort`'s permutation and the dedup), every word lies in one of `HmIn.blocks`' aligned blocks (`blocks_cover`:
in a run of consecutive words, and each run cut into blocks that cover it), and every block is `2^(e - s)` regions of
`2^s` words (`inRegions_cover`). So a public leaf's bits in a satisfying witness are its instance's public word
(`public_pinned`).
-/

set_option Elab.async false

open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace FlockSoundness.Discharge.Layout

open Flock Model

/-! ### `Array.qsort` keeps its elements -/

theorem inLoop_perm {α : Type} {n : ℕ} (lt : α → α → Bool) (lo hi : ℕ) (hhi : hi < n) (pivot : α) :
    ∀ (d : ℕ) (as : Vector α n) (i k : ℕ) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w).2.toArray.Perm as.toArray := by
  intro d
  induction d with
  | zero =>
    intro as i k ilo ik w hd
    rw [Array.qpartition.loop.eq_def]
    have : ¬ k < hi := by omega
    simp only [this, dite_false]
    exact (Vector.swap_perm _ _).toArray
  | succ d ih =>
    intro as i k ilo ik w hd
    rw [Array.qpartition.loop.eq_def]
    have hk : k < hi := by omega
    simp only [hk, dite_true]
    split
    · exact (ih _ _ _ _ _ _ (by omega)).trans (Vector.swap_perm _ _).toArray
    · exact ih _ _ _ _ _ _ (by omega)

theorem inQpartition_perm {α : Type} {n : ℕ} (as : Vector α n) (lt : α → α → Bool) (lo hi : ℕ) (w : lo ≤ hi)
    (hlo : lo < n) (hhi : hi < n) : (Array.qpartition as lt lo hi w hlo hhi).2.toArray.Perm as.toArray := by
  unfold Array.qpartition
  simp only
  refine (inLoop_perm lt lo hi hhi _ _ _ _ _ _ _ _ rfl).trans ?_
  have h1 : ∀ (b : Vector α n) (c : Bool) (i j : ℕ) (hi' : i < n) (hj : j < n),
      (if c = true then b.swap i j hi' hj else b).toArray.Perm b.toArray := by
    intro b c i j hi' hj
    split
    · exact (Vector.swap_perm _ _).toArray
    · exact Array.Perm.refl _
  exact (h1 _ _ _ _ _ _).trans ((h1 _ _ _ _ _ _).trans (h1 _ _ _ _ _ _))

theorem inSort_perm {α : Type} (lt : α → α → Bool) {n : ℕ} :
    ∀ (d : ℕ) (as : Vector α n) (lo hi : ℕ) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo = d →
      (Array.qsort.sort lt as lo hi w hlo hhi).toArray.Perm as.toArray := by
  intro d
  induction d using Nat.strong_induction_on with
  | _ d ih =>
    intro as lo hi w hlo hhi hd
    rw [Array.qsort.sort.eq_def]
    split
    · rename_i hlt
      generalize hq : Array.qpartition as lt lo hi w hlo hhi = q
      obtain ⟨⟨mid, hmid⟩, as'⟩ := q
      have hp : as'.toArray.Perm as.toArray := by
        have := inQpartition_perm as lt lo hi w hlo hhi
        rw [hq] at this
        exact this
      simp only
      split
      · exact hp
      · rename_i hge
        refine ((ih (hi - (mid + 1)) (by omega) _ _ _ _ _ _ rfl).trans
          ((ih (mid - lo) (by omega) _ _ _ _ _ _ rfl))).trans hp
    · exact Array.Perm.refl _

theorem in_mem_qsort {α : Type} (as : Array α) (lt : α → α → Bool) (x : α) : x ∈ as.qsort lt ↔ x ∈ as := by
  unfold Array.qsort
  split
  · exact Iff.rfl
  · exact (inSort_perm lt _ _ _ _ _ _ _ rfl).mem_iff

/-! ### Folds that keep and add elements -/

theorem foldl_all {α β : Type} (P : β → Prop) (xs : Array α) (f : Array β → α → Array β) {init : Array β}
    (hinit : ∀ y ∈ init, P y) (hf : ∀ acc a, a ∈ xs → (∀ y ∈ acc, P y) → ∀ y ∈ f acc a, P y) :
    ∀ y ∈ xs.foldl f init, P y :=
  Array.foldl_induction (motive := fun _ acc => ∀ y ∈ acc, P y) hinit
    (fun i acc h => hf acc _ (Array.getElem_mem i.2) h)

theorem foldl_mono {α β : Type} (xs : Array α) (f : Array β → α → Array β) {init : Array β}
    (hf : ∀ acc a, ∀ y ∈ acc, y ∈ f acc a) : ∀ y ∈ init, y ∈ xs.foldl f init := fun y hy =>
  Array.foldl_induction (motive := fun _ acc => y ∈ acc) hy (fun _ acc h => hf acc _ y h)

theorem foldl_adds {α β : Type} (xs : Array α) (f : Array β → α → Array β) (init : Array β)
    (hf : ∀ acc a, ∀ y ∈ acc, y ∈ f acc a) {i : ℕ} (hi : i < xs.size) {y : β} (hy : ∀ acc, y ∈ f acc xs[i]) :
    y ∈ xs.foldl f init := by
  have := Array.foldl_induction (as := xs) (motive := fun k acc => i < k → y ∈ acc) (init := init) (f := f)
    (fun h => absurd h (Nat.not_lt_zero _)) (fun k acc h hk => by
      by_cases hik : i = k.1
      · subst hik
        exact hy acc
      · exact hf acc _ y (h (by omega)))
  exact this hi

/-! ### `HmIn.words` -/

theorem words_mem (c : Circuit) (pi : HmIn.Ports) {u jj : ℕ} (hq : (pi.leaves.getD u #[]).getD jj (-1) ≥ 0) :
    inCol c + jj / 8 ∈ HmIn.words c pi := by
  have hu : u < pi.leaves.size := by
    by_contra h
    simp [Array.getD, h] at hq
  have hq1 : pi.leaves[u].getD jj (-1) ≥ 0 := by simpa [Array.getD, hu] using hq
  have hj : jj < pi.leaves[u].size := by
    by_contra h
    simp [Array.getD, h] at hq1
  have hv : pi.leaves[u][jj]! ≥ 0 := by
    rw [getElem!_pos pi.leaves[u] jj hj]
    simpa [Array.getD, hj] using hq1
  unfold HmIn.words
  dsimp only
  have hin : ∀ (l : Array ℤ) (acc : Array ℕ), ∀ y ∈ acc, y ∈ (Array.range l.size).foldl
      (fun acc j => if l[j]! ≥ 0 then acc.push ((c.unit.inGroups.getD 0 default).col + j / 8) else acc) acc :=
    fun l acc => foldl_mono _ _ fun acc j y hy => by
      split
      · exact Array.mem_push_of_mem _ hy
      · exact hy
  have h1 := foldl_adds pi.leaves _ #[] (fun acc l => hin l acc) hu (y := inCol c + jj / 8) (fun acc =>
    foldl_adds _ _ acc (fun acc j y hy => by
        split
        · exact Array.mem_push_of_mem _ hy
        · exact hy)
      (by simpa using hj) (by
        intro acc'
        rw [Array.getElem_range]
        simp only [hv, ↓reduceIte]
        exact Array.mem_push_self))
  obtain ⟨i, hi, hgi⟩ := Array.mem_iff_getElem.1 ((in_mem_qsort _ _ _).2 h1)
  refine foldl_adds _ _ #[] (fun acc w y hy => by
      split
      · exact hy
      · exact Array.mem_push_of_mem _ hy) hi (fun acc => ?_)
  rw [hgi]
  split
  · rename_i hb
    exact Array.mem_of_back? (by simpa using hb)
  · exact Array.mem_push_self

theorem words_ge (c : Circuit) (pi : HmIn.Ports) : ∀ w ∈ HmIn.words c pi, inCol c ≤ w := by
  unfold HmIn.words
  dsimp only
  have hws : ∀ w ∈ pi.leaves.foldl (fun acc l => (Array.range l.size).foldl (fun acc j =>
      if l[j]! ≥ 0 then acc.push ((c.unit.inGroups.getD 0 default).col + j / 8) else acc) acc) #[], inCol c ≤ w :=
    foldl_all _ _ _ (by simp) fun acc l _ hacc => foldl_all _ _ _ hacc fun acc j _ hacc y hy => by
      split at hy
      · rcases Array.mem_push.1 hy with hy | rfl
        · exact hacc y hy
        · unfold inCol
          omega
      · exact hacc y hy
  refine foldl_all _ _ _ (by simp) fun acc w hw hacc y hy => ?_
  split at hy
  · exact hacc y hy
  · rcases Array.mem_push.1 hy with hy | rfl
    · exact hacc y hy
    · exact hws _ ((in_mem_qsort _ _ _).1 hw)

/-! ### `HmIn.blocks` -/

/-- The runs' fold step: extend the last run by `w` when `w` is its end, else start a run at `w`. -/
def runsStep (acc : Array (ℕ × ℕ)) (w : ℕ) : Array (ℕ × ℕ) :=
  match acc.back? with
  | some (a, b) => if w == b then acc.pop.push (a, b + 1) else acc.push (w, w + 1)
  | none => #[(w, w + 1)]

/-- Some run or block of `out` holds `z`. -/
def RunCov (out : Array (ℕ × ℕ)) (z : ℕ) : Prop := ∃ r ∈ out.toList, r.1 ≤ z ∧ z < r.2

/-- Some block of `out` (`(w, e)`, words `[w, w + 2^e)`) holds `z`. -/
def BlkCov (out : Array (ℕ × ℕ)) (z : ℕ) : Prop := ∃ r ∈ out.toList, r.1 ≤ z ∧ z < r.1 + 2 ^ r.2

theorem mem_of_back?_some {acc : Array (ℕ × ℕ)} {a b : ℕ} (h : acc.back? = some (a, b)) {r : ℕ × ℕ}
    (hr : r ∈ acc.toList) : r ∈ acc.pop.toList ∨ r = (a, b) := by
  have hl : acc.toList.getLast? = some (a, b) := by rw [Array.getLast?_toList]; exact h
  have := List.dropLast_append_getLast? (a, b) hl
  rw [← this] at hr
  simpa [Array.toList_pop] using hr

theorem runsStep_mono {acc : Array (ℕ × ℕ)} {z : ℕ} (w : ℕ) (h : RunCov acc z) : RunCov (runsStep acc w) z := by
  obtain ⟨r, hr, h1, h2⟩ := h
  unfold runsStep
  split
  · rename_i a b hb
    split
    · rcases mem_of_back?_some hb hr with hr | hr
      · refine ⟨r, ?_, h1, h2⟩
        simp only [Array.toList_push, List.mem_append]
        exact Or.inl hr
      · subst hr
        simp only at h1 h2
        exact ⟨(a, b + 1), by simp, by simp only; omega, by simp only; omega⟩
    · exact ⟨r, by simp [hr], h1, h2⟩
  · rename_i hn
    rw [Array.back?_eq_none_iff] at hn
    subst hn
    simp at hr

/-- Every run of `acc` is non-empty and starts at `L` or above. -/
def RunsOk (L : ℕ) (acc : Array (ℕ × ℕ)) : Prop := ∀ r ∈ acc.toList, L ≤ r.1 ∧ r.1 < r.2

theorem runsStep_ok {L : ℕ} {acc : Array (ℕ × ℕ)} {w : ℕ} (h : RunsOk L acc) (hw : L ≤ w) :
    RunsOk L (runsStep acc w) := by
  unfold runsStep
  split
  · rename_i a b hb
    have hab := h _ (by simpa using Array.mem_of_back? hb)
    split
    · intro r hr
      simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hr
      rcases hr with hr | rfl
      · rw [Array.toList_pop] at hr
        exact h r (List.mem_of_mem_dropLast hr)
      · simp only at hab ⊢
        omega
    · intro r hr
      simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hr
      rcases hr with hr | rfl
      · exact h r hr
      · simp only
        omega
  · intro r hr
    simp only [List.mem_singleton] at hr
    subst hr
    simp only
    omega

theorem runsStep_self {L : ℕ} {acc : Array (ℕ × ℕ)} (h : RunsOk L acc) (w : ℕ) : RunCov (runsStep acc w) w := by
  unfold runsStep
  split
  · rename_i a b hb
    have hab := h _ (by simpa using Array.mem_of_back? hb)
    split
    · rename_i hw
      have hw : w = b := by simpa using hw
      simp only at hab
      exact ⟨(a, b + 1), by simp, by simp only; omega, by simp only; omega⟩
    · exact ⟨(w, w + 1), by simp, le_refl _, by simp⟩
  · exact ⟨(w, w + 1), by simp, le_refl _, by simp⟩

theorem runs_cover (ws : Array ℕ) (L : ℕ) (hL : ∀ w ∈ ws, L ≤ w) :
    RunsOk L (ws.foldl runsStep #[]) ∧ ∀ y ∈ ws, RunCov (ws.foldl runsStep #[]) y := by
  have := Array.foldl_induction (as := ws) (init := #[]) (f := runsStep)
    (motive := fun k acc => RunsOk L acc ∧ ∀ i (hi : i < ws.size), i < k → RunCov acc ws[i])
    ⟨fun r hr => by simp at hr, fun i _ h => absurd h (Nat.not_lt_zero _)⟩ (fun k acc ⟨hok, hcov⟩ => by
      refine ⟨runsStep_ok hok (hL _ (Array.getElem_mem k.2)), fun i hi hik => ?_⟩
      by_cases hk : i = k.1
      · subst hk
        exact runsStep_self hok _
      · exact runsStep_mono _ (hcov i hi (by omega)))
  refine ⟨this.1, fun y hy => ?_⟩
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hy
  exact this.2 i hi hi

theorem forIn_range'_idx {β : Type} (f : ℕ → β → Id (ForInStep β)) : ∀ (k s : ℕ) (I : ℕ → β → Prop) (b : β),
    (∀ i < k, ∀ b, I i b → ∃ b', f (s + i) b = ForInStep.yield b' ∧ I (i + 1) b') → I 0 b →
      I k (forIn (List.range' s k) b f)
  | 0, s, I, b, _, hb => by
    rw [List.range'_zero, List.forIn_nil]
    exact hb
  | k + 1, s, I, b, hf, hb => by
    rw [List.range'_succ, List.forIn_cons]
    obtain ⟨b', h1, h2⟩ := hf 0 (by omega) b hb
    rw [Nat.add_zero] at h1
    rw [h1]
    exact forIn_range'_idx f k (s + 1) (fun i => I (i + 1)) b' (fun i hi b hb => by
      obtain ⟨b'', e, h⟩ := hf (i + 1) (by omega) b hb
      exact ⟨b'', by rw [← e, Nat.add_assoc, Nat.add_comm 1 i], h⟩) h2

theorem forIn_range_idx {β : Type} (I : ℕ → β → Prop) (n : ℕ) (f : ℕ → β → Id (ForInStep β))
    (hf : ∀ i < n, ∀ b, I i b → ∃ b', f i b = ForInStep.yield b' ∧ I (i + 1) b') {b : β} (hb : I 0 b) :
    I n (forIn [:n] b f) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  exact forIn_range'_idx f n 0 I b (fun i hi b hb => by rw [Nat.zero_add]; exact hf i hi b hb) hb

theorem forIn_list_each {α β : Type} (R : β → Prop) (Q : α → β → Prop) (f : α → β → Id (ForInStep β)) :
    ∀ (l : List α) (b : β),
      (∀ a ∈ l, ∀ b, R b → ∃ b', f a b = ForInStep.yield b' ∧ R b' ∧ Q a b' ∧ ∀ a', Q a' b → Q a' b') → R b →
        R (forIn l b f) ∧ (∀ a, Q a b → Q a (forIn l b f)) ∧ ∀ a ∈ l, Q a (forIn l b f)
  | [], b, _, hb => by
    rw [List.forIn_nil]
    exact ⟨hb, fun _ h => h, fun _ h => by simp at h⟩
  | a :: l, b, hf, hb => by
    rw [List.forIn_cons]
    obtain ⟨b', h1, hR, hQ, hmono⟩ := hf a List.mem_cons_self b hb
    rw [h1]
    obtain ⟨r1, r2, r3⟩ := forIn_list_each R Q f l b' (fun a ha => hf a (List.mem_cons_of_mem _ ha)) hR
    refine ⟨r1, fun a' h => r2 a' (hmono a' h), fun a' ha' => ?_⟩
    rcases List.mem_cons.1 ha' with rfl | ha'
    · exact r2 _ hQ
    · exact r3 a' ha'

/-- The cut of run `(a0, b)` after `i` steps, `s = (out, a)`: `a` past `a0` and at least `i` words on or at `b`, every
block from `L` up, `[a0, a)` covered, and whatever `out0` covered still covered. -/
def CutInv (L a0 b : ℕ) (out0 : Array (ℕ × ℕ)) (i : ℕ) (s : Array (ℕ × ℕ) × ℕ) : Prop :=
  a0 ≤ s.2 ∧ min b (a0 + i) ≤ s.2 ∧ (∀ r ∈ s.1.toList, L ≤ r.1) ∧ (∀ z, a0 ≤ z → z < s.2 → BlkCov s.1 z) ∧
    ∀ z, BlkCov out0 z → BlkCov s.1 z

theorem blkCov_push {out : Array (ℕ × ℕ)} {z : ℕ} (x : ℕ × ℕ) (h : BlkCov out z) : BlkCov (out.push x) z := by
  obtain ⟨r, hr, h1, h2⟩ := h
  exact ⟨r, by simp [hr], h1, h2⟩

theorem cutInv_step {L a0 b i : ℕ} {out0 : Array (ℕ × ℕ)} {s : Array (ℕ × ℕ) × ℕ} (e : ℕ)
    (h : CutInv L a0 b out0 i s) (hL : L ≤ a0) : CutInv L a0 b out0 (i + 1) (s.1.push (s.2, e), s.2 + 2 ^ e) := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  have he := Nat.one_le_two_pow (n := e)
  refine ⟨by simp only; omega, by simp only; omega, fun r hr => ?_, fun z hz1 hz2 => ?_,
    fun z hz => blkCov_push _ (h5 z hz)⟩
  · simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hr
    rcases hr with hr | rfl
    · exact h3 r hr
    · simp only
      omega
  · simp only at hz2
    by_cases hz : z < s.2
    · exact blkCov_push _ (h4 z hz1 hz)
    · exact ⟨(s.2, e), by simp, by simp only; omega, by simp only; omega⟩

theorem cutInv_stop {L a0 b i : ℕ} {out0 : Array (ℕ × ℕ)} {s : Array (ℕ × ℕ) × ℕ}
    (h : CutInv L a0 b out0 i s) (hb : ¬ s.2 < b) : CutInv L a0 b out0 (i + 1) (s.1, s.2) := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  exact ⟨h1, by simp only; omega, h3, h4, h5⟩

/-- One run `x = (a0, b)` cut into blocks: each step at `a < b` adds a block `(a, e)` and moves `a` past it. -/
theorem cut_run {L : ℕ} {x : ℕ × ℕ} (hLx : L ≤ x.1) (out : Array (ℕ × ℕ)) (hout : ∀ r ∈ out.toList, L ≤ r.1)
    (G : ℕ → Array (ℕ × ℕ) × ℕ → Id (ForInStep (Array (ℕ × ℕ) × ℕ)))
    (hG : ∀ i s, (s.2 < x.2 → ∃ e, G i s = ForInStep.yield (s.1.push (s.2, e), s.2 + 2 ^ e)) ∧
      (¬ s.2 < x.2 → G i s = ForInStep.yield (s.1, s.2))) :
    (∀ r ∈ (forIn [:x.2 - x.1] (out, x.1) G).1.toList, L ≤ r.1) ∧
      (∀ z, x.1 ≤ z → z < x.2 → BlkCov (forIn [:x.2 - x.1] (out, x.1) G).1 z) ∧
      ∀ z, BlkCov out z → BlkCov (forIn [:x.2 - x.1] (out, x.1) G).1 z := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := forIn_range_idx (CutInv L x.1 x.2 out) (x.2 - x.1) G (fun i _ s hs => by
      by_cases h : s.2 < x.2
      · obtain ⟨e, he⟩ := (hG i s).1 h
        exact ⟨_, he, cutInv_step e hs hLx⟩
      · exact ⟨_, (hG i s).2 h, cutInv_stop hs h⟩)
    (b := (out, x.1)) ⟨le_refl _, by simp only; omega, hout, fun z hz1 hz2 => absurd hz2 (by simp only; omega),
      fun z h => h⟩
  exact ⟨h3, fun z hz1 hz2 => h4 z hz1 (by omega), h5⟩

/-- The runs cut one after the other: every block from `L` up, and every word some run holds in a block. -/
theorem cut_loop (L : ℕ) (ws : Array ℕ) (runs : List (ℕ × ℕ)) (hcov : ∀ y ∈ ws, ∃ x ∈ runs, x.1 ≤ y ∧ y < x.2)
    (f : ℕ × ℕ → Array (ℕ × ℕ) → Id (ForInStep (Array (ℕ × ℕ))))
    (hf : ∀ x ∈ runs, ∀ out, (∀ r ∈ out.toList, L ≤ r.1) → ∃ out', f x out = ForInStep.yield out' ∧
      (∀ r ∈ out'.toList, L ≤ r.1) ∧ (∀ z, x.1 ≤ z → z < x.2 → BlkCov out' z) ∧ ∀ z, BlkCov out z → BlkCov out' z) :
    (∀ r ∈ (forIn (m := Id) runs (#[] : Array (ℕ × ℕ)) f).toList, L ≤ r.1) ∧
      ∀ y ∈ ws, BlkCov (forIn (m := Id) runs (#[] : Array (ℕ × ℕ)) f) y := by
  obtain ⟨r1, -, r3⟩ := forIn_list_each (fun out => ∀ r ∈ out.toList, L ≤ r.1)
    (fun x out => ∀ z, x.1 ≤ z → z < x.2 → BlkCov out z) f runs #[] (fun x hx out hout => by
      obtain ⟨out', h1, h2, h3, h4⟩ := hf x hx out hout
      exact ⟨out', h1, h2, h3, fun a' h z hz1 hz2 => h4 z (h z hz1 hz2)⟩) (by simp)
  refine ⟨r1, fun y hy => ?_⟩
  obtain ⟨x, hx, h1, h2⟩ := hcov y hy
  exact r3 x hx y h1 h2

/-- Every word of `ws`, all from `L` up, is in a block of `HmIn.blocks ws`, every block from `L` up. -/
theorem blocks_cover (ws : Array ℕ) (L : ℕ) (hL : ∀ w ∈ ws, L ≤ w) :
    (∀ r ∈ (HmIn.blocks ws).toList, L ≤ r.1) ∧ ∀ y ∈ ws, BlkCov (HmIn.blocks ws) y := by
  obtain ⟨hok, hcov⟩ := runs_cover ws L hL
  unfold HmIn.blocks
  simp only [Id.run, bind_pure]
  rw [← Array.forIn_toList]
  exact cut_loop L ws _ (fun y hy => hcov y hy) _ (fun x hx out hout =>
    ⟨_, rfl, cut_run (hok x hx).1 out hout _ (fun i s =>
      ⟨fun h => ⟨_, by simp only [h, ↓reduceIte]; rfl⟩,
        fun h => by simp only [h, ↓reduceIte]; rfl⟩)⟩)

/-! ### `HmIn.inRegions` -/

theorem split_cover {w0 e s y : ℕ} (hse : s ≤ e) (h1 : w0 ≤ y) (h2 : y < w0 + 2 ^ e) :
    (y - w0) / 2 ^ s < 2 ^ (e - s) ∧ w0 + (y - w0) / 2 ^ s * 2 ^ s ≤ y ∧
      y < w0 + (y - w0) / 2 ^ s * 2 ^ s + 2 ^ s := by
  have hP : 0 < 2 ^ s := Nat.two_pow_pos s
  have he : 2 ^ e = 2 ^ s * 2 ^ (e - s) := by rw [← Nat.pow_add]; congr 1; omega
  have hdm := Nat.div_add_mod (y - w0) (2 ^ s)
  have hm := Nat.mod_lt (y - w0) hP
  refine ⟨Nat.div_lt_of_lt_mul (by omega), ?_, ?_⟩ <;> rw [Nat.mul_comm] <;> omega

/-- `HmIn.inRegions`' fold over blocks `bs`: block `i`, `(w0, e)`, gives region `(w0 + r 2^s, s)` for each `r < 2^(e - s)`. -/
theorem regFold_mem (bs : Array (ℕ × ℕ)) (T : ℕ) {i : ℕ} (hi : i < bs.size) {w0 e r : ℕ} (hgi : bs[i] = (w0, e))
    (hr : r < 2 ^ (e - min e T)) :
    (w0 + r * 2 ^ min e T, min e T) ∈ bs.foldl (fun acc (w, e) =>
      let s := min e T
      (Array.range (2 ^ (e - s))).foldl (fun acc r => acc.push (w + r * 2 ^ s, s)) acc) #[] := by
  refine foldl_adds _ _ #[] (fun acc we z hz => ?_) hi (fun acc => ?_)
  · obtain ⟨w, e⟩ := we
    exact foldl_mono _ _ (fun acc r z hz => Array.mem_push_of_mem _ hz) z hz
  · rw [hgi]
    exact foldl_adds _ _ acc (fun acc r z hz => Array.mem_push_of_mem _ hz) (by simpa using hr)
      (fun acc' => by rw [Array.getElem_range]; exact Array.mem_push_self)

theorem inRegions_eq (c : Circuit) (pi : HmIn.Ports) : HmIn.inRegions c pi =
    (HmIn.blocks (HmIn.words c pi)).foldl (fun acc (w, e) =>
      let s := min e (PT_LOCAL - (7 + log2ceil c.unitRange.count))
      (Array.range (2 ^ (e - s))).foldl (fun acc r => acc.push (w + r * 2 ^ s, s)) acc) #[] := rfl

theorem getElem!_of_eq {α : Type} [Inhabited α] {xs : Array α} {i : ℕ} {a : α} (h : i < xs.size) (he : xs[i] = a) :
    xs[i]! = a := by
  rw [getElem!_pos xs i h]
  exact he

/-- Every word of `HmIn.words` is in an in region `(w, s)` from a word of the leaf group. -/
theorem inRegions_cover (c : Circuit) (pi : HmIn.Ports) {y : ℕ} (hy : y ∈ HmIn.words c pi) :
    ∃ x w s, x < (HmIn.inRegions c pi).size ∧ (HmIn.inRegions c pi)[x]! = (w, s) ∧ inCol c ≤ w ∧ w ≤ y ∧
      y < w + 2 ^ s := by
  obtain ⟨hL, hcov⟩ := blocks_cover _ (inCol c) (words_ge c pi)
  obtain ⟨⟨w0, e⟩, hblk, h1, h2⟩ := hcov y hy
  have hw0 := hL _ hblk
  simp only at h1 h2 hw0
  obtain ⟨i, hi, hgi⟩ := List.mem_iff_getElem.1 hblk
  rw [Array.getElem_toList] at hgi
  have hi' : i < (HmIn.blocks (HmIn.words c pi)).size := by simpa using hi
  generalize hT : PT_LOCAL - (7 + log2ceil c.unitRange.count) = T
  obtain ⟨hr, hlo, hhi⟩ := split_cover (Nat.min_le_left e T) h1 h2
  have hmem := regFold_mem _ T hi' hgi hr
  rw [← hT, ← inRegions_eq] at hmem
  obtain ⟨x, hx, hgx⟩ := Array.mem_iff_getElem.1 hmem
  rw [hT] at hgx
  exact ⟨x, _, _, hx, getElem!_of_eq hx hgx, by omega, hlo, hhi⟩

/-- Every public leaf `jj` of a unit's public leaf map is in an in region `x`, `(w, s)`, from a word of the leaf group:
`in_pinned`'s hypotheses on `x`. -/
theorem in_cover (c : Circuit) (pi : HmIn.Ports) {u jj : ℕ} (hq : (pi.leaves.getD u #[]).getD jj (-1) ≥ 0) :
    ∃ x w s, x < (HmIn.inRegions c pi).size ∧ (HmIn.inRegions c pi)[x]! = (w, s) ∧ inCol c ≤ w ∧
      w ≤ inCol c + jj / 8 ∧ inCol c + jj / 8 < w + 2 ^ s :=
  inRegions_cover c pi (words_mem c pi hq)

/-- **A public leaf's bits are its instance's public word**: in a satisfying witness of the model's statement at the
executable's regions, unit slot `k`'s public leaf `jj` (public word `q` of the unit's leaf map) at column `16 jj + t`
of the leaf group holds bit `t` of public word `q` of the slot's instance `i` (`in_cover`, `in_pinned`,
`inVal_public`). -/
theorem public_pinned {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    {pi : HmIn.Ports} {ri : Array Region} (hR : HmIn.regions st.c pi st.pub = .ok ri)
    (hri : ∀ r ∈ ri.toList, r ∈ st.regions.toList) (hne : pi.ports.isEmpty = false)
    {z : Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {k jj t i : ℕ} {q : ℤ} (hq : (pi.leaves.getD (k % st.c.upv) #[]).getD jj (-1) = q) (hq0 : 0 ≤ q)
    (hk : k < st.c.unitRange.count) (hi : st.c.instOf st.pub.n o.val (k / st.c.upv) = some i) (ht : t < 16) :
    bit ((L.model regs).block z o (inBlockOr st.c.kLog (inAddr st.c k jj t))) =
      (st.pub.outputs.getD (i * pi.words + q.toNat) 0).testBit t := by
  obtain ⟨x, w, s, hx, hws, hcol, hlo, hhi⟩ := in_cover st.c pi (u := k % st.c.upv) (jj := jj) (by rw [hq]; exact hq0)
  rw [in_pinned L regs hregs hR hri hne hz o hx hws hcol hlo hhi hk ht, inVal_public hk hi hq hq0]

end FlockSoundness.Discharge.Layout
