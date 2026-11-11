module

public import Definitions.Claims.SamplingTwo

/-!
# The two-stage scope, decided in passes over the program

`SpotFlock.scopeOk` decides the two-stage scope as the claims state it, and as stated its checks walk lists by
position: the program's gates are a list that `topoOk`, the coverage check and `Prog.Boundary` index at every gate;
`Prog.Unread` scans every gate's reads at every gate; `Prog.Boundary` scans every gate at each gate it is asked
about, and `unitClass` asks `committed` once for each size class it tries; `Nodup` compares every pair; and the
replay check asks `replayOf` at every proof block, which scans the replay blocks by position. So the size classes take
time cubic in the program's gates and the rest quadratic. These decide the same from arrays marked in one pass each,
equal to the statement's checks; `SpotFlock.Code` compiles `topoOk`, `partitionsOk` and `scopeOk` as them
(`@[csimp]`):

* `markWhere`, `markEach`: the entries of a Boolean array that a list names, or that each element of a list names,
  marked (`getD_markWhere`, `getD_markEach`).
* `firstMarks n l`: the entries `l` names, each marked once, none where `l` repeats one or names one at or past `n`
  (`firstMarks_isSome`, `getD_firstMarks`); `nodupFast`, `Nodup` by it (`nodupFast_eq`).
* `blockIdx n bs`: each gate's block, `k + 1` for block `k` and `0` for none, where no two blocks share a gate
  (`getD_blockIdx`, `getD_blockIdx_of_not_mem`, `mem_iff_getD_blockIdx`).
* `topoFast`, `partitionsFast` and `replaysFast`: `topoOk`, `partitionsOk` and the replay check, reading the gates
  from an array and every gate's returns, reads, block and cross-block reads from marks (`topoFast_eq`,
  `partitionsFast_eq`, `replaysFast_eq`), each proof block's committed values counted once (`committedAt_eq`).
-/

@[expose] public section

namespace SpotFlock

open Claims.Sampling

theorem all_congr_of_mem {α : Type} {l : List α} {p q : α → Bool} (h : ∀ x ∈ l, p x = q x) : l.all p = l.all q := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [List.all_cons, List.all_cons, h x List.mem_cons_self, ih fun y hy => h y (List.mem_cons_of_mem _ hy)]

theorem any_congr_of_mem {α : Type} {l : List α} {p q : α → Bool} (h : ∀ x ∈ l, p x = q x) : l.any p = l.any q := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [List.any_cons, List.any_cons, h x List.mem_cons_self, ih fun y hy => h y (List.mem_cons_of_mem _ hy)]

theorem getD_setIfInBounds_of_lt {α : Type} (a : Array α) (i j : ℕ) (v d : α) (hj : j < a.size) :
    (a.setIfInBounds i v).getD j d = if i = j then v else a.getD j d := by
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds]
  by_cases h : i = j
  · subst h
    simp [hj]
  · simp [h]

theorem getD_of_size_le {α : Type} (a : Array α) {i : ℕ} (d : α) (h : a.size ≤ i) : a.getD i d = d := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none h, Option.getD_none]

/-! ## Marks -/

/-- **The entries `l` names that `p` holds of, marked** in `a`: each set `true`, any past `a`'s end left out. -/
def markWhere (p : ℕ → Bool) (l : List ℕ) (a : Array Bool) : Array Bool :=
  l.foldl (fun a i => if p i then a.setIfInBounds i true else a) a

theorem size_markWhere (p : ℕ → Bool) (l : List ℕ) (a : Array Bool) : (markWhere p l a).size = a.size := by
  induction l generalizing a with
  | nil => rfl
  | cons i l ih =>
    show (markWhere p l (if p i then a.setIfInBounds i true else a)).size = a.size
    rw [ih]
    split <;> simp

theorem getD_markWhere (p : ℕ → Bool) (l : List ℕ) (a : Array Bool) {i : ℕ} (hi : i < a.size) :
    (markWhere p l a).getD i false = (a.getD i false || (l.contains i && p i)) := by
  induction l generalizing a with
  | nil => simp [markWhere]
  | cons j l ih =>
    have hi' : i < (if p j then a.setIfInBounds j true else a).size := by
      split <;> simpa using hi
    show (markWhere p l (if p j then a.setIfInBounds j true else a)).getD i false = _
    rw [ih _ hi']
    by_cases hp : p j = true
    · rw [ite_eq_left hp, getD_setIfInBounds_of_lt _ _ _ _ _ hi]
      by_cases hji : j = i
      · subst hji
        simp [hp]
      · simp [hji, Ne.symm hji]
    · rw [ite_eq_right hp]
      have hp' : p j = false := by simpa using hp
      by_cases hji : j = i
      · subst hji
        simp [hp']
      · simp [Ne.symm hji]

/-- **The entries each element of `xs` names, marked** in `a`: for each `x`, the entries `f x` that `q x` holds of. -/
def markEach {β : Type} (q : β → ℕ → Bool) (f : β → List ℕ) (xs : List β) (a : Array Bool) : Array Bool :=
  xs.foldl (fun a x => markWhere (q x) (f x) a) a

theorem getD_markEach {β : Type} (q : β → ℕ → Bool) (f : β → List ℕ) (xs : List β) (a : Array Bool) {i : ℕ}
    (hi : i < a.size) :
    (markEach q f xs a).getD i false = (a.getD i false || xs.any fun x => (f x).contains i && q x i) := by
  induction xs generalizing a with
  | nil => simp [markEach]
  | cons x xs ih =>
    have hi' : i < (markWhere (q x) (f x) a).size := by rw [size_markWhere]; exact hi
    show (markEach q f xs (markWhere (q x) (f x) a)).getD i false = _
    rw [ih _ hi', getD_markWhere _ _ _ hi, List.any_cons, Bool.or_assoc]

/-- One entry of `firstMarks`: none once entry `i` is marked already or past `a`'s end, else the rest marked after
it. -/
def firstMarksIn : List ℕ → Array Bool → Option (Array Bool)
  | [], a => some a
  | i :: l, a => if a.getD i true then none else firstMarksIn l (a.setIfInBounds i true)

/-- **The entries `l` names, each marked once**, in an array of `n`: none where `l` names one twice or one at or past
`n` (`firstMarks_isSome`, `getD_firstMarks`). -/
def firstMarks (n : ℕ) (l : List ℕ) : Option (Array Bool) := firstMarksIn l (Array.replicate n false)

theorem lt_of_getD_true {a : Array Bool} {i : ℕ} (h : a.getD i true = false) : i < a.size := by
  by_cases hn : i < a.size
  · exact hn
  · rw [getD_of_size_le a true (Nat.le_of_not_lt hn)] at h
    exact absurd h (by decide)

theorem firstMarksIn_isSome (l : List ℕ) (a : Array Bool) :
    (firstMarksIn l a).isSome = true ↔ l.Nodup ∧ ∀ i ∈ l, a.getD i true = false := by
  induction l generalizing a with
  | nil => simp [firstMarksIn]
  | cons i l ih =>
    rw [firstMarksIn]
    by_cases h : a.getD i true = true
    · rw [ite_eq_left h]
      simp only [Option.isSome_none, Bool.false_eq_true, false_iff, not_and]
      intro _ hall
      rw [hall i List.mem_cons_self] at h
      exact absurd h (by decide)
    · rw [ite_eq_right h, ih]
      have h' : a.getD i true = false := by simpa using h
      have hi : i < a.size := lt_of_getD_true h'
      have key : ∀ j, (a.setIfInBounds i true).getD j true = false ↔ j ≠ i ∧ a.getD j true = false := by
        intro j
        by_cases hj : j < a.size
        · rw [getD_setIfInBounds_of_lt _ _ _ _ _ hj]
          by_cases hij : i = j
          · subst hij
            simp
          · simp [hij, Ne.symm hij]
        · have e1 : (a.setIfInBounds i true).getD j true = true :=
            getD_of_size_le _ true (by simpa using Nat.le_of_not_lt hj)
          rw [e1, getD_of_size_le a true (Nat.le_of_not_lt hj)]
          simp
      constructor
      · rintro ⟨hnd, hall⟩
        refine ⟨List.nodup_cons.2 ⟨fun hm => ((key i).1 (hall i hm)).1 rfl, hnd⟩, fun j hj => ?_⟩
        rcases List.mem_cons.1 hj with rfl | hj
        · exact h'
        · exact ((key j).1 (hall j hj)).2
      · rintro ⟨hnd, hall⟩
        obtain ⟨hni, hnd'⟩ := List.nodup_cons.1 hnd
        exact ⟨hnd', fun j hj => (key j).2 ⟨fun hji => hni (hji ▸ hj), hall j (List.mem_cons_of_mem _ hj)⟩⟩

theorem firstMarks_isSome (n : ℕ) (l : List ℕ) :
    (firstMarks n l).isSome = true ↔ l.Nodup ∧ ∀ i ∈ l, i < n := by
  have key : ∀ i, (Array.replicate n false).getD i true = false ↔ i < n := by
    intro i
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
    by_cases hi : i < n
    · simp [hi]
    · simp [hi]
  rw [firstMarks, firstMarksIn_isSome]
  simp only [key]

theorem getD_firstMarksIn (l : List ℕ) (a b : Array Bool) (h : firstMarksIn l a = some b) {i : ℕ} (hi : i < a.size) :
    b.getD i false = (a.getD i false || l.contains i) := by
  induction l generalizing a with
  | nil =>
    rw [firstMarksIn, Option.some.injEq] at h
    subst h
    simp
  | cons j l ih =>
    rw [firstMarksIn] at h
    by_cases hj : a.getD j true = true
    · rw [ite_eq_left hj] at h
      cases h
    · rw [ite_eq_right hj] at h
      rw [ih _ h (by simpa using hi), getD_setIfInBounds_of_lt _ _ _ _ _ hi]
      by_cases hji : j = i
      · subst hji
        simp
      · simp [hji, Ne.symm hji]

theorem getD_firstMarks {n : ℕ} {l : List ℕ} {b : Array Bool} (h : firstMarks n l = some b) {i : ℕ} (hi : i < n) :
    b.getD i false = l.contains i := by
  rw [getD_firstMarksIn l _ b h (by simpa using hi)]
  simp [hi]

theorem foldl_max_bound : ∀ (l : List ℕ) (a : ℕ), a ≤ l.foldl max a ∧ ∀ i ∈ l, i ≤ l.foldl max a
  | [], a => ⟨Nat.le_refl a, fun _ h => absurd h List.not_mem_nil⟩
  | x :: l, a => by
    obtain ⟨h1, h2⟩ := foldl_max_bound l (max a x)
    refine ⟨Nat.le_trans (Nat.le_max_left a x) h1, fun i hi => ?_⟩
    rcases List.mem_cons.1 hi with rfl | hi
    · exact Nat.le_trans (Nat.le_max_right _ _) h1
    · exact h2 i hi

/-- **`Nodup`, by marks**: `l` names no entry twice, each marked once in an array one past its largest
(`nodupFast_eq`). -/
def nodupFast (l : List ℕ) : Bool := (firstMarks (l.foldl max 0 + 1) l).isSome

theorem nodupFast_eq (l : List ℕ) : nodupFast l = decide l.Nodup := by
  rw [Bool.eq_iff_iff, decide_eq_true_eq, nodupFast, firstMarks_isSome]
  exact ⟨fun h => h.1, fun h => ⟨h, fun i hi => Nat.lt_succ_of_le ((foldl_max_bound l 0).2 i hi)⟩⟩

/-! ## Each gate's block -/

/-- **Each gate of `u` given block index `v`** in `a`; a gate past `a`'s end left out. -/
def setBlock (v : ℕ) (u : List ℕ) (a : Array ℕ) : Array ℕ := u.foldl (fun a g => a.setIfInBounds g v) a

theorem size_setBlock (v : ℕ) (u : List ℕ) (a : Array ℕ) : (setBlock v u a).size = a.size := by
  induction u generalizing a with
  | nil => rfl
  | cons g u ih =>
    show (setBlock v u (a.setIfInBounds g v)).size = a.size
    rw [ih, Array.size_setIfInBounds]

theorem getD_setBlock (v : ℕ) (u : List ℕ) (a : Array ℕ) {g : ℕ} (hg : g < a.size) :
    (setBlock v u a).getD g 0 = if g ∈ u then v else a.getD g 0 := by
  induction u generalizing a with
  | nil => simp [setBlock]
  | cons x u ih =>
    have hg' : g < (a.setIfInBounds x v).size := by simpa using hg
    show (setBlock v u (a.setIfInBounds x v)).getD g 0 = _
    rw [ih _ hg', getD_setIfInBounds_of_lt _ _ _ _ _ hg]
    by_cases hx : x = g
    · subst hx
      simp
    · simp [hx, Ne.symm hx]

/-- Each gate's block among `bs`, `bs`' first block numbered `j + 1`. -/
def blockIdxFrom : ℕ → List (List ℕ) → Array ℕ → Array ℕ
  | _, [], a => a
  | j, u :: bs, a => blockIdxFrom (j + 1) bs (setBlock (j + 1) u a)

/-- **Each gate's block** among `bs`, for the gates under `n`: `k + 1` for a gate of block `k` and `0` for a gate of
none, where no two blocks share a gate (`getD_blockIdx`, `getD_blockIdx_of_not_mem`). -/
def blockIdx (n : ℕ) (bs : List (List ℕ)) : Array ℕ := blockIdxFrom 0 bs (Array.replicate n 0)

theorem getD_blockIdxFrom_of_not_mem : ∀ (j : ℕ) (bs : List (List ℕ)) (a : Array ℕ) (g : ℕ), g < a.size →
    g ∉ bs.flatten → (blockIdxFrom j bs a).getD g 0 = a.getD g 0
  | _, [], _, _, _, _ => rfl
  | j, u :: bs, a, g, hg, hn => by
    rw [List.flatten_cons, List.mem_append, not_or] at hn
    rw [blockIdxFrom, getD_blockIdxFrom_of_not_mem (j + 1) bs (setBlock (j + 1) u a) g
      (by rw [size_setBlock]; exact hg) hn.2, getD_setBlock _ _ _ hg, ite_eq_right hn.1]

theorem getD_blockIdxFrom : ∀ (j : ℕ) (bs : List (List ℕ)) (a : Array ℕ), bs.flatten.Nodup → ∀ (k : ℕ) (u : List ℕ),
    bs[k]? = some u → ∀ g, g < a.size → g ∈ u → (blockIdxFrom j bs a).getD g 0 = j + k + 1
  | _, [], _, _, _, _, hk, _, _, _ => by simp at hk
  | j, x :: bs, a, hnd, k, u, hk, g, hg, hu => by
    rw [List.flatten_cons, List.nodup_append] at hnd
    rw [blockIdxFrom]
    cases k with
    | zero =>
      rw [List.getElem?_cons_zero, Option.some.injEq] at hk
      subst hk
      rw [Nat.add_zero, getD_blockIdxFrom_of_not_mem (j + 1) bs (setBlock (j + 1) x a) g
        (by rw [size_setBlock]; exact hg) (fun hm => hnd.2.2 g hu g hm rfl), getD_setBlock _ _ _ hg, ite_eq_left hu]
    | succ k =>
      rw [List.getElem?_cons_succ] at hk
      rw [getD_blockIdxFrom (j + 1) bs (setBlock (j + 1) x a) hnd.2.1 k u hk g (by rw [size_setBlock]; exact hg) hu]
      omega

theorem getD_blockIdx {n : ℕ} {bs : List (List ℕ)} (hnd : bs.flatten.Nodup) {k : ℕ} {u : List ℕ}
    (hk : bs[k]? = some u) {g : ℕ} (hg : g < n) (hu : g ∈ u) : (blockIdx n bs).getD g 0 = k + 1 := by
  rw [blockIdx, getD_blockIdxFrom 0 bs (Array.replicate n 0) hnd k u hk g (by simpa using hg) hu, Nat.zero_add]

theorem getD_blockIdx_of_not_mem {n : ℕ} {bs : List (List ℕ)} {g : ℕ} (hg : g < n) (hn : g ∉ bs.flatten) :
    (blockIdx n bs).getD g 0 = 0 := by
  rw [blockIdx, getD_blockIdxFrom_of_not_mem 0 bs (Array.replicate n 0) g (by simpa using hg) hn]
  simp [hg]

theorem mem_iff_getD_blockIdx {n : ℕ} {bs : List (List ℕ)} (hnd : bs.flatten.Nodup) {k : ℕ} {u : List ℕ}
    (hk : bs[k]? = some u) {g : ℕ} (hg : g < n) : g ∈ u ↔ (blockIdx n bs).getD g 0 = k + 1 := by
  refine ⟨getD_blockIdx hnd hk hg, fun he => ?_⟩
  by_cases hm : g ∈ bs.flatten
  · obtain ⟨u', hu', hgu'⟩ := List.mem_flatten.1 hm
    obtain ⟨k', hk'⟩ := List.getElem?_of_mem hu'
    have hkk : k' = k := by
      have := getD_blockIdx hnd hk' hg hgu'
      omega
    subst hkk
    rw [hk] at hk'
    exact Option.some.inj hk' ▸ hgu'
  · rw [getD_blockIdx_of_not_mem hg hm] at he
    omega

/-! ## The scope's checks -/

/-- **Gate `g` is a constant**, read from the gates' array `gs` (`SpotFlock.constAt`'s check). -/
def constArr (gs : Array BGate) (g : ℕ) : Bool :=
  match gs[g]? with
  | some (.const _) => true
  | _ => false

/-- **`topoOk`, reading the gates from an array**: every gate reads only earlier gates, and every returned gate is
one of them (`topoFast_eq`). -/
def topoFast (C : Prog) : Bool :=
  let gs := C.gates.toArray
  (List.range gs.size).all (fun g => ((gs[g]?.map BGate.reads).getD []).all (· < g)) && C.outs.all (· < gs.size)

theorem topoFast_eq (C : Prog) : topoFast C =
    ((List.range C.gates.length).all (fun g => ((C.gates[g]?.map BGate.reads).getD []).all (· < g)) &&
      C.outs.all (· < C.gates.length)) := by
  simp only [topoFast, List.getElem?_toArray, List.size_toArray]

/-- **The gates `C` returns**, marked, in an array of `n`. -/
def outMarks (C : Prog) (n : ℕ) : Array Bool := markWhere (fun _ => true) C.outs (Array.replicate n false)

/-- **The gates some gate of `C` reads**, marked, in an array of `n`. -/
def readMarks (C : Prog) (n : ℕ) : Array Bool :=
  markEach (fun _ _ => true) BGate.reads C.gates (Array.replicate n false)

/-- **The gates a gate of another block reads**: each gate under `n` that a gate of `gs` under `n` reads from another
block than its own (`blk`, `blockIdx`'s), marked. -/
def crossMarks (gs : Array BGate) (blk : Array ℕ) (n : ℕ) : Array Bool :=
  markEach (fun g i => blk.getD g 0 != blk.getD i 0) (fun g => (gs[g]?.map BGate.reads).getD []) (List.range n)
    (Array.replicate n false)

theorem getD_outMarks (C : Prog) {n g : ℕ} (hg : g < n) : (outMarks C n).getD g false = C.outs.contains g := by
  have hg' : g < (Array.replicate n false).size := by simpa using hg
  rw [outMarks, getD_markWhere _ _ _ hg']
  simp [hg]

theorem getD_readMarks (C : Prog) {n g : ℕ} (hg : g < n) :
    (readMarks C n).getD g false = C.gates.any fun x => x.reads.contains g := by
  have hg' : g < (Array.replicate n false).size := by simpa using hg
  rw [readMarks, getD_markEach _ _ _ _ hg']
  simp [hg]

theorem getD_crossMarks (gs : Array BGate) (blk : Array ℕ) {n i : ℕ} (hi : i < n) :
    (crossMarks gs blk n).getD i false =
      (List.range n).any fun g => ((gs[g]?.map BGate.reads).getD []).contains i && (blk.getD g 0 != blk.getD i 0) := by
  have hi' : i < (Array.replicate n false).size := by simpa using hi
  rw [crossMarks, getD_markEach _ _ _ _ hi']
  simp [hi]

/-- **Gate `g` is an unread input** (`Prog.Unread`), read from the gates' array `gs`, the returned gates `out` and the
gates read `rd`. -/
def unreadAt (gs : Array BGate) (out rd : Array Bool) (g : ℕ) : Bool :=
  decide (gs[g]? = some .input) && !out.getD g false && !rd.getD g false

theorem unreadAt_eq {C : Prog} {g : ℕ} (hg : g < C.gates.length) :
    unreadAt C.gates.toArray (outMarks C C.gates.length) (readMarks C C.gates.length) g = decide (C.Unread g) := by
  rw [unreadAt, getD_outMarks C hg, getD_readMarks C hg, List.getElem?_toArray, Bool.eq_iff_iff, decide_eq_true_iff]
  simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not,
    List.any_eq_false, Prog.Unread, and_assoc]

/-- **Every gate of `C` is covered**: a constant, an unread input (`unreadAt`) or marked in `inP`, the gates read from
the array `gs`. -/
def coverFast (gs : Array BGate) (out rd inP : Array Bool) : Bool :=
  (List.range gs.size).all fun g => constArr gs g || unreadAt gs out rd g || inP.getD g false

theorem coverFast_eq {C : Prog} {P : Partition} {inP : Array Bool} (hm : firstMarks C.gates.length P.gates = some inP) :
    coverFast C.gates.toArray (outMarks C C.gates.length) (readMarks C C.gates.length) inP =
      (List.range C.gates.length).all fun g =>
        constArr C.gates.toArray g || decide (C.Unread g) || P.gates.contains g := by
  rw [coverFast, List.size_toArray]
  apply all_congr_of_mem
  intro g hg
  rw [List.mem_range] at hg
  rw [getD_firstMarks hm hg, unreadAt_eq hg]

/-- **The values proof block `u` commits** (`committed`), from marks: its gates returned (`out`), inputs, or read from
another block (`cross`), the gates read from the array `gs`. -/
def committedAt (gs : Array BGate) (out cross : Array Bool) (u : List ℕ) : ℕ :=
  (u.filter fun g => out.getD g false || decide (gs[g]? = some .input) || cross.getD g false).length

theorem committedAt_eq {C : Prog} {P : Partition} (hnd : P.gates.Nodup) (hlt : ∀ g ∈ P.gates, g < C.gates.length)
    {u : List ℕ} (hu : u ∈ P.blocks) :
    committedAt C.gates.toArray (outMarks C C.gates.length)
      (crossMarks C.gates.toArray (blockIdx C.gates.length P.blocks) C.gates.length) u = committed C u := by
  obtain ⟨k, hk⟩ := List.getElem?_of_mem hu
  rw [committedAt, committed]
  congr 1
  apply List.filter_congr
  intro g hg
  have hgn : g < C.gates.length := hlt g (List.mem_flatten.2 ⟨u, hu, hg⟩)
  rw [getD_outMarks C hgn, getD_crossMarks _ _ hgn, List.getElem?_toArray, getD_blockIdx hnd hk hgn hg, Prog.Boundary]
  congr 1
  apply any_congr_of_mem
  intro g' hg'
  rw [List.mem_range] at hg'
  rw [List.getElem?_toArray, Bool.and_comm]
  congr 1
  rw [Bool.eq_iff_iff, bne_iff_ne, decide_eq_true_eq, ne_eq, mem_iff_getD_blockIdx hnd hk hg']

/-- **Every proof block of `bs` fits a size class**, its committed values (`committedAt`) counted once. -/
def classesFast (gs : Array BGate) (out cross : Array Bool) (bs : List (List ℕ)) : Bool :=
  bs.all fun u =>
    let c := committedAt gs out cross u
    (sizeClasses.find? fun w => decide (c ≤ w)).any fun w => decide (c ≤ w)

theorem classesFast_eq {C : Prog} {P : Partition} (hnd : P.gates.Nodup) (hlt : ∀ g ∈ P.gates, g < C.gates.length) :
    classesFast C.gates.toArray (outMarks C C.gates.length)
        (crossMarks C.gates.toArray (blockIdx C.gates.length P.blocks) C.gates.length) P.blocks =
      P.blocks.all fun u => (unitClass C u).any fun w => decide (committed C u ≤ w) := by
  rw [classesFast]
  apply all_congr_of_mem
  intro u hu
  rw [committedAt_eq hnd hlt hu]
  rfl

/-- **`partitionsOk`, in passes over the gates**: `P`'s gates marked once each (`firstMarks`: none where one repeats
or is past the program's gates), then the coverage read from the marks and the returned and read gates
(`coverFast`), and each proof block's size class from each gate's block (`blockIdx`) and the gates read from another
block (`classesFast`); equal to `partitionsOk` (`partitionsFast_eq`). -/
def partitionsFast (P : Partition) (C : Prog) : Bool :=
  let gs := C.gates.toArray
  let n := gs.size
  match firstMarks n P.gates with
  | none => false
  | some inP =>
    let out := outMarks C n
    coverFast gs out (readMarks C n) inP && P.blocks.all (fun u => !u.isEmpty) &&
      classesFast gs out (crossMarks gs (blockIdx n P.blocks) n) P.blocks

theorem partitionsFast_eq (P : Partition) (C : Prog) : partitionsFast P C =
    (decide P.gates.Nodup && P.gates.all (· < C.gates.length) &&
      (List.range C.gates.length).all
        (fun g => constArr C.gates.toArray g || decide (C.Unread g) || P.gates.contains g) &&
      P.blocks.all (fun u => !u.isEmpty) &&
      P.blocks.all fun u => (unitClass C u).any fun w => decide (committed C u ≤ w)) := by
  simp only [partitionsFast, List.size_toArray]
  cases hm : firstMarks C.gates.length P.gates with
  | none =>
    have h1 : (decide P.gates.Nodup && P.gates.all (· < C.gates.length)) = false := by
      rw [Bool.eq_false_iff]
      intro h
      have h' : P.gates.Nodup ∧ ∀ i ∈ P.gates, i < C.gates.length := by
        simpa only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] using h
      have hs := (firstMarks_isSome _ _).2 h'
      rw [hm] at hs
      exact absurd hs (by simp)
    simp only [h1, Bool.false_and]
  | some inP =>
    have hs : (firstMarks C.gates.length P.gates).isSome = true := by simp [hm]
    obtain ⟨hnd, hlt⟩ := (firstMarks_isSome _ _).1 hs
    have h1 : (decide P.gates.Nodup && P.gates.all (· < C.gates.length)) = true := by
      simpa only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] using And.intro hnd hlt
    simp only [h1, Bool.true_and]
    rw [coverFast_eq hm, classesFast_eq hnd hlt]

/-- **Proof block `u` lies inside one replay block**, read from each gate's replay block `rb` (`blockIdx`'s): its first
gate is in one and every gate of it in the same; an empty one where `nb`, there being a replay block. -/
def inOneBlock (rb : Array ℕ) (nb : Bool) : List ℕ → Bool
  | [] => nb
  | g :: us => rb.getD g 0 != 0 && (g :: us).all (rb.getD · 0 == rb.getD g 0)

theorem inOneBlock_iff {Pρ : Partition} {n : ℕ} (hnd : Pρ.gates.Nodup) {u : List ℕ} (hun : ∀ g ∈ u, g < n) :
    inOneBlock (blockIdx n Pρ.blocks) (!Pρ.blocks.isEmpty) u = true ↔ ∃ r ∈ Pρ.blocks, ∀ g ∈ u, g ∈ r := by
  cases u with
  | nil =>
    rw [inOneBlock]
    cases h : Pρ.blocks with
    | nil => exact iff_of_false (by simp) fun ⟨_, hr, _⟩ => absurd hr List.not_mem_nil
    | cons r rs => exact iff_of_true (by simp) ⟨r, List.mem_cons_self, fun _ hg => absurd hg List.not_mem_nil⟩
  | cons g us =>
    have hg : g < n := hun g List.mem_cons_self
    rw [inOneBlock]
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, List.all_eq_true, beq_iff_eq]
    constructor
    · rintro ⟨hk, hall⟩
      have hm : g ∈ Pρ.blocks.flatten := by
        by_cases hm : g ∈ Pρ.blocks.flatten
        · exact hm
        · exact absurd (getD_blockIdx_of_not_mem hg hm) hk
      obtain ⟨r, hr, hgr⟩ := List.mem_flatten.1 hm
      obtain ⟨j, hj⟩ := List.getElem?_of_mem hr
      have hgj := getD_blockIdx hnd hj hg hgr
      refine ⟨r, hr, fun x hx => ?_⟩
      have hx' := hall x hx
      rw [hgj] at hx'
      exact (mem_iff_getD_blockIdx hnd hj (hun x hx)).2 hx'
    · rintro ⟨r, hr, hsub⟩
      obtain ⟨j, hj⟩ := List.getElem?_of_mem hr
      rw [getD_blockIdx hnd hj hg (hsub g List.mem_cons_self)]
      exact ⟨by omega, fun x hx => getD_blockIdx hnd hj (hun x hx) (hsub x hx)⟩

theorem replayOf_isSome_iff {Pt Pρ : Partition} {v : ℕ} (hv : v < Pt.blocks.length) :
    (replayOf Pt Pρ v).isSome = true ↔ ∃ r ∈ Pρ.blocks, ∀ g ∈ Pt.blocks[v], g ∈ r := by
  rw [replayOf, List.find?_isSome]
  constructor
  · rintro ⟨ρ, hρ, h⟩
    rw [List.mem_range] at hρ
    rw [List.getElem?_eq_getElem hv, List.getElem?_eq_getElem hρ] at h
    simp only [List.all_eq_true, List.contains_eq_mem, decide_eq_true_eq] at h
    exact ⟨Pρ.blocks[ρ], List.getElem_mem hρ, h⟩
  · rintro ⟨r, hr, hsub⟩
    obtain ⟨ρ, hρ, rfl⟩ := List.getElem_of_mem hr
    refine ⟨ρ, List.mem_range.2 hρ, ?_⟩
    rw [List.getElem?_eq_getElem hv, List.getElem?_eq_getElem hρ]
    simp only [List.all_eq_true, List.contains_eq_mem, decide_eq_true_eq]
    exact hsub

/-- **The replay check, by blocks**: each proof block of `Pt` lies inside one replay block of `Pρ` (`inOneBlock`), each
gate's replay block marked once for the gates under `n`; equal to the check by positions where `Pρ`'s gates are
distinct and `Pt`'s under `n` (`replaysFast_eq`). -/
def replaysFast (Pt Pρ : Partition) (n : ℕ) : Bool :=
  let rb := blockIdx n Pρ.blocks
  Pt.blocks.all (inOneBlock rb !Pρ.blocks.isEmpty)

theorem replaysFast_eq {Pt Pρ : Partition} {n : ℕ} (hnd : Pρ.gates.Nodup) (hlt : ∀ g ∈ Pt.gates, g < n) :
    replaysFast Pt Pρ n = (List.range Pt.blocks.length).all fun v => (replayOf Pt Pρ v).isSome := by
  rw [Bool.eq_iff_iff]
  simp only [replaysFast, List.all_eq_true, List.mem_range]
  constructor
  · intro h v hv
    rw [replayOf_isSome_iff hv]
    exact (inOneBlock_iff hnd fun g hg => hlt g (List.mem_flatten.2 ⟨_, List.getElem_mem hv, hg⟩)).1
      (h _ (List.getElem_mem hv))
  · intro h u hu
    obtain ⟨v, hv, rfl⟩ := List.getElem_of_mem hu
    rw [inOneBlock_iff hnd fun g hg => hlt g (List.mem_flatten.2 ⟨_, hu, hg⟩)]
    exact (replayOf_isSome_iff hv).1 (h v hv)

end SpotFlock
