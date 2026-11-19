module

public import Proofs.Properties.InferenceOnly.LeafWidths
public import Proofs.Properties.InferenceOnly.LeafWidthsRange
public import Proofs.Properties.InferenceOnly.Walk
public import Mathlib.Tactic

/-!
# A part the width check passes names leaves of the widths its type gives them

`partOk_sound`: where `Widths.partOk` returns `true` for a part read as `t`'s leaves from leaf `a`, each reference of
the part names a leaf as wide as `t`'s leaf at its place, wherever it names a source the check reads. A strided part is
decided in closed form: by the values of its pattern's corners, between which every reference lies, and by the residues
its two strides reach in the period of the array holding them (`residues_mem`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.SecurityProofs.LeafWidths
open Flock.Program

/-! ## The residues of a stride -/

/-- **Every multiple of a stride is among its residues**: `Widths.residues stride count L bound` holds
`k · stride mod L` for each `k < count`. -/
theorem residues_mem (stride : ℤ) (count L bound : ℕ) (A : List ℕ) (hL : 0 < L)
    (h : Widths.residues stride count L bound = some A) (k : ℕ) (hk : k < count) :
    ((k : ℤ) * stride % (L : ℤ)).toNat ∈ A := by
  have hLz : (0 : ℤ) < L := by exact_mod_cast hL
  have hs : ((stride % (L : ℤ)).toNat : ℤ) = stride % L := Int.toNat_of_nonneg (Int.emod_nonneg _ (by omega))
  have hkey : ((k : ℤ) * stride % (L : ℤ)).toNat = k * (stride % (L : ℤ)).toNat % L := by
    have e : (k : ℤ) * stride % L = ((k * (stride % (L : ℤ)).toNat % L : ℕ) : ℤ) := by
      push_cast
      rw [hs]
      conv_rhs => rw [Int.mul_emod, Int.emod_emod_of_dvd _ (dvd_refl _), ← Int.mul_emod]
    rw [e, Int.toNat_natCast]
  rw [hkey]
  unfold Widths.residues at h
  dsimp only at h
  generalize (stride % (L : ℤ)).toNat = s at h ⊢
  have hg : 0 < Nat.gcd s L := Nat.gcd_pos_of_pos_right _ hL
  by_cases hc : count ≥ L / Nat.gcd s L
  · rw [ite_eq_left hc] at h
    by_cases hbound : L / Nat.gcd s L ≤ bound
    · rw [ite_eq_left hbound] at h
      injection h with h
      subst h
      rw [List.mem_map]
      refine ⟨k * s % L / Nat.gcd s L, List.mem_range.2 ?_, ?_⟩
      · exact Nat.div_lt_div_of_lt_of_dvd (Nat.gcd_dvd_right s L) (Nat.mod_lt _ hL)
      · exact Nat.div_mul_cancel
          ((Nat.dvd_mod_iff (Nat.gcd_dvd_right s L)).2 (Dvd.dvd.mul_left (Nat.gcd_dvd_left s L) k))
    · rw [ite_eq_right hbound] at h
      exact absurd h (by simp)
  · rw [ite_eq_right hc] at h
    by_cases hbound : count ≤ bound
    · rw [ite_eq_left hbound] at h
      injection h with h
      subst h
      rw [List.mem_map]
      exact ⟨k, List.mem_range.2 hk, rfl⟩
    · rw [ite_eq_right hbound] at h
      exact absurd h (by simp)

/-! ## The sources a body's check reads -/

/-- A reference of body `B` that names a source of the check names a leaf of that source's type, whose one width, if
it has one, is the one the check holds for it. -/
theorem src_ty (B : Body) (below : ℕ) (r : Ref) (S : Ty) (hr : BodiesOk.RefIn B r)
    (h : (srcOf B below).ty? r.p r.i = some S) :
    r.j < S.leaves ∧ (srcOf B below).homo? r.p r.i = Widths.homo S := by
  obtain ⟨hp, hn⟩ := hr
  unfold Widths.Src.ty? at h
  unfold Widths.Src.homo?
  cases hrp : r.p with
  | true =>
    obtain ⟨h1, h2⟩ := hp hrp
    rw [hrp] at h
    simp only [srcOf, ite_true, Array.getElem?_map, Array.getElem?_eq_getElem h1, Option.map_some] at h ⊢
    rw [getElem!_pos B.params r.i h1] at h2
    injection h with h
    subst h
    exact ⟨h2, rfl⟩
  | false =>
    obtain ⟨h1, h2⟩ := hn hrp
    rw [hrp] at h
    simp only [srcOf, Bool.false_eq_true, ite_false, Array.getElem?_map, Array.getElem?_eq_getElem h1,
      Option.map_some] at h ⊢
    rw [getElem!_pos B.nodes r.i h1] at h2
    split at h
    · injection h with h
      subst h
      exact ⟨h2, rfl⟩
    · exact absurd h (by simp)

/-- **Leaves of one width have it each.** -/
theorem sliceWidth_wAt (t : Ty) (a n w : ℕ) (h : Widths.sliceWidth t a n = some w) (k : ℕ) (hk : k < n)
    (hak : a + k < t.leaves) : Widths.wAt t (a + k) = w := by
  have hm := wAt_mem_rangeWidths t a (a + n) (a + k) (Nat.le_add_right _ _) (by omega) hak
  unfold Widths.sliceWidth at h
  split at h
  · rename_i w' hw'
    rw [hw'] at hm
    rw [List.mem_singleton.1 hm]
    exact Option.some.inj h
  · exact absurd h (by simp)

/-- Where the check finds a slice and a source of one same width, a leaf of the one is as wide as a leaf of the
other. -/
theorem homo_same (S : Ty) (ht o : Option ℕ) (wt : ℕ) (h : (ht.isSome && ht == o) = true)
    (ho : o = Widths.homo S) (hht : ∀ w, ht = some w → wt = w) (j : ℕ) (hj : j < S.leaves) :
    Widths.wAt S j = wt := by
  rw [Bool.and_eq_true] at h
  obtain ⟨h1, h2⟩ := h
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 h1
  have h3 : ht = o := by simpa using h2
  rw [hht w hw, homo_wAt S w (by rw [← ho, ← h3, hw]) j hj]

/-! ## A strided part's corners -/

theorem foldl_min_le : ∀ (l : List ℕ) (init x : ℕ), (x ∈ l ∨ x = init) → l.foldl min init ≤ x
  | [], init, x, h => by
    rcases h with h | h
    · exact absurd h (by simp)
    · rw [h]; exact le_refl _
  | y :: ys, init, x, h => by
    rw [List.foldl_cons]
    rcases h with h | h
    · rcases List.mem_cons.1 h with h | h
      · exact le_trans (foldl_min_le ys _ _ (Or.inr rfl)) (by rw [h]; exact min_le_right _ _)
      · exact foldl_min_le ys _ x (Or.inl h)
    · exact le_trans (foldl_min_le ys _ _ (Or.inr rfl)) (by rw [h]; exact min_le_left _ _)

theorem le_foldl_max : ∀ (l : List ℕ) (init x : ℕ), (x ∈ l ∨ x = init) → x ≤ l.foldl max init
  | [], init, x, h => by
    rcases h with h | h
    · exact absurd h (by simp)
    · rw [h]; exact le_refl _
  | y :: ys, init, x, h => by
    rw [List.foldl_cons]
    rcases h with h | h
    · rcases List.mem_cons.1 h with h | h
      · exact le_trans (by rw [h]; exact le_max_right _ _) (le_foldl_max ys _ _ (Or.inr rfl))
      · exact le_foldl_max ys _ x (Or.inl h)
    · exact le_trans (by rw [h]; exact le_max_left _ _) (le_foldl_max ys _ _ (Or.inr rfl))

theorem foldl_max_mem : ∀ (l : List ℕ) (init : ℕ), l.foldl max init ∈ l ∨ l.foldl max init = init
  | [], init => Or.inr rfl
  | y :: ys, init => by
    rw [List.foldl_cons]
    rcases foldl_max_mem ys (max init y) with h | h
    · exact Or.inl (List.mem_cons_of_mem _ h)
    · rw [h]
      rcases max_choice init y with h' | h'
      · exact Or.inr h'
      · exact Or.inl (by rw [h']; exact List.mem_cons_self)

/-- The corners of a strided part's pattern, as `Widths.partOk` and the parser take them. -/
def corners (inner n : ℕ) : List (ℕ × ℕ) :=
  [(0, 0), (0, min inner n - 1), ((n - 1) / inner, 0), ((n - 1) / inner, (n - 1) % inner)] ++
    (if (n - 1) / inner > 0 then [((n - 1) / inner - 1, inner - 1)] else [])

/-- Every corner of a pattern is one of its elements. -/
theorem corner_elem (inner n : ℕ) (hi : 0 < inner) (hn : 0 < n) (c : ℕ × ℕ) (hc : c ∈ corners inner n) :
    ∃ k < n, k / inner = c.1 ∧ k % inner = c.2 := by
  have hq : (n - 1) / inner * inner ≤ n - 1 := Nat.div_mul_le_self _ _
  unfold corners at hc
  rw [List.mem_append] at hc
  rcases hc with hc | hc
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl | rfl | rfl | rfl
    · exact ⟨0, hn, Nat.zero_div _, Nat.zero_mod _⟩
    · exact ⟨min inner n - 1, by omega, Nat.div_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)⟩
    · exact ⟨(n - 1) / inner * inner, by omega, Nat.mul_div_cancel _ hi, Nat.mul_mod_left _ _⟩
    · exact ⟨n - 1, by omega, rfl, rfl⟩
  · by_cases hpos : (n - 1) / inner > 0
    · rw [ite_eq_left hpos, List.mem_singleton] at hc
      subst hc
      obtain ⟨m, hm⟩ : ∃ m, (n - 1) / inner = m + 1 := ⟨(n - 1) / inner - 1, by omega⟩
      rw [hm, Nat.succ_mul] at hq
      rw [hm]
      refine ⟨inner - 1 + m * inner, by omega, ?_, ?_⟩
      · rw [Nat.add_mul_div_right _ _ hi, Nat.div_eq_of_lt (by omega)]
        omega
      · rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
    · rw [ite_eq_right hpos] at hc
      exact absurd hc (by simp)

/-- The position of element `k` of a strided pattern. -/
def atOf (base : ℤ) (inner : ℕ) (os is : ℤ) (k : ℕ) : ℕ :=
  (base + ((k / inner : ℕ) : ℤ) * os + ((k % inner : ℕ) : ℤ) * is).toNat

/-- **Every element of a strided pattern lies between the least and the greatest of its corners' positions**, and
the greatest is the position of one of its elements. -/
theorem strided_between (base : ℤ) (inner : ℕ) (os is : ℤ) (n : ℕ) (hi : 0 < inner) (k : ℕ) (hk : k < n)
    (vals : List ℕ) (hv : vals = (corners inner n).map fun c => (base + (c.1 : ℤ) * os + (c.2 : ℤ) * is).toNat) :
    vals.foldl min vals.head! ≤ atOf base inner os is k ∧ atOf base inner os is k ≤ vals.foldl max vals.head! ∧
      ∃ k' < n, vals.foldl max vals.head! = atOf base inner os is k' := by
  have hmem : ∀ c ∈ corners inner n, (base + (c.1 : ℤ) * os + (c.2 : ℤ) * is).toNat ∈ vals := fun c hc => by
    rw [hv]
    exact List.mem_map.2 ⟨c, hc, rfl⟩
  obtain ⟨⟨c1, hc1, hle1⟩, ⟨c2, hc2, hle2⟩⟩ := Walk.strided_corners base os is hi hk
  refine ⟨le_trans (foldl_min_le vals _ _ (Or.inl (hmem c1 hc1))) (Int.toNat_le_toNat hle1),
    le_trans (Int.toNat_le_toNat hle2) (le_foldl_max vals _ _ (Or.inl (hmem c2 hc2))), ?_⟩
  have hhead : vals.head! ∈ vals := by
    rw [hv]
    exact List.mem_cons_self
  have hmax : vals.foldl max vals.head! ∈ vals := by
    rcases foldl_max_mem vals vals.head! with h | h
    · exact h
    · rw [h]; exact hhead
  rw [hv] at hmax
  obtain ⟨c, hc, hcv⟩ := List.mem_map.1 hmax
  obtain ⟨k', hk', h1, h2⟩ := corner_elem inner n hi (by omega) c hc
  refine ⟨k', hk', ?_⟩
  rw [hv, ← hcv]
  unfold atOf
  rw [h1, h2]

/-! ## A lane's residues -/

/-- Residues add mod `L`. -/
theorem toNat_add_emod (u v : ℤ) (L : ℕ) (hL : 0 < L) :
    ((u % (L : ℤ)).toNat + (v % (L : ℤ)).toNat) % L = ((u + v) % (L : ℤ)).toNat := by
  have hLz : (0 : ℤ) < L := by exact_mod_cast hL
  have h1 : (((u % (L : ℤ)).toNat : ℕ) : ℤ) = u % L := Int.toNat_of_nonneg (Int.emod_nonneg _ (by omega))
  have h2 : (((v % (L : ℤ)).toNat : ℕ) : ℤ) = v % L := Int.toNat_of_nonneg (Int.emod_nonneg _ (by omega))
  rw [← Int.toNat_natCast (((u % (L : ℤ)).toNat + (v % (L : ℤ)).toNat) % L)]
  congr 1
  rw [Int.natCast_mod, Nat.cast_add, h1, h2, ← Int.add_emod]

/-- `sumMod` holds the sum of any two of its summands' residues. -/
theorem sumMod_mem (L : ℕ) (A B : List ℕ) (x y : ℕ) (hx : x ∈ A) (hy : y ∈ B) :
    (x + y) % L ∈ Widths.sumMod L A B :=
  List.mem_flatMap.2 ⟨x, hx, List.mem_map.2 ⟨y, hy, rfl⟩⟩

/-- A product of radices folded from `c` is `c` times the product folded from 1. -/
theorem radix_foldl (l : List (ℕ × ℤ)) (c : ℕ) :
    l.foldl (fun a (r, _) => a * r) c = c * l.foldl (fun a (r, _) => a * r) 1 := by
  induction l generalizing c with
  | nil => simp
  | cons x xs ih =>
    obtain ⟨r, d⟩ := x
    simp only [List.foldl_cons]
    rw [ih (c * r), ih (1 * r)]
    ring

/-- **A lane's levels reach, at each period they count, a residue `levelRes` holds.** -/
theorem levelRes_mem (L cap : ℕ) (hL : 0 < L) : ∀ (lv : List (ℕ × ℤ)) (R : List ℕ) (t : ℕ),
    Widths.levelRes L cap lv = some R → t < lv.foldl (fun a (r, _) => a * r) 1 →
    ((Lane.leafAt.go lv t) % (L : ℤ)).toNat ∈ R
  | [], R, t, h, _ => by
    simp only [Widths.levelRes, Option.some.injEq] at h
    subst h
    simp [Lane.leafAt.go]
  | [(r, d)], R, t, h, ht => by
    simp only [Widths.levelRes] at h
    simp only [List.foldl_cons, List.foldl_nil, Nat.one_mul] at ht
    simp only [Lane.leafAt.go]
    exact residues_mem d r L cap R hL h t ht
  | (r, d) :: x :: rest, R, t, h, ht => by
    rw [List.foldl_cons, radix_foldl] at ht
    dsimp only at ht
    have hr : 0 < r := by
      rcases Nat.eq_zero_or_pos r with h0 | h0
      · rw [h0] at ht; simp at ht
      · exact h0
    rw [Widths.levelRes] at h
    on_goal 2 => simp
    cases hA : Widths.residues d r L cap with
    | none => rw [hA] at h; exact absurd h (by simp)
    | some A =>
      cases hB : Widths.levelRes L cap (x :: rest) with
      | none => rw [hA, hB] at h; exact absurd h (by simp)
      | some B =>
        rw [hA, hB] at h
        by_cases hc : A.length * B.length ≤ cap
        · simp only [Option.bind_eq_bind, Option.bind_some, ite_eq_left hc, Option.some.injEq] at h
          subst h
          have hxA := residues_mem d r L cap A hL hA (t % r) (Nat.mod_lt _ hr)
          have hyB := levelRes_mem L cap hL (x :: rest) B (t / r) hB
            (Nat.div_lt_of_lt_mul (by rw [Nat.one_mul] at ht; exact ht))
          have hm := sumMod_mem L A B _ _ hxA hyB
          rw [toNat_add_emod _ _ L hL] at hm
          simpa only [Lane.leafAt.go] using hm
        · simp only [Option.bind_eq_bind, Option.bind_some, ite_eq_right hc] at h
          exact absurd h (by simp)

/-- **A lane reaches, at each period and offset, a residue `laneRes` holds**, from leaf `off` on. -/
theorem laneRes_mem (b : ℤ) (off run : ℕ) (e : ℤ) (lv : List (ℕ × ℤ)) (L cap : ℕ) (hL : 0 < L) (R : List ℕ)
    (h : Widths.laneRes b off run e lv L cap = some R) (t o : ℕ) (ho : o < run)
    (ht : t < lv.foldl (fun a (r, _) => a * r) 1)
    (hoff : (off : ℤ) ≤ b + (o : ℤ) * e + Lane.leafAt.go lv t) :
    ((b + (o : ℤ) * e + Lane.leafAt.go lv t).toNat - off) % L ∈ R := by
  have hLz : (0 : ℤ) < L := by exact_mod_cast hL
  unfold Widths.laneRes at h
  cases hO : Widths.residues e run L cap with
  | none => rw [hO] at h; exact absurd h (by simp)
  | some O =>
    cases hG : Widths.levelRes L cap lv with
    | none => rw [hO, hG] at h; exact absurd h (by simp)
    | some G =>
      rw [hO, hG] at h
      by_cases hc : O.length * G.length ≤ cap
      · simp only [Option.bind_eq_bind, Option.bind_some, ite_eq_left hc, Option.some.injEq] at h
        subst h
        have hx := residues_mem e run L cap O hL hO o ho
        have hy := levelRes_mem L cap hL lv G t hG ht
        have hm := sumMod_mem L O G _ _ hx hy
        rw [toNat_add_emod _ _ L hL] at hm
        have hm' := List.mem_map_of_mem (f := fun x => (((b - (off : ℤ)) % (L : ℤ)).toNat + x) % L) hm
        rw [toNat_add_emod _ _ L hL] at hm'
        convert hm' using 1
        have hnn : (0 : ℤ) ≤ b + (o : ℤ) * e + Lane.leafAt.go lv t := le_trans (by positivity) hoff
        rw [← Int.toNat_natCast (((b + (o : ℤ) * e + Lane.leafAt.go lv t).toNat - off) % L)]
        congr 1
        rw [Int.natCast_mod, Nat.cast_sub (by omega), Int.toNat_of_nonneg hnn]
        congr 1
        ring
      · simp only [Option.bind_eq_bind, Option.bind_some, ite_eq_right hc] at h
        exact absurd h (by simp)

/-- **A lane `laneWidth` passes reaches leaves of width `w` only** between its extremes `lo … hi`, within `S`. -/
theorem laneWidth_sound {S : Ty} {w lo hi run : ℕ} {l : Lane} {exact : Bool}
    (h : Widths.laneWidth S w lo hi l.b run l.e l.lv exact = some true) (hS : hi < S.leaves)
    {t o : ℕ} (ht : t < l.periods) (ho : o < run)
    (hlo : (lo : ℤ) ≤ l.leafAt t o) (hhi : l.leafAt t o ≤ hi) :
    Widths.wAt S (l.leafAt t o).toNat = w := by
  have hnn : (0 : ℤ) ≤ l.leafAt t o := le_trans (Int.natCast_nonneg _) hlo
  have hc : (((l.leafAt t o).toNat : ℕ) : ℤ) = l.leafAt t o := Int.toNat_of_nonneg hnn
  have hj1 : lo ≤ (l.leafAt t o).toNat := by omega
  have hj2 : (l.leafAt t o).toNat ≤ hi := by omega
  unfold Widths.laneWidth at h
  split at h
  next x hx =>
    have hm := wAt_mem_rangeWidths S lo (hi + 1) _ hj1 (by omega) (by omega)
    rw [hx, List.mem_singleton] at hm
    rw [hm]
    simpa using h
  next =>
    obtain ⟨c1, c2, -, c4⟩ := cover_spec S lo (hi + 1) (by omega) (by omega)
    split at h
    next m f off hcs =>
      rw [hcs] at c1 c2 c4
      dsimp only at c1 c2 c4
      split at h
      next => exact absurd h (by simp)
      next hL0 =>
        have hL : 0 < f.leaves := Nat.pos_of_ne_zero (by simpa using hL0)
        split at h
        next R hR =>
          split at h
          next hall =>
            rw [List.all_eq_true] at hall
            have hmem := laneRes_mem l.b off run l.e l.lv f.leaves Widths.RCAP hL R hR t o ho ht
              (show (off : ℤ) ≤ l.leafAt t o from le_trans (by exact_mod_cast c1) hlo)
            have hmem' : ((l.leafAt t o).toNat - off) % f.leaves ∈ R := hmem
            rw [c4 _ (by omega) (by omega), wAt_array m f _ hL]
            exact beq_iff_eq.1 (hall _ hmem')
          next => split at h <;> simp at h
        next => exact absurd h (by simp)
    next => exact absurd h (by simp)

/-- The reference of a grid at lane `m`, offset `o` and period `t`. -/
theorem grid_index (size run m o t : ℕ) (hm : m < size) (ho : o < run) :
    (t * (size * run) + (m * run + o)) % (size * run) / run = m ∧
      (t * (size * run) + (m * run + o)) / (size * run) = t ∧ (t * (size * run) + (m * run + o)) % run = o := by
  have hx : m * run + o < size * run := by
    calc m * run + o < m * run + run := by omega
      _ = (m + 1) * run := by ring
      _ ≤ size * run := Nat.mul_le_mul_right _ hm
  have hN : 0 < size * run := by omega
  have e1 : (t * (size * run) + (m * run + o)) % (size * run) = m * run + o := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hx]
  refine ⟨?_, ?_, ?_⟩
  · rw [e1, Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt ho, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hx, Nat.zero_add]
  · rw [← Nat.mod_mod_of_dvd _ (Dvd.intro_left size rfl), e1, Nat.add_comm, Nat.add_mul_mod_self_right,
      Nat.mod_eq_of_lt ho]

/-! ## A part -/

/-- **The plain check reads no input node at another width**: `srcOf` holds no input widths, so `recutOk` leaves every
part to the rest of `partOk`. -/
theorem recutOk_srcOf (B : Body) (below : ℕ) (t : Ty) (ht : Option ℕ) (a : ℕ) (q : Part) :
    Widths.recutOk (srcOf B below) t ht a q = pure none := by
  cases q <;> simp [Widths.recutOk, Widths.Src.inW?, srcOf]

/-- **A part the width check passes names leaves of the widths its type gives them.** -/
theorem partOk_sound (B : Body) (below : ℕ) (t : Ty) (a : ℕ) (q : Part) (w w' : ℕ) (hq : BodiesOk.PartNonneg q)
    (hi : PartInner q) (ha : a + q.count ≤ t.leaves) (hin : ∀ k < q.count, BodiesOk.RefIn B (q.get k))
    (h : (Widths.partOk (srcOf B below) t (Widths.sliceWidth t a q.count) a q).run w = .ok true w') :
    ∀ k < q.count, ∀ S, (srcOf B below).ty? (q.get k).p (q.get k).i = some S →
      Widths.wAt S (q.get k).j = Widths.wAt t (a + k) := by
  intro k hk S hS
  have hjr := hin k hk
  obtain ⟨hj, hhomo⟩ := src_ty B below (q.get k) S hjr hS
  have hht : ∀ wd, Widths.sliceWidth t a q.count = some wd → Widths.wAt t (a + k) = wd := fun wd hwd =>
    sliceWidth_wAt t a q.count wd hwd k hk (by omega)
  generalize Widths.sliceWidth t a q.count = ht at h hht
  cases q with
  | explicit items =>
    unfold Widths.partOk at h
    rw [recutOk_srcOf, pure_bind] at h
    have hall := (m_pure h).1
    rw [List.all_eq_true] at hall
    have hk' := hall k (List.mem_range.2 hk)
    change (srcOf B below).ty? (items[k]!).p (items[k]!).i = some S at hS
    dsimp only at hk'
    rw [hS] at hk'
    exact beq_iff_eq.1 hk'
  | affine p i base n =>
    change (srcOf B below).ty? p i = some S at hS
    change (srcOf B below).homo? p i = Widths.homo S at hhomo
    change base + k < S.leaves at hj
    change k < n at hk
    change a + n ≤ t.leaves at ha
    unfold Widths.partOk at h
    rw [recutOk_srcOf, pure_bind] at h
    dsimp only at h
    rw [hS] at h
    dsimp only at h
    by_cases hh : (ht.isSome && ht == (srcOf B below).homo? p i) = true
    · exact homo_same S ht _ _ hh hhomo hht _ hj
    · rw [ite_eq_right hh] at h
      have hb : base + n ≤ S.leaves := by
        have hlast : n - 1 < (Part.affine p i base n).count := by change n - 1 < n; omega
        have := (src_ty B below ((Part.affine p i base n).get (n - 1)) S (hin (n - 1) hlast) hS).1
        change base + (n - 1) < S.leaves at this
        omega
      exact (eqRange_sound _ t a S base n w w' ha hb h k hk).symm
  | grid run lanes =>
    have hgr : ∀ l ∈ lanes, l.periods = lanes[0]!.periods := hi
    change k < lanes.size * run * ((lanes[0]?.map Lane.periods).getD 0) at hk
    change a + lanes.size * run * ((lanes[0]?.map Lane.periods).getD 0) ≤ t.leaves at ha
    have hgpos : 0 < lanes.size := by
      rcases Nat.eq_zero_or_pos lanes.size with h0 | h0
      · rw [h0] at hk; simp at hk
      · exact h0
    have hP : (lanes[0]?.map Lane.periods).getD 0 = lanes[0]!.periods := by
      rw [Array.getElem?_eq_getElem hgpos, getElem!_pos lanes 0 hgpos]; rfl
    rw [hP] at hk ha
    have hrun : 0 < run := by
      rcases Nat.eq_zero_or_pos run with h0 | h0
      · rw [h0] at hk; simp at hk
      · exact h0
    change (srcOf B below).ty? (lanes[k % (lanes.size * run) / run]!).p
      (lanes[k % (lanes.size * run) / run]!).i = some S at hS
    change (srcOf B below).homo? (lanes[k % (lanes.size * run) / run]!).p
      (lanes[k % (lanes.size * run) / run]!).i = Widths.homo S at hhomo
    change ((lanes[k % (lanes.size * run) / run]!).leafAt (k / (lanes.size * run)) (k % run)).toNat
      < S.leaves at hj
    show Widths.wAt S ((lanes[k % (lanes.size * run) / run]!).leafAt (k / (lanes.size * run))
      (k % run)).toNat = Widths.wAt t (a + k)
    obtain ⟨m, hmdef⟩ : ∃ m, m = k % (lanes.size * run) / run := ⟨_, rfl⟩
    rw [← hmdef] at hS hhomo hj ⊢
    have hm : m < lanes.size := by
      rw [hmdef]
      exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm run]; exact Nat.mod_lt _ (Nat.mul_pos hgpos hrun))
    have hlmem : lanes[m]! ∈ lanes := by
      rw [getElem!_pos lanes _ hm]; exact Array.getElem_mem hm
    have ht' : k / (lanes.size * run) < (lanes[m]!).periods := by
      rw [hgr _ hlmem]; exact Nat.div_lt_of_lt_mul hk
    have hmem := Walk.leafAt_mem (lanes[m]!) (Nat.mod_lt k hrun) ht'
    have hnn : 0 ≤ (lanes[m]!).leafAt (k / (lanes.size * run)) (k % run) := by
      have hq' : ∀ l ∈ lanes, ∀ t < (lanes[0]?.map Lane.periods).getD 0, ∀ o < run, 0 ≤ l.leafAt t o := hq
      exact hq' _ hlmem _ (by rw [hP]; exact Nat.div_lt_of_lt_mul hk) _ (Nat.mod_lt k hrun)
    obtain ⟨t₀, ht₀, o₀, ho₀, he₀⟩ := Walk.leafAt_top (lanes[m]!) hrun (Nat.lt_of_le_of_lt (Nat.zero_le _) ht')
    obtain ⟨i1, i2, i3⟩ := grid_index lanes.size run m o₀ t₀ hm ho₀
    have hk₀ : t₀ * (lanes.size * run) + (m * run + o₀) < (Part.grid run lanes).count := by
      change _ < lanes.size * run * ((lanes[0]?.map Lane.periods).getD 0)
      rw [hP, ← hgr _ hlmem]
      have hx : m * run + o₀ < lanes.size * run := by
        calc m * run + o₀ < m * run + run := by omega
          _ = (m + 1) * run := by ring
          _ ≤ lanes.size * run := Nat.mul_le_mul_right _ hm
      calc t₀ * (lanes.size * run) + (m * run + o₀) < t₀ * (lanes.size * run) + lanes.size * run := by omega
        _ = (t₀ + 1) * (lanes.size * run) := by ring
        _ ≤ (lanes[m]!).periods * (lanes.size * run) := Nat.mul_le_mul_right _ ht₀
        _ = lanes.size * run * (lanes[m]!).periods := by ring
    have hS₀ : (srcOf B below).ty? ((Part.grid run lanes).get (t₀ * (lanes.size * run) + (m * run + o₀))).p
        ((Part.grid run lanes).get (t₀ * (lanes.size * run) + (m * run + o₀))).i = some S := by
      change (srcOf B below).ty? (lanes[(t₀ * (lanes.size * run) + (m * run + o₀)) % (lanes.size * run) / run]!).p
        (lanes[(t₀ * (lanes.size * run) + (m * run + o₀)) % (lanes.size * run) / run]!).i = some S
      rw [i1]
      exact hS
    have hj₀ := (src_ty B below _ S (hin _ hk₀) hS₀).1
    change ((lanes[(t₀ * (lanes.size * run) + (m * run + o₀)) % (lanes.size * run) / run]!).leafAt
      ((t₀ * (lanes.size * run) + (m * run + o₀)) / (lanes.size * run))
      ((t₀ * (lanes.size * run) + (m * run + o₀)) % run)).toNat < S.leaves at hj₀
    rw [i1, i2, i3, he₀] at hj₀
    unfold Widths.partOk at h
    rw [recutOk_srcOf, pure_bind] at h
    try dsimp only at h
    cases ht with
    | none =>
      try dsimp only at h
      exact (m_throw h).elim
    | some wd =>
      have hwd : Widths.wAt t (a + k) = wd := hht wd rfl
      rw [hwd]
      try dsimp only at h
      split at h
      next hall =>
        rw [Array.all_eq_true] at hall
        have hl := hall m (by rw [Array.size_map]; exact hm)
        simp only [Array.getElem_map] at hl
        rw [← getElem!_pos lanes _ hm] at hl
        generalize lanes[m]! = l at hl hmem hnn hS hhomo hj hj₀ ht' ⊢
        try dsimp only at hl
        rw [hS] at hl
        try dsimp only at hl
        split at hl
        next hh => exact homo_wAt S wd (by rw [← hhomo]; simpa using hh) _ hj
        next =>
          rcases hbd : l.bounds run with ⟨lo, hi'⟩
          simp only [hbd] at hl hmem hj₀
          try dsimp only at hl hmem hj₀
          have hl' : Widths.laneWidth S wd lo.toNat hi'.toNat l.b run l.e l.lv true = some true := by
            simpa using hl
          have hlo : ((lo.toNat : ℕ) : ℤ) ≤ l.leafAt (k / (lanes.size * run)) (k % run) := by
            rcases le_or_gt 0 lo with h0 | h0
            · rw [Int.toNat_of_nonneg h0]; exact hmem.1
            · rw [Int.toNat_of_nonpos h0.le]; exact_mod_cast hnn
          exact laneWidth_sound hl' hj₀ ht' (Nat.mod_lt k hrun) hlo (le_trans hmem.2 (Int.self_le_toNat hi'))
      next =>
        split at h
        next => exact absurd (m_pure h).1 (by simp)
        next => exact (m_throw h).elim
  | strided p i base inner os is n =>
    change (srcOf B below).ty? p i = some S at hS
    change (srcOf B below).homo? p i = Widths.homo S at hhomo
    change k < n at hk
    change a + n ≤ t.leaves at ha
    change 0 < inner at hi
    change BodiesOk.RefIn B ⟨p, i, atOf base inner os is k⟩ at hjr
    change atOf base inner os is k < S.leaves at hj
    show Widths.wAt S (atOf base inner os is k) = Widths.wAt t (a + k)
    unfold Widths.partOk at h
    rw [recutOk_srcOf, pure_bind] at h
    dsimp (config := { zeta := false }) only at h
    rw [hS] at h
    dsimp (config := { zeta := false }) only at h
    by_cases hh : (n == 0 || ht.isSome && ht == (srcOf B below).homo? p i) = true
    · have hh' : (ht.isSome && ht == (srcOf B below).homo? p i) = true := by
        rw [Bool.or_eq_true] at hh
        rcases hh with h0 | h1
        · exact absurd (by simpa using h0 : n = 0) (by omega)
        · exact h1
      exact homo_same S ht _ _ hh' hhomo hht _ hj
    rw [ite_eq_right hh] at h
    cases ht with
    | none => exact (m_throw h).elim
    | some wd =>
      have hwd : Widths.wAt t (a + k) = wd := hht wd rfl
      dsimp (config := { zeta := false }) only at h
      extract_lets q' crn vals lo hi' at h
      obtain ⟨hlo, hhi, k', hk', hmax⟩ := strided_between base inner os is n hi k hk vals rfl
      have hlo' : lo ≤ atOf base inner os is k := hlo
      have hhi' : atOf base inner os is k ≤ hi' := hhi
      have hS' : hi' < S.leaves := by
        have := (src_ty B below ((Part.strided p i base inner os is n).get k') S (hin k' hk') hS).1
        change atOf base inner os is k' < S.leaves at this
        rw [← hmax] at this
        exact this
      clear_value lo hi'
      split at h
      next hlw =>
        have hnn : 0 ≤ base + ((k / inner : ℕ) : ℤ) * os + ((k % inner : ℕ) : ℤ) * is := hq k hk
        have hkq : k / inner < (n + inner - 1) / inner := by
          have h1 : k / inner ≤ (n - 1) / inner := Nat.div_le_div_right (by omega)
          have h2 : (n - 1 + inner) / inner = (n - 1) / inner + 1 := Nat.add_div_right _ hi
          have h3 : n + inner - 1 = n - 1 + inner := by omega
          rw [h3, h2]
          omega
        have e : (⟨p, i, base, is, [((n + inner - 1) / inner, os)]⟩ : Lane).leafAt (k / inner) (k % inner) =
            base + ((k / inner : ℕ) : ℤ) * os + ((k % inner : ℕ) : ℤ) * is := by
          simp only [Lane.leafAt, Lane.leafAt.go]
          ring
        have hat : ((atOf base inner os is k : ℕ) : ℤ) =
            base + ((k / inner : ℕ) : ℤ) * os + ((k % inner : ℕ) : ℤ) * is := Int.toNat_of_nonneg hnn
        have key := laneWidth_sound (l := ⟨p, i, base, is, [((n + inner - 1) / inner, os)]⟩) hlw hS'
          (t := k / inner) (o := k % inner) (by simpa [Lane.periods] using hkq) (Nat.mod_lt _ hi)
          (by rw [e, ← hat]; exact_mod_cast hlo') (by rw [e, ← hat]; exact_mod_cast hhi')
        rw [e] at key
        change Widths.wAt S (base + ((k / inner : ℕ) : ℤ) * os + ((k % inner : ℕ) : ℤ) * is).toNat = wd at key
        rw [hwd]
        exact key
      next => exact absurd (m_pure h).1 (by simp)
      next => exact (m_throw h).elim

end InferenceOnlyVerifier.Properties.SecurityProofs.LeafWidths
