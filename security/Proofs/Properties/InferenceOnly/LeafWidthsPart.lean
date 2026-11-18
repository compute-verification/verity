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

theorem emod_three (A B C L : ℤ) : (A % L + B % L + C % L) % L = (A + B + C) % L := by
  rw [Int.add_emod (A % L + B % L) (C % L) L, ← Int.add_emod A B L, Int.emod_emod_of_dvd C (dvd_refl L),
    ← Int.add_emod (A + B) C L]

/-- The place of an element of a strided pattern in the period `L` of the array holding it: its base's, plus a
residue of each stride. -/
theorem strided_residue (base os is : ℤ) (off L x y : ℕ) (hL : 0 < L)
    (hnn : 0 ≤ base + (x : ℤ) * os + (y : ℤ) * is) (hoff : off ≤ (base + (x : ℤ) * os + (y : ℤ) * is).toNat) :
    ((base + (x : ℤ) * os + (y : ℤ) * is).toNat - off) % L =
      (((base - (off : ℤ)) % (L : ℤ) + (((x : ℤ) * os % (L : ℤ)).toNat : ℤ) +
        (((y : ℤ) * is % (L : ℤ)).toNat : ℤ)) % (L : ℤ)).toNat := by
  have hLz : (0 : ℤ) < L := by exact_mod_cast hL
  have hx : ((((x : ℤ) * os % (L : ℤ)).toNat : ℕ) : ℤ) = (x : ℤ) * os % L :=
    Int.toNat_of_nonneg (Int.emod_nonneg _ (by omega))
  have hy : ((((y : ℤ) * is % (L : ℤ)).toNat : ℕ) : ℤ) = (y : ℤ) * is % L :=
    Int.toNat_of_nonneg (Int.emod_nonneg _ (by omega))
  have hz : ((((base + (x : ℤ) * os + (y : ℤ) * is).toNat - off) % L : ℕ) : ℤ) =
      ((base - (off : ℤ)) % (L : ℤ) + (((x : ℤ) * os % (L : ℤ)).toNat : ℤ) +
        (((y : ℤ) * is % (L : ℤ)).toNat : ℤ)) % (L : ℤ) := by
    rw [hx, hy, Int.natCast_mod, Int.natCast_sub hoff, Int.toNat_of_nonneg hnn, emod_three]
    congr 1
    ring
  rw [← hz, Int.toNat_natCast]

/-- The check's loop over a part's leaves one by one: it ends having found a leaf that fails its test, or every leaf
passed. -/
theorem walk_loop (n : ℕ) (G : ℕ → Bool) (w w' : ℕ) (r : Option Bool × Unit)
    (h : (forIn (m := Widths.M) [0:n] ((none : Option Bool), ()) fun k _ => do
      Widths.step
      if G k then pure (ForInStep.done (some false, ())) else pure (ForInStep.yield (none, ()))).run w = .ok r w') :
    ((∀ k < n, G k = false) ∧ r.1 = none) ∨ r.1 = some false := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have hL := forIn_m_inv _ (fun k : ℕ => G k = false) (fun r : Option Bool × Unit => r.1 = none)
    (fun r => r.1 = some false) _ (none, ()) r w w' rfl ?step h
  case step =>
    rintro k - bb s₀ st s₀' - hf
    obtain ⟨u, s₁, -, hf⟩ := m_bind hf
    cases hG : G k with
    | true =>
      rw [hG, ite_eq_left rfl] at hf
      obtain ⟨rfl, -⟩ := m_pure hf
      rfl
    | false =>
      rw [hG, ite_eq_right (by decide)] at hf
      obtain ⟨rfl, -⟩ := m_pure hf
      exact ⟨rfl, rfl⟩
  rcases hL with ⟨hall, hr⟩ | hr
  · exact Or.inl ⟨fun k hk => hall k (by simp [List.mem_range'_1, Std.Legacy.Range.size]; omega), hr⟩
  · exact Or.inr hr

/-- A loop of any body that, run from nothing found, either stops with `some false` or goes on having found `G` of
its element. -/
theorem walk_loop_gen (n : ℕ) (f : ℕ → Option Bool × Unit → Widths.M (ForInStep (Option Bool × Unit)))
    (G : ℕ → Prop) (w w' : ℕ) (r : Option Bool × Unit)
    (hf : ∀ k s st s', (f k (none, ())).run s = .ok st s' →
      st = .done (some false, ()) ∨ (st = .yield (none, ()) ∧ G k))
    (h : (forIn (m := Widths.M) [0:n] ((none : Option Bool), ()) f).run w = .ok r w') :
    ((∀ k < n, G k) ∧ r.1 = none) ∨ r.1 = some false := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have hL := forIn_m_inv _ G (fun r : Option Bool × Unit => r = (none, ()))
    (fun r => r.1 = some false) _ (none, ()) r w w' rfl ?step h
  case step =>
    rintro k - bb s₀ st s₀' rfl hf'
    rcases hf k s₀ st s₀' hf' with rfl | ⟨rfl, hg⟩
    · rfl
    · exact ⟨hg, rfl⟩
  rcases hL with ⟨hall, hr⟩ | hr
  · exact Or.inl ⟨fun k hk => hall k (by simp [List.mem_range'_1, Std.Legacy.Range.size]; omega), by rw [hr]⟩
  · exact Or.inr hr

/-! ## A lane's width -/

/-- The product of a lane's radices. -/
def prodR : List (ℕ × ℤ) → ℕ
  | [] => 1
  | (r, _) :: rest => r * prodR rest

theorem foldl_prodR : ∀ (lv : List (ℕ × ℤ)) (a : ℕ), lv.foldl (fun a (r, _) => a * r) a = a * prodR lv
  | [], a => by simp [prodR]
  | (r, d) :: rest, a => by
    rw [List.foldl_cons, foldl_prodR rest]
    dsimp only [prodR]
    ring

theorem periods_prodR (l : Lane) : l.periods = prodR l.lv := by
  unfold Lane.periods
  rw [foldl_prodR, Nat.one_mul]

theorem sumMod_mem (L : ℕ) (A B : List ℕ) {x y : ℕ} (hx : x ∈ A) (hy : y ∈ B) :
    (x + y) % L ∈ Widths.sumMod L A B := by
  unfold Widths.sumMod
  exact List.mem_flatMap.2 ⟨x, hx, List.mem_map.2 ⟨y, hy, rfl⟩⟩

theorem toNat_add_emod (x y : ℤ) (L : ℕ) (hL : 0 < L) :
    ((x + y) % (L : ℤ)).toNat = ((x % (L : ℤ)).toNat + (y % (L : ℤ)).toNat) % L := by
  have hL0 : (L : ℤ) ≠ 0 := by exact_mod_cast Nat.pos_iff_ne_zero.1 hL
  have h1 := Int.emod_nonneg x hL0
  have h2 := Int.emod_nonneg y hL0
  have key : (x + y) % (L : ℤ) = ((((x % (L : ℤ)).toNat + (y % (L : ℤ)).toNat) % L : ℕ) : ℤ) := by
    rw [Int.natCast_mod, Nat.cast_add, Int.toNat_of_nonneg h1, Int.toNat_of_nonneg h2]
    exact Int.add_emod x y L
  rw [key, Int.toNat_natCast]

/-- **Every period's sum of levels is among `levelRes`'s residues.** -/
theorem levelRes_mem (L cap : ℕ) (hL : 0 < L) : ∀ (lv : List (ℕ × ℤ)) (G : List ℕ) (t : ℕ),
    Widths.levelRes L cap lv = some G → t < prodR lv → ((Lane.leafAt.go lv t) % (L : ℤ)).toNat ∈ G
  | [], G, t, h, _ => by
    have e : Widths.levelRes L cap [] = some [0] := rfl
    rw [e] at h
    injection h with h
    subst h
    simp [Lane.leafAt.go]
  | [(r, d)], G, t, h, ht => by
    have e : Widths.levelRes L cap [(r, d)] = Widths.residues d r L cap := rfl
    rw [e] at h
    have ht' : t < r := by simpa [prodR] using ht
    have hm := residues_mem d r L cap G hL h t ht'
    have e2 : Lane.leafAt.go [(r, d)] t = (t : ℤ) * d := rfl
    rw [e2]
    exact hm
  | (r, d) :: (r2, d2) :: rest, G, t, h, ht => by
    have e : Lane.leafAt.go ((r, d) :: (r2, d2) :: rest) t
        = ((t % r : ℕ) : ℤ) * d + Lane.leafAt.go ((r2, d2) :: rest) (t / r) := rfl
    rw [e]
    have ht0 : t < r * prodR ((r2, d2) :: rest) := ht
    have hr : 0 < r := by
      rcases Nat.eq_zero_or_pos r with h0 | h0
      · rw [h0, Nat.zero_mul] at ht0; exact absurd ht0 (Nat.not_lt_zero _)
      · exact h0
    have ht2 : t / r < prodR ((r2, d2) :: rest) := Nat.div_lt_of_lt_mul ht0
    have e3 : Widths.levelRes L cap ((r, d) :: (r2, d2) :: rest) = (do
        let A ← Widths.residues d r L cap
        let B ← Widths.levelRes L cap ((r2, d2) :: rest)
        if A.length * B.length ≤ cap then some (Widths.sumMod L A B) else none) := rfl
    rw [e3] at h
    cases hA : Widths.residues d r L cap with
    | none => rw [hA] at h; exact absurd h (by simp)
    | some A =>
      cases hB : Widths.levelRes L cap ((r2, d2) :: rest) with
      | none => rw [hA, hB] at h; exact absurd h (by simp)
      | some B =>
        rw [hA, hB] at h
        change (if A.length * B.length ≤ cap then some (Widths.sumMod L A B) else none) = some G at h
        split at h
        · injection h with h
          subst h
          rw [toNat_add_emod _ _ L hL]
          exact sumMod_mem L A B (residues_mem d r L cap A hL hA (t % r) (Nat.mod_lt t hr))
            (levelRes_mem L cap hL ((r2, d2) :: rest) B (t / r) hB ht2)
        · exact absurd h (by simp)

/-- **A lane `laneWidth` passes reads leaves of width `w`**, at every period and offset between its extremes. -/
theorem laneWidth_sound {S : Ty} {w lo hi run : Nat} {l : Lane} {exact : Bool}
    (h : Widths.laneWidth S w lo hi l.b run l.e l.lv exact = some true)
    {t o : Nat} (ht : t < l.periods) (ho : o < run)
    (hlo : (lo : Int) ≤ l.leafAt t o) (hhi : l.leafAt t o ≤ hi) :
    Widths.wAt S (l.leafAt t o).toNat = w := by
  unfold Widths.laneWidth at h
  by_cases hS : S.leaves ≤ hi
  · rw [if_pos hS] at h; exact absurd h (by simp)
  rw [ite_eq_right hS] at h
  have hnn : 0 ≤ l.leafAt t o := le_trans (by exact_mod_cast Nat.zero_le lo) hlo
  have hcast : ((l.leafAt t o).toNat : ℤ) = l.leafAt t o := Int.toNat_of_nonneg hnn
  have hj1 : lo ≤ (l.leafAt t o).toNat := by omega
  have hj2 : (l.leafAt t o).toNat ≤ hi := by omega
  have hjS : (l.leafAt t o).toNat < S.leaves := by omega
  split at h
  · rename_i x hx
    have hm := wAt_mem_rangeWidths S lo (hi + 1) _ hj1 (by omega) hjS
    rw [hx, List.mem_singleton] at hm
    rw [hm]
    simpa using h
  · have hspec := cover_spec S lo (hi + 1) (by omega) (by omega)
    generalize hc : Widths.cover S lo (hi + 1) = c at h hspec
    obtain ⟨c1, off⟩ := c
    cases c1
    case array n f =>
      dsimp only at h hspec
      obtain ⟨hoff, hend, -, hw⟩ := hspec
      split at h
      · exact absurd h (by simp)
      · rename_i hf
        have hL : 0 < f.leaves := by
          rcases Nat.eq_zero_or_pos f.leaves with h0 | h0
          · exact absurd (by rw [h0]; rfl) hf
          · exact h0
        split at h
        · rename_i R hR
          split at h
          · rename_i hall
            rw [hw _ (by omega) (by omega), wAt_array n f _ hL]
            rw [List.all_eq_true] at hall
            refine beq_iff_eq.1 (hall _ ?_)
            unfold Widths.laneRes at hR
            cases hO : Widths.residues l.e run f.leaves Widths.RCAP with
            | none => rw [hO] at hR; exact absurd hR (by simp)
            | some O =>
              cases hG : Widths.levelRes f.leaves Widths.RCAP l.lv with
              | none => rw [hO, hG] at hR; exact absurd hR (by simp)
              | some G =>
                rw [hO, hG] at hR
                change (if O.length * G.length ≤ Widths.RCAP then some ((Widths.sumMod f.leaves O G).map
                  fun x => (((l.b - (off : ℤ)) % (f.leaves : ℤ)).toNat + x) % f.leaves) else none) = some R at hR
                split at hR
                · injection hR with hR
                  subst hR
                  have hx := residues_mem l.e run f.leaves Widths.RCAP O hL hO o ho
                  have hy := levelRes_mem f.leaves Widths.RCAP hL l.lv G t hG (by rw [← periods_prodR]; exact ht)
                  refine List.mem_map.2 ⟨_, sumMod_mem f.leaves O G hx hy, ?_⟩
                  have hsub : (((l.leafAt t o).toNat - off : ℕ) : ℤ) = l.leafAt t o - off := by
                    rw [Nat.cast_sub (by omega), hcast]
                  have e : l.leafAt t o - off = (l.b - off) + ((o : ℤ) * l.e + Lane.leafAt.go l.lv t) := by
                    show l.b + (o : ℤ) * l.e + Lane.leafAt.go l.lv t - off = _
                    ring
                  rw [← toNat_add_emod _ _ _ hL, ← toNat_add_emod _ _ _ hL, ← e, ← hsub, ← Int.natCast_mod,
                    Int.toNat_natCast]
                · exact absurd hR (by simp)
          · split at h <;> exact absurd h (by simp)
        · exact absurd h (by simp)
    all_goals exact absurd h (by simp)

/-! ## A part -/

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
    have hqn : ∀ l ∈ lanes, ∀ t' < (lanes[0]?.map Lane.periods).getD 0, ∀ o < run, 0 ≤ l.leafAt t' o := hq
    change k < lanes.size * run * ((lanes[0]?.map Lane.periods).getD 0) at hk
    change a + lanes.size * run * ((lanes[0]?.map Lane.periods).getD 0) ≤ t.leaves at ha
    have hgpos : 0 < lanes.size := by
      rcases Nat.eq_zero_or_pos lanes.size with h0 | h0
      · rw [h0] at hk; simp at hk
      · exact h0
    have hP : (lanes[0]?.map Lane.periods).getD 0 = lanes[0]!.periods := by
      rw [Array.getElem?_eq_getElem hgpos, getElem!_pos lanes 0 hgpos]; rfl
    rw [hP] at hk ha hqn
    have hrun : 0 < run := by
      rcases Nat.eq_zero_or_pos run with h0 | h0
      · rw [h0] at hk; simp at hk
      · exact h0
    have hm : k % (lanes.size * run) / run < lanes.size :=
      Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm run]; exact Nat.mod_lt _ (Nat.mul_pos hgpos hrun))
    have hlmem : lanes[k % (lanes.size * run) / run]! ∈ lanes := by
      rw [getElem!_pos lanes _ hm]; exact Array.getElem_mem hm
    have ht0 : k / (lanes.size * run) < lanes[0]!.periods := Nat.div_lt_of_lt_mul hk
    have ht' : k / (lanes.size * run) < (lanes[k % (lanes.size * run) / run]!).periods := by
      rw [hgr _ hlmem]; exact ht0
    have hmem := Walk.leafAt_mem (lanes[k % (lanes.size * run) / run]!) (Nat.mod_lt k hrun) ht'
    have hnn := hqn _ hlmem _ ht0 _ (Nat.mod_lt k hrun)
    unfold Widths.partOk at h
    cases ht with
    | none => exact (m_throw h).elim
    | some wd =>
      have hwd : Widths.wAt t (a + k) = wd := hht wd rfl
      dsimp only at h
      split at h
      · rename_i hall
        rw [Array.all_eq_true] at hall
        have hv := hall _ (by rw [Array.size_map]; exact hm)
        rw [Array.getElem_map, ← getElem!_pos lanes _ hm] at hv
        change (srcOf B below).ty? (lanes[k % (lanes.size * run) / run]!).p
          (lanes[k % (lanes.size * run) / run]!).i = some S at hS
        change (srcOf B below).homo? (lanes[k % (lanes.size * run) / run]!).p
          (lanes[k % (lanes.size * run) / run]!).i = Widths.homo S at hhomo
        change ((lanes[k % (lanes.size * run) / run]!).leafAt (k / (lanes.size * run)) (k % run)).toNat
          < S.leaves at hj
        change Widths.wAt S ((lanes[k % (lanes.size * run) / run]!).leafAt (k / (lanes.size * run))
          (k % run)).toNat = Widths.wAt t (a + k)
        generalize lanes[k % (lanes.size * run) / run]! = l at hv hmem hnn hS hhomo hj ht' ⊢
        rw [hwd]
        rw [hS] at hv
        dsimp only at hv
        by_cases hh : ((srcOf B below).homo? l.p l.i == some wd) = true
        · have hw : Widths.homo S = some wd := by rw [← hhomo]; simpa using hh
          exact homo_wAt S wd hw _ hj
        · rw [ite_eq_right hh] at hv
          rcases hbd : l.bounds run with ⟨lo, hi'⟩
          rw [hbd] at hmem hv
          dsimp only at hv
          have hlw : Widths.laneWidth S wd lo.toNat hi'.toNat l.b run l.e l.lv true = some true := by
            simpa using hv
          have hlo : ((lo.toNat : ℕ) : ℤ) ≤ l.leafAt (k / (lanes.size * run)) (k % run) := by
            by_cases h0 : 0 ≤ lo
            · rw [Int.toNat_of_nonneg h0]; exact hmem.1
            · rw [Int.toNat_of_nonpos (by omega)]; exact_mod_cast hnn
          exact laneWidth_sound hlw ht' (Nat.mod_lt k hrun) hlo (le_trans hmem.2 (Int.self_le_toNat hi'))
      · split at h
        · exact absurd (m_pure h).1 (by simp)
        · exact (m_throw h).elim
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
      obtain ⟨hlo, hhi, -, -, -⟩ := strided_between base inner os is n hi k hk vals rfl
      have hlo' : lo ≤ atOf base inner os is k := hlo
      have hhi' : atOf base inner os is k ≤ hi' := hhi
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
        have key := laneWidth_sound (l := ⟨p, i, base, is, [((n + inner - 1) / inner, os)]⟩) hlw
          (t := k / inner) (o := k % inner) (by simpa [Lane.periods] using hkq) (Nat.mod_lt _ hi)
          (by rw [e, ← hat]; exact_mod_cast hlo') (by rw [e, ← hat]; exact_mod_cast hhi')
        rw [e] at key
        change Widths.wAt S (base + ((k / inner : ℕ) : ℤ) * os + ((k % inner : ℕ) : ℤ) * is).toNat = wd at key
        rw [hwd]
        exact key
      next => exact absurd (m_pure h).1 (by simp)
      next => exact (m_throw h).elim

end InferenceOnlyVerifier.Properties.SecurityProofs.LeafWidths
