module

public import Proofs.Properties.Nci.LeafWidths
public import Proofs.Properties.Nci.LeafWidthsRange
public import Proofs.Properties.Nci.Walk
public import Mathlib.Tactic

/-!
# A part the width check passes names leaves of the widths its type gives them

`partOk_sound`: where `Widths.partOk` returns `true` for a part read as `t`'s leaves from leaf `a`, each reference of
the part names a leaf as wide as `t`'s leaf at its place, wherever it names a source the check reads. A strided part is
decided in closed form: by the values of its pattern's corners, between which every reference lies, and by the residues
its two strides reach in the period of the array holding them (`residues_mem`).
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program

/-! ## The residues of a stride -/

/-- **Every multiple of a stride is among its residues**: `Widths.residues stride count L cap` holds `k · stride mod L`
for each `k < count`. -/
theorem residues_mem (stride : ℤ) (count L cap : ℕ) (A : List ℕ) (hL : 0 < L)
    (h : Widths.residues stride count L cap = some A) (k : ℕ) (hk : k < count) :
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
    by_cases hcap : L / Nat.gcd s L ≤ cap
    · rw [ite_eq_left hcap] at h
      injection h with h
      subst h
      rw [List.mem_map]
      refine ⟨k * s % L / Nat.gcd s L, List.mem_range.2 ?_, ?_⟩
      · exact Nat.div_lt_div_of_lt_of_dvd (Nat.gcd_dvd_right s L) (Nat.mod_lt _ hL)
      · exact Nat.div_mul_cancel
          ((Nat.dvd_mod_iff (Nat.gcd_dvd_right s L)).2 (Dvd.dvd.mul_left (Nat.gcd_dvd_left s L) k))
    · rw [ite_eq_right hcap] at h
      exact absurd h (by simp)
  · rw [ite_eq_right hc] at h
    by_cases hcap : count ≤ cap
    · rw [ite_eq_left hcap] at h
      injection h with h
      subst h
      rw [List.mem_eraseDups, List.mem_map]
      exact ⟨k, List.mem_range.2 hk, rfl⟩
    · rw [ite_eq_right hcap] at h
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

/-- Where the check finds a type and a source of one same width, a leaf of the one is as wide as a leaf of the
other. -/
theorem homo_same (t S : Ty) (o : Option ℕ) (h : ((Widths.homo t).isSome && Widths.homo t == o) = true)
    (ho : o = Widths.homo S) (j j' : ℕ) (hj : j < S.leaves) (hj' : j' < t.leaves) :
    Widths.wAt S j = Widths.wAt t j' := by
  rw [Bool.and_eq_true] at h
  obtain ⟨h1, h2⟩ := h
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 h1
  have h3 : Widths.homo t = o := by simpa using h2
  rw [homo_wAt t w hw j' hj', homo_wAt S w (by rw [← ho, ← h3, hw]) j hj]

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

/-! ## A part -/

/-- **A part the width check passes names leaves of the widths its type gives them.** -/
theorem partOk_sound (B : Body) (below : ℕ) (t : Ty) (a : ℕ) (q : Part) (w w' : ℕ) (hq : BodiesOk.PartNonneg q)
    (hi : PartInner q) (ha : a + q.count ≤ t.leaves) (hin : ∀ k < q.count, BodiesOk.RefIn B (q.get k))
    (h : (Widths.partOk (srcOf B below) t (Widths.homo t) a q).run w = .ok true w') :
    ∀ k < q.count, ∀ S, (srcOf B below).ty? (q.get k).p (q.get k).i = some S →
      Widths.wAt S (q.get k).j = Widths.wAt t (a + k) := by
  intro k hk S hS
  have hjr := hin k hk
  obtain ⟨hj, hhomo⟩ := src_ty B below (q.get k) S hjr hS
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
    by_cases hh : ((Widths.homo t).isSome && Widths.homo t == (srcOf B below).homo? p i) = true
    · exact homo_same t S _ hh hhomo _ _ hj (by omega)
    · rw [ite_eq_right hh] at h
      have hb : base + n ≤ S.leaves := by
        have hlast : n - 1 < (Part.affine p i base n).count := by change n - 1 < n; omega
        have := (src_ty B below ((Part.affine p i base n).get (n - 1)) S (hin (n - 1) hlast) hS).1
        change base + (n - 1) < S.leaves at this
        omega
      exact (eqRange_sound _ t a S base n w w' ha hb h k hk).symm
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
    by_cases hh : (n == 0 || (Widths.homo t).isSome && Widths.homo t == (srcOf B below).homo? p i) = true
    · have hh' : ((Widths.homo t).isSome && Widths.homo t == (srcOf B below).homo? p i) = true := by
        rw [Bool.or_eq_true] at hh
        rcases hh with h0 | h1
        · exact absurd (by simpa using h0 : n = 0) (by omega)
        · exact h1
      exact homo_same t S _ hh' hhomo _ _ hj (by omega)
    rw [ite_eq_right hh] at h
    extract_lets at_ jp q' crn vals lo hi' at h
    have hjp : ∀ w₁ w₁', (jp ()).run w₁ = .ok true w₁' →
        Widths.wAt S (atOf base inner os is k) = Widths.wAt t (a + k) := by
      intro w₁ w₁' hj'
      unfold jp at hj'
      obtain ⟨r, w₂, hloop, hj'⟩ := m_bind hj'
      obtain ⟨r1, r2⟩ := r
      rcases walk_loop n (fun k => Widths.wAt S (at_ k) != Widths.wAt t (a + k)) w₁ w₂ _ hloop with ⟨hall, -⟩ | hr
      · have e : at_ k = atOf base inner os is k := rfl
        rw [← e]
        simpa using hall k hk
      · dsimp only at hr hj'
        subst hr
        dsimp only at hj'
        exact absurd (m_pure hj').1 (by simp)
    clear_value jp
    split at h
    · rename_i wd hwd
      obtain ⟨hlo, hhi, k', hk', hmax⟩ := strided_between base inner os is n hi k hk vals rfl
      have hlo' : lo ≤ atOf base inner os is k := hlo
      have hhi' : atOf base inner os is k ≤ hi' := hhi
      have hmax' : hi' = atOf base inner os is k' := hmax
      clear_value lo hi'
      have hhiS : hi' + 1 ≤ S.leaves := by
        have hlt := (src_ty B below ((Part.strided p i base inner os is n).get k') S (hin k' hk') hS).1
        change atOf base inner os is k' < S.leaves at hlt
        omega
      by_cases hrw : (Widths.rangeWidths S lo (hi' + 1) == [wd]) = true
      · have hmem := wAt_mem_rangeWidths S lo (hi' + 1) (atOf base inner os is k) hlo' (by omega) hj
        rw [beq_iff_eq.1 hrw, List.mem_singleton] at hmem
        rw [hmem, homo_wAt t wd hwd (a + k) (by omega)]
      rw [ite_eq_right hrw] at h
      obtain ⟨c1, c2, c3, c4⟩ := cover_spec S lo (hi' + 1) (by omega) hhiS
      split at h
      · rename_i m f hcs
        extract_lets L r0 at h
        by_cases hLpos : f.leaves > 0
        · rw [ite_eq_left hLpos] at h
          split at h
          · rename_i A B' hres
            injection hres with hA hB
            by_cases hlen : A.length * B'.length ≤ 4096
            · rw [ite_eq_left hlen] at h
              by_cases hall : (A.all fun x => B'.all fun y =>
                  Widths.wAt f ((r0 + (x : ℤ) + (y : ℤ)) % (L : ℤ)).toNat == wd) = true
              · have hnn : 0 ≤ base + ((k / inner : ℕ) : ℤ) * os + ((k % inner : ℕ) : ℤ) * is := hq k hk
                have hkq : k / inner < (n + inner - 1) / inner := by
                  have h1 : k / inner ≤ (n - 1) / inner := Nat.div_le_div_right (by omega)
                  have h2 : (n - 1 + inner) / inner = (n - 1) / inner + 1 := Nat.add_div_right _ hi
                  have h3 : n + inner - 1 = n - 1 + inner := by omega
                  rw [h3, h2]
                  omega
                have hxA := residues_mem os ((n + inner - 1) / inner) f.leaves 4096 A hLpos hA (k / inner) hkq
                have hyB := residues_mem is inner f.leaves 4096 B' hLpos hB (k % inner) (Nat.mod_lt _ hi)
                rw [List.all_eq_true] at hall
                have h1 := hall _ hxA
                rw [List.all_eq_true] at h1
                have h2 := beq_iff_eq.1 (h1 _ hyB)
                have e := strided_residue base os is (Widths.cover S lo (hi' + 1)).2 f.leaves (k / inner)
                  (k % inner) hLpos hnn (le_trans c1 hlo')
                change (atOf base inner os is k - (Widths.cover S lo (hi' + 1)).2) % f.leaves = _ at e
                rw [c4 (atOf base inner os is k) (by omega) (by omega), hcs, wAt_array m f _ hLpos,
                  homo_wAt t wd hwd (a + k) (by omega), e]
                exact h2
              · rw [ite_eq_right hall] at h
                exact hjp _ _ h
            · rw [ite_eq_right hlen] at h
              exact hjp _ _ h
          · exact hjp _ _ h
        · rw [ite_eq_right hLpos] at h
          exact hjp _ _ h
      · exact hjp _ _ h
    · exact hjp _ _ h

end NciCheck.Properties.SecurityProofs.LeafWidths
