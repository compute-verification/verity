import FlockSoundness.ExecTemplateClass
import FlockSoundness.ExecLookup

/-!
# `PlacedWF` from the accepted statement (the layout facts `stmtOf_eq_placed` reads)

`Refine.stmtOf_eq_placed` needs three layout facts of a statement's nets and leaf circuit (`Refine.PlacedWF`). This file
derives them from what the verifier accepted:
- **Sparse columns inside their width.** Every matrix `Sparse.ofRows` builds refuses a column outside its width
  (`ofRows_col_lt`). The nets of an accepted typed template are the template's root and placed layouts (`netOfD`, and
  `placedNet`, which changes only the lookup), and the rest of the circuit's nets, each `Net.parse`'s or
  `Lookup.build`'s (`ParseFactsT.pre`); all of them hold `Sparse.ofRows` outputs. `Stmt.setupH` sets the leaf to
  `default`, whose matrices have no rows (`setupH_leaf_typed`).
- **A lookup inside its net.** `Lookup.build` refuses more than 32 output bits, and its low decoder's top block ends
  below its product rows (`build_lookup`).
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## `Sparse.ofRows` refuses a column outside its width -/

theorem rd32_lt (b : ByteArray) (o : ℕ) : rd32 b o < 2 ^ 32 := by
  unfold rd32
  have h0 := (b.get! o).toNat_lt
  have h1 := (b.get! (o + 1)).toNat_lt
  have h2 := (b.get! (o + 2)).toNat_lt
  have h3 := (b.get! (o + 3)).toNat_lt
  refine Nat.or_lt_two_pow (Nat.or_lt_two_pow (Nat.or_lt_two_pow ?_ ?_) ?_) ?_
  · omega
  · rw [Nat.shiftLeft_eq]; omega
  · rw [Nat.shiftLeft_eq]; omega
  · rw [Nat.shiftLeft_eq]; omega

theorem ofRows_width {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m) :
    m.width = width := by
  unfold Sparse.ofRows at h
  simp only at h
  obtain ⟨st, -, h⟩ := except_bind_ok h
  cases h
  rfl

/-- **Every stored column of a matrix `Sparse.ofRows` built is inside its width.** -/
theorem ofRows_col_lt {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m) :
    ∀ i t, i < m.rows → t < m.len[i]! → m.col i t < m.width := by
  intro i t hi ht
  rw [ofRows_width h]
  by_cases hw : width ≤ 2 ^ 32
  · obtain ⟨-, hs, hl, hc⟩ := ofRows_spec h
    have hrs : m.rows = rows.size := ofRows_rows h
    have hls : m.len.size = rows.size := by rw [← Array.length_toList, hl]; simp
    have hss : m.start.size = rows.size := by rw [← Array.length_toList, hs, rowStarts_length]; simp
    have hlen : m.len[i]! = m.len.getD i 0 := by
      simp [Array.getD_eq_getD_getElem?, show i < m.len.size by omega]
    have hst : m.start[i]! = m.start.getD i 0 := by
      simp [Array.getD_eq_getD_getElem?, show i < m.start.size by omega]
    have hsz : t < (m.row i).size := by simp only [Sparse.row, Array.size_map, Array.size_range]; omega
    have hcol : m.col i t = (m.row i)[t] := by
      simp only [Sparse.col, Sparse.row, Array.getElem_map, Array.getElem_range, hst]
    have hmem : m.col i t ∈ (rows.getD i #[]).toList := by
      rw [hcol, ← ofRows_row h hw i]
      exact Array.getElem_mem_toList _
    have hri : rows.getD i #[] = rows[i]'(by omega) := by
      simp [Array.getD_eq_getD_getElem?, show i < rows.size by omega]
    rw [hri] at hmem
    exact hc _ (Array.getElem_mem_toList _) _ hmem
  · have := rd32_lt m.data (m.start[i]! + 4 * t)
    unfold Sparse.col
    omega

/-- A net whose matrices `Sparse.ofRows` built has its columns inside their widths. -/
theorem netCols_of_ofRows {net : Net} {u : ℕ} (ha : ∃ ra, Sparse.ofRows ra (2 ^ u) = .ok net.a)
    (hb : ∃ rb, Sparse.ofRows rb (2 ^ u) = .ok net.b) :
    (∀ i t, i < net.a.rows → t < net.a.len[i]! → net.a.col i t < net.a.width) ∧
      (∀ i t, i < net.b.rows → t < net.b.len[i]! → net.b.col i t < net.b.width) :=
  ⟨let ⟨_, h⟩ := ha; ofRows_col_lt h, let ⟨_, h⟩ := hb; ofRows_col_lt h⟩

/-! ## `Lookup.build`'s lookup is inside its net -/

/-- `Lookup.decode` only adds rows. -/
theorem decode_mono (fuel : ℕ) : ∀ (bd : Lookup.Builder) (bits : List ℕ),
    bd.a.size ≤ (Lookup.decode bd bits fuel).1.a.size := by
  induction fuel with
  | zero => intro bd bits; unfold Lookup.decode; split <;> exact Nat.le_refl _
  | succ fuel ih =>
    intro bd bits
    unfold Lookup.decode
    split
    · exact Nat.le_refl _
    · exact Nat.le_refl _
    · rename_i heq _
      obtain rfl := Nat.succ.inj heq
      dsimp only
      exact foldl_inv (fun s : Lookup.Builder × Array Lookup.Form => bd.a.size ≤ s.1.a.size)
        (fun s x hs => by simp only [Lookup.row, Array.size_push]; omega) (Nat.le_trans (ih _ _) (ih _ _)) rfl

/-- **The low decoder's top block**: for `lo ≥ 2` bits, `decode` returns the first row of the top block it pushed last,
`2^lo` rows. -/
theorem decode_top {lo : ℕ} (hlo : 2 ≤ lo) (bd : Lookup.Builder) :
    (Lookup.decode bd (List.range lo) lo).2.2 + 2 ^ lo = (Lookup.decode bd (List.range lo) lo).1.a.size := by
  obtain ⟨k, rfl⟩ : ∃ k, lo = k + 1 := ⟨lo - 1, by omega⟩
  unfold Lookup.decode
  split
  · rename_i heq
    have := congrArg List.length heq
    simp at this
    omega
  · omega
  · rename_i heq _
    obtain rfl := Nat.succ.inj heq
    dsimp only
    symm
    refine (foldl_count (fun s : Lookup.Builder × Array Lookup.Form => s.1.a.size) (k := 1) (fun s x => ?_) _ _).trans ?_
    · simp [Lookup.row]
    · simp

/-- **What `Lookup.build` builds, for `PlacedWF`**: its matrices are `Sparse.ofRows`'s at its slot size, and its
lookup has at most 32 output bits and its low minterms inside the `B` width. -/
theorem build_placed {name : String} {table : ByteArray} {n lo bits : ℕ} {net : Net}
    (h : Lookup.build name table n lo bits = .ok net) :
    (∃ ra, Sparse.ofRows ra (2 ^ net.unitLog) = .ok net.a) ∧ (∃ rb, Sparse.ofRows rb (2 ^ net.unitLog) = .ok net.b) ∧
      ∀ lk, net.lookup = some lk → lk.loTop + 2 ^ lk.lo ≤ net.b.width ∧ lk.bits ≤ 32 := by
  unfold Lookup.build at h
  obtain ⟨hc, h⟩ := ite_throw_ok_not h
  have hc' : 2 ≤ lo ∧ bits ≤ 32 := by simp at hc; omega
  obtain ⟨s1, hs1, h⟩ := except_bind_ok h
  dsimp only at h
  rcases hd1 : Lookup.decode s1 (List.range lo) lo with ⟨bd1, loM, loTop⟩
  rw [hd1] at h
  rcases hd2 : Lookup.decode bd1 ((List.range (n - lo)).map (lo + ·)) (n - lo) with ⟨bd2, hiM, x⟩
  rw [hd2] at h
  dsimp only at h
  obtain ⟨s2, hs2, h⟩ := except_bind_ok h
  obtain ⟨s3, hs3, h⟩ := except_bind_ok h
  obtain ⟨s4, hs4, h⟩ := except_bind_ok h
  obtain ⟨hc2, h⟩ := ite_throw_ok_not h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨a, ha, h⟩ := except_bind_ok h
  obtain ⟨b, hb, h⟩ := except_bind_ok h
  cases h
  have htop := decode_top hc'.1 s1
  rw [hd1] at htop
  have hm2 : bd1.a.size ≤ bd2.a.size := by
    have := decode_mono (n - lo) bd1 ((List.range (n - lo)).map (lo + ·)); rwa [hd2] at this
  have e1a := forIn_range_pure (fun r : Lookup.Builder => r.a.size) (fun i r => by split <;> simp) hs1
  have e1b := forIn_range_pure (fun r : Lookup.Builder => r.b.size) (fun i r => by split <;> simp) hs1
  simp only [Array.size_empty] at e1a e1b
  have hb1 : bd1.a.size = bd1.b.size := by
    have := decode_sizes lo s1 (List.range lo) (by omega); rwa [hd1] at this
  have hb2 : bd2.a.size = bd2.b.size := by
    have := decode_sizes (n - lo) bd1 ((List.range (n - lo)).map (lo + ·)) hb1; rwa [hd2] at this
  have e2a := forIn_range_nested (fun r : Lookup.Builder => r.a.size) (fun i r => by simp) hs2
  have e2b := forIn_range_nested (fun r : Lookup.Builder => r.b.size) (fun i r => by simp) hs2
  have e3 := @while_pad s2.a.size _ _ ⟨Nat.le_refl _, by omega⟩ hs3
  have e4a := forIn_range_pure (fun r : Lookup.Builder => r.a.size) (fun i r => by split <;> simp) hs4
  refine ⟨⟨_, ha⟩, ⟨_, hb⟩, fun lk hlk => ?_⟩
  cases hlk
  refine ⟨?_, hc'.2⟩
  rw [ofRows_width hb]
  have hu := le_two_pow_log2ceil (s4.a.size + 1)
  have hp : 2 ^ log2ceil (s4.a.size + 1) ≤ 2 ^ max 7 (log2ceil (s4.a.size + 1)) :=
    Nat.pow_le_pow_right (by norm_num) (le_max_right _ _)
  simp only [Array.size_push] at hu hp ⊢
  omega

/-! ## The leaf of an accepted statement -/

set_option maxHeartbeats 4000000 in
/-- **`Stmt.setupH` sets the leaf to `default`** for a typed statement: its compression circuit's matrices have no rows. -/
theorem setupH_leaf_typed {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (ht : tags.typed = true) (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    st.leaf = default := by
  unfold Stmt.setupH at h
  simp (config := { zeta := false }) only [ht, ↓reduceIte] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨c, hc, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at h)
  obtain ⟨f0, h⟩ : ∃ f0, jpA f0 = .ok st := by
    split at h
    · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
    · split at h <;> (obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩)
  simp (config := { zeta := false }) only [jpA] at h
  split at h
  swap
  · cases h
  (have hh := except_bind_ok h; clear h; obtain ⟨pop, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at h
  obtain ⟨pub, h⟩ : ∃ pub, jpB pub = .ok st := by
    split at h <;> (obtain ⟨p, -, h⟩ := except_bind_ok h; exact ⟨p, h⟩)
  simp (config := { zeta := false }) only [jpB] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpU at h
  obtain ⟨u0, h⟩ : ∃ u, jpU u = .ok st := by
    split at h
    · split at h
      · obtain ⟨_, -, h⟩ := except_bind_ok h
        exact ⟨(), ite_throw_ok h⟩
      · exact ⟨(), h⟩
    · exact ⟨(), h⟩
  simp (config := { zeta := false }) only [jpU] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h
  obtain ⟨finding, h⟩ : ∃ f, jpC f = .ok st := by
    split at h
    · extract_lets (config := { underBinder := false, onlyGivenNames := true }) q at h
      split at h
      · obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨f, -, h⟩ := except_bind_ok h
        exact ⟨f, h⟩
      · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
    · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
  simp (config := { zeta := false }) only [jpC] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) n blocksReal nbl m jpD at h
  obtain ⟨hk, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jpD] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨pin, hpin, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) h0 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) h1 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  have hs := Except.ok.inj h
  subst hs
  rfl

end FlockSoundness
