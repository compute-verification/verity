import Proofs.Flock.Soundness.Discharge.Integrate.Sites

/-!
# The accepted statement `Placed.Accepted` reads is the one the refinement plays

`Placed.Accepted` reads an accepted statement as `Layout.model` (the matrices `placedA`, `placedB` of flock-verifier's
`FlockLevel3.Placed`); the executable's refinement (`stmtOf_fold`, `accepts_tables_hm96`) plays at `Refine.stmtOf` (`matA`,
`matB`). They read the same layout two ways:
* the covering range: `FlockLevel3.lastCover` is a left fold, `Refine.lastCover` a right recursion (`lastCover_eq`);
* Δ: a filter's length against `List.count` (`deltaEntry_eq`);
* a net's entries: `sparseEntry` counts a column at any index, `spCnt` only below the matrix's width (`sparse_eq`). At a
  net range's local columns (below `2^slotLog`) the width check holds: every net `HmRow.parse` builds is `2^unitLog`
  wide (`net_width`), and a net range's slots are its net's rows;
* a lookup slot's table rows: `lookupEntry` against `lkCnt`, which reads bit indices below 32 (`lookup_eq`, as
  `Lookup.build` refuses more than 32 output bits);
* a compression-leaf range: `setupH`'s leaf is the default one, whose matrices are empty (`setupH_leaf`).

`model_eq_stmtOf`: for a statement `Stmt.setupH` accepted on the untyped path, `Layout.model regs = stmtOf st hwf regs` at
any regions. `table_stmtOf`, `tableAt_stmtOf`: a drawn table of `sitesAt` is `stmtOf` of its statement at `regionsOf`.
-/

namespace FlockSoundness.Discharge.Integrate

open Flock FlockLevel3 Refine

/-! ### The covering range and Δ -/

theorem lastCover_foldl (c : ℕ) : ∀ (rs : List Range) (acc : Option Range),
    rs.foldl (fun acc r => if covers r c then some r else acc) acc = (Refine.lastCover rs c).or acc
  | [], _ => rfl
  | r :: rs, acc => by
      rw [List.foldl_cons, lastCover_foldl c rs, Refine.lastCover, Option.or_assoc]
      congr 1
      by_cases h : covers r c <;> simp [h]

theorem lastCover_eq (rs : List Range) (c : ℕ) : FlockLevel3.lastCover rs c = Refine.lastCover rs c :=
  (lastCover_foldl c rs none).trans Option.or_none

theorem deltaEntry_eq (pairs : Array (ℕ × ℕ)) (r j : ℕ) : deltaEntry pairs r j = (dCnt pairs r j : ZMod 2) := by
  unfold deltaEntry dCnt
  rw [List.count_eq_length_filter]
  congr 3
  funext x
  rw [beq_eq_decide]

/-! ### A slot's entries -/

theorem card_filter_range (n : ℕ) (p : ℕ → Prop) [DecidablePred p] :
    ((Finset.range n).filter p).card = ((List.range n).filter fun t => decide (p t)).length := by
  rw [Finset.card_def, Finset.filter_val, Finset.range_val, ← Multiset.coe_range, Multiset.filter_coe,
    Multiset.coe_card]

theorem sparse_eq (m : Sparse) (i c : ℕ) (hc : c < m.width ∨ m.rows ≤ i) :
    sparseEntry m i c = (spCnt m i c : ZMod 2) := by
  unfold sparseEntry spCnt
  by_cases hi : i < m.rows
  · rw [ite_eq_left hi, ite_eq_left ⟨hi, hc.resolve_right (by omega)⟩, card_filter_range]
  · rw [ite_eq_right hi, ite_eq_right fun h => hi h.1]
    simp

theorem lookup_eq (lk : LookupTable) (hb : lk.bits ≤ 32) (i c : ℕ) :
    lookupEntry (some lk) i c = (lkCnt lk i c : ZMod 2) := by
  have hnh : 0 < 2 ^ (lk.n - lk.lo) := Nat.two_pow_pos _
  unfold lookupEntry lkCnt
  dsimp only
  split_ifs with hcond
  · obtain ⟨h1, h2, h3, h4, h5⟩ := hcond
    have hq : (i - lk.prod0) / 2 ^ (lk.n - lk.lo) < lk.bits := by
      rw [Nat.div_lt_iff_lt_mul hnh]; omega
    rw [show ((Finset.range (2 ^ (lk.n - lk.lo)) ×ˢ Finset.range (2 ^ lk.lo) ×ˢ Finset.range 32).filter fun x =>
        lk.loTop + x.2.1 = c ∧ (Lookup.word lk (2 ^ lk.lo) (2 ^ lk.bits - 1) x.1 x.2.1).testBit x.2.2 = true ∧
          x.2.2 < lk.bits ∧ lk.prod0 + x.2.2 * 2 ^ (lk.n - lk.lo) + x.1 = i) =
        {((i - lk.prod0) % 2 ^ (lk.n - lk.lo), c - lk.loTop, (i - lk.prod0) / 2 ^ (lk.n - lk.lo))} from ?_]
    · simp
    ext ⟨h, l, jb⟩
    simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_range, Finset.mem_singleton, Prod.mk.injEq]
    constructor
    · rintro ⟨⟨hh, -, -⟩, hc, -, -, hi⟩
      have hu : (i - lk.prod0) / 2 ^ (lk.n - lk.lo) = jb ∧ (i - lk.prod0) % 2 ^ (lk.n - lk.lo) = h :=
        (Nat.div_mod_unique hnh).2 ⟨by rw [Nat.mul_comm]; omega, hh⟩
      exact ⟨hu.2.symm, by omega, hu.1.symm⟩
    · rintro ⟨rfl, rfl, rfl⟩
      have hd := Nat.div_add_mod' (i - lk.prod0) (2 ^ (lk.n - lk.lo))
      refine ⟨⟨Nat.mod_lt _ hnh, by omega, by omega⟩, by omega, h5, hq, by omega⟩
  · rw [Finset.card_eq_zero.mpr]
    · simp
    rw [Finset.filter_eq_empty_iff]
    rintro ⟨h, l, jb⟩ hx ⟨hc, ht, hjb, hi⟩
    dsimp only at hc ht hjb hi
    simp only [Finset.mem_product, Finset.mem_range] at hx
    obtain ⟨hh, hl, -⟩ := hx
    have hu : (i - lk.prod0) / 2 ^ (lk.n - lk.lo) = jb ∧ (i - lk.prod0) % 2 ^ (lk.n - lk.lo) = h :=
      (Nat.div_mod_unique hnh).2 ⟨by rw [Nat.mul_comm]; omega, hh⟩
    refine hcond ⟨by omega, ?_, by omega, by omega, ?_⟩
    · have : jb * 2 ^ (lk.n - lk.lo) + 2 ^ (lk.n - lk.lo) ≤ lk.bits * 2 ^ (lk.n - lk.lo) := by
        rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hjb
      omega
    · rw [hu.1, hu.2, show c - lk.loTop = l by omega]; exact ht

/-! ### The nets `HmRow.parse` builds -/

theorem ofRows_width {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m) :
    m.width = width := by
  unfold Sparse.ofRows at h
  simp only at h
  obtain ⟨_, -, h⟩ := except_bind_ok h
  cases h
  rfl

/-- A lookup slot `Lookup.build` builds is `2^unitLog` wide, and reads at most 32 output bits. -/
theorem build_width {name : String} {table : ByteArray} {n lo bits : ℕ} {net : Net}
    (h : Lookup.build name table n lo bits = .ok net) :
    net.a.width = 2 ^ net.unitLog ∧ net.b.width = 2 ^ net.unitLog ∧ ∀ lk, net.lookup = some lk → lk.bits ≤ 32 := by
  unfold Lookup.build at h
  obtain ⟨hc, h⟩ := ite_throw_ok_not h
  have hb : bits ≤ 32 := by simp at hc; omega
  obtain ⟨s1, -, h⟩ := except_bind_ok h
  dsimp only at h
  rcases hd1 : Lookup.decode s1 (List.range lo) lo with ⟨bd1, loM, loTop⟩
  rw [hd1] at h
  rcases hd2 : Lookup.decode bd1 ((List.range (n - lo)).map (lo + ·)) (n - lo) with ⟨bd2, hiM, x⟩
  rw [hd2] at h
  dsimp only at h
  obtain ⟨s2, -, h⟩ := except_bind_ok h
  obtain ⟨s3, -, h⟩ := except_bind_ok h
  obtain ⟨s4, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_ok_not h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨a, ha, h⟩ := except_bind_ok h
  obtain ⟨b, hb', h⟩ := except_bind_ok h
  cases h
  refine ⟨ofRows_width ha, ofRows_width hb', ?_⟩
  rintro lk ⟨⟩
  exact hb

/-- Every net `HmRow.parse` accepted is `2^unitLog` wide on both sides, and a lookup slot's table reads at most 32 output
bits. -/
theorem net_width {tags : Tags} {c : Circuit} (P : ParseFacts tags c) {n : ℕ} (hn : n < c.nets.size) :
    (c.nets.getD n default).2.a.width = 2 ^ (c.nets.getD n default).2.unitLog ∧
      (c.nets.getD n default).2.b.width = 2 ^ (c.nets.getD n default).2.unitLog ∧
      ∀ lk, (c.nets.getD n default).2.lookup = some lk → lk.bits ≤ 32 := by
  have hmem : c.nets.getD n default ∈ c.nets.toList := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hn, Option.getD_some]
    exact Array.mem_def.1 (Array.getElem_mem hn)
  rcases P.nets _ hmem with hp | ⟨_, _, _, _, _, hb⟩
  · obtain ⟨hlk, -, -, -, -, _, _, -, -, ha, hb⟩ := built_spec hp
    exact ⟨ofRows_width ha, ofRows_width hb, fun lk h => by rw [hlk] at h; cases h⟩
  · exact build_width hb

set_option maxHeartbeats 2000000 in
/-- **`Stmt.setupH`'s leaf is the default one**: an hm96 row statement has no compression leaf. -/
theorem setupH_leaf {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st) : st.leaf = default := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

/-! ### The statement -/

section Model

/-- Every net stores `A` and `B` at its slot's width and reads at most 32 lookup bits (`net_width`). -/
abbrev NetWidths (c : Circuit) : Prop :=
  ∀ n < c.nets.size, (c.nets.getD n default).2.a.width = 2 ^ (c.nets.getD n default).2.unitLog ∧
    (c.nets.getD n default).2.b.width = 2 ^ (c.nets.getD n default).2.unitLog ∧
    ∀ lk, (c.nets.getD n default).2.lookup = some lk → lk.bits ≤ 32

/-- Every net range names a net. -/
abbrev NetKinds (c : Circuit) : Prop := ∀ r ∈ c.ranges.toList, ∀ i, r.kind = .net i → i < c.nets.size

theorem netWidths_of {tags : Tags} {c : Circuit} (P : ParseFacts tags c) : NetWidths c := fun _ hn => net_width P hn

variable {st : Stmt} (hW : NetWidths st.c) (hK : NetKinds st.c) (L : Layout st) (hleaf : st.leaf = default)
include hW hK L hleaf

theorem slotA_eqW {rg : Range} (hrg : rg ∈ st.c.ranges.toList) (i : ℕ) {c : ℕ} (hc : c < 2 ^ rg.slotLog) :
    slotEntryA st rg.kind i c = (slotA st rg.kind i c : ZMod 2) := by
  cases hk : rg.kind with
  | net n =>
      have hw := hW n (hK rg hrg n hk)
      rw [L.slotLog rg hrg n hk] at hc
      exact sparse_eq _ _ _ (Or.inl (by rw [hw.1]; exact hc))
  | mask =>
      show (if i = c then 1 else 0) = (((if i = c then 1 else 0 : ℕ)) : ZMod 2)
      split_ifs <;> simp
  | comp =>
      show sparseEntry st.leaf.comp.a i c = (spCnt st.leaf.comp.a i c : ZMod 2)
      rw [hleaf]
      exact sparse_eq _ _ _ (Or.inr (Nat.zero_le _))

theorem slotB_eqW {rg : Range} (hrg : rg ∈ st.c.ranges.toList) (i : ℕ) {c : ℕ} (hc : c < 2 ^ rg.slotLog) :
    slotEntryB st rg.kind i c = (slotB st rg.kind i c : ZMod 2) := by
  cases hk : rg.kind with
  | net n =>
      have hw := hW n (hK rg hrg n hk)
      rw [L.slotLog rg hrg n hk] at hc
      show sparseEntry _ i c + lookupEntry _ i c = ((spCnt _ i c + lkSide _ _ i c : ℕ) : ZMod 2)
      rw [Nat.cast_add, sparse_eq _ _ _ (Or.inl (by rw [hw.2.1]; exact hc))]
      congr 1
      cases hl : (st.c.nets.getD n default).2.lookup with
      | none => simp [lookupEntry, lkSide]
      | some lk =>
          rw [lookup_eq lk (hw.2.2 lk hl)]
          simp only [lkSide, ite_eq_left (show c < (st.c.nets.getD n default).2.b.width by rw [hw.2.1]; exact hc)]
  | mask =>
      show (if i = c then 1 else 0) = (((if i = c then 1 else 0 : ℕ)) : ZMod 2)
      split_ifs <;> simp
  | comp =>
      show sparseEntry st.leaf.comp.b i c = (spCnt st.leaf.comp.b i c : ZMod 2)
      rw [hleaf]
      exact sparse_eq _ _ _ (Or.inr (Nat.zero_le _))

omit hW hK L hleaf in
theorem rangesEntry_eq (entry : Kind → ℕ → ℕ → ZMod 2) (side : Kind → ℕ → ℕ → ℕ)
    (hside : ∀ rg ∈ st.c.ranges.toList, ∀ i c, c < 2 ^ rg.slotLog → entry rg.kind i c = (side rg.kind i c : ZMod 2))
    (r j : ℕ) : rangesEntry st entry r j = (rangeCnt side st.c.ranges.toList r j : ZMod 2) := by
  unfold rangesEntry rangeCnt
  rw [lastCover_eq]
  cases h : Refine.lastCover st.c.ranges.toList j with
  | none => simp
  | some rg =>
      dsimp only
      split_ifs
      · exact hside rg (lastCover_spec _ _ _ h).1 _ _ (Nat.mod_lt _ (Nat.two_pow_pos _))
      · simp

theorem placedA_eqW (r j : Fin (2 ^ st.c.kLog)) : placedA st r j = matA st r.val j.val := by
  unfold placedA matA
  rw [Nat.cast_add, rangesEntry_eq _ _ (fun rg hrg i c hc => slotA_eqW hW hK L hleaf hrg i hc), deltaEntry_eq]

theorem placedB_eqW (r j : Fin (2 ^ st.c.kLog)) : placedB st r j = matB st r.val j.val := by
  unfold placedB matB
  rw [Nat.cast_add, rangesEntry_eq _ _ (fun rg hrg i c hc => slotB_eqW hW hK L hleaf hrg i hc), deltaEntry_eq]

theorem model_eqW (hwf : StmtWF st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    L.model regs = stmtOf st hwf regs := by
  unfold Layout.model stmtOf
  congr 1 <;> funext r j
  · exact placedA_eqW hW hK L hleaf r j
  · exact placedB_eqW hW hK L hleaf r j

end Model

section ModelFlat

variable {st : Stmt} {tags : Tags} (P : ParseFacts tags st.c) (L : Layout st) (hleaf : st.leaf = default)
include P L hleaf

theorem slotA_eq {rg : Range} (hrg : rg ∈ st.c.ranges.toList) (i : ℕ) {c : ℕ} (hc : c < 2 ^ rg.slotLog) :
    slotEntryA st rg.kind i c = (slotA st rg.kind i c : ZMod 2) :=
  slotA_eqW (netWidths_of P) P.kinds L hleaf hrg i hc

theorem slotB_eq {rg : Range} (hrg : rg ∈ st.c.ranges.toList) (i : ℕ) {c : ℕ} (hc : c < 2 ^ rg.slotLog) :
    slotEntryB st rg.kind i c = (slotB st rg.kind i c : ZMod 2) :=
  slotB_eqW (netWidths_of P) P.kinds L hleaf hrg i hc

theorem placedA_eq (r j : Fin (2 ^ st.c.kLog)) : placedA st r j = matA st r.val j.val :=
  placedA_eqW (netWidths_of P) P.kinds L hleaf r j

theorem placedB_eq (r j : Fin (2 ^ st.c.kLog)) : placedB st r j = matB st r.val j.val :=
  placedB_eqW (netWidths_of P) P.kinds L hleaf r j

theorem model_eq (hwf : StmtWF st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    L.model regs = stmtOf st hwf regs :=
  model_eqW (netWidths_of P) P.kinds L hleaf hwf regs

end ModelFlat

/-- **The accepted statement is `stmtOf`'s.** For a statement `Stmt.setupH` accepted on the untyped path, the
statement `Placed.Accepted` reads (`Layout.model`: `placedA`, `placedB`) is the one the executable's refinement plays
(`Refine.stmtOf`: `matA`, `matB`), at any regions. -/
theorem model_eq_stmtOf {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) (hwf : StmtWF st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    (setupH_layout ht h).model regs = stmtOf st hwf regs :=
  model_eq (parse_facts (setupH_spec ht h).1) (setupH_layout ht h) (setupH_leaf h) hwf regs

/-- **A drawn table of `sitesAt` is `stmtOf` of its statement**, at the statement's own regions (`regionsOf`). -/
theorem table_stmtOf {I : Exec.Inputs} {mPts n : ℕ} (ht : I.tags.typed = false) {S : Finset (Fin n)} {d : Lean.Json}
    (x : DrawSetup I mPts S d) : (x.table ht).T.S = stmtOf x.st x.wf.1 (regionsOf x.st) :=
  model_eq_stmtOf ht x.h x.wf.1 _

/-- **`tableAt`'s table at a draw that sets up is `stmtOf` of a statement `Stmt.setupH` accepted** on the session's inputs
and the draw's JSON, with the same statement as any other setup of that draw. -/
theorem tableAt_stmtOf {I : Exec.Inputs} {mPts n : ℕ} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀)) {S : Finset (Fin n)} (x : DrawSetup I mPts S (dj S)) :
    ∃ y : DrawSetup I mPts S (dj S), y.st = x.st ∧
      (tableAt ht dj x₀ S).T.S = stmtOf y.st y.wf.1 (regionsOf y.st) := by
  obtain ⟨y, hy, e⟩ := tableAt_stmt ht dj x₀ x
  exact ⟨y, hy, by rw [e]; exact table_stmtOf ht y⟩

end FlockSoundness.Discharge.Integrate
