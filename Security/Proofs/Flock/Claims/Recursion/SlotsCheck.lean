import Proofs.Flock.Claims.Recursion.SlotsCommit

/-!
# The hm96 slots' commit strings at the outer relation

Toward `HmSlotsCommit` from `StagedPart` and `Flock.check`: at a staged part, the outer relation's model is satisfied at
the witness's table share (`layout_of_check`, `zOf`), and there `Hm.row_bc_flat` gives each held row's `b ‖ c` as the
commit string of its row and salt, at the model's columns (`bcCol`, `saltCol`, `Typed.msgCol`). This file maps those
columns onto the witness's hidden values that `SlotCommits` reads (`slotHid`, `rowHidden`).
-/

namespace RecDeploy

/-- **A block's bit at the table share is the witness's hidden value** at the share's interleaved address. -/
theorem blockBit_zOf (S : FlockSoundness.Model.Statement) (w : Claims.Flock.Wit) (J t : ℕ) (o : Fin (2 ^ S.nbl))
    {y : ℕ} (hy : y < 2 ^ S.kLog) :
    FlockSoundness.Discharge.Hm.blockBit S (Flock.Readable.zOf J w t S.m) o y =
      w.hid.getD ((o.val * 2 ^ S.kLog + y) * J + t) false := by
  rw [FlockSoundness.Discharge.Hm.blockBit_of_lt S _ o hy]
  exact Flock.Readable.bit_toZ _

/-- At one table, the share's address is the block's own: `slotHid`'s form. -/
theorem blockBit_zOf_one (S : FlockSoundness.Model.Statement) (w : Claims.Flock.Wit) (o : Fin (2 ^ S.nbl))
    {y : ℕ} (hy : y < 2 ^ S.kLog) :
    FlockSoundness.Discharge.Hm.blockBit S (Flock.Readable.zOf 1 w 0 S.m) o y =
      w.hid.getD (o.val * 2 ^ S.kLog + y) false := by
  rw [blockBit_zOf S w 1 0 o hy, Nat.mul_one, Nat.add_zero]

/-- The hm96 net's output column, as `Hm.bcCol` reads it, is `hmOutCol`'s. -/
theorem hmOut_eq (c : Flock.Circuit) : FlockSoundness.Discharge.Hm.hmOut c = hmOutCol c := by
  unfold FlockSoundness.Discharge.Hm.hmOut hmOutCol
  rw [Array.getD_eq_getD_getElem?]
  cases (Flock.portCols (c.nets.getD c.hmNet default).2.outGroups)[0]? <;> rfl

/-- The hm96 net's `k`-th input column, as `Hm.saltCol` reads it, is `hmIn`'s. -/
theorem hmIn_eq (c : Flock.Circuit) (k : ℕ) : FlockSoundness.Discharge.Hm.hmIn c k = hmIn c k := by
  unfold FlockSoundness.Discharge.Hm.hmIn hmIn
  rw [Array.getD_eq_getD_getElem?]
  cases (Flock.portCols (c.nets.getD c.hmNet default).2.inGroups)[k]? <;> rfl

/-- **Port `q`'s `b ‖ c` at instance `n`, bit `t`**, as `Hm.row_bc_flat` reads it in block `n / g` (VU `n % g`), is
the hidden value at the slot's out column `t` (`hmCommit`'s). -/
theorem bc_bit (S : FlockSoundness.Model.Statement) (w : Claims.Flock.Wit) (c : Flock.Circuit) {q n t : ℕ}
    (hk : 2 ^ S.kLog = 2 ^ c.kLog) (hb : n / c.g < 2 ^ S.nbl) (hy : hmBase c q n + (hmOutCol c + t) < 2 ^ c.kLog)
    (ht : t < 1024) :
    FlockSoundness.Discharge.Hm.bitL (FlockSoundness.Discharge.Hm.bytesOf
      (FlockSoundness.Discharge.Hm.blockBit S (Flock.Readable.zOf 1 w 0 S.m) ⟨n / c.g, hb⟩) 128
      (FlockSoundness.Discharge.Hm.bcCol c (n % c.g) q)) t =
      w.hid.getD (slotHid c n (hmBase c q n) (hmOutCol c + t)) false := by
  have he : FlockSoundness.Discharge.Hm.bcCol c (n % c.g) q (t / 8) (t % 8) = hmBase c q n + (hmOutCol c + t) := by
    unfold FlockSoundness.Discharge.Hm.bcCol hmBase
    rw [hmOut_eq]
    omega
  rw [FlockSoundness.Discharge.Hm.bitL_bytesOf _ _ (by omega), he, blockBit_zOf_one S w _ (by rw [hk]; exact hy), hk]
  rfl

/-- **Port `q`'s salt at instance `n`, bit `j`**, as `Hm.row_bc_flat` reads it in block `n / g`, is the hidden value
at the slot's salt column `j` (`hmSalt`'s). -/
theorem salt_bit (S : FlockSoundness.Model.Statement) (w : Claims.Flock.Wit) (c : Flock.Circuit) {q n j : ℕ}
    (hk : 2 ^ S.kLog = 2 ^ c.kLog) (hb : n / c.g < 2 ^ S.nbl) (hy : hmBase c q n + (hmIn c 2 + j) < 2 ^ c.kLog)
    (hj : j < 1536) :
    FlockSoundness.Discharge.Hm.bitL (FlockSoundness.Discharge.Hm.bytesOf
      (FlockSoundness.Discharge.Hm.blockBit S (Flock.Readable.zOf 1 w 0 S.m) ⟨n / c.g, hb⟩) 192
      (FlockSoundness.Discharge.Hm.saltCol c (n % c.g) q)) j =
      w.hid.getD (slotHid c n (hmBase c q n) (hmIn c 2 + j)) false := by
  have he : FlockSoundness.Discharge.Hm.saltCol c (n % c.g) q (j / 8) (j % 8) = hmBase c q n + (hmIn c 2 + j) := by
    unfold FlockSoundness.Discharge.Hm.saltCol hmBase
    rw [hmIn_eq]
    omega
  rw [FlockSoundness.Discharge.Hm.bitL_bytesOf _ _ (by omega), he, blockBit_zOf_one S w _ (by rw [hk]; exact hy), hk]
  rfl

/-! ## A row bit's input column copies its message bit -/

theorem words_take_mono {α : Type} (f : α → ℕ) (l : List α) {a b : ℕ} (h : a ≤ b) :
    ((l.take a).map f).sum ≤ ((l.take b).map f).sum := by
  have e := List.take_append_drop a (l.take b)
  rw [List.take_take, Nat.min_eq_left h] at e
  conv => rhs; rw [← e]
  rw [List.map_append, List.sum_append]
  omega

theorem words_take_succ {α : Type} (f : α → ℕ) (l : List α) {q : ℕ} (hq : q < l.length) :
    ((l.take (q + 1)).map f).sum = ((l.take q).map f).sum + f l[q] := by
  rw [List.take_add_one, List.map_append, List.sum_append, List.getElem?_eq_getElem hq]
  simp

theorem foldl_words {α : Type} (f : α → ℕ) : ∀ (l : List α) (a : ℕ),
    l.foldl (fun a p => a + f p) a = a + (l.map f).sum
  | [], a => by simp
  | p :: l, a => by rw [List.foldl_cons, foldl_words f l]; simp [Nat.add_assoc]

theorem extract_words {α : Type} (f : α → ℕ) (as : Array α) (q : ℕ) :
    (as.extract 0 q).foldl (fun a p => a + f p) 0 = ((as.toList.take q).map f).sum := by
  rw [← Array.foldl_toList, Array.toList_extract, foldl_words]
  simp [List.extract]

/-- Two runs of consecutive lengths that both hold `x` start at one index. -/
theorem words_disj {α : Type} (f : α → ℕ) (l : List α) {q q' x : ℕ} (hq : q < l.length) (hq' : q' < l.length)
    (h1 : ((l.take q).map f).sum ≤ x) (h2 : x < ((l.take q).map f).sum + f l[q])
    (h1' : ((l.take q').map f).sum ≤ x) (h2' : x < ((l.take q').map f).sum + f l[q']) : q' = q := by
  rcases Nat.lt_trichotomy q q' with h | h | h
  · have := words_take_mono f l (show q + 1 ≤ q' by omega)
    rw [words_take_succ f l hq] at this
    omega
  · exact h.symm
  · have := words_take_mono f l (show q' + 1 ≤ q by omega)
    rw [words_take_succ f l hq'] at this
    omega

/-- **The port `Flat.leafSrc` reads an input word from** is the one whose words hold it. -/
theorem starts_find_eq (c : Flock.Circuit) {q x : ℕ} (hq : q < c.ports.size)
    (h1 : (FlockSoundness.Typed.starts c)[q]! ≤ x)
    (h2 : x < (FlockSoundness.Typed.starts c)[q]! + c.ports[q].words) :
    ((List.range (FlockSoundness.Typed.starts c).size).reverse.find?
      (fun q => (FlockSoundness.Typed.starts c)[q]! ≤ x)).getD 0 = q := by
  have hql : q < c.ports.toList.length := by simpa using hq
  rw [FlockSoundness.Discharge.Layout.starts_get c hq] at h1 h2
  have hx : x < c.inWords := by
    have := words_take_mono (·.words) c.ports.toList (show q + 1 ≤ c.ports.toList.length by omega)
    rw [words_take_succ _ _ hql, List.take_length] at this
    rw [FlockSoundness.Discharge.Layout.inWords_eq]
    simp only [Array.getElem_toList] at this
    omega
  obtain ⟨hq', h1', h2'⟩ := FlockSoundness.Discharge.Layout.starts_find c hx rfl
  rw [FlockSoundness.Discharge.Layout.starts_get c hq'] at h1' h2'
  exact words_disj (·.words) c.ports.toList hql (by simpa using hq') h1 (by simpa using h2) h1' (by simpa using h2')

section copy

variable {tags : Flock.Tags} {st : Flock.Stmt} (L : FlockSoundness.Layout st)
  (regs : List (FlockSoundness.Model.Region st.c.kLog (st.m - st.c.kLog)))
  (F : FlockSoundness.Discharge.Aliased.UnitFacts tags st)

/-- **Row bit `i` of port `q` at its input column is its message bit**, in every block of a satisfying witness: with
unit `0`'s input leaves its `N` input words in order, the column (`portCol`'s, in template instance `g`'s first unit)
is the leaf reading the port's word `i / 16`, whose copy position (`Aliased.copyAt`) is the row's message bit `i`
(`Flat.leafSrc`), or a forced-zero row when that lies past the block, where the message bit reads 0 too. -/
include F in
theorem leaf_copy {N g q i : ℕ} (hl : st.c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat)
    (hupv : 0 < st.c.upv) (hg : g < st.c.g) (hq : q < st.c.ports.size)
    (hN : 16 * (st.c.ports.extract 0 q).foldl (· + ·.words) 0 + st.c.ports[q].bits ≤ 16 * N)
    (hi : i < st.c.ports[q].bits) {z : FlockSoundness.Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z)
    (o : Fin (2 ^ (L.model regs).nbl)) :
    FlockSoundness.Discharge.Hm.blockBit (L.model regs) z o (st.c.unitRange.slot (g * st.c.upv) +
      ((st.c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * (st.c.ports.extract 0 q).foldl (· + ·.words) 0 +
        i)) =
      FlockSoundness.Discharge.Hm.blockBit (L.model regs) z o
        (FlockSoundness.Typed.msgCol st.c g q (i / 8) (i % 8)) := by
  have C := F.check
  set start := (st.c.ports.extract 0 q).foldl (· + ·.words) 0 with hstart
  have hst : start = ((st.c.ports.toList.take q).map (·.words)).sum := extract_words (·.words) st.c.ports q
  have hl0 : st.c.leavesIn.getD 0 #[] = st.c.leavesIn[0]! := by
    have h0 : 0 < st.c.leavesIn.size := by rw [C.leaves.1]; exact hupv
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h0, Option.getD_some, getElem!_pos st.c.leavesIn 0 h0]
  have hsz : (st.c.unit.inGroups.getD 0 default).bits.size = N := by
    rw [← C.leaves.2 0 hupv, ← hl0, hl]
    simp
  set x := start + i / 16 with hx
  have hxN : x < N := by omega
  have hxg : x < (st.c.unit.inGroups.getD 0 default).bits.size := by omega
  obtain ⟨hxp, hcol, hw⟩ := FlockSoundness.firstGroup_port C.unit_groups.1 hxg
  have h16 : (st.c.unit.inGroups.getD 0 default).bits[x]! = 16 := C.unit_groups.2 _ (by
    rw [getElem!_pos (st.c.unit.inGroups.getD 0 default).bits x hxg]
    exact Array.getElem_mem_toList _)
  have hc1 : (Flock.portCols st.c.unit.inGroups)[x]!.1 =
      (st.c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * x := by
    rw [← hcol, ← FlockSoundness.Discharge.HiddenExec.portCols_16_get C.unit_groups.2 hxg,
      Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_map]
    cases (Flock.portCols #[st.c.unit.inGroups.getD 0 default])[x]? <;> rfl
  set t := i % 16 with ht
  have ht16 : t < (Flock.portCols st.c.unit.inGroups)[x]!.2 := by rw [hw, h16]; omega
  have hoffe : (st.c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * start + i =
      (Flock.portCols st.c.unit.inGroups)[x]!.1 + t := by
    rw [hc1]
    omega
  have hs₀ : g * st.c.upv < st.c.unitRange.count := by simpa using L.slot_lt hg hupv
  have hoff := L.port_lt hxp ht16
  have hy := L.unit_slot_lt hs₀ (lt_of_lt_of_le hoff (L.fit st.c.unitNet).2)
  rw [hoffe, FlockSoundness.Discharge.Hm.blockBit_of_lt (L.model regs) z o hy,
    FlockSoundness.Discharge.Aliased.copyAt_block L regs F hs₀ hoff ⟨_, hy⟩ rfl hz o]
  have hm : (Flock.portCols st.c.unit.inGroups)[x]!.1 + t ∈ FlockSoundness.portBits st.c.unit.inGroups := by
    unfold FlockSoundness.portBits
    refine List.mem_flatMap.2 ⟨(Flock.portCols st.c.unit.inGroups)[x]!, ?_, ?_⟩
    · rw [getElem!_pos (Flock.portCols st.c.unit.inGroups) x hxp]
      exact Array.getElem_mem_toList _
    · rw [List.mem_range'_1]
      omega
  have hstq : (FlockSoundness.Typed.starts st.c)[q]! = start := by
    rw [FlockSoundness.Discharge.Layout.starts_get st.c hq, hst]
  have hfind := starts_find_eq st.c hq (x := x) (by omega) (by omega)
  have hsrc : FlockSoundness.flatSrcAt st.c (g * st.c.upv) ((Flock.portCols st.c.unit.inGroups)[x]!.1 + t) =
      FlockSoundness.Typed.msgCol st.c g q (i / 8) (i % 8) := by
    unfold FlockSoundness.flatSrcAt
    rw [FlockSoundness.portOf_eq (FlockSoundness.Discharge.Aliased.portBits_nodup L.unit_built) hxp ht16,
      Nat.add_sub_cancel_left]
    unfold FlockSoundness.Flat.flatSrc FlockSoundness.Flat.leafSrc
    rw [if_pos hxg, Nat.mul_mod_left, hl]
    have hlx : ((Array.range N).map Int.ofNat)[x]! = (x : Int) := by
      rw [getElem!_pos _ x (by simpa using hxN)]
      simp
    dsimp only
    have hb : 16 * (x - start) + t < st.c.ports[q]!.bits := by rw [getElem!_pos st.c.ports q hq]; omega
    rw [hlx, if_neg (show ¬ ((x : Int) < 0) by omega), Int.toNat_natCast, hfind, hstq, if_pos hb,
      Nat.mul_div_cancel _ hupv]
    congr 1 <;> omega
  unfold FlockSoundness.Discharge.Aliased.copyAt
  rw [if_pos hm]
  by_cases hlt : FlockSoundness.flatSrcAt st.c (g * st.c.upv) ((Flock.portCols st.c.unit.inGroups)[x]!.1 + t) <
      2 ^ st.c.kLog
  · rw [dif_pos hlt, ← FlockSoundness.Discharge.Hm.blockBit_of_lt (L.model regs) z o hlt, hsrc]
  · rw [dif_neg hlt, FlockSoundness.Discharge.Aliased.zeroPos_block L regs F hs₀ hz o]
    rw [hsrc] at hlt
    unfold FlockSoundness.Discharge.Hm.blockBit
    split
    · rename_i h
      exact absurd h hlt
    · rfl

end copy

end RecDeploy
