import Proofs.Flock.Claims.Recursion.SlotsCommit
import Proofs.Flock.Claims.Recursion.VStarLinks
import Proofs.Flock.Claims.Recursion.DeployComplete

/-!
# The hm96 slots' commit strings at the outer relation

`HmSlotsCommit` from `StagedPart` and `Flock.check` (`hmSlotsCommit_of_full`): at a staged part, the outer relation's
model is satisfied at the witness's table share (`layout_of_check`, `zOf`), and there `Hm.row_bc_flat` gives each slot's
`b ‖ c` as the commit string of its row and salt, at the model's columns (`bcCol`, `saltCol`, `Typed.msgCol`). This file
maps those columns onto the witness's hidden values that `SlotCommits` reads (`slotHid`, `rowHidden`): the out and salt
bits (`bc_bit`, `salt_bit`), and each row bit's input column, which the unit copies from its message bit (`leaf_copy`),
in either of `setupOne`'s branches (`table_rows`, `slotCommits_table`).

The parse facts come from the verifier's checks at the staged part: one table's circuit is `partCircuit`'s
(`partCircuit_staged`), in `ProvedScope.scopeOk` (`staged_scope`) and V\*'s parse (`checkParsed_ok`); every port a held
row or RecCap reads is an input row (`in_block`, `ownPorts_held`), and a held row is rep `0` or `1`. The staged part
is one table holding its whole population (`oneFull_staged`, `capFull_staged`), which the verifier's setup refuses
otherwise (`RecSetup.oneFull`). `ksLin2_deployed_full` and `bitsLin2_deployed_full` drop `HmSlotsCommit`.
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

theorem getD_map_fst (a : Array (ℕ × ℕ)) (j : ℕ) : (a.map (·.1)).getD j 0 = (a.getD j default).1 := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map]
  cases a[j]? <;> rfl

section copy

variable {tags : Flock.Tags} {st : Flock.Stmt} (L : FlockSoundness.Layout st)
  (regs : List (FlockSoundness.Model.Region st.c.kLog (st.m - st.c.kLog)))
  (F : FlockSoundness.Discharge.Aliased.UnitFacts tags st)

include F in
/-- **Row bit `i` of port `q` at its input column is its message bit**, in every block of a satisfying witness, and
that column is in the block: with unit `0`'s input leaves its `N` input words in order, the column (`portCol`'s, in
template instance `g`'s first unit) is the leaf reading the port's word `i / 16`, whose copy position
(`Aliased.copyAt`) is the row's message bit `i` (`Flat.leafSrc`), or a forced-zero row when that lies past the block,
where the message bit reads 0 too. -/
theorem leaf_copy {N g q i : ℕ} (hl : st.c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat)
    (hupv : 0 < st.c.upv) (hg : g < st.c.g) (hq : q < st.c.ports.size)
    (hbw : st.c.ports[q].bits = 16 * st.c.ports[q].words)
    (hN : 16 * (st.c.ports.extract 0 q).foldl (· + ·.words) 0 + st.c.ports[q].bits ≤ 16 * N)
    (hi : i < st.c.ports[q].bits) {z : FlockSoundness.Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z)
    (o : Fin (2 ^ (L.model regs).nbl)) :
    st.c.unitRange.slot (g * st.c.upv) + ((st.c.unit.inGroups.getD 0 default).col * Flock.WORD +
      16 * (st.c.ports.extract 0 q).foldl (· + ·.words) 0 + i) < 2 ^ st.c.kLog ∧
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
    rw [← hcol, getD_map_fst, FlockSoundness.Discharge.HiddenExec.portCols_16_get C.unit_groups.2 hxg]
  set t := i % 16 with ht
  have ht16 : t < (Flock.portCols st.c.unit.inGroups)[x]!.2 := by rw [hw, h16]; omega
  have hoffe : (st.c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * start + i =
      (Flock.portCols st.c.unit.inGroups)[x]!.1 + t := by
    rw [hc1]
    omega
  have hs₀ : g * st.c.upv < st.c.unitRange.count := by simpa using L.slot_lt hg hupv
  have hoff := L.port_lt hxp ht16
  have hy := L.unit_slot_lt hs₀ (lt_of_lt_of_le hoff (L.fit st.c.unitNet).2)
  refine ⟨by rw [hoffe]; exact hy, ?_⟩
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
    rw [ite_eq_left hxg, Nat.mul_mod_left, hl]
    have hlx : ((Array.range N).map Int.ofNat)[x]! = (x : Int) := by
      rw [getElem!_pos _ x (by simpa using hxN)]
      simp
    dsimp only
    have hb : 16 * (x - start) + t < st.c.ports[q]!.bits := by rw [getElem!_pos st.c.ports q hq]; omega
    rw [hlx, ite_eq_right (show ¬ ((x : Int) < 0) by omega), Int.toNat_natCast, hfind, hstq, ite_eq_left hb,
      Nat.mul_div_cancel _ hupv]
    congr 1 <;> omega
  unfold FlockSoundness.Discharge.Aliased.copyAt
  rw [ite_eq_left hm]
  by_cases hlt : FlockSoundness.flatSrcAt st.c (g * st.c.upv) ((Flock.portCols st.c.unit.inGroups)[x]!.1 + t) <
      2 ^ st.c.kLog
  · rw [dite_eq_left hlt, ← FlockSoundness.Discharge.Hm.blockBit_of_lt (L.model regs) z o hlt, hsrc]
  · rw [dite_eq_right hlt, FlockSoundness.Discharge.Aliased.zeroPos_block L regs F hs₀ hz o]
    rw [hsrc] at hlt
    unfold FlockSoundness.Discharge.Hm.blockBit
    split
    · rename_i h
      exact absurd h hlt
    · rfl

end copy

/-! ## A table's rows at the outer relation -/

/-- **What a table the verifier sets up gives at a satisfying witness of its model**, in either of `setupOne`'s
branches (the hidden statement's through its flat twin, `Hidden.sat_flat`): its row ranges, a template instance, its
blocks, each row's `b ‖ c` as the commit string of its message and salt (`Hm.row_bc_flat`), and each row bit's input
column its message bit (`leaf_copy`). -/
theorem table_rows {tags : Flock.Tags} {cf pf : ByteArray} {coins : _} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Flock.Stmt} (ht : tags.typed = false)
    (hd : FlockSoundness.Discharge.HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st)
    (hupv : 0 < st.c.upv) (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (L : FlockSoundness.Layout st) (regs : List (FlockSoundness.Model.Region st.c.kLog (st.m - st.c.kLog)))
    {z : FlockSoundness.Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) :
    FlockSoundness.Discharge.Hm.RowRanges st.c ∧ 0 < st.c.g ∧
      st.nbl = max 3 (Flock.log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) ∧
      (∀ (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ}, g < st.c.g → p < st.c.ports.size →
        FlockSoundness.Discharge.Hm.bytesOf (FlockSoundness.Discharge.Hm.blockBit (L.model regs) z o) 128
          (FlockSoundness.Discharge.Hm.bcCol st.c g p) = FlockSoundness.Binding.commitString
          ((Flock.HmRow.rowPrefix st.c.ports[p]!.words).data.toList ++
            FlockSoundness.Discharge.Hm.bytesOf (FlockSoundness.Discharge.Hm.blockBit (L.model regs) z o)
              (2 * st.c.ports[p]!.words) (FlockSoundness.Typed.msgCol st.c g p))
          (FlockSoundness.Discharge.Hm.bytesOf (FlockSoundness.Discharge.Hm.blockBit (L.model regs) z o) 192
            (FlockSoundness.Discharge.Hm.saltCol st.c g p))) ∧
      ∀ (o : Fin (2 ^ (L.model regs).nbl)) {N g q i : ℕ},
        st.c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat → g < st.c.g → ∀ hq : q < st.c.ports.size,
        st.c.ports[q].bits = 16 * st.c.ports[q].words →
        16 * (st.c.ports.extract 0 q).foldl (· + ·.words) 0 + st.c.ports[q].bits ≤ 16 * N →
        i < st.c.ports[q].bits →
        st.c.unitRange.slot (g * st.c.upv) + ((st.c.unit.inGroups.getD 0 default).col * Flock.WORD +
          16 * (st.c.ports.extract 0 q).foldl (· + ·.words) 0 + i) < 2 ^ st.c.kLog ∧
        FlockSoundness.Discharge.Hm.blockBit (L.model regs) z o (st.c.unitRange.slot (g * st.c.upv) +
          ((st.c.unit.inGroups.getD 0 default).col * Flock.WORD +
            16 * (st.c.ports.extract 0 q).foldl (· + ·.words) 0 + i)) =
          FlockSoundness.Discharge.Hm.blockBit (L.model regs) z o
            (FlockSoundness.Typed.msgCol st.c g q (i / 8) (i % 8)) := by
  unfold FlockSoundness.Discharge.HiddenExec.setupOne at hd
  cases hh : tags.hiddenOutputs
  · rw [hh] at hd
    simp only [Bool.false_eq_true, ↓reduceIte] at hd
    have F := FlockSoundness.Discharge.Aliased.setupH_unitFacts ht hd
    obtain ⟨-, -, hda, -⟩ := FlockSoundness.setupH_spec ht hd
    obtain ⟨ls, hcheck⟩ := F.parse.check
    exact ⟨FlockSoundness.Discharge.Hm.RowRanges.ofL L F.parse F.check,
      (FlockSoundness.Discharge.Layout.check_cuts F.parse.typed hcheck).2,
      (FlockSoundness.Discharge.Integrate.setupH_n hd).2,
      fun o g p hg hp => FlockSoundness.Discharge.Hm.row_bc_flat L F hda hupv hv1 hz o hg hp,
      fun o N g q i hl hg hq hbw hN hi => leaf_copy L regs F hl hupv hg hq hbw hN hi hz o⟩
  · rw [hh] at hd
    simp only [↓reduceIte] at hd
    have s := FlockSoundness.Discharge.Hidden.setupHidden_spec' hd
    obtain ⟨hp, hcr⟩ := FlockSoundness.Discharge.Hidden.parse_untyped ht s.parse
    have P := FlockSoundness.parse_facts hp
    have L0 : FlockSoundness.Layout (FlockSoundness.Discharge.Hidden.flatTwin st) :=
      FlockSoundness.layout_of (st := FlockSoundness.Discharge.Hidden.flatTwin st) P s.pin rfl rfl s.m s.kLog
    have F0 : FlockSoundness.Discharge.Aliased.UnitFacts tags (FlockSoundness.Discharge.Hidden.flatTwin st) :=
      FlockSoundness.Discharge.Hidden.unitFacts_of_parse (st := FlockSoundness.Discharge.Hidden.flatTwin st) hp rfl
    have hda : st.da = (Flock.HmRow.delta st.c st.pin).1 ++
        (FlockSoundness.Discharge.Hidden.outPairs st.c).toArray := by
      rw [s.da, FlockSoundness.Discharge.Hidden.hidden_delta']
    have hrows := FlockSoundness.Discharge.Hidden.out_rows_v1 hp hcr fun p hp => (hv1 p hp).1
    have hz0 := FlockSoundness.Discharge.Hidden.sat_flat L L0 F0 hv1 hda hrows hz
    obtain ⟨ls, hcheck⟩ := P.check
    refine ⟨FlockSoundness.Discharge.Hm.RowRanges.ofL (st := FlockSoundness.Discharge.Hidden.flatTwin st) L0 F0.parse
      F0.check, (FlockSoundness.Discharge.Layout.check_cuts P.typed hcheck).2,
      (FlockSoundness.Discharge.HiddenExec.setupHidden_n hd).2, fun o g p hg hp => ?_,
      fun o N g q i hl hg hq hbw hN hi => ?_⟩
    · exact FlockSoundness.Discharge.Hm.row_bc_flat L0 F0 rfl hupv hv1 hz0 o hg hp
    · exact leaf_copy L0 regs F0 hl hupv hg hq hbw hN hi hz0 o

theorem mapM_eq_some_map {α β : Type} {f : α → Option β} {g : α → β} :
    ∀ {l : List α}, (∀ a ∈ l, f a = some (g a)) → l.mapM f = some (l.map g)
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons, h a (by simp), mapM_eq_some_map fun a' ha' => h a' (List.mem_cons_of_mem _ ha')]
    rfl

theorem findIdx_range_eq {N w : ℕ} (hw : w < N) :
    ((Array.range N).map Int.ofNat).findIdx? (· == (w : Int)) = some w :=
  Array.findIdx?_eq_some_iff_getElem.2 ⟨by simpa using hw, by simp, fun j hj => by simp; omega⟩

/-- With unit `0`'s input leaves its `N` input words in order, row bit `i` of port `q` has its input column wherever
its word is one of them. -/
theorem portCol_eq {c : Flock.Circuit} {N q i : ℕ} (hl : c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat)
    (hw : (c.ports.extract 0 q).foldl (· + ·.words) 0 + i / 16 < N) :
    portCol c 0 q i = some ((c.unit.inGroups.getD 0 default).col * Flock.WORD +
      16 * ((c.ports.extract 0 q).foldl (· + ·.words) 0 + i / 16) + i % 16) := by
  unfold portCol
  rw [hl]
  dsimp only
  rw [findIdx_range_eq hw]
  rfl

/-- **Port `nm`'s row's hidden values at instance `n`**, wherever its words are among unit `0`'s input leaves. -/
theorem rowHidden_eq {c : Flock.Circuit} {N q : ℕ} {nm : String} (n : ℕ)
    (hl : c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat) (hq : c.ports.findIdx? (·.name == nm) = some q)
    (hqs : q < c.ports.size) (hN : 16 * (c.ports.extract 0 q).foldl (· + ·.words) 0 + c.ports[q].bits ≤ 16 * N) :
    rowHidden c nm n = some ((List.range c.ports[q].bits).map fun i => n / c.g * 2 ^ c.kLog +
      (c.unitRange.slot (n % c.g * c.upv) + ((c.unit.inGroups.getD 0 default).col * Flock.WORD +
        16 * (c.ports.extract 0 q).foldl (· + ·.words) 0 + i))) := by
  unfold rowHidden
  refine Option.bind_eq_some_iff.2 ⟨q, hq,
    Option.bind_eq_some_iff.2 ⟨c.ports[q], Array.getElem?_eq_getElem hqs, ?_⟩⟩
  dsimp only
  refine mapM_eq_some_map fun i hi => ?_
  have hi' := List.mem_range.1 hi
  have e : ∀ C S : ℕ, C + 16 * (S + i / 16) + i % 16 = C + 16 * S + i := fun C S => by omega
  rw [portCol_eq hl (by omega), Option.map_some, e]

/-- **Port `q`'s hm96 slot at instance `n` outputs its commit string**, at a table the verifier sets up whose model
the witness's one-table share satisfies: its out columns are the model's `b ‖ c` (`bc_bit`), the commit string of the
row's message and salt (`table_rows`), the message at the row's input columns (`rowHidden`, `leaf_copy`) and the
salt at its salt columns (`salt_bit`). -/
theorem slotCommits_table {tags : Flock.Tags} {cf pf : ByteArray} {coins : _} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Flock.Stmt} (ht : tags.typed = false)
    (hd : FlockSoundness.Discharge.HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st)
    (hupv : 0 < st.c.upv) (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {N : ℕ} (hl : st.c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat)
    (L : FlockSoundness.Layout st) (regs : List (FlockSoundness.Model.Region st.c.kLog (st.m - st.c.kLog)))
    {w : Claims.Flock.Wit} (hz : (L.model regs).Satisfies (Flock.Readable.zOf 1 w 0 st.m)) {nm : String} {q n : ℕ}
    (hq : st.c.ports.findIdx? (·.name == nm) = some q)
    (hN : 16 * (st.c.ports.extract 0 q).foldl (· + ·.words) 0 + st.c.ports[q]!.bits ≤ 16 * N)
    (hb : n / st.c.g < 2 ^ st.nbl) :
    SlotCommits st.c nm q n w := by
  obtain ⟨R, hg0, -, hbc, hleaf⟩ := table_rows ht hd hupv hv1 L regs hz
  have hqs : q < st.c.ports.size := (Array.findIdx?_eq_some_iff_getElem.1 hq).1
  have hgn : n % st.c.g < st.c.g := Nat.mod_lt _ hg0
  have hbw : st.c.ports[q].bits = 16 * st.c.ports[q].words := (hv1 _ (Array.getElem_mem_toList hqs)).2.1
  have hq! : st.c.ports[q]! = st.c.ports[q] := getElem!_pos st.c.ports q hqs
  rw [hq!] at hN
  replace hb : n / st.c.g < 2 ^ (L.model regs).nbl := by
    show n / st.c.g < 2 ^ (st.m - st.c.kLog)
    rw [L.m, Nat.add_sub_cancel_left]
    exact hb
  let o : Fin (2 ^ (L.model regs).nbl) := ⟨n / st.c.g, hb⟩
  have hpfx : rowPfx st.c q = (Flock.HmRow.rowPrefix st.c.ports[q]!.words).data.toList := by
    unfold rowPfx
    rw [Array.getElem?_eq_getElem hqs, hq!]
    rfl
  have hrow : Claims.Recursion.VStarB.readForms w (((rowHidden st.c nm n).getD []).map Flock.HmNets.Lin.var) =
      FlockSoundness.Discharge.Hm.bytesOf (FlockSoundness.Discharge.Hm.blockBit (L.model regs)
        (Flock.Readable.zOf 1 w 0 st.m) o) (2 * st.c.ports[q]!.words)
        (FlockSoundness.Typed.msgCol st.c (n % st.c.g) q) := by
    rw [rowHidden_eq n hl hq hqs hN, Option.getD_some]
    refine readForms_eq_of_bits (by simp [FlockSoundness.Discharge.Hm.length_bytesOf, hq!, hbw]; omega)
      fun i hi => ?_
    simp only [List.length_map, List.length_range] at hi
    simp only [List.getElem_map, List.getElem_range, formBit_var]
    obtain ⟨hy, he⟩ := hleaf o hl hgn hqs hbw hN hi
    rw [FlockSoundness.Discharge.Hm.bitL_bytesOf _ _ (by rw [hq!]; omega), ← he]
    exact (blockBit_zOf_one (L.model regs) w o hy).symm
  have hsalt : Claims.Recursion.VStarB.readForms w (hmSalt st.c q n) =
      FlockSoundness.Discharge.Hm.bytesOf (FlockSoundness.Discharge.Hm.blockBit (L.model regs)
        (Flock.Readable.zOf 1 w 0 st.m) o) 192 (FlockSoundness.Discharge.Hm.saltCol st.c (n % st.c.g) q) := by
    refine readForms_eq_of_bits (by simp [hmSalt, FlockSoundness.Discharge.Hm.length_bytesOf]) fun j hj => ?_
    have hj' : j < 1536 := by simpa [hmSalt] using hj
    have hy : hmBase st.c q n + (hmIn st.c 2 + j) < 2 ^ st.c.kLog := by
      have := FlockSoundness.Discharge.Hm.saltCol_lt R hgn hqs (j := j / 8) (t := j % 8) (by omega) (by omega)
      unfold FlockSoundness.Discharge.Hm.saltCol at this
      rw [hmIn_eq] at this
      unfold hmBase
      omega
    simp only [hmSalt, List.getElem_map, List.getElem_range, formBit_var]
    exact (salt_bit (L.model regs) w st.c rfl hb hy hj').symm
  refine slotCommits_of_bits fun t ht => ?_
  have hy : hmBase st.c q n + (hmOutCol st.c + t) < 2 ^ st.c.kLog := by
    have := FlockSoundness.Discharge.Hm.bcCol_lt R hgn hqs (j := t / 8) (t := t % 8) (by omega) (by omega)
    unfold FlockSoundness.Discharge.Hm.bcCol at this
    rw [hmOut_eq] at this
    unfold hmBase
    omega
  rw [hpfx, hrow, hsalt, ← hbc o hgn hqs]
  exact (bc_bit (L.model regs) w st.c rfl hb hy ht).symm

/-! ## The checks a staged part passed -/

theorem sum_rows_le : ∀ (a : List String) (b : List ℕ) (k : ℕ),
    ((((a.zip b).map fun p => (p.1, p.2, 16 * p.2)).take k).map (·.2.1)).sum ≤ b.sum
  | [], _, _ => by simp
  | _ :: _, [], _ => by simp
  | _ :: _, _ :: _, 0 => by simp
  | _ :: a, y :: b, k + 1 => by
    have := sum_rows_le a b k
    simp only [List.zip_cons_cons, List.map_cons, List.take_succ_cons, List.sum_cons]
    omega

/-- Rows that run an input block of `16 · w`-bit rows, then rows named among `outs`: a row named outside `outs` is in
the input block, so its words end within the block's. -/
theorem in_block_list {β : Type} (f : β → String × ℕ × ℕ) (l : List β) (ins outs : List String)
    (ip : List ℕ)
    (B : List (String × ℕ × ℕ)) (hB : ∀ r ∈ B, r.1 ∈ outs)
    (hr : l.map f = (ins.zip ip).map (fun p => (p.1, p.2, 16 * p.2)) ++ B)
    {q : ℕ} (hq : q < l.length) (hnm : (f l[q]).1 ∉ outs) :
    16 * ((l.take q).map fun p => (f p).2.1).sum + (f l[q]).2.2 ≤ 16 * ip.sum := by
  have hq' : q < (l.map f).length := by simpa using hq
  have hqA : q < ((ins.zip ip).map (fun p => (p.1, p.2, 16 * p.2))).length := by
    refine Nat.lt_of_not_le fun hge => hnm (hB _ ?_)
    have h := List.getElem_append_right (as := ((ins.zip ip).map (fun p => (p.1, p.2, 16 * p.2)))) (bs := B)
      (i := q) (h₂ := by rw [← hr]; exact hq') hge
    simp only [← hr, List.getElem_map] at h
    rw [h]
    exact List.getElem_mem _
  have hbits : (f l[q]).2.2 = 16 * (f l[q]).2.1 := by
    have h := List.getElem_append_left (bs := B) (h' := by rw [← hr]; exact hq') hqA
    simp only [← hr, List.getElem_map] at h
    rw [h]
  have htake : (l.take (q + 1)).map (fun p => (f p).2.1) =
      (((ins.zip ip).map (fun p => (p.1, p.2, 16 * p.2))).take (q + 1)).map (·.2.1) := by
    have h := congrArg (fun m => (m.take (q + 1)).map (·.2.1)) hr
    rw [List.take_append_of_le_length hqA, ← List.map_take, List.map_map] at h
    exact h
  have hs := sum_rows_le ins ip (q + 1)
  have hsucc : ((l.take (q + 1)).map fun p => (f p).2.1).sum =
      ((l.take q).map fun p => (f p).2.1).sum + (f l[q]).2.1 := by
    rw [List.take_add_one, List.map_append, List.sum_append, List.getElem?_eq_getElem hq]; simp
  rw [← htake, hsucc] at hs
  omega

/-- **A port named outside V\*'s outputs is one of its input rows**, at the circuit V\*'s parse check passed
(`checkParsed_ok`'s rows, `vstarRows`): its row ends within the unit's input words. -/
theorem in_block {c : Flock.Circuit} {sub : VStar.Sub} {u : VStar.UnitC}
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = Flock.RecVStar.vstarRows sub u)
    {nm : String} (hnm : nm ∉ sub.outputs) {q : ℕ} (hq : c.ports.findIdx? (·.name == nm) = some q) :
    16 * (c.ports.extract 0 q).foldl (· + ·.words) 0 + c.ports[q]!.bits ≤ 16 * u.inWords := by
  obtain ⟨hqs, hname, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hql : q < c.ports.toList.length := by simpa using hqs
  have hr' : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      (sub.inputs.zip u.inPorts).map (fun p => (p.1, p.2, 16 * p.2)) ++
        (sub.outputs.zip u.outPorts).map fun p => (p.1, (p.2 + 63) / 64 * 64, 16 * ((p.2 + 63) / 64 * 64)) := hr
  have h := in_block_list (fun p => (p.name, p.words, p.bits)) c.ports.toList sub.inputs sub.outputs u.inPorts _
    (fun r hr => by
      obtain ⟨⟨a, b⟩, hm, rfl⟩ := List.mem_map.1 hr
      exact (List.of_mem_zip hm).1) hr' hql (by simpa [beq_iff_eq.1 hname] using hnm)
  have hst : (c.ports.extract 0 q).foldl (· + ·.words) 0 = ((c.ports.toList.take q).map (·.words)).sum :=
    extract_words (·.words) c.ports q
  rw [hst, getElem!_pos c.ports q hqs, VStar.UnitC.inWords, FlockSoundness.foldl_add_eq_sum]
  simpa using h

theorem mIndex_out : Flock.RecVStar.mIndex "out" = none := by
  unfold Flock.RecVStar.mIndex
  exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

theorem claimIndex_out : Flock.RecVStar.claimIndex "out" = none := by
  unfold Flock.RecVStar.claimIndex
  exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

theorem toNat_laim : ("laim" : String).toNat? = none := by
  rw [← String.toNat?_toSlice, String.Slice.toNat?_eq_none_iff, String.isNat_toSlice, ← Bool.not_eq_true,
    String.isNat_iff]
  simp

theorem cIndex_claim : Flock.RecVStar.cIndex Flock.RecVStar.CLAIM_VALUE = none := by
  unfold Flock.RecVStar.cIndex Flock.RecVStar.CLAIM_VALUE
  refine ite_eq_right_iff.2 fun _ => ?_
  rw [show ("rec-claim".drop 5).toString = "laim" by decide +kernel]
  exact toNat_laim

/-- **A statement's own ports are none of its outputs**, `out` (no `m<k>` nor `claim<k>`), and each row one registered
under a `rec-c<k>` reads is rep `0` or `1`. -/
theorem ownPorts_held {sv : VStar.Stmt} {q : String × String × Array (Option ℕ)}
    (hq : q ∈ Flock.RecVStar.ownPorts sv) :
    q.1 ∉ sv.sub.outputs ∧
      ∀ k, Flock.RecVStar.cIndex q.2.1 = some k → ∀ rep, some rep ∈ q.2.2.toList → rep < 2 := by
  cases sv with
  | inner _ _ _ =>
    unfold Flock.RecVStar.ownPorts at hq
    obtain ⟨⟨nm, _⟩, -, h⟩ := List.mem_filterMap.1 hq
    dsimp only at h
    obtain ⟨k, hk, rfl⟩ := Option.map_eq_some_iff.1 h
    refine ⟨fun hm => ?_, fun _ _ rep hr => ?_⟩
    · rw [show nm = "out" from List.mem_singleton.1 hm, mIndex_out] at hk
      cases hk
    · simp at hr
      omega
  | sparse _ _ =>
    unfold Flock.RecVStar.ownPorts at hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl
    · refine ⟨by show "h" ∉ ["out"]; decide, fun _ _ rep hr => ?_⟩
      simp at hr
      omega
    · refine ⟨by show "claim" ∉ ["out"]; decide, fun _ hk => ?_⟩
      rw [cIndex_claim] at hk
      cases hk
  | opening _ _ _ _ _ _ =>
    unfold Flock.RecVStar.ownPorts at hq
    obtain ⟨⟨nm, _⟩, -, h⟩ := List.mem_filterMap.1 hq
    dsimp only at h
    cases hm : Flock.RecVStar.mIndex nm with
    | some k =>
      rw [hm] at h
      cases h
      refine ⟨fun ho => ?_, fun _ _ rep hr => ?_⟩
      · rw [show nm = "out" from List.mem_singleton.1 ho, mIndex_out] at hm
        cases hm
      · simp at hr
        omega
    | none =>
      cases hc : Flock.RecVStar.claimIndex nm with
      | some k =>
        rw [hm, hc] at h
        cases h
        refine ⟨fun ho => ?_, fun _ hk => ?_⟩
        · rw [show nm = "out" from List.mem_singleton.1 ho, claimIndex_out] at hc
          cases hc
        · rw [cIndex_claim] at hk
          cases hk
      | none =>
        rw [hm, hc] at h
        cases h
  | recOpen _ _ _ _ => simp [Flock.RecVStar.ownPorts] at hq
  | recOpenStacked _ _ _ _ => simp [Flock.RecVStar.ownPorts] at hq
  | cap _ _ => simp [Flock.RecVStar.ownPorts] at hq

/-- What `heldRows` holds: a row a statement's own port reads, at the port's index among its circuit's ports. -/
theorem mem_heldRows {r : XR} {s : ℕ} {sv : VStar.Stmt} {c : Flock.Circuit}
    (hs : (Flock.RecVStar.statements r.shape).toOption.bind (·[s]?) = some sv)
    (hc : r.w.parts[s]?.bind partCircuit = some c) {h : ℕ × String × ℕ × ℕ} (hh : h ∈ heldRows r s) :
    ∃ q ∈ Flock.RecVStar.ownPorts sv, ∃ k, Flock.RecVStar.cIndex q.2.1 = some k ∧
      some h.2.2.2 ∈ q.2.2.toList ∧
      h.2.1 = q.1 ∧ c.ports.findIdx? (·.name == q.1) = some h.2.2.1 := by
  unfold heldRows at hh
  rw [hs, hc] at hh
  cases hrs : repSched r.shape with
  | none =>
    rw [hrs] at hh
    simp at hh
  | some rs =>
    rw [hrs] at hh
    dsimp only at hh
    obtain ⟨q, hq, hh⟩ := List.mem_flatMap.1 hh
    split at hh
    · rename_i k hk
      obtain ⟨row, hrow, hh⟩ := List.mem_filterMap.1 hh
      cases row with
      | none => simp at hh
      | some rep =>
        cases hj : msgRound rs rep k with
        | none => simp [hj] at hh
        | some j =>
          cases hqi : c.ports.findIdx? (·.name == q.1) with
          | none => simp [hqi] at hh
          | some qi =>
            obtain rfl : (j, q.1, qi, rep) = h := by simpa [hj, hqi] using hh
            exact ⟨q, hq, k, hk, hrow, rfl, hqi⟩
    · simp at hh

theorem checkStmt_scopeOk {i : Flock.RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {part : Option ℕ} {bindings : Option VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecVStar.checkStmt (some i) tags tables st circuitB partition program registered part bindings rows =
      .ok ()) :
    Flock.ProvedScope.scopeOk st.c = true := by
  unfold Flock.RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
      cases h1
    · split at h
      · split at h
        · split at h
          · split at h
            · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
              cases h1
            rename_i hs
            simpa using hs
          · cases h
        · cases h
      · cases h

/-- **What a staged part's tables passed**: each is in `ProvedScope.scopeOk`, and its population is the instance
count of V\*'s statement `s` at the inner shape (`checkStmt`, `checkCircuit_ok`, `check_ok`). -/
theorem staged_scope {st : Stage} {s : ℕ} {x xO : Claims.Msg} (h : StagedPart st s x xO)
    {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) {t : ℕ} (ht : t < p.sts.size) :
    Flock.ProvedScope.scopeOk p.sts[t].c = true ∧ ∃ r ss sv, readX x = some r ∧
      Flock.RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some sv ∧
      Flock.RecVStar.population p.sts[t] = sv.instances := by
  obtain ⟨ms, i, spec, sts, rfl, hsu, -⟩ := h
  have e : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  subst e
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain ⟨r, hr, hinner, -⟩ := insO_staged hins
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  have hck := checks_stmt hC t ht
  refine ⟨checkStmt_scopeOk hck, ?_⟩
  obtain ⟨j, f, sv, hj, -, hc⟩ := checkStmt_circuit hck
  cases hj
  obtain ⟨hck', -⟩ := FlockSoundness.Discharge.Recursion.checkCircuit_ok hc
  obtain ⟨ss, hss, hs, -, hn, -⟩ := FlockSoundness.Discharge.Recursion.check_ok hck'
  exact ⟨r, ss, sv, hr, hss, hs, hn⟩

/-! ## A staged part is one table holding its whole population -/

/-- A setup that decodes has passed `RecSetup.oneFull` on its tables. -/
theorem RecIns.setup_oneFull {s : RecIns} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (h : s.setup = .ok (spec, sts)) : Flock.RecSetup.oneFull s.inner sts = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

/-- Under an inner shape, `RecSetup.oneFull` passes only one table, whose instances are its population's. -/
theorem oneFull_ok {i : Flock.RecVStar.Inner} {sts : Array Flock.Stmt}
    (h : Flock.RecSetup.oneFull (some i) sts = .ok ()) :
    sts.size = 1 ∧ ∀ t (ht : t < sts.size), Flock.RecVStar.population sts[t] ≤ sts[t].n := by
  unfold Flock.RecSetup.oneFull at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hs
    refine ⟨by simpa using hs, fun t ht => ?_⟩
    have h1 := Flock.Readable.array_forM_ok h t ht
    split at h1
    · cases h1
    · omega

/-- **A staged part sets up one table holding its whole population**: its setup ran `RecSetup.oneFull` under the
verifier's inner shape, which the staged statement carries (`insO_staged`). -/
theorem oneFull_staged {st : Stage} {s : ℕ} {x xO : Claims.Msg} {i : RecIns} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (h : StagedPart st s x xO) (hsu : setupO xO = some (i, spec, sts)) :
    sts.size = 1 ∧ ∀ t (ht : t < sts.size), Flock.RecVStar.population sts[t] ≤ sts[t].n := by
  obtain ⟨ms, -, -, -, rfl, -, -⟩ := h
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain ⟨r, -, hinner, -⟩ := insO_staged hins
  have h1 := RecIns.setup_oneFull hset
  rw [hinner] at h1
  exact oneFull_ok h1

/-- **A staged RecCap part's table holds all its `N` instances**: `checkStmt` makes its population `N`
(`staged_scope`), and `RecSetup.oneFull` its instances at least its population's (`oneFull_staged`), so a table drawn
short, which proves only its drawn units (`HmRow.drawn`), never stages. -/
theorem capFull_staged {st : Stage} {s : ℕ} {x xO : Claims.Msg} {i : RecIns} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} {r : XR} {l N : ℕ} (hS : StagedPart st s x xO) (hsu : setupO xO = some (i, spec, sts))
    (hr : readX x = some r) (hcap : (Flock.RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.cap l N))
    (h0 : 0 < sts.size) : N ≤ sts[0].n := by
  have hx := parse_of_setupO hsu
  obtain ⟨-, r', ss, sv, hr', hss, hsv, hn⟩ := staged_scope hS hx (t := 0) h0
  rw [hr] at hr'
  cases hr'
  rw [hss] at hcap
  have e : sv = .cap l N := Option.some.inj (hsv.symm.trans hcap)
  subst e
  have hle := (oneFull_staged hS hsu).2 0 h0
  have hp : Flock.RecVStar.population sts[0] = N := hn
  omega

/-! ## The slots commit at a staged part -/

theorem div_lt_blocks {n m g e : ℕ} (hg : 0 < g) (hn : n < m)
    (he : e = max 3 (Flock.log2ceil (max ((m + g - 1) / g) 1))) : n / g < 2 ^ e := by
  have h1 : (m + g - 1) / g ≤ 2 ^ e :=
    le_trans (le_trans (le_max_left _ 1) (FlockSoundness.le_two_pow_log2ceil _))
      (Nat.pow_le_pow_right (by norm_num) (he ▸ le_max_right _ _))
  have h2 : n / g < (m + g - 1) / g := by
    rw [Nat.div_lt_iff_lt_mul hg]
    have := Nat.div_add_mod' (m + g - 1) g
    have := Nat.mod_lt (m + g - 1) hg
    omega
  omega

theorem div_lt_blocks_two {n m g e : ℕ} (hn : n < 2)
    (he : e = max 3 (Flock.log2ceil (max ((m + g - 1) / g) 1))) :
    n / g < 2 ^ e := by
  have h8 : 2 ^ 3 ≤ 2 ^ e := Nat.pow_le_pow_right (by norm_num) (he ▸ le_max_left _ _)
  have := Nat.div_le_self n g
  have : (2 : ℕ) ^ 3 = 8 := by norm_num
  omega

/-- **V\*'s hm96 slots output their commit strings at the outer relation**: a staged part sets up one table, and a
RecCap part's table holds all its instances (`oneFull_staged`, `capFull_staged`), so at `Flock.check` its one table's
model is satisfied at the witness (`layout_of_check`), the table is `setupOne`'s (`setupTables_setupOne`), in scope
(`staged_scope`) and V\*'s parse (`checkParsed_ok`), and each held row's port and RecCap's `top` is an input row of
V\*'s (`in_block`, `ownPorts_held`) at a block of the table (`div_lt_blocks`), so `slotCommits_table` applies. -/
theorem hmSlotsCommit_of_full {st : Stage} : HmSlotsCommit (deployZkRec Flock.Readable.relation) st := by
  intro x ms s w r c hS hck hr hpc
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss, hss, hsv⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hS' := hS
  obtain ⟨-, i, spec, sts, -, hsu, -⟩ := hS'
  have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have hsz : p.sts.size = 1 := by rw [hp]; exact (oneFull_staged hS hsu).1
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0
  obtain ⟨L, regs, hz⟩ := Flock.Readable.layout_of_check hck hx ht hpi h0
  have hz1 : (L.model regs).Satisfies (Flock.Readable.zOf 1 w 0 p.sts[0].m) := by
    have e := congrArg (fun J => Flock.Readable.zOf J w 0 p.sts[0].m) hsz
    rw [← e]
    exact hz
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨spec0, hset⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall 0 h0
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨u, -, hupv, -, hl, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨-, hg0, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L regs hz1
  have hsv' : (Flock.RecVStar.statements r.shape).toOption.bind (·[s]?) = some sv := by
    rw [hss]
    exact hsv
  subst hc0
  refine ⟨fun h hh => ?_, fun l N hN q hq n hn => ?_⟩
  · obtain ⟨q, hq, k, hk, hrep, hnm, hqi⟩ := mem_heldRows hsv' hpc hh
    obtain ⟨hout, hrep2⟩ := ownPorts_held hq
    rw [hnm]
    exact slotCommits_table ht' hd (by omega) hv1 hl L regs hz1 hqi (in_block hrows hout hqi)
      (div_lt_blocks_two (hrep2 k hk _ hrep) hnbl)
  · rw [hsv'] at hN
    have e : sv = .cap l N := Option.some.inj hN
    subst e
    have hfull : N ≤ p.sts[0].n := by
      subst hp
      exact capFull_staged hS hsu hr hsv' h0
    have htop : "top" ∉ (VStar.Stmt.cap l N).sub.outputs := by show "top" ∉ ["out"]; decide
    exact slotCommits_table ht' hd (by omega) hv1 hl L regs hz1 hq (in_block hrows htop hq)
      (div_lt_blocks hg0 (by omega) hnbl)

section Corollaries

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`KSLin2` at the model of the coin server's record (`recTr`), with the slots' commit strings proved**:
`ksLin2_deployed_slots` with `HmSlotsCommit` from `hmSlotsCommit_of_full`. -/
theorem ksLin2_deployed_full (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf hd rest }
    let L := layoutBTops (finOpen fin)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    let bL : Brings := { bR with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D bL hP
  exact ksLin2_deployed_slots b aux nOf hd rest fin wE join final w hP hmSlotsCommit_of_full

/-- **`BitsLin2` at the model of the coin server's record (`recTr`), with the slots' commit strings proved**:
`bitsLin2_deployed_slots` with `HmSlotsCommit` from `hmSlotsCommit_of_full`. -/
theorem bitsLin2_deployed_full (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf hd rest }
    let L := layoutBTops (finOpen fin)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    let bL : Brings := { bR with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D bL hP
  exact bitsLin2_deployed_slots b aux nOf hd rest fin wE join final w hP hmSlotsCommit_of_full

end Corollaries

end RecDeploy
