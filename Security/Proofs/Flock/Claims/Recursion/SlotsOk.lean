import Proofs.Flock.Claims.Recursion.HpEarly
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.RowCommit
import Proofs.Flock.Soundness.Discharge.Hm.Row
import Proofs.Flock.Soundness.Discharge.Hm.Slot
import Proofs.Flock.Soundness.Discharge.Hm.Place
import Proofs.Flock.Soundness.Discharge.Hm.Copy
import Proofs.Flock.Soundness.ExecRows
import Proofs.Flock.Soundness.SparseRows

/-!
# The trace assigner's hm96 slots commit (`slotsOk_assign`)

`HpEarly.SlotsOk` at the trace assigner `VStarTrace.A mask`, from the assigner's definition: port `q`'s hm96 slot at
instance `n` is `evalNet` of the hm96 net on the inputs `hmSlot` writes (`hm_bit`), the row's last chaining value at
`cv`, its message's last block at `pad` and its salt at `y` (`slot_in`). `evalNet` computes every row of the net in
order, so the slot holds the net's rows (`evalNet_rows`, `LaidRows`), and its inputs are the last chaining value and
block of the row's `sha512/row/v1` message (`chain_cvs`, `block_eq`): `slotCommits_of_hm96` gives its commit string.

What it reads besides `TraceOk`: the circuit's row nets are the verifier's (`HmNets.check`), the hm96 net's columns
fit `Sparse`'s 32-bit encoding (`kLog ≤ 32`), and every port is a `sha512/row/v1` row of whole 128-byte chunks
(`2 · words = 128 · chunks`), as the v1 parse makes it.
-/

namespace RecDeploy.SlotsOkProof

open _root_.Flock (Circuit Net WORD portCols Port)
open FlockSoundness.Discharge
open Claims.Recursion.VStarB (readForms formBit)
open RecDeploy.VStarTrace (A hmN TraceOk getD_set! getD_set!_ne size_set! foldl_range_spec foldl_keeps getD_map_range
  lt_mul_of blk_lt slot_eq bitAt_eq hid_getD lt_two_pow_m slots_hm blk_div evalNet_input name_of_findIdx
  count_of_portRow portIn_some wordBits_get wordBits_length wordBytes_get putBytes_spec putWords_size placesR_assign
  ite_t ite_f)
open _root_.VStar.Trace (evalNet hmSlot portIn wordBytes putWords putBytes blockWords rowIns rowAt zeros chain)

set_option hygiene false in
local notation "𝑾" => (c.ports.getD q default).words
set_option hygiene false in
local notation "𝑵" => (c.ports.getD q default).chunks
set_option hygiene false in
local notation "𝑫" => (wordBytes rw.words (c.ports.getD q default).words).data.toList
set_option hygiene false in
local notation "𝑷" => Hm.padData (Hm.rowMsg (c.ports.getD q default).words
  (wordBytes rw.words (c.ports.getD q default).words).data.toList)

/-! ## XOR folds -/

theorem foldl_xor (g : ℕ → Bool) : ∀ (l : List ℕ) (s : Bool),
    l.foldl (fun s k => s ^^ g k) s = (s ^^ l.foldr (fun j b => g j ^^ b) false)
  | [], s => by simp
  | k :: l, s => by rw [List.foldl_cons, foldl_xor g l, List.foldr_cons, Bool.xor_assoc]

theorem foldl_xorRow (v : ℕ → Bool) (cs : Array ℕ) : cs.foldl (fun s k => s ^^ v k) false = Hm.xorRow v cs := by
  rw [← Array.foldl_toList, foldl_xor, Bool.false_xor]
  rfl

theorem xorRow_congr {v v' : ℕ → Bool} {cs : Array ℕ} (h : ∀ j ∈ cs.toList, v j = v' j) :
    Hm.xorRow v cs = Hm.xorRow v' cs := by
  unfold Hm.xorRow
  generalize cs.toList = l at h
  induction l with
  | nil => rfl
  | cons j l ih =>
    rw [List.foldr_cons, List.foldr_cons, h j List.mem_cons_self, ih fun k hk => h k (List.mem_cons_of_mem _ hk)]

/-! ## `evalNet` computes the net's rows -/

/-- One step of `evalNet`'s fold: row `i` set to its `A` and `B` XORs, the constant left. -/
def evStep (net : Net) (z : Array Bool) (i : ℕ) : Array Bool :=
  if i == net.constPos then z
  else z.set! i ((net.a.row i).foldl (fun s k => s ^^ z.getD k false) false &&
    (net.b.row i).foldl (fun s k => s ^^ z.getD k false) false)

/-- `evalNet`'s start: the input rows' bits and the constant. -/
def ev0 (net : Net) (ins : Array Bool) : Array Bool :=
  (Array.range (2 ^ net.unitLog)).map fun i =>
    if i == net.constPos then true else i < net.inWords * WORD && (net.a.row i).size == 1 && ins.getD i false

theorem evalNet_eq (net : Net) (ins : Array Bool) :
    evalNet net ins =
      (List.range' (net.inWords * WORD) (net.useful - net.inWords * WORD)).foldl (evStep net) (ev0 net ins) := rfl

theorem evStep_size (net : Net) (z : Array Bool) (i : ℕ) : (evStep net z i).size = z.size := by
  unfold evStep
  split
  · rfl
  · exact size_set!

theorem evFold_size (net : Net) : ∀ (l : List ℕ) (z : Array Bool), (l.foldl (evStep net) z).size = z.size
  | [], _ => rfl
  | i :: l, z => by rw [List.foldl_cons, evFold_size net l, evStep_size]

theorem evStep_keep {net : Net} {z : Array Bool} {i k : ℕ} (h : k ≠ i ∨ i = net.constPos) :
    (evStep net z i).getD k false = z.getD k false := by
  unfold evStep
  split
  · rfl
  · rename_i hne
    rcases h with h | h
    · exact getD_set!_ne (Ne.symm h)
    · exact absurd (beq_iff_eq.2 h) hne

theorem evFold_keep {net : Net} {k : ℕ} (l : List ℕ) (z : Array Bool) (h : ∀ j ∈ l, k ≠ j ∨ j = net.constPos) :
    (l.foldl (evStep net) z).getD k false = z.getD k false :=
  foldl_keeps (p := fun j => k ≠ j ∨ j = net.constPos) (fun _ _ hj => evStep_keep hj) l z h

/-- **`evalNet` holds the net's rows**: its constant is 1, and each computed row `i` (from the input rows to the
constant) is the AND of its `A` and `B` XORs, the rows `ra` and `rb` the net's `A` and `B` read back, in the order
`Net.checkOrder` checks. -/
theorem evalNet_rows {net : Net} {ins : Array Bool} {ra rb : Array (Array ℕ)}
    (hra : ∀ i, net.a.row i = ra.getD i #[]) (hrb : ∀ i, net.b.row i = rb.getD i #[])
    (hord : FlockSoundness.RowOrder (net.inWords * WORD) net.constPos ra rb) (huse : net.useful = net.constPos + 1)
    (hc : net.inWords * WORD ≤ net.constPos) (hl : net.constPos < 2 ^ net.unitLog) :
    (evalNet net ins).getD net.constPos false = true ∧
      ∀ i, net.inWords * WORD ≤ i → i < net.constPos →
        (Hm.xorRow (fun j => (evalNet net ins).getD j false) (ra.getD i #[]) &&
          Hm.xorRow (fun j => (evalNet net ins).getD j false) (rb.getD i #[])) = (evalNet net ins).getD i false := by
  have h0s : (ev0 net ins).size = 2 ^ net.unitLog := by simp [ev0]
  rw [evalNet_eq]
  refine ⟨?_, fun i hi1 hi2 => ?_⟩
  · rw [evFold_keep _ _ fun j _ => (Decidable.em (j = net.constPos)).elim Or.inr fun h => Or.inl fun e => h e.symm]
    unfold ev0
    rw [getD_map_range hl]
    simp
  · have e1 : net.useful - net.inWords * WORD = (i - net.inWords * WORD) + (net.constPos - i + 1) := by omega
    rw [e1, ← List.range'_append_1, show net.inWords * WORD + (i - net.inWords * WORD) = i by omega,
      List.range'_succ, List.foldl_append, List.foldl_cons]
    generalize hZ : (List.range' (net.inWords * WORD) (i - net.inWords * WORD)).foldl (evStep net) (ev0 net ins) = zr
    have hZs : zr.size = 2 ^ net.unitLog := by rw [← hZ, evFold_size, h0s]
    have hrest : ∀ k, (k < i + 1 ∨ k = net.constPos) →
        ((List.range' (i + 1) (net.constPos - i)).foldl (evStep net) (evStep net zr i)).getD k false =
          (evStep net zr i).getD k false := by
      intro k hk
      refine evFold_keep _ _ fun j hj => ?_
      have := List.mem_range'_1.1 hj
      rcases hk with hk | hk
      · exact Or.inl (by omega)
      · by_cases hj' : j = net.constPos
        · exact Or.inr hj'
        · exact Or.inl (by omega)
    have hi' : i ≠ net.constPos := by omega
    have hZi : (evStep net zr i).getD i false =
        ((net.a.row i).foldl (fun s k => s ^^ zr.getD k false) false &&
          (net.b.row i).foldl (fun s k => s ^^ zr.getD k false) false) := by
      unfold evStep
      rw [ite_f (by simpa using hi'), getD_set! (by omega)]
      exact if_pos rfl
    have hkeep : ∀ k, (k = net.constPos ∨ k < i) →
        ((List.range' (i + 1) (net.constPos - i)).foldl (evStep net) (evStep net zr i)).getD k false = zr.getD k false := by
      intro k hk
      rw [hrest k (hk.elim Or.inr fun h => Or.inl (by omega)), evStep_keep (Or.inl (by rcases hk with h | h <;> omega))]
    rw [hrest i (Or.inl (by omega)), hZi, hra i, hrb i, foldl_xorRow (fun k => zr.getD k false),
      foldl_xorRow (fun k => zr.getD k false)]
    congr 1
    · refine xorRow_congr fun k hk => ?_
      exact hkeep k ((hord i hi1 hi2 k (List.mem_append_left _ hk)).imp id And.left)
    · refine xorRow_congr fun k hk => ?_
      exact hkeep k ((hord i hi1 hi2 k (List.mem_append_right _ hk)).imp id And.left)

/-! ## The slot's values and inputs -/

/-- Port `q`'s hm96 slot at instance `n`, as the trace assigner computes it. -/
abbrev slotE (c : Circuit) (pin : _root_.VStar.In.PartIn) (n q : ℕ) : Array Bool :=
  evalNet (hmN c) (hmSlot c pin ((Array.range c.ports.size).map fun p => chain c pin p n) n q)

/-- The trace's witness bits at block `n / g`'s in-block columns. -/
def slotW (mask : ℕ → ℕ → Bool) (c : Circuit) (pin : _root_.VStar.In.PartIn) (n : ℕ) (y : ℕ) : Bool :=
  (A mask c pin).hid.getD (n / c.g * 2 ^ c.kLog + y) false

variable {c : Circuit} {pin : _root_.VStar.In.PartIn}

/-- **The trace at port `q`'s hm96 slot** of instance `n`: column `t` of the slot holds the slot's value `t`. -/
theorem hm_bit (h : TraceOk c pin) (mask : ℕ → ℕ → Bool) {q n t : ℕ} (hq : q < c.ports.size)
    (hn : n < _root_.VStar.Trace.count pin) (ht : t < 2 ^ (hmN c).unitLog) :
    slotW mask c pin n (c.slotCol c.hmNet (n % c.g) q + t) = (slotE c pin n q).getD t false := by
  unfold slotW Circuit.slotCol
  have hP := h.ports_hm
  have hcount := h.hm_count
  have hslot := h.hm_slot
  have hblock := h.hm_block
  have hk' : n % c.g * c.perVu.getD c.hmNet 0 + q < (c.rangeOf c.hmNet).count := by
    have h1 := Nat.mod_lt n h.g_pos
    have h2 := Nat.mul_le_mul_right (c.perVu.getD c.hmNet 0) (Nat.succ_le_of_lt h1)
    rw [Nat.succ_mul] at h2
    omega
  have ht' : t < 2 ^ (c.rangeOf c.hmNet).slotLog := by rw [hslot]; exact ht
  have hblk := blk_lt h.g_pos hn
  have hspan := lt_mul_of hk' ht'
  have hb : (c.rangeOf c.hmNet).slot (n % c.g * c.perVu.getD c.hmNet 0 + q) + t < 2 ^ c.kLog := by
    rw [slot_eq]; omega
  rw [hid_getD h mask (lt_two_pow_m h hn hb)]
  rw [bitAt_eq h.hm_lt hk' ht' ht ?hz hb ?hd]
  case hz =>
    rw [slots_hm h, Array.size_map, Array.size_range]
    exact lt_mul_of hblk hk'
  case hd =>
    intro x hx hne hin
    refine h.disjoint x c.hmNet _ hx h.hm_lt hne hin ⟨?_, ?_⟩
    · rw [slot_eq]; omega
    · rw [slot_eq]; omega
  rw [slots_hm h, getD_map_range (lt_mul_of hblk hk'), (blk_div hk').1, (blk_div hk').2,
    getD_map_range hblk]
  have hPpos : 0 < c.perVu.getD c.hmNet 0 := by omega
  have hqP : q < c.perVu.getD c.hmNet 0 := by omega
  have hd1 : (n % c.g * c.perVu.getD c.hmNet 0 + q) / c.perVu.getD c.hmNet 0 = n % c.g := by
    rw [Nat.mul_comm, Nat.mul_add_div hPpos, Nat.div_eq_of_lt hqP, Nat.add_zero]
  have hd2 : (n % c.g * c.perVu.getD c.hmNet 0 + q) % c.perVu.getD c.hmNet 0 = q := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hqP]
  have hi : n / c.g * c.g + n % c.g = n := Nat.div_add_mod' n c.g
  have hkg : n % c.g * c.perVu.getD c.hmNet 0 + q < c.g * c.perVu.getD c.hmNet 0 := by
    have h1 := Nat.mod_lt n h.g_pos
    have h2 := Nat.mul_le_mul_right (c.perVu.getD c.hmNet 0) (Nat.succ_le_of_lt h1)
    rw [Nat.succ_mul] at h2
    omega
  unfold rowIns
  simp only []
  rw [getD_map_range hkg, hd1, hd2, getD_map_range (Nat.mod_lt n h.g_pos), hi]

theorem putWords_spec (w : Array Bool) (col : ℕ) (ws : Array UInt64) :
    (putWords w col ws).size = w.size ∧ ∀ xc < w.size, (putWords w col ws).getD xc false =
      if col ≤ xc ∧ xc < col + 64 * ws.size then
        (ws.getD ((xc - col) / 64) 0 >>> ((xc - col) % 64).toUInt64) &&& 1 == 1
      else w.getD xc false := by
  have := foldl_range_spec (col := col) (cond := fun _ => true)
    (val := fun t => (ws.getD (t / 64) 0 >>> (t % 64).toUInt64) &&& 1 == 1)
    (f := fun acc t => acc.set! (col + t) ((ws.getD (t / 64) 0 >>> (t % 64).toUInt64) &&& 1 == 1))
    (fun acc t => ⟨size_set!, fun xc hX => by rw [getD_set! hX]; simp⟩) (64 * ws.size) w
  exact ⟨this.1, fun xc hX => by unfold putWords; rw [this.2 xc hX]; simp⟩

theorem getD_u64 (a : Array UInt64) (i : ℕ) : a.getD i 0 = a[i]! := Array.getElem!_eq_getD.symm

/-- **The slot's inputs**: `cv` the chain's last value, `pad` the message's last block. -/
theorem slot_in (hck : _root_.Flock.HmNets.check c = .ok ()) {q n : ℕ} (hq : q < c.ports.size)
    (hC8 : ((chain c pin q n).2.getD (c.ports.getD q default).chunks #[]).size = 8) :
    (∀ i < 512, (hmSlot c pin ((Array.range c.ports.size).map fun p => chain c pin p n) n q).getD i false =
      ((((chain c pin q n).2.getD (c.ports.getD q default).chunks #[]).getD (i / 64) 0 >>> (i % 64).toUInt64)
        &&& 1 == 1)) ∧
    (∀ i < 1024, (hmSlot c pin ((Array.range c.ports.size).map fun p => chain c pin p n) n q).getD (512 + i) false =
      (((blockWords (chain c pin q n).1 (c.ports.getD q default).chunks).getD (i / 64) 0 >>> (i % 64).toUInt64)
        &&& 1 == 1)) := by
  have hW : (c.nets.getD c.hmNet default).2.inWords * WORD = 4096 := Hm.hmNet_inRows hck
  have hi0 : ((portCols (c.nets.getD c.hmNet default).2.inGroups).getD 0 default).1 = 0 := (Hm.hmIn_eq hck).1
  have hi1 : ((portCols (c.nets.getD c.hmNet default).2.inGroups).getD 1 default).1 = 512 := (Hm.hmIn_eq hck).2.1
  have hi2 : ((portCols (c.nets.getD c.hmNet default).2.inGroups).getD 2 default).1 = 1536 := (Hm.hmIn_eq hck).2.2
  have hB : (blockWords (chain c pin q n).1 (c.ports.getD q default).chunks).size = 16 := by
    simp [blockWords, _root_.Flock.HmRow.wordsBE]
  unfold hmSlot
  rw [getD_map_range hq, ite_t hq]
  generalize chain c pin q n = ch at hC8 hB ⊢
  obtain ⟨msg, cvs⟩ := ch
  dsimp only at hC8 hB ⊢
  rw [hi0, hi1, hi2]
  generalize hw0 : Array.replicate ((c.nets.getD c.hmNet default).2.inWords * WORD) false = w0
  generalize hCV : cvs.getD (c.ports.getD q default).chunks #[] = CV at hC8 ⊢
  generalize hBW : blockWords msg (c.ports.getD q default).chunks = BW at hB ⊢
  generalize hS : (portIn c pin q n).2 = S
  have hs0 : w0.size = 4096 := by rw [← hw0, Array.size_replicate, hW]
  have hs1 : (putWords w0 0 CV).size = 4096 := by rw [putWords_size, hs0]
  have hs2 : (putWords (putWords w0 0 CV) 512 BW).size = 4096 := by rw [putWords_size, hs1]
  refine ⟨fun i hi => ?_, fun i hi => ?_⟩
  · rw [(putBytes_spec (putWords (putWords w0 0 CV) 512 BW) 1536 S).2 i (by omega), ite_f (by omega),
      (putWords_spec (putWords w0 0 CV) 512 BW).2 i (by omega), ite_f (by omega),
      (putWords_spec w0 0 CV).2 i (by omega), ite_t (by omega), Nat.sub_zero]
  · rw [(putBytes_spec (putWords (putWords w0 0 CV) 512 BW) 1536 S).2 (512 + i) (by omega), ite_f (by omega),
      (putWords_spec (putWords w0 0 CV) 512 BW).2 (512 + i) (by omega), ite_t (by omega), Nat.add_sub_cancel_left]

/-! ## The row's message and chain -/

theorem wordBits_bytes (ws : Array ℕ) :
    Claims.Recursion.VStar.bytesOf (_root_.VStar.In.wordBits ws) = (wordBytes ws ws.size).data.toList := by
  have hl : (wordBytes ws ws.size).data.toList.length = 2 * ws.size := by simp [wordBytes]
  conv_rhs => rw [← Claims.Recursion.VStar.bytesOf_bitL (wordBytes ws ws.size).data.toList]
  congr 1
  apply List.ext_getElem?
  intro i
  rw [List.getElem?_map, hl]
  by_cases hi : i < 16 * ws.size
  · rw [wordBits_get hi, List.getElem?_range (by omega), Option.map_some]
    congr 1
    unfold Hm.bitL
    rw [← Hm.get!_eq_getD, wordBytes_get (by omega), show i / 8 / 2 = i / 16 by omega]
    rcases Nat.mod_two_eq_zero_or_one (i / 8) with h2 | h2
    · rw [h2, Nat.pow_zero, Nat.div_one, show (256 : ℕ) = 2 ^ 8 from rfl, Nat.testBit_mod_two_pow,
        decide_eq_true (Nat.mod_lt _ (by decide)), Bool.true_and]
      congr 1
      omega
    · rw [h2, Nat.pow_one, show (256 : ℕ) = 2 ^ 8 from rfl, Nat.testBit_mod_two_pow,
        decide_eq_true (Nat.mod_lt _ (by decide)), Bool.true_and, Nat.testBit_div_two_pow]
      congr 1
      omega
  · rw [List.getElem?_eq_none (by rw [wordBits_length]; omega), List.getElem?_eq_none (by simp; omega),
      Option.map_none]

theorem getD_extract (b : ByteArray) {s e k : ℕ} (hk : k < e - s) :
    (b.extract s e).data.toList.getD k 0 = b.data.toList.getD (s + k) 0 := by
  rw [ByteArray.data_extract, Array.toList_extract, List.extract_eq_take_drop]
  simp only [List.getD_eq_getElem?_getD, List.getElem?_take, List.getElem?_drop, hk, if_true]

theorem fold_push (g : Array UInt64 → ℕ → Array UInt64) (a : ℕ → Array UInt64) (ha : ∀ k, a (k + 1) = g (a k) k) :
    ∀ m, ((List.range m).foldl (fun cvs k => cvs.push (g cvs.back! k)) #[a 0]).size = m + 1 ∧
      ((List.range m).foldl (fun cvs k => cvs.push (g cvs.back! k)) #[a 0]).back! = a m ∧
      ∀ k ≤ m, ((List.range m).foldl (fun cvs k => cvs.push (g cvs.back! k)) #[a 0]).getD k #[] = a k
  | 0 => ⟨rfl, by simp [Array.back!], fun k hk => by rw [Nat.le_zero.1 hk]; simp⟩
  | m + 1 => by
    obtain ⟨hs, hb, hg⟩ := fold_push g a ha m
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    generalize (List.range m).foldl (fun cvs k => cvs.push (g cvs.back! k)) #[a 0] = F at hs hb hg
    rw [hb, ← ha]
    refine ⟨by rw [Array.size_push, hs], Array.back!_push, fun k hk => ?_⟩
    rcases Nat.lt_or_ge k (m + 1) with h | h
    · have e := hg k (by omega)
      rw [Array.getD_eq_getD_getElem?] at e ⊢
      rw [Array.getElem?_push, if_neg (by omega), e]
    · obtain rfl : k = m + 1 := by omega
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_push, if_pos hs.symm]
      rfl

theorem hmOutCol_eq (c : Circuit) : Hm.hmOut c = hmOutCol c := by
  unfold Hm.hmOut hmOutCol
  rw [Array.getD_eq_getD_getElem?]
  cases (portCols (c.nets.getD c.hmNet default).2.outGroups)[0]? <;> rfl

section Row

variable {q n : ℕ} {rw : _root_.VStar.In.Row} (hn : n < _root_.VStar.Trace.count pin)
  (hr : rowAt pin (c.ports.getD q default).name n = some rw) (hv2 : (c.ports.getD q default).v2 = false)
  (hsg : (c.ports.getD q default).segs = none) (hwN : 2 * 𝑾 = 128 * 𝑵)
include hn hr hv2 hsg hwN

theorem data_len : 𝑫.length = 128 * 𝑵 := by
  rw [← hwN]; simp [wordBytes]

/-- **The row's message after its prefix**: its bytes, then the padding of `128 (chunks + 1)` bytes. -/
theorem msg_data : (chain c pin q n).1.data.toList = 𝑫 ++ Hm.rowPadT 𝑵 := by
  have hsz : (wordBytes rw.words 𝑾).size = 2 * 𝑾 := by simp [wordBytes, ByteArray.size]
  have hby : (c.ports.getD q default).bytes = 2 * 𝑾 := by
    unfold Port.bytes; simp only [hv2, Bool.false_eq_true, ↓reduceIte]
  have he : (wordBytes rw.words 𝑾).extract 0 (2 * 𝑾) = wordBytes rw.words 𝑾 := by
    have := @ByteArray.extract_zero_size (wordBytes rw.words 𝑾)
    rwa [hsz] at this
  have ht : (_root_.Flock.HmRow.tailOf (c.ports.getD q default)).data.toList = Hm.rowPadT 𝑵 := by
    simp only [_root_.Flock.HmRow.tailOf, hsg]
    rw [hby, Hidden.padTail_data, Hm.rowPadT, hwN,
      show Hm.zeroPad (128 + 128 * 𝑵) = 111 by unfold Hm.zeroPad; omega]
  show ((portIn c pin q n).1.extract 0 (c.ports.getD q default).bytes ++
    zeros ((c.ports.getD q default).bytes - (portIn c pin q n).1.size) ++
      _root_.Flock.HmRow.tailOf (c.ports.getD q default)).data.toList = _
  rw [portIn_some hn hr]
  simp only []
  have hz : (zeros 0).data.toList = [] := rfl
  rw [hby, he, hsz, Nat.sub_self, ByteArray.toList_data_append, ByteArray.toList_data_append, ht, hz, List.append_nil]

/-- The padded `sha512/row/v1` message past its prefix block is the row's message. -/
theorem pad_msg (y : ℕ) : 𝑷.get! (128 + y) = (chain c pin q n).1.get! y := by
  rw [Hm.get!_eq_getD, Hm.get!_eq_getD, Hm.rowMsg_pad (data_len hn hr hv2 hsg hwN), List.append_assoc,
    ← msg_data hn hr hv2 hsg hwN, List.getD_append_right _ _ _ _ (by rw [Hm.length_rowPrefix]; omega),
    Hm.length_rowPrefix, Nat.add_sub_cancel_left]

/-- **The chain's last value** is the chaining value of the padded message before its last block. -/
theorem chain_cvs : (chain c pin q n).2.getD 𝑵 #[] = Hm.chain 𝑷 (𝑵 + 1) := by
  have hd := data_len hn hr hv2 hsg hwN
  have hpre : _root_.Flock.HmRow.prefixOf (c.ports.getD q default) = _root_.Flock.HmRow.rowPrefix 𝑾 := by
    unfold _root_.Flock.HmRow.prefixOf
    simp only [hsg, hv2, Option.isSome_none, Bool.false_eq_true, ↓reduceIte]
  have h1 : _root_.Flock.Sha512.compress _root_.Flock.Sha512.iv 𝑷 0 =
      _root_.Flock.Sha512.compress _root_.Flock.Sha512.iv (_root_.Flock.HmRow.rowPrefix 𝑾) 0 := by
    refine Hm.compress_congr _ fun k hk => ?_
    rw [Hm.get!_eq_getD, Hm.get!_eq_getD, Hm.rowMsg_pad hd, Nat.zero_add, Hm.getD_rowPad_pre hd hk]
  have h0 : _root_.Flock.HmRow.midstateOf (c.ports.getD q default) = Hm.chain 𝑷 1 := by
    rw [Hm.chain_succ, show Hm.chain 𝑷 0 = _root_.Flock.Sha512.iv from rfl, Nat.mul_zero, h1,
      _root_.Flock.HmRow.midstateOf, hpre]
  have ha : ∀ k, Hm.chain 𝑷 (k + 1 + 1) =
      _root_.Flock.Sha512.compress (Hm.chain 𝑷 (k + 1)) (chain c pin q n).1 (128 * k) := by
    intro k
    rw [Hm.chain_succ 𝑷 (k + 1)]
    refine Hm.compress_congr _ fun j _ => ?_
    rw [show 128 * (k + 1) + j = 128 + (128 * k + j) by omega, pad_msg hn hr hv2 hsg hwN]
  have hcv : (chain c pin q n).2 = (List.range 𝑵).foldl
      (fun cvs k => cvs.push (_root_.Flock.Sha512.compress cvs.back! (chain c pin q n).1 (128 * k)))
      #[_root_.Flock.HmRow.midstateOf (c.ports.getD q default)] := rfl
  rw [hcv, h0]
  exact (fold_push (fun h k => _root_.Flock.Sha512.compress h (chain c pin q n).1 (128 * k))
    (fun k => Hm.chain 𝑷 (k + 1)) ha _).2.2 _ le_rfl

/-- **The message's last block** is the padded message's last block. -/
theorem block_eq {j : ℕ} (hj : j < 16) :
    (blockWords (chain c pin q n).1 𝑵)[j]! = Hm.wordBE 𝑷 (128 * (𝑵 + 1)) j := by
  unfold blockWords
  rw [Hm.wordsBE_getElem! hj, Hm.wordBE_eq_wordL]
  refine Hm.wordL_congr fun k hk => ?_
  rw [getD_extract _ (by omega), ← Hm.get!_eq_getD, ← Hm.get!_eq_getD,
    show 128 * (𝑵 + 1) + 8 * j + k = 128 + (128 * 𝑵 + (0 + 8 * j + k)) by omega, pad_msg hn hr hv2 hsg hwN]

end Row

/-! ## The slots commit -/

/-- **The trace assigner's hm96 slots commit** (`HpEarly.SlotsOk`): where `TraceOk` holds, the circuit's row nets are
the verifier's (`HmNets.check`), its blocks' columns fit 32 bits and every port is a `sha512/row/v1` row of whole
chunks, each port's hm96 slot at an instance with a placed row outputs the commit string of its row and salt. -/
theorem _root_.RecDeploy.HpEarly.slotsOk_assign (h : TraceOk c pin) (hck : _root_.Flock.HmNets.check c = .ok ())
    (hk : c.kLog ≤ 32)
    (hv1 : ∀ q (hq : q < c.ports.size), c.ports[q].v2 = false ∧ c.ports[q].segs = none ∧
      2 * c.ports[q].words = 128 * c.ports[q].chunks)
    (mask : ℕ → ℕ → Bool) : HpEarly.SlotsOk mask c pin := by
  intro nm q n rw ps hq hrw hps
  obtain ⟨hqs, hname, hget⟩ := name_of_findIdx hq
  have hn := count_of_portRow h hrw
  have hr' : rowAt pin (c.ports.getD q default).name n = some rw := by rw [hget, hname]; exact hrw
  obtain ⟨hws, -⟩ := h.rows q hqs n rw (hname ▸ hrw)
  obtain ⟨hv2, hsg, hwN⟩ := hv1 q hqs
  rw [← hget] at hws hv2 hsg hwN
  -- the hm96 net's rows
  obtain ⟨-, ha, hb, huse, hcp, -, -, -, -, -, hordc, -, -, -, -⟩ := Net.ofRows_ok (Hm.hmNet_ofRows hck)
  have ha' : _root_.Flock.Sparse.ofRows (_root_.Flock.HmNets.hmLaid ()).ra (2 ^ (hmN c).unitLog) = .ok (hmN c).a := ha
  have hb' : _root_.Flock.Sparse.ofRows (_root_.Flock.HmNets.hmLaid ()).rb (2 ^ (hmN c).unitLog) = .ok (hmN c).b := hb
  have hcp' : (hmN c).constPos = (_root_.Flock.HmNets.hmLaid ()).constPos := hcp
  have hW : (hmN c).inWords * WORD = 4096 := Hm.hmNet_inRows hck
  have hw32 : 2 ^ (hmN c).unitLog ≤ 2 ^ 32 := by
    have hc1 : 0 < (c.rangeOf c.hmNet).count := by
      have := h.hm_count
      have := h.ports_hm
      have : 0 < c.g * c.perVu.getD c.hmNet 0 := Nat.mul_pos h.g_pos (by omega)
      omega
    have e := Nat.le_mul_of_pos_left (2 ^ (c.rangeOf c.hmNet).slotLog) hc1
    have hb := h.hm_block
    rw [← h.hm_slot]
    calc 2 ^ (c.rangeOf c.hmNet).slotLog ≤ 2 ^ c.kLog := by omega
      _ ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hk
  have hra : ∀ i, (hmN c).a.row i = (_root_.Flock.HmNets.hmLaid ()).ra.getD i #[] :=
    FlockSoundness.SparseRows.ofRows_row ha' hw32
  have hrb : ∀ i, (hmN c).b.row i = (_root_.Flock.HmNets.hmLaid ()).rb.getD i #[] :=
    FlockSoundness.SparseRows.ofRows_row hb' hw32
  have hord : FlockSoundness.RowOrder ((hmN c).inWords * WORD) (hmN c).constPos (_root_.Flock.HmNets.hmLaid ()).ra
      (_root_.Flock.HmNets.hmLaid ()).rb := by
    rw [hcp']; exact FlockSoundness.checkOrder_ok hordc
  have hlu : ∀ st (i o : Array ℕ) outs, (_root_.Flock.HmNets.layout st i o outs).useful =
      (_root_.Flock.HmNets.layout st i o outs).constPos + 1 := fun _ _ _ _ => rfl
  have huse' : (hmN c).useful = (hmN c).constPos + 1 := by
    have e1 : (hmN c).useful = (_root_.Flock.HmNets.hmLaid ()).useful := huse
    rw [e1, hcp', _root_.Flock.HmNets.hmLaid_eq]
    exact hlu _ _ _ _
  have evR := evalNet_rows (ins := hmSlot c pin ((Array.range c.ports.size).map fun p => chain c pin p n) n q)
    hra hrb hord huse' h.hm_const h.hm_log
  -- the slot holds them
  have hv : ∀ t < 2 ^ (hmN c).unitLog,
      slotW mask c pin n (c.slotCol c.hmNet (n % c.g) q + t) = (slotE c pin n q).getD t false :=
    fun t ht => hm_bit h mask hqs hn ht
  have hhm : Hm.LaidRows (_root_.Flock.HmNets.hmLaid ()) 4096
      fun j => slotW mask c pin n (c.slotCol c.hmNet (n % c.g) q + j) := by
    refine ⟨?_, fun i hi1 hi2 => ?_⟩
    · dsimp only
      rw [← hcp', hv _ h.hm_log]
      exact evR.1
    · dsimp only
      have hi2' : i < (hmN c).constPos := by rw [hcp']; exact hi2
      have hiu : i < 2 ^ (hmN c).unitLog := lt_trans hi2' h.hm_log
      rw [xorRow_congr (v' := fun j => (slotE c pin n q).getD j false)
          (cs := (_root_.Flock.HmNets.hmLaid ()).ra.getD i #[]) fun j hj => hv j (Hm.ofRows_cols_lt ha' i j hj),
        xorRow_congr (v' := fun j => (slotE c pin n q).getD j false)
          (cs := (_root_.Flock.HmNets.hmLaid ()).rb.getD i #[]) fun j hj => hv j (Hm.ofRows_cols_lt hb' i j hj),
        hv i hiu]
      exact evR.2 i (by rw [hW]; exact hi1) hi2'
  -- the slot's inputs
  have hzin := Hidden.hm_zh_in c (slotW mask c pin n) (n % c.g) q
  have hzE : ∀ v < 3072, slotW mask c pin n (c.slotCol c.hmNet (n % c.g) q + v) =
      (hmSlot c pin ((Array.range c.ports.size).map fun p => chain c pin p n) n q).getD v false := by
    intro v hv'
    have hc := h.hm_const
    have hl := h.hm_log
    rw [hm_bit h mask hqs hn (by omega)]
    exact evalNet_input (by omega) hc hl (by rw [hra, (Hm.hmLaid_self hv').1]; rfl)
  have hC8 : ((chain c pin q n).2.getD (c.ports.getD q default).chunks #[]).size = 8 := by
    rw [chain_cvs hn hr' hv2 hsg hwN, Hm.chain_succ, Hm.size_compress]
  have hin := slot_in hck hqs hC8
  -- the row's message
  have hP := placesR_assign h mask
  have hr : readForms (A mask c pin) (ps.map _root_.Flock.HmNets.Lin.var) =
      Claims.Recursion.VStar.bytesOf (_root_.VStar.In.wordBits rw.words) := by
    rw [Claims.Recursion.VStarB.readForms_pos]
    unfold Claims.Recursion.VStar.readAt
    rw [hP.places.row nm n rw ps hrw hps, hP.take_full hrw hps]
  have hpfx : rowPfx c q = (_root_.Flock.HmRow.rowPrefix (c.ports.getD q default).words).data.toList := by
    rw [hget]; simp [rowPfx, Array.getElem?_eq_getElem hqs]
  have hd := data_len hn hr' hv2 hsg hwN
  have hmsgE : (⟨(rowPfx c q ++ readForms (A mask c pin)
      (((rowHidden c nm n).getD []).map _root_.Flock.HmNets.Lin.var)).toArray⟩ : ByteArray) =
      Hm.rowMsg (c.ports.getD q default).words (wordBytes rw.words (c.ports.getD q default).words).data.toList := by
    rw [hps, Option.getD_some, hr, hpfx, wordBits_bytes, hws]
    rfl
  have hlast : (Hm.padData (Hm.rowMsg (c.ports.getD q default).words
      (wordBytes rw.words (c.ports.getD q default).words).data.toList)).size / 128 - 1 =
      (c.ports.getD q default).chunks + 1 := by
    rw [Hm.size_padData_rowMsg hd, Nat.mul_div_cancel_left _ (by norm_num)]
    rfl
  have hin2 : hmIn c 2 = 1536 := by rw [VStarTrace.hmIn_eq]; exact (Hm.hmIn_eq hck).2.2
  refine slotCommits_of_hm96 _ (Hidden.hm_zh c (slotW mask c pin n) (n % c.g) q hhm) ?_ ?_ ?_ ?_
  · intro i hi
    have hi' : @LT.lt ℕ _ i 512 := hi
    rw [hmsgE, Hm.lastCv_eq, hlast]
    refine (hzin i (by omega)).trans ((hzE i (by omega)).trans ?_)
    rw [hin.1 i hi, getD_u64, chain_cvs hn hr' hv2 hsg hwN]
    exact Hm.wordsBit_eq _ (n := 8) (by omega)
  · intro i hi
    have hi' : @LT.lt ℕ _ i 1024 := hi
    rw [hmsgE, Hm.lastBlock_eq, hlast]
    refine (hzin (512 + i) (by omega)).trans ((hzE (512 + i) (by omega)).trans ?_)
    rw [hin.2 i hi, getD_u64]
    refine (Hm.wordsBit_eq (blockWords (chain c pin q n).1 (c.ports.getD q default).chunks) (n := 16)
      (by omega)).trans ?_
    congr 1
    unfold Hm.blockV
    refine List.map_congr_left fun j hj => ?_
    rw [block_eq hn hr' hv2 hsg hwN (List.mem_range.1 hj)]
  · intro j hj
    have hj' : @LT.lt ℕ _ j 1536 := hj
    refine (hzin (1536 + j) (by omega)).trans ?_
    rw [bitL_readForms (fs := hmSalt c q n) (by simp [hmSalt]; omega)]
    simp only [hmSalt, List.getElem_map, List.getElem_range]
    rw [formBit_var, hin2]
    rfl
  · intro t ht
    have e := Hidden.hm_bc c (slotW mask c pin n) (n % c.g) q hck hhm t ht
    rw [hmOutCol_eq, Nat.add_assoc] at e
    exact e

end RecDeploy.SlotsOkProof
