import Proofs.Flock.Soundness.Discharge.Inst.Keys
import Proofs.Flock.Soundness.Discharge.Layout.OutPin
import Proofs.Flock.Soundness.Discharge.Layout.RowEnc
import Proofs.Flock.Soundness.Discharge.Composed.RegWords

/-!
# The returned outputs at the output net's slots, a tail's included

`Layout/Out*` pin the returned outputs where the unit returns them (`c.outNet = c.unitNet`). A tail returns them from its
last stage (GLM's tailed RMSNorm): `HmRow.regions` then ends with a region "out" over the out range's slots, one per VU,
each the VU's instance's `nOut` output words, two bytes each, at the last stage's only output group (`tailRegion`,
`regions_outT`). Here both are read at the output net's group `(netOf c c.outNet).outGroups` (`outGrp`) and at its
slots' columns `slotCol c.outNet g i + outColT c j t`:

* `hm_regions_wholeT`, `setupH_wholeT`: the regions are whole-word, a tail's region "out" when it has a whole word per
  slot (`7 ≤ tz (16 · nOut)`, `seven_le_tz`);
* `check_outT`, `OutShapeT`, `outShape_of_parseT`: the returned outputs' shape, from the parse (`check_tail` at a tail);
* `tail_pinned`, `out_pinnedT`, `out_drawnT`, `out_siteT`: a satisfying witness holds the population's output words at
  the output net's slots, under `PinShape` (`pinShape_of`, for a circuit whose tail has one slot per VU, `TailOne`);
* `OutsOutT`, `instOutBitOf`, `instProg_hout`: at `instProg`'s places, a unit's committed gates on the returned outputs'
  columns (`OutColAtT`: `(order c).getD x 0 = code c.outNet i (outColT c j t)`) hold the registration's bit, which is
  `instProg_valPinned`'s `hout`; `instRowsEnc` is `instHmRows` with `enc := rowEnc`, `encO := bitCode` and that bit.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Inst

open Flock FlockLevel3

abbrev outGrp (c : Circuit) : Group := (netOf c c.outNet).outGroups.getD 0 default

def tz (x : ℕ) : ℕ := ((List.range 64).find? fun i => x % 2 ^ (i + 1) != 0).getD 0

theorem tz_two_pow {a : ℕ} (ha : a < 64) : tz (2 ^ a) = a := by
  have h : (List.range 64).find? (fun i => 2 ^ a % 2 ^ (i + 1) != 0) = some a := by
    rw [List.find?_range_eq_some]
    refine ⟨?_, List.mem_range.2 ha, fun j hj => ?_⟩
    · rw [Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (Nat.lt_succ_self a))]
      simp
    · rw [Nat.mod_eq_zero_of_dvd (Nat.pow_dvd_pow 2 (by omega))]
      rfl
  unfold tz
  rw [h]
  rfl

def tailVal (c : Circuit) (pub : Public) (b g o : ℕ) : ℕ :=
  match c.instOf pub.n b g with
  | some i => pub.outputs.getD (i * c.outWords + o) 0
  | none => c.dummyOutputs.getD o 0

def tailBytes (c : Circuit) (pub : Public) (b : ℕ) : ByteArray :=
  (Array.range c.g).foldl (fun acc g => (Array.range c.outWords).foldl (fun acc o =>
    (acc.push (tailVal c pub b g o % 256).toUInt8).push (tailVal c pub b g o / 256).toUInt8) acc) ByteArray.empty

def tailRegion (c : Circuit) (pub : Public) : V Region :=
  mkRegion c.kLog "out" ((Array.range (tz (16 * c.outWords))).map (0 + ·) ++
      (Array.range (tz (c.rangeOf c.outNet).count)).map ((c.rangeOf c.outNet).slotLog + ·))
    ((c.rangeOf c.outNet).start + (outGrp c).col * WORD) (tailBytes c pub)

def outRegionT (c : Circuit) (pub : Public) : V Region :=
  if c.outNet = c.unitNet then c.outRegion pub else tailRegion c pub

set_option maxHeartbeats 2000000 in
theorem regions_outT {c : Circuit} {pub : Public} {regs : Array Region} (h : HmRow.regions c pub = .ok regs)
    (ht : c.typed = none) : ∃ r ∈ regs.toList, outRegionT c pub = .ok r := by
  revert regs
  show Refine.Ret (HmRow.regions c pub) fun regs => ∃ r ∈ regs.toList, outRegionT c pub = .ok r
  unfold HmRow.regions
  dsimp only
  refine Refine.ret_bind' _ _ _ fun out => ?_
  split
  · rename_i hc
    have hne : c.outNet ≠ c.unitNet := by simpa [ht] using hc
    refine Refine.ret_bind _ _ _ fun r hr => Refine.ret_pure _ _ ⟨r, by simp, ?_⟩
    rw [outRegionT, ite_eq_right hne]
    exact hr
  · rename_i hc
    have heq : c.outNet = c.unitNet := by simpa [ht] using hc
    refine Refine.ret_bind _ _ _ fun r hr => Refine.ret_pure _ _ ⟨r, by simp, ?_⟩
    rw [outRegionT, ite_eq_left heq]
    exact hr

theorem foldl_push2 (x y : ℕ → UInt8) : ∀ (L : List ℕ) (A : ByteArray),
    (L.foldl (fun acc o => (acc.push (x o)).push (y o)) A).data.toList =
      A.data.toList ++ (L.map fun o => [x o, y o]).flatten
  | [], A => by simp
  | o :: L, A => by
      rw [List.foldl_cons, foldl_push2 x y L]
      simp

theorem foldl_push2_outer (N : ℕ) (x y : ℕ → ℕ → UInt8) : ∀ (L : List ℕ) (A : ByteArray),
    (L.foldl (fun acc g => (List.range N).foldl (fun acc o => (acc.push (x g o)).push (y g o)) acc) A).data.toList =
      A.data.toList ++ (L.map fun g => ((List.range N).map fun o => [x g o, y g o]).flatten).flatten
  | [], A => by simp
  | g :: L, A => by
      rw [List.foldl_cons, foldl_push2_outer N x y L, foldl_push2]
      simp

theorem tailBytes_data (c : Circuit) (pub : Public) (b : ℕ) :
    (tailBytes c pub b).data.toList = ((List.range c.g).map fun g => ((List.range c.outWords).map fun o =>
      [(tailVal c pub b g o % 256).toUInt8, (tailVal c pub b g o / 256).toUInt8]).flatten).flatten := by
  unfold tailBytes
  simp only [← Array.foldl_toList, Array.toList_range]
  rw [foldl_push2_outer]
  rfl

theorem toUInt8_toNat (n : ℕ) : n.toUInt8.toNat = n % 256 := by
  rw [Nat.toUInt8, UInt8.toNat_ofNat']

theorem tailBytes_bit (c : Circuit) (pub : Public) (b : ℕ) {g j t : ℕ} (hg : g < c.g) (hj : j < c.outWords)
    (ht : t < 16) :
    ((tailBytes c pub b).data.toList.getD (g * (2 * c.outWords) + (2 * j + t / 8)) default).toNat.testBit (t % 8) =
      (tailVal c pub b g j).testBit t := by
  have hlen : ∀ (f : ℕ → List UInt8), (∀ o, (f o).length = 2) → ∀ N, ((List.range N).map f).flatten.length = 2 * N := by
    intro f hf N
    induction N with
    | zero => simp
    | succ N ih => rw [List.range_succ, List.map_append, List.flatten_append, List.length_append, ih]; simp [hf]; ring
  rw [tailBytes_data, Layout.getD_flatten_const default g _ (fun l hl => by
      obtain ⟨g', -, rfl⟩ := List.mem_map.1 hl
      exact hlen _ (fun _ => rfl) _) (by simpa using hg) (by omega)]
  have hmap : ∀ {α : Type} (F : ℕ → List α) {N k : ℕ}, k < N → ((List.range N).map F).getD k [] = F k := by
    intro α F N k hk
    rw [List.getD_eq_getElem _ _ (by simpa using hk)]
    simp
  rw [hmap _ hg, show 2 * j + t / 8 = j * 2 + t / 8 by ring, Layout.getD_flatten_const default j _ (fun l hl => by
      obtain ⟨o, -, rfl⟩ := List.mem_map.1 hl
      rfl) (by simpa using hj) (by omega), hmap _ hj]
  generalize tailVal c pub b g j = v
  have h256 : (256 : ℕ) = 2 ^ 8 := rfl
  rcases Nat.lt_or_ge t 8 with h8 | h8
  · rw [show t / 8 = 0 by omega, show t % 8 = t by omega, List.getD_cons_zero, toUInt8_toNat, h256,
      Nat.testBit_mod_two_pow, Nat.testBit_mod_two_pow]
    simp [h8]
  · rw [show t / 8 = 0 + 1 by omega, List.getD_cons_succ, List.getD_cons_zero, toUInt8_toNat, h256,
      Nat.testBit_mod_two_pow, Nat.testBit_div_two_pow, show t % 8 + 8 = t by omega]
    simp only [Nat.mod_lt _ (by norm_num : 0 < 8), decide_true, Bool.true_and]


theorem tz_of_pow2 {x : ℕ} (h : isPow2 x = true) (hx : x < 2 ^ 64) : 2 ^ tz x = x := by
  simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  have hl : x.log2 < 64 := (Nat.log2_lt (by omega)).2 hx
  conv_lhs => rw [← h.2, tz_two_pow hl]
  exact h.2

/-- **What `HmRow.check` checks of a tail that returns the outputs**, in an untyped circuit: the last stage's output
ports are one per output word, each 16 bits, in its only output group; `16 · nOut` and `vus_per_block` are powers of
two. -/
theorem check_tail {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none) (ho : c.outNet ≠ c.unitNet)
    (h : HmRow.check c tags ls = .ok ()) :
    (portCols (netOf c c.outNet).outGroups).size = c.outWords ∧
      (∀ x ∈ (portCols (netOf c c.outNet).outGroups).toList, x.2 = 16) ∧ (netOf c c.outNet).outGroups.size = 1 ∧
      isPow2 (16 * c.outWords) = true ∧ isPow2 c.g = true := by
  unfold HmRow.check at h
  obtain ⟨spans, -, h⟩ := except_bind_ok h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  simp only [ht] at h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  have hc : (c.outNet != c.unitNet) = true := by simpa using ho
  rw [ite_eq_left hc] at h
  obtain ⟨hbad, h⟩ := Layout.ite_throw_cond h
  obtain ⟨u1, -, h⟩ := except_bind_ok h
  obtain ⟨u2, -, h⟩ := except_bind_ok h
  obtain ⟨wired, -, h⟩ := except_bind_ok h
  obtain ⟨wired2, -, h⟩ := except_bind_ok h
  obtain ⟨u3, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨seen, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨hg, -⟩ := Layout.ite_throw_cond h
  simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, not_not, Array.any_eq_true', not_exists, not_and,
    Bool.not_eq_true', Bool.not_eq_false] at hbad
  simp only [Bool.not_eq_true', Bool.not_eq_false] at hg
  obtain ⟨⟨⟨⟨⟨⟨-, -⟩, hsz⟩, h16⟩, h1⟩, hp⟩, -⟩ := hbad
  exact ⟨hsz, fun x hx => h16 x (Array.mem_toList_iff.1 hx), h1, hp, hg⟩

/-- **A lookup slot's output group**: one word of its output bits. -/
theorem lookup_outGroups {name : String} {table : ByteArray} {n lo bits : ℕ} {net : Net}
    (h : Lookup.build name table n lo bits = .ok net) : ∃ col, net.outGroups = #[⟨col, 1, #[bits]⟩] := by
  unfold Lookup.build at h
  obtain ⟨-, h⟩ := ite_throw_ok_not h
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
  obtain ⟨a, -, h⟩ := except_bind_ok h
  obtain ⟨b, -, h⟩ := except_bind_ok h
  cases h
  exact ⟨_, rfl⟩

theorem eq_singleton_getD {α : Type} [Inhabited α] {a : Array α} (h : a.size = 1) : a = #[a.getD 0 default] := by
  obtain ⟨l⟩ := a
  match l, h with
  | [x], _ => rfl

/-- Slot `i`'s returned output words: its unit copy's leaf where the unit returns the outputs, else every output word
(the tail's one slot per VU). -/
def outsOf (c : Circuit) (i : ℕ) : Array ℕ :=
  if c.outNet = c.unitNet then c.leavesOut.getD i #[] else Array.range c.outWords

theorem outGrp_unit {c : Circuit} (ho : c.outNet = c.unitNet) : outGrp c = c.unit.outGroups.getD 0 default := by
  unfold outGrp netOf Circuit.unit
  rw [ho]

/-- **What `HmRow.check` checks of the returned outputs, at the output net's group**, in an untyped circuit: where the
unit returns them, `Layout.check_out` (16-bit ports; one leaf map per unit, with one output per port, each below
`nOut`); at a tail, `check_tail`. -/
theorem check_outT {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none)
    (h : HmRow.check c tags ls = .ok ()) :
    (c.outNet = c.unitNet → (∀ b ∈ (outGrp c).bits.toList, b = 16) ∧ c.leavesOut.size = c.upv ∧
      ∀ u < c.upv, (outsOf c u).size = (outGrp c).bits.size ∧ ∀ o ∈ (outsOf c u).toList, o < c.outWords) ∧
    (c.outNet ≠ c.unitNet → (portCols (netOf c c.outNet).outGroups).size = c.outWords ∧
      (∀ x ∈ (portCols (netOf c c.outNet).outGroups).toList, x.2 = 16) ∧ (netOf c c.outNet).outGroups.size = 1 ∧
      isPow2 (16 * c.outWords) = true ∧ isPow2 c.g = true) := by
  refine ⟨fun ho => ?_, fun ho => check_tail ht ho h⟩
  obtain ⟨h16, hsz, hlo⟩ := Layout.check_out ht ho h
  rw [outGrp_unit ho]
  refine ⟨h16, hsz, fun u hu => ?_⟩
  have e : outsOf c u = c.leavesOut[u]! := by
    rw [outsOf, ite_eq_left ho, Array.getD_eq_getD_getElem?, getElem!_pos c.leavesOut u (by omega),
      Array.getElem?_eq_getElem (by omega), Option.getD_some]
  rw [e]
  exact hlo u hu

/-- **The returned outputs' shape, at the output net's group** (`Layout.OutShape` where the unit returns them): where
the unit returns them, the group's words are a power of two, and each copy's leaf's outputs are 16-bit ports, output
words below `nOut`, that fit in the group's words; at a tail, the last stage's only output group is one 16-bit port
per output word, `16 · nOut` (a power of two) fits in its slot, and `vus_per_block` is a power of two. -/
def OutShapeT (c : Circuit) : Prop :=
  (c.outNet = c.unitNet → (outGrp c).words = 2 ^ (outGrp c).words.log2 ∧
    ∀ u' < c.upv, (∀ j < (outsOf c u').size, (outGrp c).bits.getD j 0 = 16 ∧ (outsOf c u')[j]! < c.outWords) ∧
      16 * (outsOf c u').size ≤ 128 * (outGrp c).words) ∧
  (c.outNet ≠ c.unitNet → (netOf c c.outNet).outGroups.size = 1 ∧ (outGrp c).bits.size = c.outWords ∧
    (∀ j < c.outWords, (outGrp c).bits.getD j 0 = 16) ∧ isPow2 (16 * c.outWords) = true ∧
    16 * c.outWords ≤ 2 ^ (netOf c c.outNet).unitLog ∧ isPow2 c.g = true)

/-- **The returned outputs' shape is the parse's**, in an untyped circuit: `Layout.outShape_of_parse` where the unit
returns them, and at a tail `check_tail`, with `Net.ofRows`' (`built_out`, its groups in the slot) or a lookup slot's
one output word. -/
theorem outShape_of_parseT {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) : OutShapeT c := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  refine ⟨fun ho => ?_, fun ho => ?_⟩
  · have C := check_facts P.typed hls
    have hU : Built c.unit := by
      rcases P.nets _ (getD_mem_toList P.unitNet) with hb | ⟨nm, tb, n, lo, bits, hb⟩
      · exact hb
      · have h1 := (build_facts hb).1
        have h2 := C.unit_lookup
        unfold Circuit.unit at h2
        rw [h1] at h2
        cases h2
    obtain ⟨hw, hsh⟩ := Layout.outShape_of_parse h hU ho
    rw [outGrp_unit ho]
    refine ⟨hw, fun u' hu' => ?_⟩
    have e : outsOf c u' = c.leavesOut.getD u' #[] := ite_eq_left ho
    rw [e]
    obtain ⟨hb, hfit⟩ := hsh u' hu'
    exact ⟨fun j hj => hb j hj, hfit⟩
  · obtain ⟨hsz, h16, h1, hp, hg⟩ := check_tail P.typed ho hls
    have hog := eq_singleton_getD h1
    rw [hog] at hsz h16
    have hb16 : ∀ b ∈ (outGrp c).bits.toList, b = 16 := by
      rw [← portCols_single]
      intro b hb
      obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hb
      exact h16 x hx
    have hbs : (outGrp c).bits.size = c.outWords := by
      have := congrArg List.length (portCols_single (outGrp c))
      rw [List.length_map, Array.length_toList, Array.length_toList, hsz] at this
      exact this.symm
    have hn : c.outNet < c.nets.size := by
      by_contra hn
      have h0 : (netOf c c.outNet).outGroups.size = 0 := by
        unfold netOf
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]
        rfl
      omega
    have hbound : 16 * c.outWords ≤ 2 ^ (netOf c c.outNet).unitLog := by
      rcases P.nets _ (getD_mem_toList hn) with hB | ⟨nm, tb, n, lo, bits, hB⟩
      · obtain ⟨-, hgs⟩ := Layout.built_out hB
        have hmem : outGrp c ∈ (netOf c c.outNet).outGroups.toList := getD_mem_toList (by omega)
        obtain ⟨-, hfit⟩ := hgs _ hmem
        obtain ⟨name, useful, constPos, ig, og, ra, rb, hr⟩ := hB
        obtain ⟨-, -, -, -, -, -, hog', -, -, -, -, -, -, -, hd11⟩ := Net.ofRows_ok hr
        have hc : (outGrp c).col + (outGrp c).words ≤ 2 ^ (netOf c c.outNet).unitLog / WORD :=
          hd11 (outGrp c) (by rw [← hog']; exact hmem)
        have hsum := Layout.foldl_add_const hb16 0
        rw [Array.foldl_toList] at hsum
        rw [hsum, Array.length_toList, hbs] at hfit
        exact le_trans (by omega : 16 * c.outWords ≤ (outGrp c).words * WORD)
          (le_trans (Nat.mul_le_mul_right _ (by omega)) (Nat.div_mul_le_self _ _))
      · obtain ⟨col, hcol⟩ := lookup_outGroups hB
        have hl := (netOK_lookup hB).2.1
        have e : outGrp c = ⟨col, 1, #[bits]⟩ := by
          unfold outGrp netOf
          rw [hcol]
          rfl
        rw [e] at hbs
        simp only [List.size_toArray, List.length_cons, List.length_nil, zero_add] at hbs
        rw [← hbs]
        calc 16 * 1 ≤ 2 ^ 7 := by norm_num
          _ ≤ 2 ^ (netOf c c.outNet).unitLog := Nat.pow_le_pow_right (by norm_num) hl
    refine ⟨h1, hbs, fun j hj => ?_, hp, hbound, hg⟩
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some]
    exact hb16 _ (Array.getElem_mem_toList _)

/-! ## The regions are whole-word -/

/-- `tz` of a power of two of at least `2^7`, below `2^64`, is at least 7. -/
theorem seven_le_tz {x : ℕ} (h : isPow2 x = true) (hx : x < 2 ^ 64) (h7 : 128 ≤ x) : 7 ≤ tz x := by
  have e := tz_of_pow2 h hx
  by_contra hc
  have : 2 ^ tz x < 2 ^ 7 := Nat.pow_lt_pow_right (by norm_num) (by omega)
  omega

open Refine in
set_option maxHeartbeats 2000000 in
/-- **`HmRow.regions`' regions are whole-word** in an untyped circuit (`Composed.hm_regions_whole` without
`c.outNet = c.unitNet`), when a tail's region "out" has a whole word per slot (`7 ≤ tz (16 · nOut)`): each port's
"digest p", and region "out", the unit's (`Circuit.outRegion`) or the tail's (`tailRegion`). -/
theorem hm_regions_wholeT {c : Flock.Circuit} {pub : Flock.Public} {regs : Array Flock.Region}
    (h : HmRow.regions c pub = .ok regs) (ht : c.typed = none)
    (hw : c.outNet ≠ c.unitNet → 7 ≤ tz (16 * c.outWords)) : ∀ x ∈ regs.toList, Composed.WholeWord x := by
  unfold HmRow.regions at h
  dsimp only at h
  try simp only [throw_bind', pure_bind] at h
  obtain ⟨out, hloop, h⟩ := ebind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  obtain ⟨es, hes, heq⟩ := list_forIn_push _ (fun _ v => v) _ _ _ hloop
  have hI : ∀ x ∈ out.toList, Composed.WholeWord x := by
    intro x hx
    have hx' : x ∈ es := by simpa [heq] using hx
    obtain ⟨a, -, v, hv, rfl⟩ := Layout.forall₂_mem hes.flip x hx'
    exact Composed.mkRegion_whole hv (Composed.hmFree_low c)
  split at h
  · rename_i hc
    have hne : c.outNet ≠ c.unitNet := by simpa [ht] using hc
    obtain ⟨r, hr, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact Composed.whole_push hI (Composed.mkRegion_whole hr (Composed.low_mem (hw hne) _))
  · obtain ⟨r, hr, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact Composed.whole_push hI (Composed.outRegion_whole hr)

/-- **A `setupH` statement's regions are whole-word** once its circuit is untyped and a tail's region "out" has a
whole word per slot (`Composed.setupH_whole` without `st.c.outNet = st.c.unitNet`). -/
theorem setupH_wholeT {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st)
    (ht : st.c.typed = none) (hw : st.c.outNet ≠ st.c.unitNet → 7 ≤ tz (16 * st.c.outWords)) :
    Composed.WholeWords st :=
  hm_regions_wholeT (Layout.setupH_regions h) ht hw

/-! ## Where region "out" pins them at a tail -/

/-- Column `16 j + t` of the output net's returned-output group, in its slot. -/
def outColT (c : Circuit) (j t : ℕ) : ℕ := (outGrp c).col * WORD + (16 * j + t)

theorem slotCol_outColT (c : Circuit) (g i j t : ℕ) :
    c.slotCol c.outNet g i + outColT c j t = (c.rangeOf c.outNet).start + (outGrp c).col * WORD + (16 * j + t) +
      (g * c.perVu.getD c.outNet 0 + i) * 2 ^ (c.rangeOf c.outNet).slotLog := by
  simp only [Circuit.slotCol, Range.slot, Range.start, outColT]
  ring

/-- Region bit `k·2^tz(16 nOut) + m` (slot `k`'s column `m`) of a tail's region "out" sits at the slot's column `m`. -/
theorem inBlock_tail {c : Circuit} {pub : Public} {r : Region} (nbl : ℕ) (h : tailRegion c pub = .ok r) {k m : ℕ}
    (hk : k < 2 ^ tz (c.rangeOf c.outNet).count) (hm : m < 2 ^ tz (16 * c.outWords)) :
    (Refine.regionOf c.kLog nbl r).inBlock (k * 2 ^ tz (16 * c.outWords) + m) =
      (c.rangeOf c.outNet).start + (outGrp c).col * WORD + m + k * 2 ^ (c.rangeOf c.outNet).slotLog := by
  unfold tailRegion at h
  rw [Layout.inBlock_mkRegion h]
  simp only [Array.toList_append, Array.toList_map, Array.toList_range, Layout.spread_append, Layout.spread_range,
    List.length_map, List.length_range, pow_zero, mul_one]
  have hP : 0 < 2 ^ tz (16 * c.outWords) := Nat.two_pow_pos _
  have h1 : (k * 2 ^ tz (16 * c.outWords) + m) % 2 ^ tz (16 * c.outWords) = m := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hm]
  have h2 : (k * 2 ^ tz (16 * c.outWords) + m) / 2 ^ tz (16 * c.outWords) = k := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hP, Nat.div_eq_of_lt hm, zero_add]
  rw [h1, h2, Nat.mod_eq_of_lt hk]
  ring

section pin

open Model

/-- **The output words a satisfying witness holds at a tail's output columns** are the ones the statement pins: in an
untyped circuit whose last stage returns the outputs, at block `o` and VU `g` (below `2^tz count`, all of them when the
output range has one slot per VU and `vus_per_block` is a power of two), the bit at column `16 j + t` of the slot's
returned-output group is bit `t` of output word `j` at the VU's instance (the pinned dummy past the population), when
`16 · nOut` is a power of two (`hA`). -/
theorem tail_pinned {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    (hR : HmRow.regions st.c st.pub = .ok st.regions) (ho : st.c.outNet ≠ st.c.unitNet) (hty : st.c.typed = none)
    {z : Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g j t : ℕ} (hA : 2 ^ tz (16 * st.c.outWords) = 16 * st.c.outWords) (hg : g < st.c.g)
    (hB : g < 2 ^ tz (st.c.rangeOf st.c.outNet).count) (hj : j < st.c.outWords) (ht : t < 16) :
    bit ((L.model regs).block z o (Layout.inBlockOr st.c.kLog ((st.c.rangeOf st.c.outNet).start +
        (outGrp st.c).col * WORD + (16 * j + t) + g * 2 ^ (st.c.rangeOf st.c.outNet).slotLog))) =
      (tailVal st.c st.pub o.val g j).testBit t := by
  obtain ⟨r, hr, hmk⟩ := regions_outT hR hty
  rw [outRegionT, ite_eq_right ho] at hmk
  have hmem : Refine.regionOf (L.model regs).kLog ((L.model regs).m - (L.model regs).kLog) r ∈
      (L.model regs).regions := hregs r hr
  have hmk' := hmk
  unfold tailRegion at hmk'
  obtain ⟨hrf, -, -, hnd, -, hlt⟩ := Layout.mkRegion_spec hmk'
  have hfreeA := congrArg Flock.Region.free hrf
  simp only at hfreeA
  have hlen : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.length =
      tz (16 * st.c.outWords) + tz (st.c.rangeOf st.c.outNet).count := by
    rw [← List.length_map (f := Fin.val), Layout.regionOf_free_map _ _ _ (fun b hb => hlt b (by
      rw [hfreeA] at hb; exact Array.mem_toList_iff.1 hb)), hfreeA]
    simp
  have hm : 16 * j + t < 2 ^ tz (16 * st.c.outWords) := by rw [hA]; omega
  have hi : g * 2 ^ tz (16 * st.c.outWords) + (16 * j + t) <
      2 ^ (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.length := by
    rw [hlen, pow_add]
    calc g * 2 ^ tz (16 * st.c.outWords) + (16 * j + t) < g * 2 ^ tz (16 * st.c.outWords) +
          2 ^ tz (16 * st.c.outWords) := by omega
      _ = (g + 1) * 2 ^ tz (16 * st.c.outWords) := by ring
      _ ≤ 2 ^ tz (st.c.rangeOf st.c.outNet).count * 2 ^ tz (16 * st.c.outWords) := Nat.mul_le_mul_right _ hB
      _ = 2 ^ tz (16 * st.c.outWords) * 2 ^ tz (st.c.rangeOf st.c.outNet).count := by ring
  have hib := inBlock_tail (st.m - st.c.kLog) hmk hB hm
  have hnd' : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.Nodup := by
    unfold Refine.regionOf
    rw [hfreeA]
    exact Refine.filterMap_fin_nodup _ _ hnd
  have ha : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).inBlock (g * 2 ^ tz (16 * st.c.outWords) + (16 * j + t)) <
      2 ^ st.c.kLog :=
    Layout.inBlock_lt _ hnd' _
  have hread := Layout.region_read hz hmem o _ hi ha
  have hfin : Layout.inBlockOr st.c.kLog ((st.c.rangeOf st.c.outNet).start + (outGrp st.c).col * WORD + (16 * j + t) +
      g * 2 ^ (st.c.rangeOf st.c.outNet).slotLog) = ⟨_, ha⟩ := by
    simp only [Layout.inBlockOr, ← hib, ha, ↓reduceDIte]
  rw [hfin]
  refine hread.trans ?_
  have hby : r.bytes = tailBytes st.c st.pub := by rw [hrf]
  have e1 : g * 2 ^ tz (16 * st.c.outWords) = 8 * (g * (2 * st.c.outWords)) := by rw [hA]; ring
  rw [Layout.get!_eq_getD, hby, e1,
    show (8 * (g * (2 * st.c.outWords)) + (16 * j + t)) / 8 = g * (2 * st.c.outWords) + (2 * j + t / 8) by omega,
    show (8 * (g * (2 * st.c.outWords)) + (16 * j + t)) % 8 = t % 8 by omega]
  exact tailBytes_bit st.c st.pub o.val hg hj ht

/-- Output `j` of slot `i`'s returned output words at VU `g` of block `b`: the VU's instance's output word
`(outsOf c i)[j]`, or the pinned dummy past the population. -/
def outValT (c : Circuit) (pub : Public) (b g i j : ℕ) : ℕ :=
  match c.instOf pub.n b g with
  | some k => pub.outputs.getD (k * c.outWords + (outsOf c i)[j]!) 0
  | none => c.dummyOutputs.getD (outsOf c i)[j]! 0

theorem outValT_unit {c : Circuit} {pub : Public} {b g i j : ℕ} (ho : c.outNet = c.unitNet) (hi : i < c.upv) :
    outValT c pub b g i j = Layout.outVal c pub b (g * c.upv + i) j := by
  have hm : (g * c.upv + i) % c.upv = i := by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
  have hd : (g * c.upv + i) / c.upv = g := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi, zero_add]
  have e : outsOf c i = c.leavesOut.getD i #[] := ite_eq_left ho
  unfold outValT Layout.outVal
  rw [hm, hd, e]
  cases c.instOf pub.n b g <;> rfl

theorem outValT_tail {c : Circuit} {pub : Public} {b g i j : ℕ} (ho : c.outNet ≠ c.unitNet) (hj : j < c.outWords) :
    outValT c pub b g i j = tailVal c pub b g j := by
  have e : outsOf c i = Array.range c.outWords := ite_eq_right ho
  have hr : (Array.range c.outWords)[j]! = j := by
    rw [getElem!_pos _ j (by simpa using hj), Array.getElem_range]
  unfold outValT tailVal
  rw [e, hr]

/-- **A tail has one slot of its last stage per VU** (a condition on the circuit: the prover lays a tail out so, and
`HmRow.regions` pins the out range's slot `g` with VU `g`'s outputs). -/
def TailOne (c : Circuit) : Prop := c.outNet ≠ c.unitNet → c.perVu.getD c.outNet 0 = 1

/-- **What region "out" needs to pin the returned outputs at a slot of the output net**: their shape (`OutShapeT`);
where the unit returns them, a power of two of unit slots, `upv` per VU; at a tail, one slot per VU (`TailOne`), and
`16 · nOut` and the out range's count powers of two that `tz` reads. -/
def PinShape (c : Circuit) : Prop :=
  OutShapeT c ∧
  (c.outNet = c.unitNet → 2 ^ c.unitRange.count.log2 = c.unitRange.count ∧ c.perVu.getD c.unitNet 0 = c.upv) ∧
  (c.outNet ≠ c.unitNet → c.perVu.getD c.outNet 0 = 1 ∧ 2 ^ tz (16 * c.outWords) = 16 * c.outWords ∧
    2 ^ tz (c.rangeOf c.outNet).count = c.g)

theorem outNet_lt {c : Circuit} (h1 : (netOf c c.outNet).outGroups.size = 1) : c.outNet < c.nets.size := by
  by_contra hn
  have h0 : (netOf c c.outNet).outGroups.size = 0 := by
    unfold netOf
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]
    rfl
  omega

/-- **`PinShape` from the parse and the layout**, for a circuit whose tail has one slot per VU. -/
theorem pinShape_of {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {st : Stmt}
    (L : FlockSoundness.Layout st) (h : HmRow.parse tags bytes tables = .ok st.c) (hone : TailOne st.c) :
    PinShape st.c := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have hS := outShape_of_parseT h
  refine ⟨hS, fun ho => ?_, fun ho => ?_⟩
  · have hpow := Layout.unitRange_pow2 h ho L.unit_count
    have hg0 : 0 < st.c.g := by
      have := Nat.one_le_two_pow (n := st.c.unitRange.count.log2)
      rw [hpow, L.unit_count] at this
      exact Nat.pos_of_ne_zero fun h0 => by rw [h0, zero_mul] at this; omega
    exact ⟨hpow, perVu_unit C L.unit_mem L.unit_kind L.unit_count hg0⟩
  · obtain ⟨h1, -, -, hp16, hbound, hgp⟩ := hS.2 ho
    have hpv := hone ho
    have hn := outNet_lt h1
    have hg0 : 0 < st.c.g := by
      simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hgp
      exact hgp.1
    have hlog := netLog_le L C hn hg0 (show 0 < st.c.perVu.getD st.c.outNet 0 by omega)
    have h64 : (2 : ℕ) ^ 32 < 2 ^ 64 := by norm_num
    have hk32 : 2 ^ st.c.kLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) L.kLog
    refine ⟨hpv, tz_of_pow2 hp16 ?_, ?_⟩
    · have := Nat.pow_le_pow_right (show 0 < 2 by norm_num) hlog
      omega
    · obtain ⟨hm, -, hcount, -⟩ := net_range L C hn
      rw [hcount, hpv, mul_one]
      refine tz_of_pow2 hgp ?_
      have h1 := count_mul_le_spanH (st.c.rangeOf st.c.outNet)
      have h2 := L.bound _ hm
      rw [hcount, hpv, mul_one] at h1
      have h3 := Nat.le_mul_of_pos_right st.c.g (Nat.two_pow_pos (st.c.rangeOf st.c.outNet).slotLog)
      omega

theorem outsOf_lt {c : Circuit} (hP : PinShape c) {i : ℕ} (hi : i < c.perVu.getD c.outNet 0) :
    ∀ j < (outsOf c i).size, (outsOf c i)[j]! < c.outWords := by
  intro j hj
  by_cases ho : c.outNet = c.unitNet
  · have hpv := (hP.2.1 ho).2
    rw [ho, hpv] at hi
    exact (((hP.1.1 ho).2 i hi).1 j hj).2
  · have e : outsOf c i = Array.range c.outWords := ite_eq_right ho
    rw [e] at hj ⊢
    rw [getElem!_pos _ j hj, Array.getElem_range]
    simpa using hj

/-- **The returned outputs a satisfying witness holds at the output net's slots** are the ones the statement pins, in
an untyped circuit, whether the unit returns them (`Layout.out_pinned`) or a tail does (`tail_pinned`): at block `o`,
VU `g` and slot `i` of the output net, the bit at column `16 j + t` of the slot's returned-output group (`outColT`) is
bit `t` of output `j` of the slot's returned output words at the VU's instance (`outValT`), under `PinShape`, for
16-bit values (`hv`). -/
theorem out_pinnedT {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    (hR : HmRow.regions st.c st.pub = .ok st.regions) (hty : st.c.typed = none)
    {z : Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g i j t : ℕ} (hg : g < st.c.g) (hi : i < st.c.perVu.getD st.c.outNet 0) (hP : PinShape st.c)
    (hv : ∀ j < (outsOf st.c i).size, outValT st.c st.pub o.val g i j < 2 ^ 16)
    (hj : j < (outsOf st.c i).size) (ht : t < 16) :
    bit ((L.model regs).block z o (Layout.inBlockOr st.c.kLog (st.c.slotCol st.c.outNet g i + outColT st.c j t))) =
      (outValT st.c st.pub o.val g i j).testBit t := by
  rw [slotCol_outColT]
  by_cases ho : st.c.outNet = st.c.unitNet
  · obtain ⟨hpow, hpv⟩ := hP.2.1 ho
    obtain ⟨hw, hsh⟩ := hP.1.1 ho
    have hpv' : st.c.perVu.getD st.c.outNet 0 = st.c.upv := by rw [ho]; exact hpv
    have hi' : i < st.c.upv := hpv' ▸ hi
    have haddr : (st.c.rangeOf st.c.outNet).start + (outGrp st.c).col * WORD + (16 * j + t) +
        (g * st.c.perVu.getD st.c.outNet 0 + i) * 2 ^ (st.c.rangeOf st.c.outNet).slotLog =
        Layout.outAddr st.c (g * st.c.upv + i) j t := by
      rw [hpv', outGrp_unit ho, Layout.outAddr, Circuit.unitRange, ← ho]
    rw [haddr, outValT_unit ho hi']
    have hm : (g * st.c.upv + i) % st.c.upv = i := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi']
    have hk : g * st.c.upv + i < st.c.unitRange.count := by
      rw [L.unit_count]
      exact Rows.mul_add_lt hg hi'
    obtain ⟨hb, hfit⟩ := hsh i hi'
    have e : outsOf st.c i = st.c.leavesOut.getD i #[] := ite_eq_left ho
    rw [e] at hb hfit hj hv
    rw [outGrp_unit ho] at hw hb hfit
    refine Layout.out_pinned L regs hregs hR ho hty hz o hk (by rw [hpow]; exact hk) hw
      (by rw [hm]; exact fun j hj => (hb j hj).1) (by rw [hm]; exact fun j hj => (outValT_unit ho hi') ▸ hv j hj)
      (by rw [hm]; exact hfit) (by rw [hm]; exact hj) ht
  · obtain ⟨hpv, hA, hB⟩ := hP.2.2 ho
    have hi0 : i = 0 := by rw [hpv] at hi; omega
    subst hi0
    have e : outsOf st.c 0 = Array.range st.c.outWords := ite_eq_right ho
    have hj' : j < st.c.outWords := by rw [e] at hj; simpa using hj
    rw [hpv, mul_one, add_zero, outValT_tail ho hj']
    exact tail_pinned L regs hregs hR ho hty hz o hA hg (by rw [hB]; exact hg) hj' ht

/-- **The returned output words at a proved VU are the population's**: at block `o` and VU `g` whose instance
`o · c.g + g` is proved, output `j` of slot `i`'s returned output words (a word below `nOut`) is proved unit
`D.proved[o · c.g + g]`'s in the population (`Layout.outVal_drawn` at the output net's slots). -/
theorem outValT_drawn {c : Circuit} {pop pub : Public} {d : Lean.Json} {D : Draw.UnitDraw}
    (h : HmRow.drawn c pop d = .ok pub) (hD : Draw.UnitDraw.ofJson d = .ok D)
    (hsz : pop.outputs.size = pop.n * c.outWords) (hlt : ∀ u ∈ D.proved, u < D.population)
    {o g i j : ℕ} (hr : o * c.g + g < D.proved.size) (hlo : (outsOf c i)[j]! < c.outWords) :
    outValT c pub o g i j = pop.outputs.getD (D.proved[o * c.g + g] * c.outWords + (outsOf c i)[j]!) 0 := by
  obtain ⟨D', hD', hpopn, hn, -, -⟩ := Layout.drawn_spec h
  obtain ⟨D'', hD'', hout⟩ := Layout.drawn_outputs h
  rw [hD] at hD' hD''
  cases hD'
  cases hD''
  unfold outValT Circuit.instOf
  rw [hn]
  simp only [hr, ↓reduceIte]
  rw [hout]
  refine Layout.foldl_extract_getD pop.outputs D.proved (fun u hu => ?_) 0 hr hlo
  have := hlt u hu
  rw [hsz, ← hpopn]
  nlinarith

/-- **The returned outputs a satisfying witness of a drawn statement holds at a proved VU's output slot** are the VU's
proved unit's in the registered population's public file, in an untyped circuit: `out_pinnedT` at `outValT_drawn`. -/
theorem out_drawnT {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {d : Lean.Json} {partition : Option Lean.Json} {program : Option ByteArray}
    {st : Stmt} (h : Stmt.setupH tags circuitFile publicFile coins tables (some d) partition program = .ok st)
    {pop : Public} (hpop : HmRow.loadPublic tags st.c publicFile = .ok pop)
    {D : Draw.UnitDraw} (hD : Draw.UnitDraw.ofJson d = .ok D) (hlt : ∀ u ∈ D.proved, u < D.population)
    (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) (hty : st.c.typed = none)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g i j t : ℕ} (hg : g < st.c.g) (hr : o.val * st.c.g + g < D.proved.size)
    (hi : i < st.c.perVu.getD st.c.outNet 0) (hP : PinShape st.c) (hj : j < (outsOf st.c i).size) (ht : t < 16) :
    bit ((L.model regs).block z o (Layout.inBlockOr st.c.kLog (st.c.slotCol st.c.outNet g i + outColT st.c j t))) =
      (pop.outputs.getD (D.proved[o.val * st.c.g + g] * st.c.outWords + (outsOf st.c i)[j]!) 0).testBit t := by
  obtain ⟨pop', hpop', hdr⟩ := Layout.setupH_drawn h
  rw [hpop] at hpop'
  cases hpop'
  obtain ⟨hsz, h16⟩ := Layout.loadPublic_outputs hpop
  have hlo := outsOf_lt hP hi
  rw [← outValT_drawn hdr hD hsz hlt hr (hlo j hj)]
  refine out_pinnedT L regs hregs (Layout.setupH_regions h) hty hz o hg hi hP (fun j' hj' => ?_) hj ht
  rw [outValT_drawn hdr hD hsz hlt hr (hlo j' hj')]
  exact Layout.getD_lt_of_forall h16 (Nat.two_pow_pos _) _

variable {T : Model.Statement}

/-- **The returned outputs at a drawn unit's site are the population's**: in a satisfying witness of an accepted drawn
statement of an untyped circuit, at a site (`BcSite`: block `o`, template instance `g`, proved unit `u`), the bit at
column `16 j + t` of the returned-output group of the output net's slot `i` is bit `t` of the population's output word
`(outsOf c i)[j]` of `u`, under `PinShape`. -/
theorem out_siteT (a : Placed.Accepted T) {pop : Public} {o : Fin (2 ^ T.nbl)} {g u : ℕ} (hs : Layout.BcSite a pop o g u)
    (hty : a.st.c.typed = none) {z : Witness T.m} (hz : T.Satisfies z) (hg : g < a.st.c.g) {i j t : ℕ}
    (hi : i < a.st.c.perVu.getD a.st.c.outNet 0) (hP : PinShape a.st.c) (hj : j < (outsOf a.st.c i).size)
    (ht : t < 16) :
    bit (z (Layout.siteAddr a o (a.st.c.slotCol a.st.c.outNet g i + outColT a.st.c j t))) =
      (pop.outputs.getD (u * a.st.c.outWords + (outsOf a.st.c i)[j]!) 0).testBit t := by
  obtain ⟨d, D, hd, hD, hlt, hregs, hpop, -, hu⟩ := hs
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, regs, hT⟩ := a
  subst hT
  dsimp only at hd hregs hpop hu hg hi hP hj hty ⊢
  subst hd
  obtain ⟨hr, rfl⟩ := Array.getElem?_eq_some_iff.1 hu
  exact out_drawnT h hpop hD hlt (setupH_layout ht' h) regs hregs hty hz o hg hr hi hP hj ht

end pin

/-! ## The template instance's committed gates at `instProg`'s places -/

section prog

variable (c : Flock.Circuit) (n : ℕ)

/-- **Composite column `x` of a template instance is a returned output's bit**: past the composite inputs, on the
output net's slot `i` at column `outColT j t`, for output `j` of the slot's returned output words and bit `t < 16`. -/
def OutColAtT (x i j t : ℕ) : Prop :=
  j < (outsOf c i).size ∧ t < 16 ∧ (instRows c).nIn ≤ x ∧ x < (order c).length ∧
    (order c).getD x 0 = code c.outNet i (outColT c j t)

/-- The slot, output and bit a composite column is on (`OutColAtT` read back). -/
def outDecodeT (x : ℕ) : ℕ × ℕ × ℕ :=
  (codeSlot ((order c).getD x 0), (codeOff ((order c).getD x 0) - (outGrp c).col * WORD) / 16,
    (codeOff ((order c).getD x 0) - (outGrp c).col * WORD) % 16)

variable {c} in
theorem outDecodeT_of {x i j t : ℕ} (h : OutColAtT c x i j t) : outDecodeT c x = (i, j, t) := by
  obtain ⟨-, ht, -, -, he⟩ := h
  unfold outDecodeT
  rw [he, codeSlot_code, codeOff_code]
  simp only [outColT, Nat.add_sub_cancel_left, Prod.mk.injEq]
  exact ⟨trivial, by omega, by omega⟩

variable {Reg : Type} (outs : Finset (Fin (instN c n))) (popOf : Reg → Public)

open Classical in
/-- **The registration's bit at a unit's computed gate** of `instProg`: bit `t` of the population's output word
`(outsOf c i)[j]` of the unit, where the gate's column in the unit's instance is the output net's slot `i`'s output
`j`, bit `t` (`outDecodeT`). -/
noncomputable def instOutBitOf (R : Reg) (g : Fin (instN c n)) : Bool :=
  match (instProg c n).unit g with
  | some u =>
    if h : ∃ x, (Const.Prog.instC (instProg c n) outs u).wire x = g then
      ((popOf R).outputs.getD (u.val * c.outWords +
        (outsOf c (outDecodeT c (Classical.choose h).val).1)[(outDecodeT c (Classical.choose h).val).2.1]!)
          0).testBit (outDecodeT c (Classical.choose h).val).2.2
    else false
  | none => false

/-- **The committed gates a unit of `instProg` computes are returned outputs' bits** (a condition on the circuit's
outputs `outs`): each column of the unit's instance on one is a returned output's bit (`OutColAtT`). -/
def OutsOutT : Prop :=
  ∀ u g, g ∈ ((instProg c n).partition outs).io u → (instProg c n).unit g = some u →
    ∀ x : Fin ((instProg c n).rows u).w, (Const.Prog.instC (instProg c n) outs u).wire x = g →
      ∃ i j t, OutColAtT c x.val i j t

variable {c n outs popOf} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : Placed.Sites plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)

/-- **A drawn unit's composite column `x`** (one of `order c`) sits at its code's place in the unit's template
instance (`pos`). -/
theorem instCol_val {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) (x : Fin ((instProg c n).rows u).w)
    (hx : x.val < (order c).length) :
    (Sites.instCol hparse S R u hu x).val =
      pos (si.acc S R u hu).st.c (si.vu S R u hu) ((order c).getD x.val 0) := by
  have hcc := (si.acc S R u hu).c_eq (hparse S R u hu)
  unfold Sites.instCol
  rw [Fin.val_cast, composePlace_val, Fin.val_cast]
  have hlen : x.val < (logical (si.acc S R u hu).st.c).order.length := by
    show _ < (order (si.acc S R u hu).st.c).length
    rw [hcc]
    exact hx
  rw [ite_eq_left hlen]
  show pos (si.acc S R u hu).st.c (si.vu S R u hu) ((order (si.acc S R u hu).st.c).getD _ 0) = _
  rw [hcc]

include hparse in
/-- **`hout` at `instProg`'s places**: in an untyped circuit, whether its unit or a tail returns the outputs, at each
drawn unit's site (`BcSite`), a satisfying witness holds at every committed gate the unit computes the registration's
bit `instOutBitOf`, the population's returned output's, when those gates are returned outputs' bits (`OutsOutT`) and a
tail has one slot per VU (`TailOne`); the shape region "out" pins is the parse's (`pinShape_of`). -/
theorem instProg_hout
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      Layout.BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (houts : OutsOutT c n outs) (hone : TailOne c) :
    ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ ((instProg c n).partition outs).io u,
      (instProg c n).unit g = some u →
        bit (Binding.wit plan tab S R u M (Layout.gateAt (instPlace hparse outs) S R u g)) =
          instOutBitOf c n outs popOf R g := by
  intro S R u hu M hM g hg hgu
  have hc' := Layout.exists_col_of_io ((instPlace hparse outs) S R u hu).inst hg
  have hc : ∃ x, (Const.Prog.instC (instProg c n) outs u).wire x = g := hc'
  obtain ⟨i, j, t, hoc⟩ := houts u g hg hgu (Classical.choose hc') (Classical.choose_spec hc')
  have hob : instOutBitOf c n outs popOf R g =
      ((popOf R).outputs.getD (u.val * c.outWords + (outsOf c i)[j]!) 0).testBit t := by
    unfold instOutBitOf
    rw [hgu]
    dsimp only
    rw [dite_eq_left hc, outDecodeT_of (show OutColAtT c (Classical.choose hc).val i j t from hoc)]
  obtain ⟨hj, ht, hnx, hx, hcode⟩ := hoc
  have hcc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hcol : (Sites.instCol hparse S R u hu (Classical.choose hc')).val =
      (si.acc S R u hu).st.c.slotCol (si.acc S R u hu).st.c.outNet (si.vu S R u hu) i +
        outColT (si.acc S R u hu).st.c j t := by
    rw [instCol_val hparse hu (Classical.choose hc') hx, hcode, hcc, pos_code]
  have hlt : (si.acc S R u hu).st.c.slotCol (si.acc S R u hu).st.c.outNet (si.vu S R u hu) i +
      outColT (si.acc S R u hu).st.c j t < 2 ^ (si.acc S R u hu).st.c.kLog := by
    rw [← hcol]
    exact lt_of_lt_of_eq (Fin.isLt _) (si.acc S R u hu).kLog_eq.symm
  have hga : Layout.gateAt (instPlace hparse outs) S R u g = Layout.siteAddr (si.acc S R u hu) (si.o S R u hu)
      ((si.acc S R u hu).st.c.slotCol (si.acc S R u hu).st.c.outNet (si.vu S R u hu) i +
        outColT (si.acc S R u hu).st.c j t) := by
    unfold Layout.gateAt
    rw [dite_eq_left hu, dite_eq_left hc']
    unfold Layout.siteAddr
    congr 1
    apply Fin.ext
    have hin : (Layout.inBlockOr (si.acc S R u hu).st.c.kLog
        ((si.acc S R u hu).st.c.slotCol (si.acc S R u hu).st.c.outNet (si.vu S R u hu) i +
          outColT (si.acc S R u hu).st.c j t)).val =
        (si.acc S R u hu).st.c.slotCol (si.acc S R u hu).st.c.outNet (si.vu S R u hu) i +
          outColT (si.acc S R u hu).st.c j t := by
      simp only [Layout.inBlockOr, hlt, ↓reduceDIte]
    exact hcol.trans hin.symm
  rw [hga, hob]
  have hrow := order_row (c := c) hx hnx
  rw [hcode] at hrow
  unfold row at hrow
  rw [codeNet_code, codeSlot_code, codeOff_code] at hrow
  obtain ⟨-, -, -, hi, -⟩ := rowAt_dom hrow
  have hpar := hparse S R u hu
  rw [← hcc] at hpar
  have hP := pinShape_of (si.acc S R u hu).layout hpar (by rw [hcc]; exact hone)
  have := out_siteT (si.acc S R u hu) (hsite S R u hu M hM) (parse_facts hpar).typed hM (si.hvu S R u hu)
    (i := i) (j := j) (t := t) (by rw [hcc]; exact hi) hP (by rw [hcc]; exact hj) ht
  rw [this, hcc]

variable (rowW : Fin c.ports.size → ℕ) (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ)

/-- **`rs` at `instProg`'s places, with the returned outputs and the row bytes `sha512/row/v1` hashes**: `instHmRows`
with `enc := rowEnc`, `bitCode` for the other commit strings, the registration's bit `instOutBitOf` and its `hout`
from `instProg_hout`. -/
noncomputable def instRowsEnc
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      Layout.BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : InstKeysRow c rowW) (houts : OutsOutT c n outs) (hone : TailOne c)
    (hc1 : ((instProg c n).partition outs).ConstCols plan tab (instPlace hparse outs) {instOneG (c := c) (n := n)})
    (hz1 : ((instProg c n).partition outs).ZeroCols plan tab (instPlace hparse outs) {instZeroG (c := c) (n := n)}) :
    Binding.HmRows (instLayout rowW hparse outs saltCol popOf (instOutBitOf c n outs popOf) Layout.bitCode) :=
  instHmRows rowW hparse outs saltCol popOf (instOutBitOf c n outs popOf) Layout.rowEnc Layout.rowEnc_inj
    Layout.bitCode Layout.bitCode_inj hsite hkeys hc1 hz1 (instProg_hout hparse hsite houts hone)

/-- `instRowsEnc`'s `enc` is `rowEnc`. -/
theorem instRowsEnc_enc
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      Layout.BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : InstKeysRow c rowW) (houts : OutsOutT c n outs) (hone : TailOne c)
    (hc1 : ((instProg c n).partition outs).ConstCols plan tab (instPlace hparse outs) {instOneG (c := c) (n := n)})
    (hz1 : ((instProg c n).partition outs).ZeroCols plan tab (instPlace hparse outs) {instZeroG (c := c) (n := n)}) :
    (instRowsEnc hparse rowW saltCol hsite hkeys houts hone hc1 hz1).enc = Layout.rowEnc := rfl

end prog

end FlockSoundness.Discharge.Inst
