import Proofs.Flock.Soundness.Discharge.Layout.OutRegion

/-!
# The public words that the statement pins at the unit slots' public leaves

A statement with public input ports (§16.10, `HmIn`) reads them from the in regions (`HmIn.inRegions`): region "in x",
`(w, s)`, is `2^s` words from word `w` of every unit slot (`regions_in`). Its free bits are a slot's `7 + s` column bits
and the slot's `log2ceil count` bits above the unit range's slot bits, at word `w` (`inBlock_in`); its bytes are, slot by
slot, each of its `8·2^s` leaves' 16-bit words, u16 LE (`inBytes_bit`): a public leaf its instance's public word, else 0
(`inVal`). So in a satisfying witness of the model's statement at the executable's regions, the bit at slot `k`'s leaf
`jj` column `16 jj + t` of the unit's leaf group (`inAddr`) is bit `t` of that word, for each leaf inside an in region
(`in_pinned`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Layout

open Flock Model

/-! ### Region "in x" -/

/-- The first column word of the unit's leaf group. -/
def inCol (c : Circuit) : ℕ := (c.unit.inGroups.getD 0 default).col

/-- The first leaf of an in region from word `w`. -/
def inJ0 (c : Circuit) (w : ℕ) : ℕ := 8 * (w - inCol c)

/-- An in region's free bits: a slot's `7 + s` column bits, then the slot's `log2ceil count` bits above the unit range's
slot bits. -/
def inFree (c : Circuit) (s : ℕ) : Array ℕ :=
  Array.map (fun x => 0 + x) (Array.range (7 + s)) ++
    Array.map (fun x => c.unitRange.slotLog + x) (Array.range (log2ceil c.unitRange.count))

/-- Leaf `jj` of unit slot `k` at block `b`, as the in regions hold it: the slot's instance's public word when the leaf
is public, 0 for a zero leaf, a padding instance or a slot past the units. -/
def inVal (c : Circuit) (pi : HmIn.Ports) (pub : Public) (b k jj : ℕ) : ℕ :=
  let q := (pi.leaves.getD (k % c.upv) #[]).getD jj (-1)
  match (if k < c.unitRange.count then c.instOf pub.n b (k / c.upv) else none) with
  | some i => if q ≥ 0 then pub.outputs.getD (i * pi.words + q.toNat) 0 else 0
  | none => 0

/-- Region "in x"'s bytes at block `b`, `(w, s)`: every unit slot's `8·2^s` leaves from `inJ0 c w`, u16 LE. -/
def inBytes (c : Circuit) (pi : HmIn.Ports) (pub : Public) (w s : ℕ) : ℕ → ByteArray := fun b =>
  (Array.range (2 ^ log2ceil c.unitRange.count)).foldl (fun acc k =>
    (Array.range (8 * 2 ^ s)).foldl (fun acc j =>
      acc.push (inVal c pi pub b k (inJ0 c w + j) % 256).toUInt8 |>.push
        (inVal c pi pub b k (inJ0 c w + j) / 256).toUInt8) acc) ByteArray.empty

theorem in_list_mapM_mem {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' →
      l'.length = l.length ∧ ∀ (j : ℕ) (hj : j < l.length) (hj' : j < l'.length), f l[j] = .ok l'[j]
  | [], l', h => by
      simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
  | a :: l, l', h => by
      rw [List.mapM_cons] at h
      obtain ⟨b, hb, h⟩ := Refine.ebind_ok h
      obtain ⟨bs, hbs, h⟩ := Refine.ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨hlen, hall⟩ := in_list_mapM_mem hbs
      refine ⟨by simp [hlen], fun j hj hj' => ?_⟩
      rcases j with _ | j
      · exact hb
      · exact hall j (by simpa using hj) (by simpa using hj')

theorem in_range_mapM {β ε : Type} {f : ℕ → Except ε β} {n : ℕ} {ys : Array β}
    (h : (Array.range n).mapM f = .ok ys) : ys.size = n ∧ ∀ (j : ℕ) (hj : j < ys.size), f j = .ok ys[j] := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : (Array.range n).toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
      rw [hl] at h
      simp only [Functor.map, Except.map, Except.ok.injEq] at h
      subst h
      obtain ⟨hlen, hall⟩ := in_list_mapM_mem hl
      simp only [Array.toList_range, List.length_range] at hlen
      refine ⟨by simpa using hlen, fun j hj => ?_⟩
      have hj' : j < l.length := by simpa using hj
      have := hall j (by rw [Array.toList_range, List.length_range]; omega) hj'
      simpa [Array.toList_range] using this

set_option maxHeartbeats 2000000 in
/-- **`HmIn.regions` has region "in x" for each in region `x`, `(w, s)`**, as `mkRegion` built it. -/
theorem regions_in {c : Circuit} {pi : HmIn.Ports} {pub : Public} {regs : Array Region}
    (h : HmIn.regions c pi pub = .ok regs) (hne : pi.ports.isEmpty = false) {x w s : ℕ}
    (hx : x < (HmIn.inRegions c pi).size) (hws : (HmIn.inRegions c pi)[x]! = (w, s)) :
    ∃ r ∈ regs.toList,
      mkRegion c.kLog s!"in {x}" (inFree c s) (c.unitRange.start + w * WORD) (inBytes c pi pub w s) = .ok r := by
  unfold HmIn.regions at h
  dsimp only at h
  simp only [hne, Bool.false_eq_true, ↓reduceIte] at h
  obtain ⟨hsz, hall⟩ := in_range_mapM h
  have hx' : x < regs.size := by rw [hsz]; exact hx
  refine ⟨regs[x], Array.getElem_mem_toList hx', ?_⟩
  have hf := hall x hx'
  try beta_reduce at hf
  rw [hws] at hf
  exact hf

/-! ### Where region "in x" pins slot `k`'s column `m` -/

/-- The in-block address of bit `t` of leaf `jj` of unit slot `k`: column `16 jj + t` of the slot's leaf group. -/
def inAddr (c : Circuit) (k jj t : ℕ) : ℕ :=
  c.unitRange.start + inCol c * WORD + (16 * jj + t) + k * 2 ^ c.unitRange.slotLog

/-- Region bit `k·2^(7+s) + m` (slot `k`'s column `m` from word `w`) of region "in x" sits there. -/
theorem inBlock_in {c : Circuit} {name : String} {w s : ℕ} {bytes : ℕ → ByteArray} {r : Region} (nbl : ℕ)
    (h : mkRegion c.kLog name (inFree c s) (c.unitRange.start + w * WORD) bytes = .ok r) {k m : ℕ}
    (hk : k < 2 ^ log2ceil c.unitRange.count) (hm : m < 2 ^ (7 + s)) :
    (Refine.regionOf c.kLog nbl r).inBlock (k * 2 ^ (7 + s) + m) =
      c.unitRange.start + w * WORD + m + k * 2 ^ c.unitRange.slotLog := by
  rw [inBlock_mkRegion h]
  simp only [inFree, Array.toList_append, Array.toList_map, Array.toList_range, spread_append, spread_range,
    List.length_map, List.length_range, pow_zero, mul_one]
  have hP : 0 < 2 ^ (7 + s) := Nat.two_pow_pos _
  have h1 : (k * 2 ^ (7 + s) + m) % 2 ^ (7 + s) = m := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hm]
  have h2 : (k * 2 ^ (7 + s) + m) / 2 ^ (7 + s) = k := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hP, Nat.div_eq_of_lt hm, zero_add]
  rw [h1, h2, Nat.mod_eq_of_lt hk]
  ring

/-! ### Its bytes -/

/-- A 16-bit word's two bytes, u16 LE. -/
def inPair (v : ℕ) : List UInt8 := [(v % 256).toUInt8, (v / 256).toUInt8]

theorem foldl_inPairs (f : ℕ → ℕ) : ∀ (L : List ℕ) (A : ByteArray),
    (L.foldl (fun acc j => acc.push (f j % 256).toUInt8 |>.push (f j / 256).toUInt8) A).data.toList =
      A.data.toList ++ (L.map fun j => inPair (f j)).flatten
  | [], A => by simp
  | j :: L, A => by
      rw [List.foldl_cons, foldl_inPairs f L]
      simp [ByteArray.data_push, inPair]

theorem foldl_inSlots (f : ℕ → ℕ → ℕ) (J : List ℕ) : ∀ (K : List ℕ) (A : ByteArray),
    (K.foldl (fun acc k =>
        J.foldl (fun acc j => acc.push (f k j % 256).toUInt8 |>.push (f k j / 256).toUInt8) acc) A).data.toList =
      A.data.toList ++ (K.map fun k => (J.map fun j => inPair (f k j)).flatten).flatten
  | [], A => by simp
  | k :: K, A => by
      rw [List.foldl_cons, foldl_inSlots f J K, foldl_inPairs (f k)]
      simp

/-- **Region "in x"'s bytes**: every unit slot's leaves' words, u16 LE, slot by slot. -/
theorem inBytes_data (c : Circuit) (pi : HmIn.Ports) (pub : Public) (w s b : ℕ) :
    (inBytes c pi pub w s b).data.toList = ((List.range (2 ^ log2ceil c.unitRange.count)).map fun k =>
      ((List.range (8 * 2 ^ s)).map fun j => inPair (inVal c pi pub b k (inJ0 c w + j))).flatten).flatten := by
  unfold inBytes
  simp only [← Array.foldl_toList, Array.toList_range]
  rw [foldl_inSlots (fun k j => inVal c pi pub b k (inJ0 c w + j))]
  rfl

theorem length_inPairs (f : ℕ → ℕ) : ∀ n, ((List.range n).map fun j => inPair (f j)).flatten.length = 2 * n
  | 0 => rfl
  | n + 1 => by
      rw [List.range_succ, List.map_append, List.flatten_append, List.length_append, length_inPairs f n]
      simp [inPair]
      omega

theorem toUInt8_toNat (n : ℕ) : n.toUInt8.toNat = n % 2 ^ 8 := by
  rw [Nat.toUInt8, UInt8.toNat_ofNat']

/-- Bit `t % 8` of byte `t / 8` of a word's two bytes is bit `t` of the word. -/
theorem inPair_bit (v t : ℕ) (ht : t < 16) :
    ((inPair v).getD (t / 8) default).toNat.testBit (t % 8) = v.testBit t := by
  rcases Nat.lt_or_ge t 8 with h | h
  · rw [show t / 8 = 0 by omega, show t % 8 = t by omega, inPair, List.getD_cons_zero, toUInt8_toNat,
      show (256 : ℕ) = 2 ^ 8 from rfl]
    simp only [Nat.testBit_mod_two_pow, h, decide_true, Bool.true_and]
  · rw [show t / 8 = 0 + 1 by omega, show t % 8 = t - 8 by omega, inPair, List.getD_cons_succ,
      List.getD_cons_zero, toUInt8_toNat, show (256 : ℕ) = 2 ^ 8 from rfl]
    simp only [Nat.testBit_mod_two_pow, show t - 8 < 8 by omega, decide_true, Bool.true_and,
      Nat.testBit_div_two_pow, show t - 8 + 8 = t by omega]

/-- **Region "in x"'s bit at slot `k`'s column `16 j + t`** is bit `t` of the slot's leaf `inJ0 c w + j`. -/
theorem inBytes_bit {c : Circuit} {pi : HmIn.Ports} {pub : Public} {w s b k j t : ℕ}
    (hk : k < 2 ^ log2ceil c.unitRange.count) (hj : j < 8 * 2 ^ s) (ht : t < 16) :
    ((inBytes c pi pub w s b).get! ((k * 2 ^ (7 + s) + (16 * j + t)) / 8)).toNat.testBit
      ((k * 2 ^ (7 + s) + (16 * j + t)) % 8) = (inVal c pi pub b k (inJ0 c w + j)).testBit t := by
  have hX : k * 2 ^ (7 + s) = 128 * (k * 2 ^ s) := by rw [pow_add]; ring
  have hY : k * (2 * (8 * 2 ^ s)) = 16 * (k * 2 ^ s) := by ring
  have e1 : (k * 2 ^ (7 + s) + (16 * j + t)) / 8 = k * (2 * (8 * 2 ^ s)) + (j * 2 + t / 8) := by
    rw [hX, hY]; omega
  have e2 : (k * 2 ^ (7 + s) + (16 * j + t)) % 8 = t % 8 := by rw [hX]; omega
  have hslot : ((List.range (2 ^ log2ceil c.unitRange.count)).map fun k =>
      ((List.range (8 * 2 ^ s)).map fun j => inPair (inVal c pi pub b k (inJ0 c w + j))).flatten).getD k [] =
      ((List.range (8 * 2 ^ s)).map fun j => inPair (inVal c pi pub b k (inJ0 c w + j))).flatten := by
    rw [List.getD_eq_getElem _ _ (by simpa using hk)]
    simp
  have hpair : ((List.range (8 * 2 ^ s)).map fun j => inPair (inVal c pi pub b k (inJ0 c w + j))).getD j [] =
      inPair (inVal c pi pub b k (inJ0 c w + j)) := by
    rw [List.getD_eq_getElem _ _ (by simpa using hj)]
    simp
  rw [get!_eq_getD, inBytes_data, e1, e2,
    getD_flatten_const (B := 2 * (8 * 2 ^ s)) default k (j * 2 + t / 8) (fun l hl => by
      obtain ⟨k', -, rfl⟩ := List.mem_map.1 hl
      exact length_inPairs _ _) (by simpa using hk) (by omega), hslot,
    getD_flatten_const (B := 2) default j (t / 8) (fun l hl => by
      obtain ⟨j', -, rfl⟩ := List.mem_map.1 hl
      rfl) (by simpa using hj) (by omega), hpair]
  exact inPair_bit _ t ht

/-! ### What a satisfying witness holds there -/

theorem slot_lt_two_pow_log2ceil {k n : ℕ} (hk : k < n) : k < 2 ^ log2ceil n := by
  unfold log2ceil
  split
  · rw [pow_zero]; omega
  · have := @Nat.lt_log2_self (n - 1)
    omega

/-- **The public words a satisfying witness holds at the unit slots' public leaves** are the ones the statement pins:
for a statement with public input ports, at block `o`, unit slot `k` and a leaf `jj` whose word `inCol + jj / 8` lies in
in region `x`, `(w, s)` (at or past the leaf group's first word, `hcol`), the bit at column `16 jj + t` of the slot's leaf
group (`inAddr`) is bit `t` of the leaf's word there (`inVal`: the slot's instance's public word when the leaf is public).
It holds in the model's statement at a layout whose regions contain the executable's (`hregs`), once the statement's
regions contain `HmIn.regions` (`hri`; `SpecP.regions` has them). -/
theorem in_pinned {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    {pi : HmIn.Ports} {ri : Array Region} (hR : HmIn.regions st.c pi st.pub = .ok ri)
    (hri : ∀ r ∈ ri.toList, r ∈ st.regions.toList) (hne : pi.ports.isEmpty = false)
    {z : Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {x w s k jj t : ℕ} (hx : x < (HmIn.inRegions st.c pi).size) (hws : (HmIn.inRegions st.c pi)[x]! = (w, s))
    (hcol : inCol st.c ≤ w) (hlo : w ≤ inCol st.c + jj / 8) (hhi : inCol st.c + jj / 8 < w + 2 ^ s)
    (hk : k < st.c.unitRange.count) (ht : t < 16) :
    bit ((L.model regs).block z o (inBlockOr st.c.kLog (inAddr st.c k jj t))) =
      (inVal st.c pi st.pub o.val k jj).testBit t := by
  obtain ⟨j, rfl⟩ : ∃ j, jj = inJ0 st.c w + j := ⟨jj - inJ0 st.c w, by unfold inJ0; omega⟩
  have hj : j < 8 * 2 ^ s := by unfold inJ0 at hhi; omega
  obtain ⟨r, hr, hmk⟩ := regions_in hR hne hx hws
  have hmem : Refine.regionOf (L.model regs).kLog ((L.model regs).m - (L.model regs).kLog) r ∈
      (L.model regs).regions := hregs r (hri r hr)
  obtain ⟨hrf, -, -, hnd, -, hlt⟩ := mkRegion_spec hmk
  have hfree : r.free = inFree st.c s := by rw [hrf]
  have hbytes : r.bytes = inBytes st.c pi st.pub w s := by rw [hrf]
  have hsb : k < 2 ^ log2ceil st.c.unitRange.count := slot_lt_two_pow_log2ceil hk
  have hm : 16 * j + t < 2 ^ (7 + s) := by
    rw [pow_add, show (2 : ℕ) ^ 7 = 128 by norm_num]; omega
  have hlen : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.length =
      (7 + s) + log2ceil st.c.unitRange.count := by
    rw [← List.length_map (f := Fin.val), regionOf_free_map _ _ _ (fun b hb => hlt b (by
      rw [hfree] at hb; exact Array.mem_toList_iff.1 hb)), hfree]
    simp [inFree]
  have hi : k * 2 ^ (7 + s) + (16 * j + t) <
      2 ^ (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.length := by
    rw [hlen, Nat.pow_add (2 : ℕ) (7 + s)]
    calc k * 2 ^ (7 + s) + (16 * j + t) < k * 2 ^ (7 + s) + 2 ^ (7 + s) := by omega
      _ = (k + 1) * 2 ^ (7 + s) := by ring
      _ ≤ 2 ^ log2ceil st.c.unitRange.count * 2 ^ (7 + s) := Nat.mul_le_mul_right _ hsb
      _ = 2 ^ (7 + s) * 2 ^ log2ceil st.c.unitRange.count := by ring
  have hib : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).inBlock (k * 2 ^ (7 + s) + (16 * j + t)) =
      inAddr st.c k (inJ0 st.c w + j) t := by
    rw [inBlock_in (st.m - st.c.kLog) hmk hsb hm, inAddr, inJ0, show WORD = 128 from rfl]
    omega
  have hnd' : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.Nodup := by
    unfold Refine.regionOf
    rw [hfree]
    exact Refine.filterMap_fin_nodup _ _ hnd
  have ha : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).inBlock (k * 2 ^ (7 + s) + (16 * j + t)) <
      2 ^ st.c.kLog :=
    inBlock_lt _ hnd' _
  have hread := region_read hz hmem o _ hi ha
  have hfin : inBlockOr st.c.kLog (inAddr st.c k (inJ0 st.c w + j) t) = ⟨_, ha⟩ := by
    simp only [inBlockOr, ← hib, ha, ↓reduceDIte]
  rw [hfin]
  refine hread.trans ?_
  rw [hbytes]
  exact inBytes_bit hsb hj ht

/-- A public leaf's word in the in regions: the slot's instance's public word. -/
theorem inVal_public {c : Circuit} {pi : HmIn.Ports} {pub : Public} {b k jj i : ℕ} {q : ℤ}
    (hk : k < c.unitRange.count) (hi : c.instOf pub.n b (k / c.upv) = some i)
    (hq : (pi.leaves.getD (k % c.upv) #[]).getD jj (-1) = q) (hq0 : 0 ≤ q) :
    inVal c pi pub b k jj = pub.outputs.getD (i * pi.words + q.toNat) 0 := by
  simp only [inVal, hk, ↓reduceIte, hi]
  simp only [hq, ge_iff_le, hq0, ↓reduceIte]

end FlockSoundness.Discharge.Layout
