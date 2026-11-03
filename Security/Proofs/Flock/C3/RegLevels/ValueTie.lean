import Proofs.Flock.C3.RegLevels.OpensRegs
import Proofs.Flock.VBridge.Family.Registrants
import Proofs.Flock.Recursive.ValueSrc

/-!
# The value tie: a registrant's row at a value the verifier registers itself is the verifier's, or SHA-512 collides

`checkValue` (`RecCoins.checkValue`) accepts a port the verifier registers itself (`coef`, `dirs`) only when the
registered root is the frame-v3-sha512 root of the verifier's own rows (`checkRows_root`, at any salt: `checkValue`'s
`verifierSalt`, `checkZero`'s `zeroSalt`, whose zero rows the same ties cover). A registrant's record opens
that root at the row's position (`ReadsX.committed`), so the registrant's row is the verifier's, or one of three
SHA-512 bindings breaks, each with both preimages in hand:

* the opening (frame-v3-sha512 against the root the verifier computes): `FV3.root` is a full tree (`lev`, `root_eq`), so
  every leaf climbs to it through its siblings in the verifier's tree (`sibsV`, `climb_sibsV`), and two climbs from two
  leaves to one root collide in a leaf's or a node's frame (`openPairV`, `climbPair_collides`);
* the leaf preimage (`Hm96.Default512.treeLeaf`, `Registered.rowLeaf`) and `RecCoins.commitRow`'s row digest and salt
  digest: one hm96 leaf over two rows' bytes is `hm96Pair`'s collision (`treeLeaf_commitRow`, `hm96Pair_spec`).

`valueColl` computes the pair from public data and the registrant's record, and `valueColl_spec`, `tie_read`,
`value_tie` are its spec, in knowledge form: the registrant's bits are the verifier's row's (`bits_of_rowInV`), or
`valueColl` outputs two different SHA-512 inputs with one digest. `dirs_tie` is `dirs_of_check` without its `hval`.
At the reads it takes `hone`, `hpos` and `hseg`, VBridge's `PartPortsOne`, `ReadsOwn` and `ReadsUnsegmented` at the
read, which acceptance does not give.

`tie_root` is `tie_read` at a root given as the verifier's tree over `rows` (`hroot`), in place of `checkValue`'s.
`salt_tie` is link (d) at R's level 0: a read whose leaf is a node's commit string under the held salts (`TopOf`) is
the node's row, or `saltFinder`'s pair collides (`Recursive/SaltFinder.lean`, where `rowInV` lives so the guarantee
reads it). `valueColl` and the tree it climbs (`lev`, `sibsV`, `openPairV`, `leavesV`) are defined in
`Recursive/ValueSrc.lean`, for the same reason.

`valueColl` evaluates SHA-512 at most `3n + 2^(D+1) + 2D + 7` times, `n` the rows and `D = log2ceil n`: the
verifier's tree once (`4n` for the leaves, row digest, salt digest, `treeLeaf` and `rowLeaf` each; `2^D - n` pads;
`2^D - 1` nodes), the two hm96 leaves (3 each), the two frame-v3 leaves (2) and the climb (2 a level, `2D`). A row of
`w` words hashes `128 · (1 + ⌈w/64⌉)` bytes; every other input is at most 3 blocks.
-/

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockSoundness.Refine (H512)
open FlockSoundness.Binding (hm96Input hm96Pair hm96Pair_spec hm96_treeLeaf)
open FlockSoundness.Discharge.HiddenExec (encPortH)

/-! ## `FV3.root`'s tree -/

theorem fold_lev (dom : ByteArray) (leaves : Array ByteArray) : ∀ k s,
    (List.range' s k).foldl (fun a d => up dom d a) (lev dom leaves s) = lev dom leaves (s + k)
  | 0, _ => rfl
  | k + 1, s => by
    rw [List.range'_succ, List.foldl_cons, show s + (k + 1) = s + 1 + k by omega]
    exact fold_lev dom leaves k (s + 1)

theorem log2ceil_pow (D : ℕ) : log2ceil (2 ^ D) = D := by
  unfold log2ceil
  rcases D with _ | D
  · simp
  · have h1 : ¬ 2 ^ (D + 1) ≤ 1 := by have := Nat.one_lt_two_pow (n := D + 1) (by omega); omega
    simp only [h1, ↓reduceIte]
    have hne : 2 ^ (D + 1) - 1 ≠ 0 := by omega
    have hp : 2 ^ D ≤ 2 ^ (D + 1) - 1 := by rw [Nat.pow_succ]; have := Nat.one_le_two_pow (n := D); omega
    have hl := (Nat.le_log2 hne).mpr hp
    have hu := (Nat.log2_lt hne).mpr (show 2 ^ (D + 1) - 1 < 2 ^ (D + 1) by omega)
    omega

theorem le_nextPow2 (n : ℕ) : n ≤ nextPow2 n := by
  unfold nextPow2 log2ceil
  split
  · simp; omega
  · have := @Nat.lt_log2_self (n - 1)
    rw [Nat.pow_succ]
    omega

/-- **`FV3.root` is the top of the tree** `lev`. -/
theorem root_eq (dom : ByteArray) (leaves : Array ByteArray) :
    FV3.root dom leaves Sha512.hash = (lev dom leaves (log2ceil leaves.size)).getD 0 ByteArray.empty := by
  unfold FV3.root
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind]
  have hs : [:log2ceil (nextPow2 leaves.size)].size = log2ceil leaves.size := by
    simp [Std.Legacy.Range.size, nextPow2, log2ceil_pow]
  rw [hs]
  have := congrArg (·.getD 0 ByteArray.empty) (fold_lev dom leaves (log2ceil leaves.size) 0)
  simp only [Nat.zero_add] at this
  exact this

theorem lev_size (dom : ByteArray) (leaves : Array ByteArray) :
    ∀ l ≤ log2ceil leaves.size, (lev dom leaves l).size = 2 ^ (log2ceil leaves.size - l)
  | 0, _ => by have := le_nextPow2 leaves.size; simp [lev, nextPow2] at *; omega
  | l + 1, h => by
    rw [lev, up, Array.size_map, Array.size_range, lev_size dom leaves l (by omega),
      show log2ceil leaves.size - l = (log2ceil leaves.size - (l + 1)) + 1 by omega, Nat.pow_succ]
    simp

theorem lev_width (dom : ByteArray) (leaves : Array ByteArray) (h : ∀ x ∈ leaves, x.size = 64) :
    ∀ l, ∀ x ∈ lev dom leaves l, x.size = 64
  | 0, x, hx => by
    simp only [lev, Array.mem_append, Array.mem_map] at hx
    rcases hx with hx | ⟨k, -, rfl⟩
    · exact h x hx
    · exact Sha512.hash_size _
  | l + 1, x, hx => by
    simp only [lev, up, Array.mem_map] at hx
    obtain ⟨i, -, rfl⟩ := hx
    exact Sha512.hash_size _

theorem sibsFrom_lev (dom : ByteArray) (leaves : Array ByteArray) (i : ℕ) : ∀ k s,
    sibsFrom dom i k s (lev dom leaves s) = (List.range' s k).map (sib dom leaves i)
  | 0, _ => rfl
  | k + 1, s => by
    rw [List.range'_succ, List.map_cons]
    exact congrArg (List.cons _) (sibsFrom_lev dom leaves i k (s + 1))

theorem sibsV_eq (dom : ByteArray) (leaves : Array ByteArray) (i : ℕ) :
    sibsV dom leaves i = (List.range (log2ceil leaves.size)).map (sib dom leaves i) := by
  rw [sibsV, sibsFrom_lev, List.range_eq_range']

theorem div_lt_lev {D s i : ℕ} (hs : s ≤ D) (hi : i < 2 ^ D) : i / 2 ^ s < 2 ^ (D - s) := by
  rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos s), ← Nat.pow_add, Nat.sub_add_cancel hs]
  exact hi

theorem climb_lev (dom : ByteArray) (leaves : Array ByteArray) {i : ℕ} (hi : i < 2 ^ log2ceil leaves.size) :
    ∀ k s, s + k = log2ceil leaves.size →
      Registered.climbL dom (lev dom leaves s)[i / 2 ^ s]! (i / 2 ^ s) s ((List.range' s k).map (sib dom leaves i)) =
        (lev dom leaves (s + k))[i / 2 ^ (s + k)]!
  | 0, s, _ => by simp [Registered.climbL]
  | k + 1, s, hk => by
    have hsz := lev_size dom leaves s (by omega)
    have hlt := div_lt_lev (show s ≤ log2ceil leaves.size by omega) hi
    have hD : 2 ^ (log2ceil leaves.size - s) = 2 * 2 ^ (log2ceil leaves.size - (s + 1)) := by
      rw [show log2ceil leaves.size - s = (log2ceil leaves.size - (s + 1)) + 1 by omega, Nat.pow_succ]; ring
    have hj : i / 2 ^ (s + 1) = i / 2 ^ s / 2 := by rw [Nat.pow_succ, Nat.div_div_eq_div_mul]
    have key : (if i / 2 ^ s % 2 = 0 then
          Registered.node dom s (i / 2 ^ s / 2) (lev dom leaves s)[i / 2 ^ s]! (sib dom leaves i s)
        else Registered.node dom s (i / 2 ^ s / 2) (sib dom leaves i s) (lev dom leaves s)[i / 2 ^ s]!) =
        (lev dom leaves (s + 1))[i / 2 ^ (s + 1)]! := by
      have hlt2 : i / 2 ^ s / 2 < (up dom s (lev dom leaves s)).size := by
        rw [up, Array.size_map, Array.size_range]; omega
      rw [lev, hj, getElem!_pos (up dom s (lev dom leaves s)) (i / 2 ^ s / 2) hlt2]
      simp only [up, Array.getElem_map, Array.getElem_range, sib, sibIdx]
      generalize i / 2 ^ s = x
      split
      · rw [show 2 * (x / 2) = x by omega]
      · rw [show 2 * (x / 2) = x - 1 by omega, show x - 1 + 1 = x by omega]
    rw [List.range'_succ, List.map_cons, Registered.climbL, key, ← hj, show s + (k + 1) = s + 1 + k by omega]
    exact climb_lev dom leaves hi k (s + 1) (by omega)

/-- **Leaf `i` climbs to `FV3.root`** through its siblings in the tree (`sibsV`). -/
theorem climb_sibsV (dom : ByteArray) (leaves : Array ByteArray) {i : ℕ} (hi : i < leaves.size) :
    Registered.climbL dom leaves[i]! i 0 (sibsV dom leaves i) = FV3.root dom leaves Sha512.hash := by
  have hp := le_nextPow2 leaves.size
  have h := climb_lev dom leaves (i := i) (by unfold nextPow2 at hp; omega) (log2ceil leaves.size) 0 (by omega)
  have h0 : (lev dom leaves 0)[i]! = leaves[i]! := by
    rw [getElem!_pos leaves i hi, lev, getElem!_pos _ i (by simp; omega), Array.getElem_append_left hi]
  simp only [Nat.pow_zero, Nat.div_one, Nat.zero_add, h0] at h
  rw [sibsV_eq, List.range_eq_range', h, root_eq,
    Nat.div_eq_of_lt (show i < 2 ^ log2ceil leaves.size by unfold nextPow2 at hp; omega)]
  have h1 := lev_size dom leaves _ le_rfl
  simp only [Nat.sub_self, Nat.pow_zero] at h1
  rw [getElem!_pos _ 0 (by omega), Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega)]
  rfl

theorem sibsV_length (dom : ByteArray) (leaves : Array ByteArray) (i : ℕ) :
    (sibsV dom leaves i).length = log2ceil leaves.size := by simp [sibsV_eq]

theorem sibsV_size (dom : ByteArray) (leaves : Array ByteArray) (h : ∀ x ∈ leaves, x.size = 64) {i : ℕ}
    (hi : i < leaves.size) : ∀ x ∈ sibsV dom leaves i, x.size = 64 := by
  intro x hx
  simp only [sibsV_eq, List.mem_map, List.mem_range] at hx
  obtain ⟨d, hd, rfl⟩ := hx
  have hp := le_nextPow2 leaves.size
  have hsz := lev_size dom leaves d hd.le
  have hlt := div_lt_lev hd.le (i := i) (by unfold nextPow2 at hp; omega)
  have hD : 2 ^ (log2ceil leaves.size - d) = 2 * 2 ^ (log2ceil leaves.size - (d + 1)) := by
    rw [show log2ceil leaves.size - d = (log2ceil leaves.size - (d + 1)) + 1 by omega, Nat.pow_succ]; ring
  have hlt2 : sibIdx (i / 2 ^ d) < (lev dom leaves d).size := by
    rw [hsz, hD]
    unfold sibIdx
    generalize i / 2 ^ d = x at *
    generalize 2 ^ (log2ceil leaves.size - (d + 1)) = g at *
    split <;> omega
  unfold sib
  rw [getElem!_pos (lev dom leaves d) _ hlt2]
  exact lev_width dom leaves h d _ (Array.getElem_mem _)

/-- Two different 64-byte payloads whose leaves at `pos` climb to one root: `openPairV` collides
(`Registered.opensLeaf_binding`'s argument, on sibling lists). -/
theorem openPairV_collides {dom : ByteArray} {pos : ℕ} {x y : ByteArray} {s t : List ByteArray} (hx : x.size = 64)
    (hy : y.size = 64) (hne : x ≠ y) (hl : s.length = t.length) (hs : ∀ z ∈ s, z.size = 64)
    (ht : ∀ z ∈ t, z.size = 64)
    (h : Registered.climbL dom (Registered.rowLeaf dom pos x) pos 0 s =
      Registered.climbL dom (Registered.rowLeaf dom pos y) pos 0 t) :
    Registered.Collides (openPairV dom pos x s y t) := by
  rw [Registered.rowLeaf_eq, Registered.rowLeaf_eq] at h
  unfold openPairV
  dsimp only
  split
  · rename_i hl'
    exact ⟨fun hh => hne (Registered.leafIn_inj (dom := dom) (pos := pos) (by rw [hx, hy]) hh), hl'⟩
  · rename_i hl'
    exact Registered.climbPair_collides dom _ _ _ _ pos 0 hl hs ht (Sha512.hash_size _) (Sha512.hash_size _) hl' h

/-! ## The verifier's rows -/

theorem leavesV_size (dom : ByteArray) (rows : Array (Array ℕ)) (salt : ℕ → ByteArray) :
    (leavesV dom rows salt).size = rows.size := by simp [leavesV]

theorem leavesV_width (dom : ByteArray) (rows : Array (Array ℕ)) (salt : ℕ → ByteArray) :
    ∀ x ∈ leavesV dom rows salt, x.size = 64 := by
  intro x hx
  simp only [leavesV, Array.mem_map] at hx
  obtain ⟨_, -, rfl⟩ := hx
  exact Sha512.hash_size _

theorem eq_of_not_bne {a b : ByteArray} (h : ¬ (a != b) = true) : a = b := by
  have hb : (a == b) = true := by simpa [bne] using h
  exact ByteArray.ext (beq_iff_eq.1 (hb : (a.data == b.data) = true))

/-- **What `checkRows` checked of the registered root**: the reads file's entry `reg` of the port, of `rows.size`
leaves, unit `i` reading row `i`, whose root is `FV3.root` of the verifier's rows under `salt` of META's
`program_sha512` (`checkValue`'s `verifierSalt`, `checkZero`'s `zeroSalt`). -/
theorem checkRows_root {salt : String → ℕ → ByteArray} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Registered.Port} {port value : String} {rows : Array (Array ℕ)} {what : ℕ → String}
    (h : RecCoins.checkRows salt c pub own port value rows what = .ok ()) :
    ∃ reg prog, own.find? (·.name == port) = some reg ∧ reg.leaves = rows.size ∧
      reg.reads = (Array.range rows.size).map some ∧ c.metaJ.getObjValAs? String "program_sha512" = .ok prog ∧
      FV3.root reg.domain (leavesV reg.domain rows (salt prog)) Sha512.hash = reg.root ∧
      reg.schema = HmRow.ROW_SCHEMA := by
  unfold RecCoins.checkRows at h
  dsimp only at h
  split at h
  · rename_i reg hf
    split at h
    · cases h
    split at h
    · cases h
    rename_i hl
    split at h
    · cases h
    rename_i hr
    split at h
    · rename_i prog hprog
      split at h
      · split at h
        · obtain ⟨_, -, h⟩ := Registered.except_bind_ok h
          obtain ⟨_, -, h⟩ := Registered.except_bind_ok h
          split at h
          · cases h
          rename_i hroot
          simp only [bne_iff_ne, ne_eq, Bool.or_eq_true, not_or, not_not] at hl hr
          refine ⟨reg, prog, hf, hl.1.1.1, hr, hprog, .trans ?_ (eq_of_not_bne hroot), hl.2⟩
          refine congrArg (FV3.root reg.domain · Sha512.hash) (Array.map_congr_left fun i hi => ?_)
          rw [getElem!_map_range _ (by simpa using hi)]
        · cases h
      · cases h
    · cases h
  · cases h

/-! ## What `commitRow` hashes -/

theorem foldl_bytes : ∀ (ws : List ℕ) (pre : ByteArray),
    (ws.foldl (fun b w => (b.push (w % 256).toUInt8).push (w / 256 % 256).toUInt8) pre).data.toList =
      pre.data.toList ++ wordBytes ws
  | [], pre => by simp [wordBytes]
  | w :: ws, pre => by
    rw [List.foldl_cons, foldl_bytes ws]
    simp [wordBytes, ByteArray.push]

theorem rowDigest_eq (row : Array ℕ) :
    Firewall.rowDigest (RecCoins.padTo row (Registered.rowWords row.size 16)) = Sha512.hash ⟨(rowInV row).toArray⟩ := by
  unfold Firewall.rowDigest rowInV
  refine congrArg Sha512.hash (ByteArray.ext ?_)
  apply Array.ext'
  rw [← Array.foldl_toList, foldl_bytes]
  rfl

/-- **`commitRow`'s tree leaf is hm96's leaf over `rowInV`** (`hm96_treeLeaf`). -/
theorem treeLeaf_commitRow (row : Array ℕ) (salt : ByteArray) :
    Hm96.Default512.treeLeaf (RecCoins.commitRow row salt) = (H512 (hm96Input (rowInV row) salt.data.toList)).1 := by
  rw [RecCoins.commitRow, rowDigest_eq]
  exact (hm96_treeLeaf (Val := List UInt8) id Function.injective_id (rowInV row) salt).symm

theorem topRows_of_size {top : ByteArray} (htop : top.size = 64) :
    Firewall.topRows top = [(Array.range 32).map (fun j => (top.get! (64 * 0 + 2 * j)).toNat +
      256 * (top.get! (64 * 0 + 2 * j + 1)).toNat) ++ Array.replicate 32 0] := by
  unfold Firewall.topRows
  rw [htop]
  rfl

theorem capRow_size {top : ByteArray} (htop : top.size = 64) : (capRow top).size = 64 := by
  rw [capRow, topRows_of_size htop]
  simp

/-- **A 64-byte node's commit string is `commitRow`'s of its row**: the row is 64 words, a whole block, so `commitRow`
pads nothing. -/
theorem commitRow_capRow {top : ByteArray} (htop : top.size = 64) (s : ByteArray) :
    RecCoins.commitRow (capRow top) s = Hm96.Default512.commitString (Firewall.rowDigest (capRow top)) s := by
  unfold RecCoins.commitRow RecCoins.padTo Registered.rowWords
  rw [capRow_size htop, show (64 * (16 / 16) + 63) / 64 * 64 - 64 = 0 by rfl, Array.replicate_zero,
    Array.append_empty]

theorem wordBytes_length (ws : List ℕ) : (wordBytes ws).length = 2 * ws.length := by
  induction ws with
  | nil => rfl
  | cons w ws ih => simp [wordBytes] at *; omega

theorem wordBytes_get : ∀ (ws : List ℕ) (j : ℕ) (hj : j < (wordBytes ws).length),
    (wordBytes ws)[j] = if j % 2 = 0 then (ws[j / 2]! % 256).toUInt8 else (ws[j / 2]! / 256 % 256).toUInt8
  | [], j, hj => by simp [wordBytes] at hj
  | w :: ws, 0, _ => by simp [wordBytes]
  | w :: ws, 1, _ => by simp [wordBytes]
  | w :: ws, j + 2, hj => by
    have hj' : j < (wordBytes ws).length := by simp [wordBytes_length] at hj ⊢; omega
    have e : (wordBytes (w :: ws))[j + 2] = (wordBytes ws)[j] := by simp [wordBytes]
    rw [e, wordBytes_get ws j hj', show (j + 2) / 2 = j / 2 + 1 by omega, show (j + 2) % 2 = j % 2 by omega]
    simp

theorem padTo_get! (row : Array ℕ) (n k : ℕ) : (RecCoins.padTo row n)[k]! = row[k]! := by
  unfold RecCoins.padTo
  by_cases hk : k < row.size
  · rw [getElem!_pos row k hk, getElem!_pos _ k (by simp; omega), Array.getElem_append_left hk]
  · rw [getElem!_neg row k hk]
    by_cases hk' : k < (row ++ Array.replicate (n - row.size) 0).size
    · rw [getElem!_pos _ k hk', Array.getElem_append_right (by omega)]
      simp
    · rw [getElem!_neg _ k hk']

theorem testBit_lo (w t : ℕ) (ht : t < 8) : ((w % 256).toUInt8).toNat.testBit t = w.testBit t := by
  rw [Nat.toUInt8, UInt8.toNat_ofNat', Nat.mod_mod, show 256 = 2 ^ 8 by rfl, Nat.testBit_mod_two_pow]
  simp [ht]

theorem testBit_hi (w t : ℕ) (ht : t < 8) : ((w / 256 % 256).toUInt8).toNat.testBit t = w.testBit (t + 8) := by
  rw [Nat.toUInt8, UInt8.toNat_ofNat', Nat.mod_mod, show 256 = 2 ^ 8 by rfl, Nat.testBit_mod_two_pow,
    Nat.testBit_div_two_pow]
  simp [ht]

/-- **The bits of a row whose bytes are the verifier's row's**: where a value's row bytes under a first block of 128
bytes (`rowEncP`) are what `commitRow` hashes for `row` (`rowInV`), its bit `b` is bit `b % 16` of the row's word
`b / 16`, zero past the row. -/
theorem bits_of_rowInV {pre : ℕ → ByteArray} (hl : ∀ w, (pre w).data.toList.length = 128) {W : ℕ}
    {v : Layout.BitVal W} {row : Array ℕ} (h : Layout.rowEncP pre v = rowInV row) (b : ℕ) (hb : b < W) :
    v.2 ⟨b, hb⟩ = (row[b / 16]!).testBit (b % 16) := by
  rw [← padTo_get! row (Registered.rowWords row.size 16)]
  have hv : Layout.RowShaped v := by
    by_contra hv
    have e := congrArg List.head? h
    rw [FlockVBridge.Family.rowEncP_head_other hv, rowInV, List.head?_append, FlockVBridge.Family.rowPrefix_head] at e
    cases e
  simp only [Layout.rowEncP, hv, ↓reduceIte, rowInV] at h
  obtain ⟨-, hbytes⟩ := List.append_inj h (by rw [hl, Layout.rowPrefix_length])
  generalize RecCoins.padTo row (Registered.rowWords row.size 16) = ws at hbytes ⊢
  have hlen := congrArg List.length hbytes
  simp only [Layout.rowBytes, List.length_ofFn, wordBytes_length, Array.length_toList] at hlen
  by_cases hiv : b < v.1.val
  · have hj : b / 8 < (Layout.rowBytes v).length := by
      simp only [Layout.rowBytes, List.length_ofFn]; have := hv.1; omega
    have e := List.getElem_of_eq hbytes hj
    simp only [Layout.rowBytes, List.getElem_ofFn] at e
    rw [wordBytes_get] at e
    have := Layout.byteOf_bit (fun t => v.get (8 * (b / 8) + t.val)) ⟨b % 8, Nat.mod_lt _ (by norm_num)⟩
    rw [e] at this
    simp only [Layout.BitVal.get, show 8 * (b / 8) + b % 8 = b by omega, hb, ↓reduceDIte] at this
    rw [← this, show b / 8 / 2 = b / 16 by omega, Array.getElem!_toList]
    split
    · rw [testBit_lo _ _ (Nat.mod_lt _ (by norm_num)), show b % 8 = b % 16 by omega]
    · rw [testBit_hi _ _ (Nat.mod_lt _ (by norm_num)), show b % 8 + 8 = b % 16 by omega]
  · rw [hv.2 ⟨b, hb⟩ (by simp only; omega), getElem!_neg ws (b / 16) (by have := hv.1; omega)]
    simp

/-! ## The extractor -/

theorem some_collides {o : Option (List UInt8 × List UInt8)} {p : List UInt8 × List UInt8} (h : o = some p)
    (hc : FlockSoundness.Binding.Collides p) : ∃ a b, o = some (a, b) ∧ a ≠ b ∧ H512 a = H512 b := by
  obtain ⟨a, b⟩ := p
  exact ⟨a, b, h, hc⟩

/-- **`valueColl`'s spec**: where the registered root is the verifier's (`hroot`), of `hm96-sha512/row/v1` rows (`hs`),
and the registrant's hm96 leaf of `r` under `y` opens it at `i` (`opensLeaf`), `r` is the verifier's row's bytes, or
`valueColl` collides. -/
theorem valueColl_spec {reg : Registered.Port} {rows : Array (Array ℕ)} {salt : ℕ → ByteArray} {i : ℕ}
    {r y : List UInt8} {path : ByteArray}
    (hroot : FV3.root reg.domain (leavesV reg.domain rows salt) Sha512.hash = reg.root) (hn : reg.leaves = rows.size)
    (hs : reg.schema = HmRow.ROW_SCHEMA) (ho : Registered.opensLeaf reg (H512 (hm96Input r y)).1 i path = true) :
    r = rowInV rows[i]! ∨ ∃ a b, valueColl reg rows salt i r y path = some (a, b) ∧ a ≠ b ∧ H512 a = H512 b := by
  rw [Registered.opensLeaf_row hs] at ho
  simp only [Bool.and_eq_true, decide_eq_true_eq] at ho
  obtain ⟨⟨hi, hp⟩, hc⟩ := ho
  rw [hn] at hi
  by_cases hr : r = rowInV rows[i]!
  · exact Or.inl hr
  refine Or.inr ?_
  unfold valueColl
  simp only [hr, ↓reduceIte]
  split
  · rename_i hh
    exact some_collides rfl (hm96Pair_spec _ _ _ _ hr hh)
  · rename_i hh
    have hiV : i < (leavesV reg.domain rows salt).size := by rw [leavesV_size]; exact hi
    have hleaf : (leavesV reg.domain rows salt)[i]! =
        Registered.rowLeaf reg.domain i (H512 (hm96Input (rowInV rows[i]!) (salt i).data.toList)).1 := by
      rw [leavesV, getElem!_map_range _ hi, treeLeaf_commitRow]
    have hV := climb_sibsV reg.domain (leavesV reg.domain rows salt) hiV
    rw [hroot, hleaf] at hV
    have hcl : Registered.climbL reg.domain (Registered.rowLeaf reg.domain i (H512 (hm96Input r y)).1) i 0
        (Registered.sibs path (log2ceil reg.leaves)) =
        Registered.climbL reg.domain
          (Registered.rowLeaf reg.domain i (H512 (hm96Input (rowInV rows[i]!) (salt i).data.toList)).1) i 0
          (sibsV reg.domain (leavesV reg.domain rows salt) i) := hc.trans hV.symm
    have hlen : (Registered.sibs path (log2ceil reg.leaves)).length =
        (sibsV reg.domain (leavesV reg.domain rows salt) i).length := by
      rw [Registered.sibs_length, sibsV_length, leavesV_size, hn]
    have hcol := openPairV_collides (H512 (hm96Input r y)).2
      (H512 (hm96Input (rowInV rows[i]!) (salt i).data.toList)).2 (fun e => hh (Subtype.ext e)) hlen
      (Registered.sibs_size hp) (sibsV_size _ _ (leavesV_width _ _ _) hiV) hcl
    exact some_collides rfl (FlockSoundness.Registered.collides_bytes _ hcol)

/-! ## At a registrant's record -/

section Reads

variable {C : FlockSoundness.Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : FlockSoundness.Binding.Layout P plan tab decode} {rs : FlockSoundness.Binding.HmRowsX lay}

/-- **A registered read of a value the verifier registers itself**: where `checkRows` accepted the port `port` at
the verifier's `rows` under `salt`, the read's root is the reads file's one entry of that name (`hone`), and that entry's
position for unit `u` is the read's (`hpos`), the read is of row `u`, and its row bytes are the verifier's row's
(`rowInV`), or `valueColl` collides. -/
theorem tie_read (rd : FlockSoundness.Registered.ReadsX lay rs) {p : lay.Pos} (hp : rd.isRead p)
    {salt : String → ℕ → ByteArray} {c : Flock.Circuit} {pub : Flock.Public} {own : Array Registered.Port}
    {port value : String} {rows : Array (Array ℕ)} {what : ℕ → String}
    (h : RecCoins.checkRows salt c pub own port value rows what = .ok ())
    {prog : String} (hprog : c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    (hone : ∀ reg ∈ own, reg.name = port → reg = rd.port p) {u : ℕ}
    (hpos : (rd.port p).reads[u]? = some (some (rd.readAt p))) :
    u < rows.size ∧ (rs.enc p (rd.wv p) = rowInV rows[u]! ∨
      ∃ x y, valueColl (rd.port p) rows (salt prog) u (rs.enc p (rd.wv p)) (rd.ws p)
        (rd.wpath p) = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) := by
  obtain ⟨reg, prog', hf, hn, hreads, hprog', hroot, hs⟩ := checkRows_root h
  obtain rfl : prog' = prog := Except.ok.inj (hprog'.symm.trans hprog)
  obtain rfl : reg = rd.port p := hone reg (Array.mem_of_find?_eq_some hf) (by simpa using Array.find?_some hf)
  rw [hreads] at hpos
  simp only [Array.getElem?_map, Array.getElem?_range] at hpos
  split at hpos
  · rename_i hu
    simp only [Option.map_some, Option.some.injEq] at hpos
    refine ⟨hu, valueColl_spec hroot hn hs ?_⟩
    rw [hpos]
    exact rd.committed p hp
  · cases hpos

/-- **A registered read of a root that is the verifier's tree over `rows`** (`tie_read` with the root in place of
`checkValue`): where the read's port's root is `FV3.root` of `rows` under `salt` (`hroot`) over `rows.size` leaves
(`hn`) of `hm96-sha512/row/v1` rows (`hs`), the read is of a row of `rows`, and its row bytes are that row's
(`rowInV`), or `valueColl` collides. -/
theorem tie_root (rd : FlockSoundness.Registered.ReadsX lay rs) {p : lay.Pos} (hp : rd.isRead p)
    {rows : Array (Array ℕ)} {salt : ℕ → ByteArray}
    (hroot : FV3.root (rd.port p).domain (leavesV (rd.port p).domain rows salt) Sha512.hash = (rd.port p).root)
    (hn : (rd.port p).leaves = rows.size) (hs : (rd.port p).schema = HmRow.ROW_SCHEMA) :
    rd.readAt p < rows.size ∧ (rs.enc p (rd.wv p) = rowInV rows[rd.readAt p]! ∨
      ∃ x y, valueColl (rd.port p) rows salt (rd.readAt p) (rs.enc p (rd.wv p)) (rd.ws p) (rd.wpath p) =
        some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) := by
  have ho := rd.committed p hp
  refine ⟨?_, valueColl_spec hroot hn hs ho⟩
  simp only [Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at ho
  rw [← hn]
  exact ho.1.1

/-- **A registered read of R's level-0 top at the held salts** (link (d) at level 0): where `bcs` are `cap`'s commit
strings under `salt`, the salts the prover's workers committed R's level-0 top rows under (`TopOf`,
`registrant-holds-openings`), and the read's leaf is `treeLeaf` of `bcs`'s string at its position (the table row the
verifier holds there), the read's row bytes are that node's row's (`rowInV`), or its hm96 pair against the node under
`salt` collides (`ReadsX.SaltClash`, `saltFinder`'s event). -/
theorem salt_tie (rd : FlockSoundness.Registered.ReadsX lay rs) {p : lay.Pos} (hp : rd.isRead p)
    {lv : FlockSoundness.Accounting.Level} {cap : FlockSoundness.Model.Cap FlockSoundness.Refine.Dig}
    {salt : ℕ → ByteArray} {bcs : Array ByteArray} (htop : TopOf lv cap salt bcs) (hi : rd.readAt p < bcs.size)
    (hleaf : (rd.leaf p).1 = Hm96.Default512.treeLeaf bcs[rd.readAt p]) :
    rs.enc p (rd.wv p) = rowInV (capRows lv cap)[rd.readAt p]! ∨ rd.SaltClash (capRows lv cap) salt := by
  have hrow : (capRows lv cap)[rd.readAt p]! = capRow (cap (rd.readAt p)).1 :=
    getElem!_map_range _ (by rw [← htop.1]; exact hi)
  have hbc : bcs[rd.readAt p] = RecCoins.commitRow (capRow (cap (rd.readAt p)).1) (salt (rd.readAt p)) :=
    (htop.2 _ hi).trans (commitRow_capRow (cap (rd.readAt p)).2 _).symm
  rw [hbc, treeLeaf_commitRow] at hleaf
  rw [hrow]
  by_cases hr : rs.enc p (rd.wv p) = rowInV (capRow (cap (rd.readAt p)).1)
  · exact Or.inl hr
  · refine Or.inr ⟨p, hp, ?_⟩
    rw [hrow]
    exact hm96Pair_spec _ _ _ _ hr (Subtype.ext hleaf)

end Reads

/-! ## At V*'s sessions -/

section Session

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **A registrant's row at a value the verifier registers itself is the verifier's, or SHA-512 collides**: at a
registered read `(u, q)` of a port without segments, where `checkRows` accepted the port at the verifier's `rows` under
`salt` (at any public file: the tie reads only the registered root),
the read is of row `u` and the registrant's bit `b` is bit `b % 16` of the row's word `b / 16` (R's `hval` at the read),
or `valueColl` outputs two different SHA-512 inputs with one digest. -/
theorem value_tie {salt : String → ℕ → ByteArray} {port value : String} {rows : Array (Array ℕ)} {what : ℕ → String}
    {pub : Flock.Public} (h : RecCoins.checkRows salt c pub zo.own port value rows what = .ok ()) {prog : String}
    (hprog : c.metaJ.getObjValAs? String "program_sha512" = .ok prog) {u : Fin nv} {q : Fin c.ports.size}
    (hp : zo.rd.isRead (.inl (u, q))) (hone : ∀ reg ∈ zo.own, reg.name = port → reg = zo.rd.port (.inl (u, q)))
    (hpos : (zo.rd.port (.inl (u, q))).reads[u.val]? = some (some (zo.rd.readAt (.inl (u, q)))))
    (hseg : c.ports[q.val]!.segs = none) :
    u.val < rows.size ∧
      ((∀ (b : ℕ) (hb : b < Layout.portRowW c q),
        (FlockVBridge.Decode.regVal zo (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          (rows[u.val]!)[b / 16]!.testBit (b % 16)) ∨
      ∃ x y, valueColl (zo.rd.port (.inl (u, q))) rows (salt prog) u.val
          (encPortH c.ports[q.val]! (FlockVBridge.Decode.regVal zo (.inl (u, q)))) (zo.rd.ws (.inl (u, q)))
          (zo.rd.wpath (.inl (u, q))) = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) := by
  obtain ⟨hu, hor⟩ := tie_read zo.rd hp h hprog hone hpos
  refine ⟨hu, hor.imp (fun he b hb => ?_) id⟩
  have he' : Layout.rowEncP (Hidden.rowPre c.ports[q.val]!.v2) (FlockVBridge.Decode.regVal zo (.inl (u, q))) =
      rowInV rows[u.val]! := (congrFun (FlockVBridge.Family.encPortH_of_none hseg) _).symm.trans he
  exact bits_of_rowInV (FlockVBridge.Family.rowPre_length _) he' b _

/-- **An accepted `rec-open-stacked` unit's `dirs` are its openings' positions' bits, or SHA-512 collides**
(`dirs_of_check` without `hval`): at the level `ℓ` the unit names and a registered read `(u, q)` of its `dirs` port
(without segments), the read is of opening `u`, and the registrant's bit `b < H` is bit `b` of the verifier's
position (`dirsRow pos H`'s), at every valid `H` (`H ≤ 32`), or `valueColl` collides. -/
theorem dirs_tie {c3 : RecCoins.C3} {LANES H : ℕ} (h : RecCoins.checkStackedLevel c3 c zo.pub zo.own LANES H = .ok ())
    {o : RecCoins.RegisteredOpening} (hop : RecCoins.opening c3 c.unit.name = .ok o) {ℓ : ℕ} (hℓ : ℓ < o.sh.sched.levels.size)
    (hL : 2 ^ (o.sh.sched.levels[ℓ]!).lanes = LANES)
    (hH : (o.sh.sched.levels[ℓ]!).d - (o.sh.sched.levels[ℓ]!).cap = H) (h32 : H ≤ 32) {prog : String}
    (hprog : c.metaJ.getObjValAs? String "program_sha512" = .ok prog) {u : Fin nv} {q : Fin c.ports.size}
    (hp : zo.rd.isRead (.inl (u, q))) (hone : ∀ reg ∈ zo.own, reg.name = "dirs" → reg = zo.rd.port (.inl (u, q)))
    (hpos : (zo.rd.port (.inl (u, q))).reads[u.val]? = some (some (zo.rd.readAt (.inl (u, q)))))
    (hseg : c.ports[q.val]!.segs = none) :
    u.val < (o.sh.sched.levels[ℓ]!).q ∧
      ((∀ b < H, ∀ (hb : b < Layout.portRowW c q),
        (FlockVBridge.Decode.regVal zo (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          ((o.lg.conss[ℓ]!).pos[u.val]!).testBit b) ∨
      ∃ x y, valueColl (zo.rd.port (.inl (u, q)))
          ((Array.range (o.sh.sched.levels[ℓ]!).q).map fun j => RecCoins.dirsRow (o.lg.conss[ℓ]!).pos[j]! H)
          (RecCoins.verifierSalt prog RecCoins.DIRS_VALUE) u.val
          (encPortH c.ports[q.val]! (FlockVBridge.Decode.regVal zo (.inl (u, q)))) (zo.rd.ws (.inl (u, q)))
          (zo.rd.wpath (.inl (u, q))) = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) := by
  obtain ⟨o', l, what, hop', hu, hdv⟩ := dirsLevel_of_check h
  have ho : o' = o := Except.ok.inj (hop'.symm.trans hop)
  subst ho
  have hl : ℓ = l := hu ℓ hℓ hL hH
  subst hl
  obtain ⟨hlt, hor⟩ := value_tie zo (salt := fun p => RecCoins.verifierSalt p RecCoins.DIRS_VALUE) hdv hprog hp hone
    hpos hseg
  rw [Array.size_map, Array.size_range] at hlt
  refine ⟨hlt, hor.imp (fun he b hbH hb => ?_) id⟩
  rw [he b hb, getElem!_map_range _ hlt, dirsRow_testBit _ _ hbH (by omega)]

end Session

end FlockC3.RegLevels
