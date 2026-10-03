import FlockSoundness.Discharge.PrivateCircuit.InstanceDefs
import FlockSoundness.Discharge.ZkView.Leaf

/-!
# One registration: the commit strings determine the registered root and its openings

The privacy theorem sees a registration as its commit strings `b ‖ c` (`regView`, `hm96Reg`); the soundness theorem sees
the registered root the verifier holds, the registrant's rows and salts at its reads (`Registered.Reads`), and the paths
`progCommit` opens. Here they are one registration:

* `regPort pins v`, `regPath pins v i`: the frame-v3-sha512 tree over the leaves the verifier computes from the commit
  strings `v` (`leafH`, at positions `0 … L-1`, `Flock.Registered.rowLeaf`), its root under the port's public pins, and
  each position's siblings (`treeNode`). Positions past `L` hold the empty payload and are never opened.
* `viewDig`, `viewOpening`: the leaves and the registrant's opening of `regPort pins v`, as `progCommit` reads them.
* `hm96_leaf_view`: hm96's leaf of a row under a salt's bytes is `leafH` of the row's commit string at that salt
  (`Binding.hm96_treeLeaf`, `ZkView.leafSplit`), so a `Reads`' registrant leaf at a program site is the view's leaf there
  (`reads_leaf_view`).
* `opens_view`: the honest opening opens the root, at every position (`opensLeaf_view`).
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock FlockSoundness.Discharge.ZkView FlockSoundness.Registered Refine Binding

/-- The registration's view of a program `p` at salts `r`: its rows' commit strings (`hm96Reg` at the rows' digests). -/
def regView {Prg : Type} {L : ℕ} (enc : Prg → Fin L → List UInt8) (p : Prg) (r : Fin L → Salt) :
    Fin L → Bits × ByteArray :=
  hm96Reg (fun p i => rowBits (enc p i)) p r

/-- The nodes of level `d` of the frame-v3-sha512 tree over the leaves `lf`: level 0 the leaves, level `d + 1` the node of
two children (`Flock.Registered.node`). -/
def treeNode (dom : ByteArray) (lf : ℕ → ByteArray) : ℕ → ℕ → ByteArray
  | 0, j => lf j
  | d + 1, j => Flock.Registered.node dom d j (treeNode dom lf d (2 * j)) (treeNode dom lf d (2 * j + 1))

/-- The frame-v3 leaf at position `j` of the registered value whose commit strings are `v`: `rowLeaf` of the hm96 leaf
the verifier computes from `v j` (`leafH`), the empty payload past `L`. -/
def viewLeafAt {L : ℕ} (dom : ByteArray) (v : Fin L → Bits × ByteArray) (j : ℕ) : ByteArray :=
  Flock.Registered.rowLeaf dom j (if h : j < L then leafH (v ⟨j, h⟩) else ByteArray.empty)

/-- The sibling of index `idx`. -/
def sibIdx (idx : ℕ) : ℕ := if idx % 2 = 0 then idx + 1 else idx - 1

/-- **The registered root of the commit strings `v`**: the port's public pins (`pins`: name, value, row width, domain,
reads), with `L` leaves and the root of the frame-v3 tree over `v`'s leaves, of depth `log2ceil L`. -/
def regPort (pins : Flock.Registered.Port) {L : ℕ} (v : Fin L → Bits × ByteArray) : Flock.Registered.Port :=
  { pins with root := treeNode pins.domain (viewLeafAt pins.domain v) (log2ceil L) 0, leaves := L }

/-- **The registrant's path at position `i`**: the siblings of `i`'s ancestors, leaf level first, 64 bytes each. -/
def regPath (pins : Flock.Registered.Port) {L : ℕ} (v : Fin L → Bits × ByteArray) (i : Fin L) : ByteArray :=
  ⟨((List.range (log2ceil L)).flatMap fun d =>
    (treeNode pins.domain (viewLeafAt pins.domain v) d (sibIdx (i.val / 2 ^ d))).data.toList).toArray⟩

theorem leafH_size (b : Bits × ByteArray) : (leafH b).size = 64 := sha512_size _

/-- The hm96 leaves the verifier computes from the commit strings `v`. -/
def viewDig {L : ℕ} (v : Fin L → Bits × ByteArray) (i : Fin L) : Dig := ⟨leafH (v i), leafH_size _⟩

/-- **The registrant's opening** of `regPort pins v`: its leaves and paths. -/
def viewOpening (pins : Flock.Registered.Port) {L : ℕ} (v : Fin L → Bits × ByteArray) : ProgOpening L :=
  ⟨regPort pins v, viewDig v, regPath pins v⟩

/-! ## The tree's openings -/

open Flock.Registered (climbL sibs rowLeaf rowLeaf_eq node_eq)

theorem bytes_len (b : ByteArray) : b.data.toList.length = b.size := Array.length_toList

theorem lt_two_pow_log2ceil {i L : ℕ} (h : i < L) : i < 2 ^ log2ceil L := by
  unfold log2ceil
  split_ifs with hL
  · simp only [pow_zero]; omega
  · have := Nat.lt_log2_self (n := L - 1)
    omega

theorem treeNode_size (dom : ByteArray) {L : ℕ} (v : Fin L → Bits × ByteArray) :
    ∀ d j, (treeNode dom (viewLeafAt dom v) d j).size = 64
  | 0, j => by rw [treeNode, viewLeafAt, rowLeaf_eq]; exact sha512_size _
  | d + 1, j => by rw [treeNode, node_eq]; exact sha512_size _

/-- Climbing from the node of level `d` above position `i` through the siblings of levels `d … d + k - 1` reaches the
node of level `d + k` above `i`. -/
theorem climbL_treeNode (dom : ByteArray) (lf : ℕ → ByteArray) (i : ℕ) :
    ∀ k d, climbL dom (treeNode dom lf d (i / 2 ^ d)) (i / 2 ^ d) d
        ((List.range' d k).map fun e => treeNode dom lf e (sibIdx (i / 2 ^ e))) =
      treeNode dom lf (d + k) (i / 2 ^ (d + k))
  | 0, d => by simp only [List.range'_zero, List.map_nil, climbL, Nat.add_zero]
  | k + 1, d => by
    have hidx : i / 2 ^ d / 2 = i / 2 ^ (d + 1) := by rw [Nat.div_div_eq_div_mul, pow_succ]
    have hstep : (if i / 2 ^ d % 2 = 0 then
          Flock.Registered.node dom d (i / 2 ^ d / 2) (treeNode dom lf d (i / 2 ^ d)) (treeNode dom lf d (sibIdx (i / 2 ^ d)))
        else Flock.Registered.node dom d (i / 2 ^ d / 2) (treeNode dom lf d (sibIdx (i / 2 ^ d))) (treeNode dom lf d (i / 2 ^ d))) =
        treeNode dom lf (d + 1) (i / 2 ^ (d + 1)) := by
      rw [← hidx, treeNode]
      unfold sibIdx
      split_ifs with h
      · have h1 : 2 * (i / 2 ^ d / 2) = i / 2 ^ d := by omega
        have h2 : i / 2 ^ d + 1 = 2 * (i / 2 ^ d / 2) + 1 := by omega
        rw [h1, h2]
      · have h1 : 2 * (i / 2 ^ d / 2) + 1 = i / 2 ^ d := by omega
        have h2 : i / 2 ^ d - 1 = 2 * (i / 2 ^ d / 2) := by omega
        rw [h1, h2]
    rw [List.range'_succ, List.map_cons, climbL, hstep, hidx]
    have := climbL_treeNode dom lf i k (d + 1)
    rw [show d + 1 + k = d + (k + 1) by omega] at this
    exact this

/-- The `d`-th chunk of `w` elements of a concatenation of `D` chunks of `w` elements is the `d`-th. -/
theorem chunk_flatMap {α : Type} (f : ℕ → List α) (w : ℕ) (hf : ∀ d, (f d).length = w) :
    ∀ D, ((List.range D).flatMap f).length = w * D ∧
      ∀ d < D, (((List.range D).flatMap f).drop (w * d)).take w = f d
  | 0 => by simp
  | D + 1 => by
    obtain ⟨hlen, hget⟩ := chunk_flatMap f w hf D
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
    refine ⟨by rw [List.length_append, hlen, hf]; ring, fun d hd => ?_⟩
    rcases Nat.lt_succ_iff_lt_or_eq.mp hd with hd | rfl
    · have hle : w * d ≤ ((List.range D).flatMap f).length := by
        rw [hlen]; exact Nat.mul_le_mul_left _ hd.le
      rw [List.drop_append_of_le_length hle, List.take_append_of_le_length]
      · exact hget d hd
      · rw [List.length_drop, hlen, ← Nat.mul_sub]
        exact Nat.le_mul_of_pos_right _ (by omega)
    · rw [List.drop_left' hlen, List.take_of_length_le (le_of_eq (hf d))]

theorem regPath_size (pins : Flock.Registered.Port) {L : ℕ} (v : Fin L → Bits × ByteArray) (i : Fin L) :
    (regPath pins v i).size = 64 * log2ceil L := by
  rw [regPath, ByteArray.size, List.size_toArray]
  exact (chunk_flatMap _ 64 (fun d => by rw [bytes_len, treeNode_size]) _).1

theorem sibs_regPath (pins : Flock.Registered.Port) {L : ℕ} (v : Fin L → Bits × ByteArray) (i : Fin L) :
    sibs (regPath pins v i) (log2ceil L) =
      (List.range (log2ceil L)).map fun d => treeNode pins.domain (viewLeafAt pins.domain v) d (sibIdx (i.val / 2 ^ d)) := by
  have hc := chunk_flatMap (fun d => (treeNode pins.domain (viewLeafAt pins.domain v) d (sibIdx (i.val / 2 ^ d))).data.toList)
    64 (fun d => by rw [bytes_len, treeNode_size]) (log2ceil L)
  refine List.map_congr_left fun d hd => ?_
  rw [List.mem_range] at hd
  refine ByteArray.ext (Array.toList_inj.mp ?_)
  rw [ByteArray.data_extract, Array.toList_extract, regPath]
  simp only [List.extract_eq_take_drop, Nat.add_sub_cancel_left]
  exact hc.2 d hd

/-- **The registrant's opening opens the registered root** at every position: the frame-v3 leaf of the view's hm96 leaf
there climbs through `regPath` to `regPort`'s root. -/
theorem opensLeaf_view (pins : Flock.Registered.Port) {L : ℕ} (v : Fin L → Bits × ByteArray) (i : Fin L) :
    Flock.Registered.opensLeaf (regPort pins v) (leafH (v i)) i (regPath pins v i) = true := by
  have hroot : Flock.Registered.climb pins.domain (rowLeaf pins.domain i (leafH (v i))) i (log2ceil L) (regPath pins v i) =
      treeNode pins.domain (viewLeafAt pins.domain v) (log2ceil L) 0 := by
    rw [Flock.Registered.climb, sibs_regPath, List.range_eq_range']
    have h := climbL_treeNode pins.domain (viewLeafAt pins.domain v) i.val (log2ceil L) 0
    simp only [pow_zero, Nat.div_one, Nat.zero_add] at h
    rw [Nat.div_eq_of_lt (lt_two_pow_log2ceil i.isLt)] at h
    rw [← h, treeNode, viewLeafAt]
    simp only [i.isLt, ↓reduceDIte]
  simp only [Flock.Registered.opensLeaf, regPort, i.isLt, regPath_size, hroot, decide_true, Bool.and_self]

/-- **The registrant's opening opens `progCommit` at the view's leaves.** -/
theorem opens_view (pins : Flock.Registered.Port) {L : ℕ} (v : Fin L → Bits × ByteArray) :
    (progCommit L).Opens (regPort pins v) (viewDig v) (viewOpening pins v) :=
  ⟨rfl, rfl, fun i => opensLeaf_view pins v i⟩

/-! ## The soundness model's leaves are the view's -/

/-- **hm96's leaf of a row under a salt's bytes is `leafH` of the row's commit string** at that salt. -/
theorem hm96_leaf_view {Val : Type} (enc : Val → List UInt8) (hinj : Function.Injective enc) (x : Val) (y : Salt) :
    ((ExplicitLeaf.hm96 enc hinj).leaf x (saltBytes y).data.toList).1 = leafH (rowBits (enc x) + My y, saltHash y) := by
  rw [hm96_treeLeaf]
  exact leafSplit _ _ (sha512_size _)

section Sites

variable {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay} {rd : Registered.Reads lay rs}
  {rt : Flock.Registered.Port} {L : ℕ}

/-- **A `Reads`' registrant leaf at a program site is the view's leaf there**, when the registrant's row and salt there
are the program's row and the salt's bytes. -/
theorem reads_leaf_view {Prg : Type} (ps : ProgSites rd rt L) (row : Prg → Fin L → lay.Val) (p : Prg)
    (r : Fin L → Salt) (hreg : ∀ i, rd.wv (ps.site i) = row p i)
    (hsalt : ∀ i, rd.ws (ps.site i) = (saltBytes (r i)).data.toList) (i : Fin L) :
    (rd.leaf (ps.site i)).1 = leafH (regView (fun p i => rs.enc (row p i)) p r i) := by
  rw [Reads.leaf, hreg, hsalt, hm96_leaf_view]
  rfl

end Sites

end FlockSoundness.Discharge.PrivateCircuit
