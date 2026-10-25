import Proofs.Flock.Bridge.Statement
import Proofs.Flock.Claims.Bridge.Transports
import Proofs.Flock.Soundness.Discharge.PlainReg.Checked

/-!
# A shaped registered opening opens the deployed commitment

`commitDep` (the deployed trace commitment at a session's rows) under `rootsOf` (the registration's roots) opens a
claims position whose registered row sits at a `sha512/row/v2` port of `16 w` bits to the bit of any value that is
`RowShaped` of the row's width, under a salt of `saltLen` bytes, whose hm96 leaf is the registered leaf of that row
(`opens_commitDep`). The opening's data is the salt, the row's frame-v3 path in the verifier's own tree (`sibsV`) and the
value's bits (`dataOf`). The one more condition: no earlier port has the same registered root, since the verifier opens
at the first port whose root the position names.

The restriction is the deployed check's own (`SamplingDeploy.row_ne_misshaped`, `SamplingDeploy.leaf_misshaped`): a
misshaped value's leaf is a deployed opening's leaf only by a SHA-512 collision, and so is a leaf under a salt of another
length once the path is at least one level (`saltOf_length`).
-/

namespace Flock.Bridge

open Claims Claims.Sampling
open FlockSoundness.Refine FlockSoundness.Binding
open FlockSoundness.Discharge.Layout (BitVal RowShaped rowEncP)
open Properties.NetworkCertifier.Bytes (bs)

/-- A row's bits as the deployed data carries them, one byte each. -/
def bitMsg (b : List Bool) : Msg := b.map fun x => if x then 1 else 0

/-- **The data of a transported opening**: its salt, its frame-v3 path's siblings and its row's bits. -/
def dataOf (y : Msg) (s : List ByteArray) (b : List Bool) : Msg := y ++ SamplingDeploy.pathMsg s ++ bitMsg b

section Data

variable {y : Msg} {s : List ByteArray} {b : List Bool}

theorem saltOf_dataOf (hy : y.length = saltLen) (i : ℕ) (x : Bool) : saltOf ⟨i, x, dataOf y s b⟩ = y := by
  simp only [saltOf, dataOf, List.append_assoc]
  exact List.take_left' hy

theorem pathOf_dataOf (hy : y.length = saltLen) (hs : ∀ z ∈ s, z.size = 64) (i : ℕ) (x : Bool) :
    pathOf s.length ⟨i, x, dataOf y s b⟩ = SamplingDeploy.pathMsg s := by
  simp only [pathOf, dataOf, List.append_assoc]
  rw [List.drop_left' hy]
  exact List.take_left' (by rw [SamplingDeploy.pathMsg_length s hs])

theorem bitsOf_dataOf (hy : y.length = saltLen) (hs : ∀ z ∈ s, z.size = 64) (k : ℕ) :
    bitsOf s.length (dataOf y s b) k = b[k]?.getD false := by
  unfold bitsOf dataOf
  rw [List.drop_left' (by rw [List.length_append, hy, SamplingDeploy.pathMsg_length s hs])]
  simp only [bitMsg, List.getElem?_map]
  cases b[k]? with
  | none => rfl
  | some x => cases x <;> rfl

end Data

/-- The length of a deployed opening's salt: `saltLen` bytes once its path has a level, since the path follows it. -/
theorem saltOf_length {d : ℕ} {o : Opening Bool} (h : (pathOf d o).length = 64 * d) (hd : 1 ≤ d) :
    (saltOf o).length = saltLen := by
  simp only [pathOf, List.length_take, List.length_drop] at h
  simp only [saltOf, List.length_take]
  omega

section Layout

variable {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt}

theorem placeOf_eq {i : ℕ} {r : RowAt} (hr : rowOfPos C rows i = some r) (hpl : r.place < r.width) :
    placeOf C rows i = ⟨r.width, ⟨r.place, hpl⟩⟩ := by
  simp only [placeOf, hr, hpl, ↓reduceDIte]

theorem rootsOf_length (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) : (rootsOf c pub nl).length = c.ports.size := by
  simp [rootsOf]

/-- The leaves of port `q`'s registered tree. -/
def leavesOf (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : Array ByteArray :=
  (Array.range nl).map fun u => Flock.Registered.leafAs Flock.HmRow.ROW_SCHEMA_V2 (domOf c q) u (regLeafOf c pub q u)

theorem leavesOf_size (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : (leavesOf c pub nl q).size = nl := by
  simp [leavesOf]

theorem leavesOf_width (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) :
    ∀ x ∈ leavesOf c pub nl q, x.size = 64 := by
  intro x hx
  simp only [leavesOf, Array.mem_map] at hx
  obtain ⟨_, -, rfl⟩ := hx
  exact Flock.Sha512.hash_size _

theorem rootsOf_getD (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) {q : ℕ} (hq : q < c.ports.size) :
    (rootsOf c pub nl).getD q [] = bs (Flock.FV3.root (domOf c q) (leavesOf c pub nl q) Flock.Sha512.hash) := by
  simp [rootsOf, leavesOf, List.getD_eq_getElem?_getD, hq]

/-- Port `q` as the verifier registers it (`portsOf`'s `q`-th). -/
def portAt (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) (q : ℕ) : Flock.Registered.Port :=
  { name := (c.ports[q]?.map (·.name)).getD "", value := "", root := bytes (roots.getD q []), leaves := nl,
    words := none, wordBits := 0, domain := domOf c q, readsRel := #[], schema := Flock.HmRow.ROW_SCHEMA_V2 }

theorem portsOf_eq (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) :
    portsOf c nl roots = (Array.range c.ports.size).map (portAt c nl roots) := rfl

/-- **The port the verifier opens a row of port `q` at**: `q`'s own, when no earlier port has its root. -/
theorem find?_portsOf (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) {q : ℕ} (hq : q < c.ports.size)
    (hfirst : ∀ q' < q, roots.getD q' [] ≠ roots.getD q []) :
    (portsOf c nl roots).find? (fun p => decide (bs p.root = roots.getD q [])) = some (portAt c nl roots q) := by
  rw [Array.find?_eq_some_iff_getElem]
  refine ⟨by simp [portAt, Claims.Sampling.bs_bytes], q, by simp [portsOf_eq, hq], by simp [portsOf_eq], fun j hj => ?_⟩
  simp [portsOf_eq, portAt, Claims.Sampling.bs_bytes, hfirst j hj]

end Layout

/-- **A shaped registered opening opens the deployed commitment.** At a claims position `i` whose registered row `r`
sits at port `r.port` of `16 w` bits with no earlier port of the same root, unit `r.leaf < nl`, and bit `r.place`: a
value `v` `RowShaped` of the row's width under a salt of `saltLen` bytes, whose hm96 leaf at a `sha512/row/v2` port is
the registered leaf of row `(r.leaf, r.port)`, opens `commitDep` under `rootsOf` at `i` to its bit `r.place`, with data
`dataOf`: the salt, the row's path in the verifier's tree and the value's bits. -/
theorem opens_commitDep (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (pub : Flock.Public) (nl : ℕ) {i : ℕ} {r : RowAt} (hr : rowOfPos C rows i = some r)
    (hq : r.port < c.ports.size) (hu : r.leaf < nl) (h16 : r.width % 16 = 0) (hpl : r.place < r.width)
    (hfirst : ∀ q' < r.port, (rootsOf c pub nl).getD q' [] ≠ (rootsOf c pub nl).getD r.port [])
    {W : ℕ} (hW : r.width ≤ W) {v : BitVal W} (hv : RowShaped v) (hvw : v.1.val = r.width) {y : Msg}
    (hy : y.length = saltLen)
    (hleaf : H512 (hm96Input (rowEncP (FlockSoundness.Discharge.Hidden.rowPre true) v) y) =
      H512 (K512.leafPrefix ++ (pub.digest r.leaf r.port c.ports.size).data.toList)) :
    (commitDep C rows c nl (rootsOf c pub nl)).Opens sha512Msg (rootsOf c pub nl)
      ⟨i, v.get r.place, dataOf y (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesOf c pub nl r.port) r.leaf)
        (List.ofFn fun k : Fin r.width => v.get k.val)⟩ = true := by
  have hsz : ∀ z ∈ FlockC3.RegLevels.sibsV (domOf c r.port) (leavesOf c pub nl r.port) r.leaf, z.size = 64 :=
    FlockC3.RegLevels.sibsV_size _ _ (leavesOf_width c pub nl r.port) (by rw [leavesOf_size]; exact hu)
  have hsl : (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesOf c pub nl r.port) r.leaf).length =
      Flock.log2ceil nl := by
    rw [FlockC3.RegLevels.sibsV_length, leavesOf_size]
  have hpo : placeOf C rows i = ⟨r.width, ⟨r.place, hpl⟩⟩ := placeOf_eq hr hpl
  have hp1 : (placeOf C rows i).1 = r.width := by rw [hpo]
  have hp2 : (placeOf C rows i).2.val = r.place := by rw [hpo]
  have hval : SamplingDeploy.valOf (placeOf C rows) (bitsOf (Flock.log2ceil nl))
      ⟨i, v.get r.place, dataOf y (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesOf c pub nl r.port) r.leaf)
        (List.ofFn fun k : Fin r.width => v.get k.val)⟩ W = v := by
    refine SamplingDeploy.valOf_eq (by rw [hp1]; exact hW) hv (by rw [hp1]; exact hvw) fun j => ?_
    unfold SamplingDeploy.rowFn
    dsimp only
    split
    · rename_i h
      rw [h, hp2]
    · have hj : j.val < r.width := hp1 ▸ j.isLt
      rw [← hsl, bitsOf_dataOf hy hsz]
      simp [hj]
  have hrow : SamplingDeploy.row (placeOf C rows) (bitsOf (Flock.log2ceil nl))
      ⟨i, v.get r.place, dataOf y (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesOf c pub nl r.port) r.leaf)
        (List.ofFn fun k : Fin r.width => v.get k.val)⟩ =
      rowEncP (FlockSoundness.Discharge.Hidden.rowPre true) v := by
    rw [SamplingDeploy.row_eq_rowEncP (W := W) (by dsimp only; rw [hp1]; exact h16)
      (by dsimp only; rw [hp1]; exact hW), hval]
  have hlen : r.port < (rootsOf c pub nl).length := by rw [rootsOf_length]; exact hq
  have hat : atOf C rows (rootsOf c pub nl) i = some ((rootsOf c pub nl).getD r.port [], r.leaf) := by
    have h1 : (rootsOf c pub nl)[r.port]? = some ((rootsOf c pub nl).getD r.port []) := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlen]
      rfl
    simp [atOf, hr, h1]
  unfold commitDep
  rw [SamplingDeploy.opens_pathCommit]
  refine ⟨_, _, hat, find?_portsOf c nl (rootsOf c pub nl) hq hfirst, ?_⟩
  dsimp only
  rw [hrow, saltOf_dataOf hy, hleaf, SamplingDeploy.bytes_digBytes, FlockSoundness.Discharge.PlainReg.regLeaf_bytes,
    ← hsl, pathOf_dataOf hy hsz]
  apply SamplingDeploy.opensLeaf_tree _ (leavesOf c pub nl r.port)
  · rw [leavesOf_size]
    rfl
  · exact leavesOf_width c pub nl r.port
  · show Flock.FV3.root (domOf c r.port) (leavesOf c pub nl r.port) Flock.Sha512.hash =
      bytes ((rootsOf c pub nl).getD r.port [])
    rw [rootsOf_getD c pub nl hq, Claims.Sampling.bytes_bs]
  · exact hu
  · show (leavesOf c pub nl r.port)[r.leaf]! =
      Flock.Registered.leafAs Flock.HmRow.ROW_SCHEMA_V2 (domOf c r.port) r.leaf
        (Flock.Hm96.Default512.treeLeaf (pub.digest r.leaf r.port c.ports.size))
    rw [getElem!_pos _ _ (by rw [leavesOf_size]; exact hu)]
    simp [leavesOf, regLeafOf]

end Flock.Bridge
