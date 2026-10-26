import Proofs.Flock.Bridge.Canon
import Proofs.Flock.Bridge.PortRoots
import Proofs.Flock.Claims.Bridge.Transports
import Proofs.Flock.Soundness.Discharge.PlainReg.Checked

/-!
# A shaped registered opening opens the deployed commitment

`commitDep` (the deployed trace commitment at a session's rows) under `rootsOf` (the registration's roots) opens a
claims position whose registered row sits at a `sha512/row/v2` port of `16 w` bits to the bit of any value that is
`RowShaped` of the row's width, under a salt of `saltLen` bytes, whose hm96 leaf is the registered leaf of that row
(`opens_commitDep`). The opening's data is the salt, the row's frame-v3 path in the verifier's own tree (`sibsV`) and the
value's bits (`dataOf`). The one more condition: the registration's roots separate differently named ports
(`PortRoots.RootFixesDomain`), since the verifier opens at the first port whose root the position names, and a port of
the same root and name opens the same leaves (`opensLeaf_portAt`).

The restriction is the deployed check's own: a misshaped value's leaf is the leaf of an opening `commitDep` passes only
by a SHA-512 collision (`leaf_misshaped_dep`), and so is a leaf under a salt of another length once the path has a level,
`2 ≤ nl` units (`leaf_salt`).

At the session's binding (`vbOf`), `transports_shaped` reads a registered opening of a wire with a row through
`opens_commitDep`; `transportsLiveCS` is `Bridge.TransportsLiveC`'s conclusion at canonical openings, `RowsV2` circuit
files and `PortRoots.RootFixesDomain` registrations, and `transportsLiveC` is `Bridge.TransportsLiveC`.
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
  simpa [portsOf_eq, portAt, Claims.Sampling.bs_bytes, List.getD_eq_getElem?_getD] using hfirst j hj

/-- **The port the verifier opens a row of port `q` at** has `q`'s root and `q`'s domain, when every earlier port of
`q`'s root has `q`'s domain. -/
theorem find?_portsOf_sep (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) {q : ℕ} (hq : q < c.ports.size)
    (hsep : ∀ q' < q, roots.getD q' [] = roots.getD q [] → domOf c q' = domOf c q) :
    ∃ q', (portsOf c nl roots).find? (fun p => decide (bs p.root = roots.getD q [])) = some (portAt c nl roots q') ∧
      roots.getD q' [] = roots.getD q [] ∧ domOf c q' = domOf c q := by
  have hqs : q < (portsOf c nl roots).size := by simp [portsOf_eq, hq]
  have hp : decide (bs (portsOf c nl roots)[q].root = roots.getD q []) = true := by
    simp [portsOf_eq, portAt, Claims.Sampling.bs_bytes]
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.mp
    ((Array.find?_isSome (p := fun p : Flock.Registered.Port => decide (bs p.root = roots.getD q []))).mpr
      ⟨_, Array.getElem_mem hqs, hp⟩)
  obtain ⟨hpx, j, hj, hxj, hbefore⟩ := (Array.find?_eq_some_iff_getElem).mp hx
  have hjq : j ≤ q := by
    by_contra hlt
    exact absurd (hbefore q (by omega)) (by rw [hp]; decide)
  have hjx : x = portAt c nl roots j := by
    rw [← hxj]
    simp [portsOf_eq]
  subst hjx
  have hroot : roots.getD j [] = roots.getD q [] := by
    simpa [portAt, Claims.Sampling.bs_bytes] using hpx
  refine ⟨j, hx, hroot, ?_⟩
  rcases Nat.lt_or_eq_of_le hjq with h | h
  · exact hsep j h hroot
  · rw [h]

/-- Two registered ports of one root and one domain open the same leaves. -/
theorem opensLeaf_portAt (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) {q q' : ℕ}
    (hroot : roots.getD q' [] = roots.getD q []) (hdom : domOf c q' = domOf c q) :
    Flock.Registered.opensLeaf (portAt c nl roots q') = Flock.Registered.opensLeaf (portAt c nl roots q) := by
  funext a b d
  unfold Flock.Registered.opensLeaf portAt
  dsimp only
  rw [hroot, hdom]

/-- **Under `RowsV2` a port is an unsegmented row/v2 row.** -/
theorem rowsV2_port {c : Flock.Circuit} (h : RowsV2 c) (q : Fin c.ports.size) :
    c.ports[q.val]!.segs.isSome = false ∧ c.ports[q.val]!.v2 = true := by
  have := (Array.all_eq_true.mp h) q.val q.isLt
  rw [getElem!_pos c.ports q.val q.isLt]
  cases hs : c.ports[q.val].segs <;> simp_all

end Layout

/-- **A shaped registered opening opens the deployed commitment.** At a claims position `i` whose registered row `r`
sits at port `r.port` of `16 w` bits, under roots that separate differently named ports (`PortRoots.RootFixesDomain`),
unit `r.leaf < nl`, and bit `r.place`: a value `v` `RowShaped` of the row's width under a salt of `saltLen` bytes, whose
hm96 leaf at a `sha512/row/v2` port is the registered leaf of row `(r.leaf, r.port)`, opens `commitDep` under `rootsOf`
at `i` to its bit `r.place`, with data `dataOf`: the salt, the row's path in the verifier's tree and the value's bits.
-/
theorem opens_commitDep (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (pub : Flock.Public) (nl : ℕ) {i : ℕ} {r : RowAt} (hr : rowOfPos C rows i = some r)
    (hq : r.port < c.ports.size) (hu : r.leaf < nl) (h16 : r.width % 16 = 0) (hpl : r.place < r.width)
    (hsep : PortRoots.RootFixesDomain c pub nl)
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
    unfold atOf
    rw [hr]
    show (rootsOf c pub nl)[r.port]?.map (fun rt => (rt, r.leaf)) = _
    rw [h1]
    rfl
  unfold commitDep
  rw [SamplingDeploy.opens_pathCommit]
  obtain ⟨q', hfind, hroot, hdom⟩ := find?_portsOf_sep c nl (rootsOf c pub nl) hq fun q' h => hsep r.port q' h hq
  refine ⟨_, _, hat, hfind, ?_⟩
  rw [opensLeaf_portAt c nl _ hroot hdom]
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

/-! ## The deployed check refuses other salts -/

/-- **hm96's extractor outputs a SHA-512 collision** whenever two different salts have one leaf, under any rows. -/
theorem hm96Pair_spec_salt (r y r' y' : List UInt8) (hy : y ≠ y') (h : H512 (hm96Input r y) = H512 (hm96Input r' y')) :
    Collides (hm96Pair r y r' y') := by
  by_cases hr : r = r'
  · subst hr
    unfold Collides hm96Pair
    by_cases ho : hm96Input r y = hm96Input r y'
    · simp only [ho, hy, ↓reduceIte]
      have hcs : K512.leafPrefix ++ commitString r y = K512.leafPrefix ++ commitString r y' := ho
      exact ⟨fun e => hy (List.append_cancel_left e),
        enc512.dig_inj (commitString_split (List.append_cancel_left hcs)).2⟩
    · simp only [ho, ↓reduceIte]
      exact ⟨ho, h⟩
  · exact hm96Pair_spec r y r' y' hr h

/-- **An opening the deployed check passes carries a full path**: `64 · log2ceil nl` bytes after its salt. -/
theorem pathOf_length_of_opens (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (nl : ℕ) (roots : List Msg) (o : Opening Bool) (h : (commitDep C rows c nl roots).Opens sha512Msg roots o = true) :
    (pathOf (Flock.log2ceil nl) o).length = 64 * Flock.log2ceil nl := by
  unfold commitDep at h
  rw [SamplingDeploy.opens_pathCommit] at h
  obtain ⟨ri, reg, -, hf, ho⟩ := h
  have hm := Array.mem_of_find?_eq_some hf
  rw [portsOf_eq, Array.mem_map] at hm
  obtain ⟨q, -, rfl⟩ := hm
  unfold Flock.Registered.opensLeaf at ho
  simp only [Bool.and_eq_true, decide_eq_true_eq] at ho
  have h2 := ho.1.2
  rw [size_bytes] at h2
  exact h2

/-- **The deployed check refuses a salt of another length**: once the path has a level (`2 ≤ nl` units), the leaf of
a game opening `(r, y)` whose salt isn't `saltLen` bytes is the leaf of an opening the deployed check passes only by a
SHA-512 collision. -/
theorem leaf_salt (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (nl : ℕ)
    (roots : List Msg) (o : Opening Bool) (h : (commitDep C rows c nl roots).Opens sha512Msg roots o = true)
    (hd : 1 ≤ Flock.log2ceil nl) {r y : Msg} (hy : y.length ≠ saltLen)
    (hl : H512 (hm96Input r y) =
      H512 (hm96Input (SamplingDeploy.row (placeOf C rows) (bitsOf (Flock.log2ceil nl)) o) (saltOf o))) :
    Collides (hm96Pair r y (SamplingDeploy.row (placeOf C rows) (bitsOf (Flock.log2ceil nl)) o) (saltOf o)) := by
  refine hm96Pair_spec_salt _ _ _ _ (fun e => hy ?_) hl
  rw [e]
  exact saltOf_length (pathOf_length_of_opens C rows c nl roots o h) hd

/-- **The deployed check refuses a misshaped value** (`SamplingDeploy.leaf_misshaped` at `commitDep`): at a position
whose row has `16 w` bits, a game opening of a value that isn't `RowShaped` of the row's width has the leaf of a
deployed opening's row only by a SHA-512 collision. -/
theorem leaf_misshaped_dep (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (d : ℕ)
    (o : Opening Bool) {r : RowAt} (hr : rowOfPos C rows o.pos = some r) (hpl : r.place < r.width)
    (h16 : r.width % 16 = 0) {W : ℕ} (hW : r.width ≤ W) {v : BitVal W} (hv : ¬ (RowShaped v ∧ v.1.val = r.width))
    (y : Msg)
    (h : H512 (hm96Input (rowEncP (FlockSoundness.Discharge.Hidden.rowPre true) v) y) =
      H512 (hm96Input (SamplingDeploy.row (placeOf C rows) (bitsOf d) o) (saltOf o))) :
    Collides (hm96Pair (rowEncP (FlockSoundness.Discharge.Hidden.rowPre true) v) y
      (SamplingDeploy.row (placeOf C rows) (bitsOf d) o) (saltOf o)) := by
  have hp := placeOf_eq hr hpl
  refine hm96Pair_spec _ _ _ _ (Ne.symm (SamplingDeploy.row_ne_misshaped (place := placeOf C rows) (bits := bitsOf d) (o := o) ?_ ?_ ?_)) h
  · rw [hp]; exact h16
  · rw [hp]; exact hW
  · rw [hp]; exact hv

/-! ## Wires and positions -/

theorem sum_filter_map {α : Type} (l : List α) (p : α → Bool) (f : α → ℕ) :
    ((l.filter p).map f).sum = (l.map fun x => if p x then f x else 0).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => by_cases hp : p a <;> simp [hp, ih]

theorem off_eq_sum_ite (C : FlockSoundness.Audit.Circuit) (g : ℕ) :
    off C g = ∑ h : Fin C.N, if h.val < g then gateSize (C.op h) else 0 := by
  rw [Fin.sum_univ_def, off, sum_filter_map]
  simp

theorem gateSize_pos {N : ℕ} (o : FlockSoundness.Audit.Op N) : 0 < gateSize o := by
  cases o <;> simp [gateSize]

theorem off_add_le_val (C : FlockSoundness.Audit.Circuit) {g g' : Fin C.N} (h : g.val < g'.val) :
    off C g.val + gateSize (C.op g) ≤ off C g'.val := by
  rw [off_eq_sum_ite, off_eq_sum_ite]
  calc (∑ h : Fin C.N, if h.val < g.val then gateSize (C.op h) else 0) + gateSize (C.op g)
      = ∑ h : Fin C.N, ((if h.val < g.val then gateSize (C.op h) else 0) +
          if h = g then gateSize (C.op g) else 0) := by
        rw [Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ g]
        simp
    _ ≤ ∑ h : Fin C.N, if h.val < g'.val then gateSize (C.op h) else 0 := by
        refine Finset.sum_le_sum fun x _ => ?_
        rcases eq_or_ne x g with rfl | hx
        · simp [h]
        · simp only [hx, ↓reduceIte, Nat.add_zero]
          split_ifs <;> omega

theorem posOf_lt (C : FlockSoundness.Audit.Circuit) {g g' : Fin C.N} (h : g.val < g'.val) : posOf C g < posOf C g' := by
  have h1 := off_add_le_val C h
  have h2 := gateSize_pos (C.op g)
  have h3 := gateSize_pos (C.op g')
  unfold posOf
  omega

theorem posOf_inj (C : FlockSoundness.Audit.Circuit) {g g' : Fin C.N} (h : posOf C g = posOf C g') : g = g' := by
  rcases lt_trichotomy g.val g'.val with hl | he | hl
  · exact absurd h (posOf_lt C hl).ne
  · exact Fin.ext he
  · exact absurd h (posOf_lt C hl).ne'

theorem wireOf_posOf (C : FlockSoundness.Audit.Circuit) (g : Fin C.N) : wireOf C (posOf C g) = some g := by
  unfold wireOf
  rw [List.find?_eq_some_iff_getElem]
  refine ⟨by simp, g.val, by simp, by simp, fun j hj => ?_⟩
  have hjN : j < C.N := by have := g.isLt; omega
  simp only [List.getElem_finRange, Bool.not_eq_true', decide_eq_false_iff_not]
  intro e
  have := congrArg Fin.val (posOf_inj C e)
  simp at this
  omega

theorem rowOfPos_posOf {C : FlockSoundness.Audit.Circuit} (rows : Fin C.N → Option RowAt) (g : Fin C.N) :
    rowOfPos C rows (posOf C g) = rows g := by
  simp only [rowOfPos, wireOf_posOf, Option.bind_some]

/-! ## The game's openings at a C-Flock session -/

section Game

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed FlockSoundness.Discharge.ZkHidden
  FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge FlockSoundness.Discharge.Composed

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Discharge.Layout.scopeOkH c = true)
  (hU : Built c.unit) {pub : Flock.Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

/-- **The game's value binding at a C-Flock session** (`SampledKSLive`'s `vb`). -/
noncomputable abbrev vbOf := (rsZHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub hpt).bindingZ
  (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt)

/-- The session's commit strings: unit `u`'s row of port `q`, or a single gate. -/
abbrev PosH (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) : Type :=
  (Fin n × Fin c.ports.size) ⊕ Fin ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).N

/-- The session's values: bit strings of at most the widest row's bits, with their width. -/
abbrev ValH (c : Flock.Circuit) : Type := BitVal (Finset.univ.sup (Discharge.Layout.portRowW c) + 1)

variable {dj y₀ ht hc hscope hU hpub hpt}
variable {g : Fin ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).N} {u : Fin n} {q : Fin c.ports.size}
  {k : Fin (Discharge.Layout.portRowW c q)}

theorem vbOf_pos_row (h : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g = some (u, ⟨q, k⟩)) :
    (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).pos g = (Sum.inl (u, q) : PosH c hU n) := by
  dsimp only [vbOf, HmRowsZX.bindingZ, LayoutZ.bindingZ, ZkHidden.rsZHJ, ZkHidden.layZHJ, LayoutZ.ofShadow,
    ZkHidden.layShHJ, HiddenExec.keyLayoutH, Discharge.Layout.Readers.layout, Discharge.Layout.ofBits,
    HiddenExec.keyReadersH]
  exact Discharge.Layout.ofRows_pos_some _ h

theorem vbOf_bit_row (h : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g = some (u, ⟨q, k⟩))
    (v : ValH c) : (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).bit v g = v.get k.val := by
  have hle : Discharge.Layout.portRowW c q ≤ Finset.univ.sup (Discharge.Layout.portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  dsimp only [vbOf, HmRowsZX.bindingZ, LayoutZ.bindingZ, ZkHidden.rsZHJ, ZkHidden.layZHJ, LayoutZ.ofShadow,
    ZkHidden.layShHJ, HiddenExec.keyLayoutH, Discharge.Layout.Readers.layout, Discharge.Layout.ofBits,
    HiddenExec.keyReadersH, Discharge.Layout.Readers.ofRows]
  split
  · rename_i heq
    rw [h] at heq
    simp only [Option.some.injEq, Prod.mk.injEq, Sigma.mk.inj_iff] at heq
    obtain ⟨rfl, rfl, heq⟩ := heq
    rw [heq_iff_eq] at heq
    subst heq
    simp only [FlockSoundness.Discharge.Layout.BitVal.get, lt_of_lt_of_le k.isLt hle, ↓reduceDIte]
    try rfl
  · rename_i heq
    rw [h] at heq
    exact absurd heq (by simp)

/-- The session's value binding's registered leaf at a row: hm96's leaf of the public file's `b ‖ c` there. -/
theorem vbOf_cs_inl (R : Reg) (u : Fin n) (q : Fin c.ports.size) :
    (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).cs R (Sum.inl (u, q) : PosH c hU n) =
      ((Sum.inl (u, q) : PosH c hU n), H512 (K512.leafPrefix ++ (pub.digest u.val q.val c.ports.size).data.toList)) :=
  rfl

/-- The session's value binding's leaf of an opening at a row: hm96's leaf of its port's bytes (`encPortH`). -/
theorem vbOf_com_inl (u : Fin n) (q : Fin c.ports.size) (v : ValH c) (y : List UInt8) :
    (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).com ((Sum.inl (u, q) : PosH c hU n), v, y) =
      ((Sum.inl (u, q) : PosH c hU n), H512 (hm96Input (HiddenExec.encPortH c.ports[q.val]! v) y)) :=
  rfl

theorem vbOf_com_fst (p : PosH c hU n) (v : ValH c) (y : List UInt8) :
    ((vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).com (p, v, y)).1 = p :=
  rfl

theorem vbOf_cs_fst (R : Reg) (p : PosH c hU n) :
    ((vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).cs R p).1 = p :=
  rfl

theorem vbOf_val (p : PosH c hU n) (v : ValH c) (y : List UInt8) :
    (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).val (p, v, y) = v :=
  rfl

theorem vbOf_cs_row (h : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g = some (u, ⟨q, k⟩)) (R : Reg) :
    (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).cs R ((vbOf dj y₀ ht hc hscope hU hpub hpt).pos g) =
      ((Sum.inl (u, q) : PosH c hU n), H512 (K512.leafPrefix ++ (pub.digest u.val q.val c.ports.size).data.toList)) := by
  rw [vbOf_pos_row h]
  exact vbOf_cs_inl R u q

/-- **A registered opening of a row's commit string is a leaf equation**: its hm96 leaf under the port's bytes
(`encPortH`) is the registered leaf of the public file's `b ‖ c` there. -/
theorem vbOf_leaf_row (h : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g = some (u, ⟨q, k⟩)) (R : Reg)
    (p : PosH c hU n) (v : ValH c) (y : List UInt8)
    (he : (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).com (p, v, y) =
      (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).cs R ((vbOf dj y₀ ht hc hscope hU hpub hpt).pos g)) :
    H512 (hm96Input (HiddenExec.encPortH c.ports[q.val]! v) y) =
      H512 (K512.leafPrefix ++ (pub.digest u.val q.val c.ports.size).data.toList) := by
  have he' := he.trans (vbOf_cs_row h R)
  rcases p with ⟨u', q'⟩ | g'
  · rw [vbOf_com_inl] at he'
    obtain ⟨h1, hl⟩ := Prod.mk.inj he'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Sum.inl.inj h1)
    exact hl
  · have h1 := congrArg Prod.fst he'
    rw [vbOf_com_fst] at h1
    cases h1

/-- **The deployed layout's rows** at a C-Flock session: `SampledKSLive`'s `rows`, the game's row map read off the
circuit file. -/
noncomputable def rowsOf (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) :
    Fin ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).N → Option RowAt := fun g =>
  (HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g).map fun r =>
    ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW c r.2.1, r.2.2.val⟩

/-- **A shaped registered opening of a row/v2 row opens the deployed commitment** (`TransportsLive` at the shaped
openings). In `SampledKSLive`'s setting, at a circuit file whose ports are unsegmented row/v2 rows (`RowsV2`) and roots
that separate differently named ports (`PortRoots.RootFixesDomain`), at a wire `g` held by a registered row `r`: an
opening `(p, v, y)` of `g`'s registered commit string at registration `R`, whose value `v` is `RowShaped` of the row's
width and whose salt `y` has `saltLen` bytes, opens `commitDep` under `rootsOf` at `g`'s claims position to `g`'s bit in
`v`. -/
theorem transports_shaped (R : Reg)
    (hv2 : RowsV2 c) (hsep : PortRoots.RootFixesDomain c pub n) {r : RowAt} (hr : rowsOf c hU n g = some r)
    (p : PosH c hU n) (v : ValH c) (y : List UInt8) (hv : RowShaped v) (hvw : v.1.val = r.width)
    (hy : y.length = saltLen)
    (he : (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).com (p, v, y) =
      (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).cs R ((vbOf dj y₀ ht hc hscope hU hpub hpt).pos g)) :
    ∃ d : Claims.Msg, (commitDep ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)) (rowsOf c hU n) c n
        (rootsOf c pub n)).Opens sha512Msg (rootsOf c pub n)
      ⟨posOf ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)) g,
        (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).bit
          ((vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).val (p, v, y)) g, d⟩ = true := by
  obtain ⟨⟨u, q, k⟩, hx, rfl⟩ := Option.map_eq_some_iff.mp hr
  have hle : Discharge.Layout.portRowW c q ≤ Finset.univ.sup (Discharge.Layout.portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  have hl := vbOf_leaf_row hx R p v y he
  have henc : HiddenExec.encPortH c.ports[q.val]! v = rowEncP (FlockSoundness.Discharge.Hidden.rowPre true) v := by
    unfold HiddenExec.encPortH
    rw [(rowsV2_port hv2 q).1, (rowsV2_port hv2 q).2]
    rfl
  rw [henc] at hl
  rw [vbOf_val, vbOf_bit_row hx v]
  exact ⟨_, opens_commitDep _ (rowsOf c hU n) c pub n (r := ⟨q.val, u.val, Discharge.Layout.portRowW c q, k.val⟩)
    (by rw [rowOfPos_posOf]; simp [rowsOf, hx]) q.isLt u.isLt (by simp [Discharge.Layout.portRowW]) k.isLt
    hsep hle hv hvw hy hl⟩

end Game

/-! ## `TransportsLive` at shaped openings -/

section Live

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **`TransportsLiveC` under roots that separate differently named ports.** `TransportsLiveC`'s setting (circuit files
whose ports are unsegmented row/v2 rows, `RowsV2`), commitment, roots and hash, when the registration's roots separate
differently named ports (`PortRoots.RootFixesDomain`): at every unit `u` and committed wire `g` of `u`, every opening
`o` of `g`'s registered commit string at the prover's registration that is canonical at `g` (`canonOpen`: `g` has a row,
the value is `RowShaped` of its width, the salt has `saltLen` bytes) opens `g`'s claims position under `Cd` to `o`'s bit
there, at SHA-512. The deployed check passes no other openings but by a SHA-512 collision (`leaf_misshaped_dep`,
`leaf_salt`). -/
def TransportsLiveCS : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hv2 : Bridge.RowsV2 D.unroll)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))),
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    Bridge.PortRoots.RootFixesDomain D.unroll pub n →
    ∀ u, ∀ g ∈ P.io u, ∀ o : vb.Opening, vb.com o = vb.cs (reg σ) (vb.pos g) →
      Bridge.canonOpen rows g (vb.val o) o.2.2 →
      ∃ d : Claims.Msg, Cd.Opens Claims.Sampling.sha512Msg roots ⟨Bridge.posOf Cc g, vb.bit (vb.val o) g, d⟩ = true

/-- **`TransportsLiveCS` holds** (`transports_shaped`). -/
theorem transportsLiveCS : TransportsLiveCS := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hv2 kd _ hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags hCust τ
    P vb σ Cc rows roots Cd hsep u g _ o he hcan
  obtain ⟨⟨r, hr, hv, hvw⟩, hy⟩ := hcan
  obtain ⟨p, v, y⟩ := o
  exact transports_shaped (dj := dj) (y₀ := y₀) (ht := ht) (hc := Flock.Desc.parse_flat hD) (hscope := hscope)
    (hU := hU) (hpub := hpub) (hpt := hpt) _ hv2 hsep hr p v y hv hvw hy he

/-- **`Bridge.TransportsLiveC` holds** (`transportsLiveCS`): its `PortRoots.RootFixesDomain` binder is
`TransportsLiveCS`'s one premise more, and it reads neither `dupName` nor that `g` isn't a constant wire. -/
theorem transportsLiveC : Bridge.TransportsLiveC := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hv2 _ hroots kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags hCust τ
    P vb σ Cc rows κ roots Cd u g hg _
  exact transportsLiveCS n Reg I ht D hD hU pub hpub hpt hscope hv2 kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hTags hCust
    τ hroots u g hg

end Live

end Flock.Bridge
