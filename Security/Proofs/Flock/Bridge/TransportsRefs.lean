import Proofs.Flock.Bridge.LeafMap
import Proofs.Flock.Bridge.Transports
import Proofs.Flock.Soundness.Discharge.Layout.SharedRows
import Proofs.Flock.Soundness.Discharge.Hidden.Public

set_option Elab.async false

/-!
# A shaped registered opening opens the deployed commitment at a leaf map

`Transports.lean` at a leaf map (`LeafMap.lean`): `opens_commitDepL` opens `commitDepL` under `rootsOfL` at a row's
position, with data `dataOfL`: the salt, the row's frame-v3 path in its port's own tree (`sibsV` over `leavesL`), zero
bytes up to `L.d` levels, and the value's bits. At the session's binding with or without shared rows (`vbS`, at
`ZkReg.rsZHJS`), `transports_shapedS` reads a registered opening through it, and `transportsLiveCR` is
`Bridge.TransportsLiveCR`.

What the verifier checks, at every public file it accepts (`ref_lt_cnt`): a unit's ref is below its port's count
(`HmRow.loadPublic`'s refs check, and its refusal of an empty table, `sharedRows_spec`, for a unit past the file's
instances). Under `dupName` the ports' names, so their domains, differ, and `RootFixesDomainL` makes their roots differ
(`rootsDistinctL_of_fixes`): the verifier opens a row at its own port, of its own leaf count.

`transportsLiveC_of_R`: at `pub.tables = none` the leaf map is `LeafMap.ident` and `TransportsLiveCR` is
`TransportsLiveC`.
-/

namespace Flock.Bridge

open Claims Claims.Sampling
open FlockSoundness.Refine FlockSoundness.Binding
open FlockSoundness.Discharge.Layout (BitVal RowShaped)
open Properties.NetworkCertifier.Bytes (bs)

/-! ## The data at a leaf map -/

/-- **The data of a transported opening at a leaf map**: its salt, its frame-v3 path's siblings, `k` zero levels and its
row's bits. -/
def dataOfL (y : Msg) (s : List ByteArray) (k : ℕ) (b : List Bool) : Msg :=
  y ++ SamplingDeploy.pathMsg s ++ List.replicate (64 * k) 0 ++ bitMsg b

section Data

variable {y : Msg} {s : List ByteArray} {k : ℕ} {b : List Bool}

theorem saltOf_dataOfL (hy : y.length = saltLen) (i : ℕ) (x : Bool) : saltOf ⟨i, x, dataOfL y s k b⟩ = y := by
  simp only [saltOf, dataOfL, List.append_assoc]
  exact List.take_left' hy

theorem pathOf_dataOfL (hy : y.length = saltLen) (hs : ∀ z ∈ s, z.size = 64) (i : ℕ) (x : Bool) :
    pathOf s.length ⟨i, x, dataOfL y s k b⟩ = SamplingDeploy.pathMsg s := by
  simp only [pathOf, dataOfL, List.append_assoc]
  rw [List.drop_left' hy]
  exact List.take_left' (by rw [SamplingDeploy.pathMsg_length s hs])

theorem bitsOf_dataOfL (hy : y.length = saltLen) (hs : ∀ z ∈ s, z.size = 64) {d : ℕ} (hd : s.length + k = d)
    (j : ℕ) : bitsOf d (dataOfL y s k b) j = b[j]?.getD false := by
  unfold bitsOf dataOfL
  rw [List.drop_left' (by
    rw [List.length_append, List.length_append, hy, SamplingDeploy.pathMsg_length s hs, List.length_replicate, ← hd]
    ring)]
  simp only [bitMsg, List.getElem?_map]
  cases b[j]? with
  | none => rfl
  | some x => cases x <;> rfl

end Data

/-! ## The ports at a leaf map -/

section Layout

variable {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt}

theorem rootsOfL_length (c : Flock.Circuit) (L : LeafMap) : (rootsOfL c L).length = c.ports.size := by
  simp [rootsOfL]

theorem leavesL_size (c : Flock.Circuit) (L : LeafMap) (q : ℕ) : (leavesL c L q).size = L.cnt q := by
  simp [leavesL]

theorem leavesL_width (c : Flock.Circuit) (L : LeafMap) (q : ℕ) : ∀ x ∈ leavesL c L q, x.size = 64 := by
  intro x hx
  simp only [leavesL, Array.mem_map] at hx
  obtain ⟨_, -, rfl⟩ := hx
  exact Flock.Sha512.hash_size _

theorem rootsOfL_getD (c : Flock.Circuit) (L : LeafMap) {q : ℕ} (hq : q < c.ports.size) :
    (rootsOfL c L).getD q [] = bs (Flock.FV3.root (domOf c q) (leavesL c L q) Flock.Sha512.hash) := by
  simp [rootsOfL, List.getD_eq_getElem?_getD, hq]

/-- Port `q` as the verifier registers it at a leaf map (`portsOfL`'s `q`-th). -/
def portAtL (c : Flock.Circuit) (L : LeafMap) (roots : List Msg) (q : ℕ) : Flock.Registered.Port :=
  { name := (c.ports[q]?.map (·.name)).getD "", value := "", root := bytes (roots.getD q []), leaves := L.cnt q,
    words := none, wordBits := 0, domain := domOf c q, readsRel := #[], schema := schemaOf c q }

theorem portsOfL_eq (c : Flock.Circuit) (L : LeafMap) (roots : List Msg) :
    portsOfL c L roots = (Array.range c.ports.size).map (portAtL c L roots) := rfl

/-- **The port the verifier opens a row of port `q` at**: `q`'s own, when no earlier port has its root. -/
theorem find?_portsOfL (c : Flock.Circuit) (L : LeafMap) (roots : List Msg) {q : ℕ} (hq : q < c.ports.size)
    (hfirst : ∀ q' < q, roots.getD q' [] ≠ roots.getD q []) :
    (portsOfL c L roots).find? (fun p => decide (bs p.root = roots.getD q [])) = some (portAtL c L roots q) := by
  rw [Array.find?_eq_some_iff_getElem]
  refine ⟨by simp [portAtL, Claims.Sampling.bs_bytes], q, by simp [portsOfL_eq, hq], by simp [portsOfL_eq],
    fun j hj => ?_⟩
  simpa [portsOfL_eq, portAtL, Claims.Sampling.bs_bytes, List.getD_eq_getElem?_getD] using hfirst j hj

theorem portIdxOf_eq {i : ℕ} {r : RowAt} (hr : rowOfPos C rows i = some r) : portIdxOf C rows i = r.port := by
  simp [portIdxOf, hr]

/-- **Distinct roots at a leaf map**: under `dupName` each port has its own name, so its own domain, and
`RootFixesDomainL` gives equal roots one domain. -/
theorem rootsDistinctL_of_fixes {c : Flock.Circuit} {L : LeafMap}
    (hn : Flock.Registered.dupName c = none) (hf : RootFixesDomainL c L) :
    ∀ q q' : ℕ, q' < q → q < c.ports.size → (rootsOfL c L).getD q' [] ≠ (rootsOfL c L).getD q [] := by
  intro q q' hlt hq he
  have hd := hf q q' hlt hq he
  have hname : (c.ports[q']?.map (·.name)).getD "" = (c.ports[q]?.map (·.name)).getD "" :=
    String.toByteArray_inj.1 hd
  have h1 : c.ports[(⟨q', by omega⟩ : Fin c.ports.size)].name = c.ports[(⟨q, hq⟩ : Fin c.ports.size)].name := by
    simpa [show q' < c.ports.size by omega, hq] using hname
  have := congrArg Fin.val (FlockSoundness.Discharge.ZkReg.names_once_of_dupName hn _ _ h1)
  simp at this
  omega

end Layout

/-- **A shaped registered opening opens the deployed commitment at a leaf map.** `opens_commitDep` at `L`: at a claims
position `i` whose registered row `r` sits at port `r.port`, of the port's bits (`16 w` of them), at leaf
`r.leaf < L.cnt r.port`, under roots no earlier port shares, a value `v` `RowShaped` of the row's width under a salt of
`saltLen` bytes, whose hm96 leaf at its port's bytes (`encPortH`, the port's own format) is the registered leaf of
`L.dig r.port r.leaf`, opens `commitDepL` under `rootsOfL` at `i` to its bit `r.place`. -/
theorem opens_commitDepL (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (L : LeafMap) {i : ℕ} {r : RowAt} (hr : rowOfPos C rows i = some r)
    (hq : r.port < c.ports.size) (hu : r.leaf < L.cnt r.port) (hd : Flock.log2ceil (L.cnt r.port) ≤ L.d)
    (hb : r.width = c.ports[r.port]!.bits) (h16 : r.width % 16 = 0) (hpl : r.place < r.width)
    (hfirst : ∀ q' < r.port, (rootsOfL c L).getD q' [] ≠ (rootsOfL c L).getD r.port [])
    {W : ℕ} (hW : r.width ≤ W) {v : BitVal W} (hv : RowShaped v) (hvw : v.1.val = r.width) {y : Msg}
    (hy : y.length = saltLen)
    (hleaf : H512 (hm96Input (FlockSoundness.Discharge.HiddenExec.encPortH c.ports[r.port]! v) y) =
      H512 (K512.leafPrefix ++ (L.dig r.port r.leaf).data.toList)) :
    (commitDepL C rows c L (rootsOfL c L)).Opens sha512Msg (rootsOfL c L)
      ⟨i, v.get r.place, dataOfL y (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesL c L r.port) r.leaf)
        (L.d - Flock.log2ceil (L.cnt r.port)) (List.ofFn fun k : Fin r.width => v.get k.val)⟩ = true := by
  have hsz : ∀ z ∈ FlockC3.RegLevels.sibsV (domOf c r.port) (leavesL c L r.port) r.leaf, z.size = 64 :=
    FlockC3.RegLevels.sibsV_size _ _ (leavesL_width c L r.port) (by rw [leavesL_size]; exact hu)
  have hsl : (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesL c L r.port) r.leaf).length =
      Flock.log2ceil (L.cnt r.port) := by
    rw [FlockC3.RegLevels.sibsV_length, leavesL_size]
  have hdd : (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesL c L r.port) r.leaf).length +
      (L.d - Flock.log2ceil (L.cnt r.port)) = L.d := by
    rw [hsl]; omega
  have hpo : placeOf C rows i = ⟨r.width, ⟨r.place, hpl⟩⟩ := placeOf_eq hr hpl
  have hp1 : (placeOf C rows i).1 = r.width := by rw [hpo]
  have hp2 : (placeOf C rows i).2.val = r.place := by rw [hpo]
  have hval : SamplingDeploy.valOf (placeOf C rows) (bitsOf L.d)
      ⟨i, v.get r.place, dataOfL y (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesL c L r.port) r.leaf)
        (L.d - Flock.log2ceil (L.cnt r.port)) (List.ofFn fun k : Fin r.width => v.get k.val)⟩ W = v := by
    refine SamplingDeploy.valOf_eq (by rw [hp1]; exact hW) hv (by rw [hp1]; exact hvw) fun j => ?_
    unfold SamplingDeploy.rowFn
    dsimp only
    split
    · rename_i h
      rw [h, hp2]
    · have hj : j.val < r.width := hp1 ▸ j.isLt
      rw [bitsOf_dataOfL hy hsz hdd]
      simp [hj]
  have hrow : rowDep C rows c (placeOf C rows) (bitsOf L.d)
      ⟨i, v.get r.place, dataOfL y (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesL c L r.port) r.leaf)
        (L.d - Flock.log2ceil (L.cnt r.port)) (List.ofFn fun k : Fin r.width => v.get k.val)⟩ =
      FlockSoundness.Discharge.HiddenExec.encPortH c.ports[r.port]! v := by
    rw [rowDep_eq (W := W)
      ⟨i, v.get r.place, dataOfL y (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesL c L r.port) r.leaf)
        (L.d - Flock.log2ceil (L.cnt r.port)) (List.ofFn fun k : Fin r.width => v.get k.val)⟩ hr hq hb hpl h16 hW,
      hval]
  have hπ : pathOfL C rows L
      ⟨i, v.get r.place, dataOfL y (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesL c L r.port) r.leaf)
        (L.d - Flock.log2ceil (L.cnt r.port)) (List.ofFn fun k : Fin r.width => v.get k.val)⟩ =
      SamplingDeploy.pathMsg (FlockC3.RegLevels.sibsV (domOf c r.port) (leavesL c L r.port) r.leaf) := by
    unfold pathOfL
    dsimp only
    rw [portIdxOf_eq hr, ← hsl, pathOf_dataOfL hy hsz]
  have hlen : r.port < (rootsOfL c L).length := by rw [rootsOfL_length]; exact hq
  have hat : atOf C rows (rootsOfL c L) i = some ((rootsOfL c L).getD r.port [], r.leaf) := by
    have h1 : (rootsOfL c L)[r.port]? = some ((rootsOfL c L).getD r.port []) := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlen]
      rfl
    unfold atOf
    rw [hr]
    show (rootsOfL c L)[r.port]?.map (fun rt => (rt, r.leaf)) = _
    rw [h1]
    rfl
  unfold commitDepL
  rw [SamplingDeploy.opens_pathCommitRow]
  refine ⟨_, _, hat, find?_portsOfL c L (rootsOfL c L) hq hfirst, ?_⟩
  dsimp only
  rw [hrow, saltOf_dataOfL hy, hleaf, SamplingDeploy.bytes_digBytes, FlockSoundness.Discharge.PlainReg.regLeaf_bytes,
    hπ]
  apply SamplingDeploy.opensLeaf_tree _ (leavesL c L r.port)
  · rw [leavesL_size]
    rfl
  · exact leavesL_width c L r.port
  · show Flock.FV3.root (domOf c r.port) (leavesL c L r.port) Flock.Sha512.hash =
      bytes ((rootsOfL c L).getD r.port [])
    rw [rootsOfL_getD c L hq, Claims.Sampling.bytes_bs]
  · exact hu
  · show (leavesL c L r.port)[r.leaf]! =
      Flock.Registered.leafAs (schemaOf c r.port) (domOf c r.port) r.leaf
        (Flock.Hm96.Default512.treeLeaf (L.dig r.port r.leaf))
    rw [getElem!_pos _ _ (by rw [leavesL_size]; exact hu)]
    simp [leavesL]

/-! ## The verifier's leaf map -/

/-- **`sharedRows` reads a table per port, none empty.** -/
theorem sharedRows_spec {tags : Flock.Tags} {c : Flock.Circuit} {header : Lean.Json} {counts : Array ℕ}
    (h : Flock.HmRow.sharedRows tags c header = .ok (some counts)) :
    counts.size = c.ports.size ∧ ∀ x ∈ counts, 0 < x := by
  unfold Flock.HmRow.sharedRows at h
  dsimp only at h
  simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals first
    | (simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at h; done)
    | (simp only [pure, Except.pure, Except.ok.injEq, Option.some.injEq] at h
       subst h
       refine ⟨by simp, fun x hx => Nat.pos_of_ne_zero fun h0 => ?_⟩
       subst h0
       exact ‹¬ _ = true› (Array.contains_eq_true_of_mem hx))

set_option maxHeartbeats 2000000 in
/-- **A shared-row file's tables are one per port, none empty** (`HmRow.loadPublic`, `sharedRows_spec`). -/
theorem loadPublic_counts {tags : Flock.Tags} {c : Flock.Circuit} {file : ByteArray} {pop : Flock.Public}
    (h : Flock.HmRow.loadPublic tags c file = .ok pop) :
    ∀ t, pop.tables = some t → t.counts.size = c.ports.size ∧ ∀ x ∈ t.counts, 0 < x := by
  unfold Flock.HmRow.loadPublic at h
  dsimp only at h
  simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [pure, Except.pure, Except.ok.injEq] at *
    subst_vars
  all_goals
    intro t ht
    cases ht
  all_goals exact sharedRows_spec ‹_›

theorem getD_zero_eq_getElem! (a : Array ℕ) (k : ℕ) : a.getD k 0 = a[k]! := by
  rw [Array.getD_eq_getD_getElem?]
  by_cases h : k < a.size
  · rw [Array.getElem?_eq_getElem h, Option.getD_some, getElem!_pos a k h]
  · rw [Array.getElem?_eq_none (by omega), Option.getD_none, getElem!_neg a k h]
    try rfl

/-- **The verifier's leaf map holds every unit's row**: at a public file `HmOut.loadPublic` accepts, unit `u`'s ref at
port `q` is below `q`'s leaf count, with or without shared rows. -/
theorem ref_lt_cnt {tags : Flock.Tags} {c : Flock.Circuit} {file : ByteArray} {pub : Flock.Public}
    (h : Flock.HmOut.loadPublic tags c file = .ok pub) {nl u q : ℕ} (hu : u < nl) (hq : q < c.ports.size) :
    (LeafMap.ofPublic c pub nl).ref q u < (LeafMap.ofPublic c pub nl).cnt q := by
  have h' := FlockSoundness.Discharge.Hidden.loadPublic_nil h
  cases ht : pub.tables with
  | none => rw [LeafMap.ofPublic_none nl ht]; exact hu
  | some t =>
    rw [LeafMap.ofPublic_some nl ht]
    obtain ⟨-, hlt, -, -⟩ := (FlockSoundness.Discharge.Hidden.loadPublic_shared h').2 t ht
    obtain ⟨hcs, hpos⟩ := loadPublic_counts h' t ht
    have hlt' : ∀ k < t.refs.size, t.refs[k]! < t.counts[k % c.ports.size]! := hlt
    have hcs' : t.counts.size = c.ports.size := hcs
    show t.refs.getD (u * c.ports.size + q) 0 < t.counts.getD q 0
    rw [getD_zero_eq_getElem!, getD_zero_eq_getElem!]
    by_cases hk : u * c.ports.size + q < t.refs.size
    · have hmod : (u * c.ports.size + q) % c.ports.size = q := by
        rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hq]
      have := hlt' _ hk
      rwa [hmod] at this
    · rw [getElem!_neg t.refs _ hk]
      have hqc : q < t.counts.size := by rw [hcs']; exact hq
      rw [getElem!_pos t.counts q hqc]
      exact hpos _ (Array.getElem_mem hqc)

/-- **The deepest port bounds every port's levels** at the public file's leaf map. -/
theorem LeafMap.le_d (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) {q : ℕ} (hq : q < c.ports.size) :
    Flock.log2ceil ((LeafMap.ofPublic c pub nl).cnt q) ≤ (LeafMap.ofPublic c pub nl).d := by
  unfold LeafMap.ofPublic
  cases pub.tables with
  | none => exact le_rfl
  | some t =>
    exact SamplingDeploy.le_foldr_max (fun q => Flock.log2ceil (t.counts.getD q 0)) (List.mem_range.mpr hq)

/-! ## The game's openings, with or without shared rows -/

section Game

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed FlockSoundness.Discharge.ZkHidden
  FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge FlockSoundness.Discharge.Composed
open FlockSoundness.Discharge.ZkReg (rsZHJS rsZHJS_computes)

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Discharge.Layout.scopeOkH c = true)
  (hU : Built c.unit) {pub : Flock.Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

/-- **The game's value binding at a C-Flock session, with or without shared rows** (`TransportsLiveCR`'s `vb`). -/
noncomputable abbrev vbS := (rsZHJS (Reg := Reg) dj y₀ ht hc hscope hU hpub).bindingZ
  (rsZHJS_computes dj y₀ ht hc hscope hU hpub)

variable {dj y₀ ht hc hscope hU hpub}
variable {g : Fin ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).N} {u : Fin n} {q : Fin c.ports.size}
  {k : Fin (Discharge.Layout.portRowW c q)}

theorem vbS_pos_row (h : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g = some (u, ⟨q, k⟩)) :
    (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).pos g = (Sum.inl (u, q) : PosH c hU n) := by
  dsimp only [vbS, HmRowsZX.bindingZ, LayoutZ.bindingZ, rsZHJS, ZkHidden.layZHJ, LayoutZ.ofShadow,
    ZkHidden.layShHJ, HiddenExec.keyLayoutH, Discharge.Layout.Readers.layout, Discharge.Layout.ofBits,
    HiddenExec.keyReadersH]
  exact Discharge.Layout.ofRows_pos_some _ h

theorem vbS_bit_row (h : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g = some (u, ⟨q, k⟩))
    (v : ValH c) : (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).bit v g = v.get k.val := by
  have hle : Discharge.Layout.portRowW c q ≤ Finset.univ.sup (Discharge.Layout.portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  dsimp only [vbS, HmRowsZX.bindingZ, LayoutZ.bindingZ, rsZHJS, ZkHidden.layZHJ, LayoutZ.ofShadow,
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

theorem vbS_cs_inl (R : Reg) (u : Fin n) (q : Fin c.ports.size) :
    (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).cs R (Sum.inl (u, q) : PosH c hU n) =
      ((Sum.inl (u, q) : PosH c hU n), H512 (K512.leafPrefix ++ (pub.digest u.val q.val c.ports.size).data.toList)) :=
  rfl

theorem vbS_com_inl (u : Fin n) (q : Fin c.ports.size) (v : ValH c) (y : List UInt8) :
    (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).com ((Sum.inl (u, q) : PosH c hU n), v, y) =
      ((Sum.inl (u, q) : PosH c hU n), H512 (hm96Input (HiddenExec.encPortH c.ports[q.val]! v) y)) :=
  rfl

theorem vbS_com_fst (p : PosH c hU n) (v : ValH c) (y : List UInt8) :
    ((vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).com (p, v, y)).1 = p :=
  rfl

theorem vbS_val (p : PosH c hU n) (v : ValH c) (y : List UInt8) :
    (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).val (p, v, y) = v :=
  rfl

theorem vbS_cs_row (h : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g = some (u, ⟨q, k⟩)) (R : Reg) :
    (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).cs R ((vbS dj y₀ ht hc hscope hU hpub).pos g) =
      ((Sum.inl (u, q) : PosH c hU n), H512 (K512.leafPrefix ++ (pub.digest u.val q.val c.ports.size).data.toList)) := by
  rw [vbS_pos_row h]
  exact vbS_cs_inl R u q

/-- **A registered opening of a row's commit string is a leaf equation**, with or without shared rows. -/
theorem vbS_leaf_row (h : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g = some (u, ⟨q, k⟩)) (R : Reg)
    (p : PosH c hU n) (v : ValH c) (y : List UInt8)
    (he : (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).com (p, v, y) =
      (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).cs R ((vbS dj y₀ ht hc hscope hU hpub).pos g)) :
    H512 (hm96Input (HiddenExec.encPortH c.ports[q.val]! v) y) =
      H512 (K512.leafPrefix ++ (pub.digest u.val q.val c.ports.size).data.toList) := by
  have he' := he.trans (vbS_cs_row h R)
  rcases p with ⟨u', q'⟩ | g'
  · rw [vbS_com_inl] at he'
    obtain ⟨h1, hl⟩ := Prod.mk.inj he'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Sum.inl.inj h1)
    exact hl
  · have h1 := congrArg Prod.fst he'
    rw [vbS_com_fst] at h1
    cases h1

/-- **The deployed layout's rows at a leaf map**: `TransportsLiveCR`'s `rows`, the game's row map with each row at its
unit's leaf `L.ref`. -/
noncomputable def rowsOfL (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) (L : LeafMap) :
    Fin ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).N → Option RowAt := fun g =>
  (HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g).map fun r =>
    ⟨r.2.1.val, L.ref r.2.1.val r.1.val, Discharge.Layout.portRowW c r.2.1, r.2.2.val⟩

/-- **A shaped registered opening opens the deployed commitment at the public file's leaf map** (`transports_shaped`
with or without shared rows): at a file whose ports are rows the verifier of record reads (`RowsOk`), each row in its
port's format, at roots no two ports share and refs below their ports' counts. -/
theorem transports_shapedS (R : Reg) (hrows : RowsOk c)
    (hfirst : ∀ q q' : ℕ, q' < q → q < c.ports.size →
      (rootsOfL c (LeafMap.ofPublic c pub n)).getD q' [] ≠ (rootsOfL c (LeafMap.ofPublic c pub n)).getD q [])
    (hrefs : ∀ (u : Fin n) (q : Fin c.ports.size),
      (LeafMap.ofPublic c pub n).ref q.val u.val < (LeafMap.ofPublic c pub n).cnt q.val)
    {r : RowAt} (hr : rowsOfL c hU n (LeafMap.ofPublic c pub n) g = some r)
    (p : PosH c hU n) (v : ValH c) (y : List UInt8) (hv : RowShaped v) (hvw : v.1.val = r.width)
    (hy : y.length = saltLen)
    (he : (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).com (p, v, y) =
      (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).cs R ((vbS dj y₀ ht hc hscope hU hpub).pos g)) :
    ∃ d : Claims.Msg, (commitDepL ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n))
        (rowsOfL c hU n (LeafMap.ofPublic c pub n)) c (LeafMap.ofPublic c pub n)
        (rootsOfL c (LeafMap.ofPublic c pub n))).Opens sha512Msg (rootsOfL c (LeafMap.ofPublic c pub n))
      ⟨posOf ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)) g,
        (vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).bit
          ((vbS dj y₀ ht hc hscope hU hpub (Reg := Reg)).val (p, v, y)) g, d⟩ = true := by
  obtain ⟨⟨u, q, k⟩, hx, rfl⟩ := Option.map_eq_some_iff.mp hr
  have hle : Discharge.Layout.portRowW c q ≤ Finset.univ.sup (Discharge.Layout.portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  have hl := vbS_leaf_row hx R p v y he
  rw [← LeafMap.dig_ref c pub n q.val u.val] at hl
  rw [vbS_val, vbS_bit_row hx v]
  exact ⟨_, opens_commitDepL _ (rowsOfL c hU n (LeafMap.ofPublic c pub n)) c (LeafMap.ofPublic c pub n)
    (r := ⟨q.val, (LeafMap.ofPublic c pub n).ref q.val u.val, Discharge.Layout.portRowW c q, k.val⟩)
    (by rw [rowOfPos_posOf]; simp [rowsOfL, hx]) q.isLt (hrefs u q) (LeafMap.le_d c pub n q.isLt)
    (rowsOk_bits hrows q) (by simp [Discharge.Layout.portRowW]) k.isLt (fun q' h => hfirst q.val q' h q.isLt) hle hv hvw hy hl⟩

end Game

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **`Bridge.TransportsLiveCR` holds** (`transports_shapedS`): `dupName` and `RootFixesDomainL` give distinct roots,
and the verifier's public-file check every unit's ref below its port's count (`ref_lt_cnt`). -/
theorem transportsLiveCR : Bridge.TransportsLiveCR := by
  intro n Reg I ht D hD hU pub hpub hscope hrows hdup hroots kd _ hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags hCust τ
    P vb σ Cc L rows κ roots Cd u g _ _ o he hcan
  obtain ⟨⟨r, hr, hv, hvw⟩, hy⟩ := hcan
  obtain ⟨p, v, y⟩ := o
  exact Bridge.transports_shapedS (dj := dj) (y₀ := y₀) (ht := ht) (hc := Flock.Desc.parse_flat hD)
    (hscope := hscope) (hU := hU) (hpub := hpub) _ hrows (Bridge.rootsDistinctL_of_fixes hdup hroots)
    (fun u q => Bridge.ref_lt_cnt hpub u.isLt q.isLt) hr p v y hv hvw hy he

/-- **`TransportsLiveCR` is `TransportsLiveC` at one leaf per unit**: at `pub.tables = none` the public file's leaf map
is `LeafMap.ident`, at which `rootsOfL`, `commitDepL` and `RootFixesDomainL` are `rootsOf`, `commitDep` and
`RootFixesDomain`, and `ZkReg.rsZHJS` is `rsZHJ`. -/
theorem transportsLiveC_of_R (h : Bridge.TransportsLiveCR) : Bridge.TransportsLiveC := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags
    hCust τ
  have hL := Bridge.LeafMap.ofPublic_none (c := D.unroll) n hpt
  have := h n Reg I ht D hD hU pub hpub hscope hrows hdup (by rw [hL]; exact hroots) kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb
    src hTags hCust τ
  rw [hL] at this
  exact this

end Flock.Guarantees
