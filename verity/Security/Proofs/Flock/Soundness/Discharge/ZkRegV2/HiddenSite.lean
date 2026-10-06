import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec
import Proofs.Flock.Soundness.Discharge.ZkRegV2.Site
import Proofs.Flock.Soundness.Discharge.ZkRegV2.Scope

set_option Elab.async false

/-!
# A drawn unit's site in its hidden-output table, for `sha512/row/v2` rows

`HiddenExec/Flat.lean`, `Sites.lean` and `SiteH.lean` at unsegmented rows of `16 · words` bits in whole SHA-512 blocks
with one output port per output row, the rows `ZkRegV2.scopeOkHV2` admits, in place of `sha512/row/v1` rows:

* `outPairs_ne_unitW`, `unitRowsFlatW`: no output pair of Δ is at a unit slot's row, so the unit slots' rows are the
  flat twin's (`outPairs_ne_unit`, `unitRowsFlat`, which read the rows' kind only for their block count);
* `TableH.accV2`: a drawn table's acceptance (`TableH.acc`) with `unitRowsFlatW`;
* `checkRows_wordsW`, `outRowsW_of_v2`: each output row holds its port's words, and the output rows are
  `Hidden.OutRowsW`;
* `out_copyHV2`, `AcceptedH.row_bcV2`, `bc_site_colHV2`, `bc_site_commitHV2`: `out_copyH` … `bc_site_commitH` through
  `Hidden.msg_copyAll`, `Hidden.sat_flatAll` and `Hidden.row_bc_tgt`, a row's `b ‖ c` under its own prefix
  `HmRow.prefixOf`;
* `tableH_bcSiteSV2`: `tableH_bcSiteS` at `TableH.accV2`.
-/

namespace FlockSoundness.Discharge.HiddenExec

section Flat

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

/-- **No output pair of Δ is at a unit slot's row**, for unsegmented rows of whole SHA-512 blocks (`2 · words = 128 ·
chunks`): `outPairs_ne_unit`, which reads `sha512/row/v1` for these two facts only. -/
theorem outPairs_ne_unitW (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hsha : st.c.unitNet ≠ st.c.shaNet)
    (hw : ∀ p ∈ st.c.ports.toList, p.segs = none ∧ 2 * p.words = 128 * p.chunks)
    (hle : st.c.outRowPorts.size ≤ st.c.ports.size) {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs st.c) {s₀ b : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hb : b < 2 ^ st.c.unit.unitLog) : q.1 ≠ st.c.unitRange.slot s₀ + b := by
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq1⟩ := outPairs_fst hq
  rw [hq1]
  have hp : HmOut.nIn st.c + j < st.c.ports.size := by unfold HmOut.nIn; omega
  obtain ⟨hs0, hw0⟩ := hw _ (getElem_mem_toList' hp)
  have e : st.c.ports.getD (HmOut.nIn st.c + j) default = st.c.ports[HmOut.nIn st.c + j]! := by
    rw [getElem!_pos st.c.ports _ hp, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hp, Option.getD_some]
  have hs : (st.c.ports.getD (HmOut.nIn st.c + j) default).segs = none := by rw [e]; exact hs0
  rw [e] at hk
  exact deltaMsgCol_ne_unit L P C hsha hg hp hs (by omega) (Nat.mod_lt _ (by norm_num)) hs₀ hb

/-- **An untyped hidden statement reads its flat twin's rows at every unit slot's row**, for unsegmented rows of whole
SHA-512 blocks: `unitRowsFlat` with `outPairs_ne_unitW`. -/
theorem unitRowsFlatW (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (hw : ∀ p ∈ st.c.ports.toList, p.segs = none ∧ 2 * p.words = 128 * p.chunks) : UnitRowsFlat st := by
  have s := Hidden.setupHidden_spec' h
  obtain ⟨-, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L := flat_layout ht h
  have F := flat_unitFacts ht h
  intro s₀ hs₀ b hb r j hr
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unitW L F.parse F.check F.sha hw (checkRows_le hcr) hq hs₀ hb
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by rw [s.da, Hidden.hidden_delta']
  have hdb : st.db = (HmRow.delta st.c st.pin).2 ++ (Hidden.outPairs st.c).toArray := by rw [s.db, Hidden.hidden_delta']
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [hda, Hidden.deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [hdb, Hidden.deltaEntry_append_out hout]

end Flat

section Table

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {mPts : ℕ}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
  (hv2 : ∀ p ∈ c.ports.toList, p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks)

namespace TableH

variable {S : Finset (Fin n)} (x : TableH I mPts S)

/-- **A drawn table's acceptance, at `sha512/row/v2` rows**: `TableH.acc` with its unit slots' rows its flat twin's from
`unitRowsFlatW`. -/
noncomputable def accV2 : AcceptedH x.T.S where
  tags := I.tags
  cf := I.circuitFile
  pf := I.publicFile
  coins := I.coins
  tables := I.tables
  draw := some x.d
  partition := I.partition
  program := I.program
  st := x.st
  ht := ht
  h := x.h
  L := Hidden.hidden_layout ht x.h
  hR := unitRowsFlatW ht x.h (by rw [x.c_eq hc]; exact fun p hp => ⟨(hv2 p hp).2.1, (hv2 p hp).2.2.2⟩)
  regs := x.regs
  hT := (Integrate.model_eq (parse_facts (parse_untyped ht (setupHidden_parseC x.h)).1) _
    (Hidden.setupHidden_spec x.h).leaf x.hwf x.regs).trans x.hT.symm

end TableH

/-- **The site fact at a drawn table under hidden outputs, at `sha512/row/v2` rows**: `tableH_bcSiteS` at
`TableH.accV2`, from the unit returning the outputs (`ho`). -/
theorem tableH_bcSiteSV2 (ho : c.outNet = c.unitNet) (dj : Finset (Fin n) → Lean.Json) {pub : Public}
    (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) {S : Finset (Fin n)} (x : DrawSetupH I mPts S (dj S))
    {u : Fin n} (h : u ∈ S) :
    BcSiteH (x.table.accV2 ht hc hv2) pub (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : x.st.c = c := x.table.c_eq hc
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, fun r hr => List.mem_map_of_mem hr, ?_, ?_, ?_⟩
  · show HmOut.loadPublic I.tags x.st.c I.publicFile = .ok pub
    rw [hcx]
    exact hpub
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [hcx]
    exact Layout.vu_lt_pow2 (parse_untyped ht hc).1 ho (hcx ▸ Nat.mod_lt _ x.hg)
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

end Table

section Site

open Flock FlockLevel3 Model
open FlockSoundness.Discharge.Hm (blockBit blockBit_of_lt bytesOf bcCol saltCol RowRanges)

/-- **Each output row's words hold its port's**, at one output port per output row and rows of `16 · words` bits:
`checkRows_words` for `sha512/row/v1` or `sha512/row/v2` rows (a lone port's row, its words zero-padded to whole
blocks, or a bit row of exactly its port's bits). -/
theorem checkRows_wordsW {c : Circuit} (h : HmOut.checkRows c = .ok ())
    (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) (hb : ∀ p ∈ c.ports.toList, p.bits = 16 * p.words)
    {j : ℕ} (hj : j < c.outPorts.size) :
    (c.outPorts.getD j default).2 ≤ (c.ports.getD (HmOut.nIn c + j) default).words := by
  have hj' : j < c.outRowPorts.size := by rw [hr, Array.size_replicate]; exact hj
  obtain ⟨hlt, hrow⟩ := Hidden.checkRows_out h hj'
  have hmem : c.ports.getD (HmOut.nIn c + j) default ∈ c.ports.toList := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
    exact Array.getElem_mem_toList hlt
  have hbw := hb _ hmem
  rw [Hidden.outGroup_single hr hj] at hrow
  dsimp only at hrow
  rcases hrow with ⟨-, hw, -, -⟩ | ⟨-, hbits⟩ <;> omega

/-- **The output rows of whole SHA-512 blocks are `Hidden.OutRowsW`**, at one output port per output row. -/
theorem outRowsW_of_v2 {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hcr : HmOut.checkRows c = .ok ())
    (hr : c.outRowPorts = Array.replicate c.outPorts.size 1)
    (hw : ∀ p ∈ c.ports.toList, p.segs = none ∧ 2 * p.words = 128 * p.chunks) : Hidden.OutRowsW c := by
  refine Hidden.outRowsW_of hp hcr (fun p hp => (hw p hp).1) hr fun j hj => ?_
  have hj' : j < c.outRowPorts.size := by rw [hr, Array.size_replicate]; exact hj
  obtain ⟨hlt, -⟩ := Hidden.checkRows_out hcr hj'
  have hmem : c.ports.getD (HmOut.nIn c + j) default ∈ c.ports.toList := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
    exact Array.getElem_mem_toList hlt
  have := (hw _ hmem).2
  omega

variable {T : Model.Statement}

/-- **A drawn unit's returned output holds its output row's message bit**, at `sha512/row/v2` rows of whole SHA-512
blocks with one output port per output row: `out_copyH` through `Hidden.msg_copyAll`. -/
theorem out_copyHV2 (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv2 : ∀ p ∈ a.st.c.ports.toList,
      p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks)
    (hor : a.st.c.outRowPorts = Array.replicate a.st.c.outPorts.size 1)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' j t : ℕ} (hu' : u' < a.st.c.upv) (hj : j < (a.st.c.leavesOut.getD u' #[]).size) (ht : t < 16)
    (h1 : (unitRows a.st a.L.unit_built).nIn ≤ Layout.outCol a.st.c j t)
    (h2 : Layout.outCol a.st.c j t < (unitRows a.st a.L.unit_built).nIn + (unitRows a.st a.L.unit_built).nComp)
    (hx : x.val = a.st.c.upv * (unitRows a.st a.L.unit_built).nIn +
      (u' * ((unitRows a.st a.L.unit_built).nComp + 1) + (1 + (Layout.outCol a.st.c j t -
        (unitRows a.st a.L.unit_built).nIn))))
    {jo k : ℕ} (hjo : jo < a.st.c.outPorts.size) (hk : k < (a.st.c.outPorts.getD jo default).2)
    (hw : Hidden.outAt a.st.c jo + k = (a.st.c.leavesOut.getD u' #[])[j]) :
    T.block z o (a.col hg x) =
      z (siteAddrH a o (Typed.msgCol a.st.c g (HmOut.nIn a.st.c + jo) ((16 * k + t) / 8) ((16 * k + t) % 8))) := by
  have hcol := col_val_compH a hg x hu' h1 h2 hx
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at ho hv2 hor hg x hu' hj h1 h2 hx hjo hk hw hcol ⊢
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hpa, hcr⟩ := Hidden.parse_untyped ht' s.parse
  have P := parse_facts hpa
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have L0 := flat_layout ht' h
  have F0 := flat_unitFacts ht' h
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta']
  have hrows : Hidden.OutRowsW st.c :=
    outRowsW_of_v2 hpa hcr hor fun p hp => ⟨(hv2 p hp).2.1, (hv2 p hp).2.2.2⟩
  obtain ⟨h16, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : ∀ u < st.c.upv, st.c.leavesOut.getD u #[] = st.c.leavesOut[u]! := fun u hu => by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hupv : 0 < st.c.upv := Hm.upv_pos C ho
  have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count (by omega)
  have hleaf := outCol_leaf P.typed ho (check_returned P.typed ho hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g hu' hj
  have hkw : k < (st.c.ports.getD (HmOut.nIn st.c + jo) default).words :=
    lt_of_lt_of_le hk (checkRows_wordsW hcr hor (fun p hp => (hv2 p hp).2.2.1) hjo)
  have hsrc : Hidden.outSrc st.c g jo k t = st.c.unitRange.slot (g * st.c.upv + u') + Layout.outCol st.c j t := by
    unfold Hidden.outSrc
    rw [Hidden.outGroup_single hor hjo]
    dsimp only
    simp only [hk, ↓reduceIte]
    rw [hw, hleaf]
    unfold Layout.outCol
    omega
  have hmc := Hidden.msg_copyAll L L0 F0 hda hrows hz o hg hjo hkw ht
  obtain ⟨hp, hw2⟩ := hrows.1.2.2 jo hjo
  have hb : (16 * k + t) / 8 = 2 * k + t / 8 := by omega
  have hb8 : (16 * k + t) % 8 = t % 8 := by omega
  have hin := Layout.msgCol_lt_of_parse hpa hg hp (2 * k + t / 8) (t % 8) (Nat.mod_lt _ (by norm_num))
  refine bit_inj _ _ ?_
  rw [hb, hb8, bit_siteAddrH ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, rfl⟩
    z o hin, Hidden.msgCol_delta st.c g (HmOut.nIn st.c + jo) (2 * k + t / 8) (t % 8), hmc, hsrc, ← hcol,
    blockBit_of_lt _ z o (Fin.isLt _)]

/-- **A row's `b ‖ c` under its own prefix** in an accepted statement, through its flat twin
(`Hidden.sat_flatAll`, `Hidden.row_bc_tgt`), at rows of whole SHA-512 blocks with one output port per output row. -/
theorem AcceptedH.row_bcV2 (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv2 : ∀ p ∈ a.st.c.ports.toList,
      p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks)
    (hor : a.st.c.outRowPorts = Array.replicate a.st.c.outPorts.size 1)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      ((HmRow.prefixOf a.st.c.ports[p]!).data.toList ++
        bytesOf (blockBit T z o) (2 * a.st.c.ports[p]!.words) (Typed.msgCol a.st.c g p))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at ho hv2 hor hg hp ⊢
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hpa, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L0 := flat_layout ht h
  have F0 := flat_unitFacts ht h
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta']
  have hrows : Hidden.OutRowsW st.c :=
    outRowsW_of_v2 hpa hcr hor fun p hp => ⟨(hv2 p hp).2.1, (hv2 p hp).2.2.2⟩
  have hz0 := Hidden.sat_flatAll L L0 F0 hda hrows hz
  obtain ⟨-, -, hb, hw⟩ := hv2 _ (getElem_mem_toList' hp)
  exact Hidden.row_bc_tgt L0 F0 rfl (Hm.upv_pos F0.check ho) hz0 o hg hp hb hw fun p hp => (hv2 p hp).2.1

/-- **A row's `b ‖ c` at a drawn unit's site**, under its own prefix: `bc_site_colH` with `AcceptedH.row_bcV2`. -/
theorem bc_site_colHV2 (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv2 : ∀ p ∈ a.st.c.ports.toList,
      p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks)
    (hor : a.st.c.outRowPorts = Array.replicate a.st.c.outPorts.size 1)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (bcCol a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.prefixOf a.st.c.ports[p]!).data.toList ++
        List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (siteAddrH a o (Typed.msgCol a.st.c g p j.val t.val))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrH a o (saltCol a.st.c g p j.val t.val)) := by
  have R := a.rowRanges
  rw [bytesAt_siteH a z o 128 (bcCol a.st.c g p) (fun j hj t ht => Hm.bcCol_lt R hg hp hj ht),
    bytesAt_siteH a z o 192 (saltCol a.st.c g p) (fun j hj t ht => Hm.saltCol_lt R hg hp hj ht),
    ofFn_siteH a z o _ (Typed.msgCol a.st.c g p)
      (fun j _ t ht => Layout.msgCol_lt_of_parse a.parse hg hp j t ht)]
  exact a.row_bcV2 ho hv2 hor hz o hg hp

/-- `bc_site_colHV2` at the address the regions pin (`Layout.bcAddr`). -/
theorem bc_site_commitHV2 (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv2 : ∀ p ∈ a.st.c.ports.toList,
      p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks)
    (hor : a.st.c.outRowPorts = Array.replicate a.st.c.outPorts.size 1)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.prefixOf a.st.c.ports[p]!).data.toList ++
        List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (siteAddrH a o (Typed.msgCol a.st.c g p j.val t.val))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrH a o (saltCol a.st.c g p j.val t.val)) := by
  have hbc : (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      fun j t => siteAddrH a o (bcCol a.st.c g p j.val t.val) := by
    funext j t
    rw [Layout.bcAddr_eq_slotCol _ _ _ _ _ (Hm.perVu_hm_pow2 a.rowRanges)]
    rfl
  rw [hbc]
  exact bc_site_colHV2 a ho hv2 hor hz o hg hp

end Site

end FlockSoundness.Discharge.HiddenExec
