import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed
import Proofs.Flock.Soundness.Discharge.HiddenExec.Ret
import Proofs.Flock.Soundness.Discharge.Hidden.Committed
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Shared
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.FitS
import Proofs.Flock.Soundness.Discharge.Layout.OutPin

set_option Elab.async false

/-!
# A drawn unit's site in its hidden-output table's accepted statement

`Layout/Site.lean`, `Hm/Site.lean` and `Layout.col_val_comp` at `AcceptedH T`. At template instance `g` of block `o`:

* `siteAddrH a o x`: in-block column `x` of block `o`, as an address of `T`;
* `bc_site_ofH`: in a satisfying witness, the 128 bytes at row `p`'s `hm96` output are the population's `b ‖ c` of the
  proved unit at rank `o · c.g + g`, for every row, input or output (`Hidden.bc_drawnW`);
  `BcSiteH` bundles what it reads;
* `copy_siteH`: a stacked unit input whose Δ key is message bit `bit` of byte `byte` of row `q` holds the witness's bit
  at that message column (`copyPos_blockH`, `Aliased.copyPos_key` at the flat twin);
* `out_copyH`: copy `u'`'s output `j`, bit `t` holds the witness's bit at output row `nIn + jo`'s message column for its
  bit `16 k + t`, where the copy's output word is word `k` of output port `jo` (`Hidden.Any.msg_copyF`, `outCol_leaf`);
* `bc_site_colH`, `bc_site_commitH`: the `b ‖ c` at row `p`'s `hm96` output is the commit string of the bytes the row
  hashes (`Hidden.Any.rowMsg`: its own prefix and the bytes at its message columns, or `RowSeg.enc` of them), under the
  salt at `saltCol` (`Hidden.Any.sat_flatF`, `Hidden.Any.row_bc_tgtS`), at rows each fit or `sha512/row-seg/v1`
  (`Hidden.Any.RowsFitS`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model
open FlockSoundness.Discharge.Hm (blockBit blockBit_of_lt bytesOf bcCol saltCol RowRanges)

theorem bit_inj : ∀ x y : ZMod 2, FlockSoundness.bit x = FlockSoundness.bit y → x = y := by decide

variable {T : Model.Statement}

namespace AcceptedH

variable (a : AcceptedH T)

/-- An accepted statement's circuit file parses (`HmRow.parse`) to its circuit (`parse_untyped`). -/
theorem parse : HmRow.parse a.tags a.cf a.tables = .ok a.st.c :=
  (Hidden.parse_untyped a.ht (Hidden.setupHidden_spec' a.h).parse).1

theorem checkRows : HmOut.checkRows a.st.c = .ok () :=
  (Hidden.parse_untyped a.ht (Hidden.setupHidden_spec' a.h).parse).2

theorem rowRanges : RowRanges a.st.c :=
  RowRanges.ofL (st := Hidden.flatTwin a.st) a.flatLayout a.unitFacts.parse a.unitFacts.check

end AcceptedH

/-! ## Site addresses -/

/-- **In-block column `x` of block `o`** of an accepted statement, as an address of its statement `T` (column 0 past
the block). -/
def siteAddrH (a : AcceptedH T) (o : Fin (2 ^ T.nbl)) (x : ℕ) : Fin (2 ^ T.m) :=
  T.addr o (Fin.cast a.kLog_eq (Layout.inBlockOr a.st.c.kLog x))

/-- **A site address reads the block's bit**: in-block column `y` of block `o`. -/
theorem bit_siteAddrH (a : AcceptedH T) (z : Witness T.m) (o : Fin (2 ^ T.nbl)) {y : ℕ}
    (hy : y < 2 ^ a.st.c.kLog) : FlockSoundness.bit (z (siteAddrH a o y)) = blockBit T z o y := by
  have hy' : y < 2 ^ T.kLog := by rw [← a.kLog_eq]; exact hy
  have hin : Layout.inBlockOr a.st.c.kLog y = ⟨y, hy⟩ := by
    simp only [Layout.inBlockOr, hy, ↓reduceDIte]
  unfold blockBit siteAddrH
  simp only [hy', ↓reduceDIte]
  rw [hin]
  rfl

/-- Bytes read at in-block site addresses are the block's (`bytesOf`). -/
theorem ofFn_siteH (a : AcceptedH T) (z : Witness T.m) (o : Fin (2 ^ T.nbl)) (N : ℕ) (col : ℕ → ℕ → ℕ)
    (hcol : ∀ j < N, ∀ t < 8, col j t < 2 ^ a.st.c.kLog) :
    (List.ofFn fun j : Fin N => Binding.byteOf fun t => FlockSoundness.bit (z (siteAddrH a o (col j.val t.val)))) =
      bytesOf (blockBit T z o) N col := by
  unfold bytesOf
  congr 1
  funext j
  congr 1
  funext t
  exact bit_siteAddrH a z o (hcol j j.isLt t t.isLt)

/-- `ofFn_siteH` as `Binding.bytesAt`. -/
theorem bytesAt_siteH (a : AcceptedH T) (z : Witness T.m) (o : Fin (2 ^ T.nbl)) (N : ℕ) (col : ℕ → ℕ → ℕ)
    (hcol : ∀ j < N, ∀ t < 8, col j t < 2 ^ a.st.c.kLog) :
    Binding.bytesAt z (fun (j : Fin N) (t : Fin 8) => siteAddrH a o (col j.val t.val)) =
      bytesOf (blockBit T z o) N col :=
  ofFn_siteH a z o N col hcol

/-! ## The registered `b ‖ c` at a site -/

/-- **What `bc_site_ofH` reads at a site**: block `o` and template instance `g` of an accepted statement whose draw `d`
decodes to `D`, whose proved units are in its population; whose block R1CS's regions (`a.regs`) contain the
executable's; whose public file `HmOut.loadPublic` loads to `pop`, with or without shared rows; with `g` below
`2^⌊log₂ c.g⌋`; and whose proved unit at rank `o · c.g + g` is `u`. -/
def BcSiteH (a : AcceptedH T) (pop : Public) (o : Fin (2 ^ T.nbl)) (g u : ℕ) : Prop :=
  ∃ d D, a.draw = some d ∧ Draw.UnitDraw.ofJson d = .ok D ∧ (∀ v ∈ D.proved, v < D.population) ∧
    (∀ r ∈ a.st.regions.toList, Refine.regionOf a.st.c.kLog (a.st.m - a.st.c.kLog) r ∈ a.regs) ∧
    HmOut.loadPublic a.tags a.st.c a.pf = .ok pop ∧ g < 2 ^ a.st.c.g.log2 ∧
    D.proved[o.val * a.st.c.g + g]? = some u

/-- **The `b ‖ c` at a drawn unit's site is the population's** (`Hidden.bc_drawnW`), at a `BcSiteH`, for every row
`p`. -/
theorem bc_site_ofH {a : AcceptedH T} {pop : Public} {o : Fin (2 ^ T.nbl)} {g u : ℕ} (hs : BcSiteH a pop o g u)
    {z : Witness T.m} (hz : T.Satisfies z) (hg : g < a.st.c.g) {p : ℕ} (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      (pop.digest u p a.st.c.ports.size).data.toList := by
  obtain ⟨d, D, hd, hD, hlt, hregs, hpop, hg2, hu⟩ := hs
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hd hregs hpop hu hg hg2 hp ⊢
  subst hd
  obtain ⟨hr, rfl⟩ := Array.getElem?_eq_some_iff.1 hu
  exact Hidden.bc_drawnW h hpop hD hlt L regs hregs hz o hg hg2 hp hr

/-! ## Copies at a site -/

/-- **A drawn unit's message-keyed input holds its row's bit**: in a satisfying witness of an accepted statement, at
block `o` and template instance `g`, the template instance's stacked unit input `x` whose Δ key is message bit `bit` of
byte `byte` of row `q` holds the witness's bit at that message column (`Typed.msgCol`), when the column is in the block.
-/
theorem copy_siteH (a : AcceptedH T) {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ}
    (hg : g < a.st.c.g) (x : Fin a.rows.w) (hx : x.val < a.rows.nIn) {q byte bit : ℕ}
    (hk : Aliased.srcKey a.st.c x.val = .msg q byte bit) (hin : Typed.msgCol a.st.c g q byte bit < 2 ^ a.st.c.kLog) :
    T.block z o (a.col hg x) = z (siteAddrH a o (Typed.msgCol a.st.c g q byte bit)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hg x hx hk hin ⊢
  have F := flat_unitFacts ht h
  have Lf := flat_layout ht h
  have hb := copyPos_blockH L Lf regs F hR hg x hx hz o
  have hx' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    rw [← L.unitRows_nIn]; exact hx
  rcases Aliased.copyPos_key Lf regs F hg hx' (by simp [hk]) (fun k off => by simp [hk]) with he | ⟨hn, -⟩
  · simp only [AcceptedH.col, siteAddrH, Fin.cast_eq_self]
    rw [hb]
    show z _ = z _
    congr 2
    apply Fin.ext
    rw [he]
    simp only at hk ⊢
    rw [hk]
    simp [Aliased.realize, Layout.inBlockOr, hin]
    rfl
  · exact absurd (by simp only at hk ⊢; rw [hk]; exact hin) hn

/-- **An accepted statement's template instance column of copy `u'`'s computed column `c0`** is column `c0` of the unit
slot `vu · upv + u'`, through `upos`. -/
theorem col_val_compH (a : AcceptedH T) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w) {u' c0 : ℕ}
    (hu' : u' < a.st.c.upv) (h1 : (unitRows a.st a.L.unit_built).nIn ≤ c0)
    (h2 : c0 < (unitRows a.st a.L.unit_built).nIn + (unitRows a.st a.L.unit_built).nComp)
    (hx : x.val = a.st.c.upv * (unitRows a.st a.L.unit_built).nIn +
      (u' * ((unitRows a.st a.L.unit_built).nComp + 1) + (1 + (c0 - (unitRows a.st a.L.unit_built).nIn)))) :
    (a.col hg x).val = a.st.c.unitRange.slot (g * a.st.c.upv + u') + upos a.st.c c0 := by
  unfold AcceptedH.col
  rw [Fin.val_cast]
  exact a.L.col_val_comp a.regs hg hu' h1 h2 x hx

/-- **A drawn unit's returned output holds its output row's message bit**: in a satisfying witness of an accepted
statement whose unit returns the outputs and whose rows are covered (`hrf`), at block `o` and template
instance `g`, copy `u'`'s computed column `outCol j t` (its output `j`, bit `t`) holds the witness's bit at output row
`nIn + jo`'s message column for bit `16 k + t` (`Typed.msgCol`, byte `(16 k + t) / 8`, bit `(16 k + t) % 8`), where the
copy's output word `lo[j]` is word `k` of output port `jo`. -/
theorem out_copyH (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hrf : Hidden.Any.RowsFitS a.st.c)
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
  dsimp only at ho hrf hg x hu' hj h1 h2 hx hjo hk hw hcol ⊢
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hpa, hcr⟩ := Hidden.parse_untyped ht' s.parse
  have P := parse_facts hpa
  rw [upos_flat P.tmpl] at hcol
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have L0 := flat_layout ht' h
  have F0 := flat_unitFacts ht' h
  have R : RowRanges st.c := RowRanges.ofL (st := Hidden.flatTwin st) L0 F0.parse F0.check
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta_unpinned (Hidden.setupHidden_unpinned ht' h hrf.1.1)]
  have hrows := hrf.1
  obtain ⟨h16, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : ∀ u < st.c.upv, st.c.leavesOut.getD u #[] = st.c.leavesOut[u]! := fun u hu => by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hupv : 0 < st.c.upv := Hm.upv_pos C ho
  have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count (by omega)
  have hleaf := outCol_leaf P.typed ho (check_returned P.typed ho hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g hu' hj
  have hkw : k < (st.c.ports.getD (HmOut.nIn st.c + jo) default).words :=
    lt_of_lt_of_le hk (checkRows_words hcr hrows.1 hrf.bits16 hjo)
  have hsrc : Hidden.outSrc st.c g jo k t = st.c.unitRange.slot (g * st.c.upv + u') + Layout.outCol st.c j t := by
    unfold Hidden.outSrc
    rw [Hidden.outGroup_single hrows.1 hjo]
    dsimp only
    simp only [hk, ↓reduceIte]
    rw [hw, hleaf]
    unfold Layout.outCol
    omega
  have hmc := Hidden.Any.msg_copyF L L0 F0 hda hrows hz o hg hjo hkw ht
  have hp := (hrows.2 jo hjo).1
  have hb : (16 * k + t) / 8 = 2 * k + t / 8 := by omega
  have hb8 : (16 * k + t) % 8 = t % 8 := by omega
  have hin := Layout.msgCol_lt_of_parse hpa hg hp (2 * k + t / 8) (t % 8) (Nat.mod_lt _ (by norm_num))
  refine bit_inj _ _ ?_
  rw [hb, hb8, bit_siteAddrH ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, rfl⟩
    z o hin, Hidden.msgCol_delta st.c g (HmOut.nIn st.c + jo) (2 * k + t / 8) (t % 8), hmc, hsrc, ← hcol, blockBit_of_lt _ z o (Fin.isLt _)]

/-! ## The rows' `b ‖ c` at a site -/

/-- `Hidden.Any.row_bc_tgtS` in an accepted statement `T`, through its flat twin (`Hidden.Any.sat_flatF`). -/
theorem AcceptedH.row_bc (a : AcceptedH T) (hupv : 0 < a.st.c.upv)
    (hrf : Hidden.Any.RowsFitS a.st.c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      (Hidden.Any.rowMsg a.st.c.ports[p]!
        (bytesOf (blockBit T z o) (2 * a.st.c.ports[p]!.words) (Typed.msgCol a.st.c g p)))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hupv hrf hg hp ⊢
  have s := Hidden.setupHidden_spec' h
  have L0 := flat_layout ht h
  have F0 := flat_unitFacts ht h
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta_unpinned (Hidden.setupHidden_unpinned ht h hrf.1.1)]
  have hz0 := Hidden.Any.sat_flatF L L0 F0 hda hrf.1 hz
  exact Hidden.Any.row_bc_tgtS L0 F0 rfl hupv (hz0.ex _) o hg hp (hrf.kind hp)

/-- **A row's `b ‖ c` at a drawn unit's site**: in a satisfying witness of an accepted statement with a unit per VU
(`hupv`) and whose rows are each fit or `sha512/row-seg/v1` (`hrf`), at block `o`, template instance `g` and row `p`, the
128 bytes at the `hm96` slot's output (`bcCol`) are the commit string of the bytes the row hashes for the `2 · words`
bytes at its message columns (`Typed.msgCol`, `Hidden.Any.rowMsg`), under the 192 bytes at `saltCol`. -/
theorem bc_site_colH (a : AcceptedH T) (hupv : 0 < a.st.c.upv)
    (hrf : Hidden.Any.RowsFitS a.st.c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (bcCol a.st.c g p j.val t.val)) =
      Binding.commitString (Hidden.Any.rowMsg a.st.c.ports[p]!
        (List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (siteAddrH a o (Typed.msgCol a.st.c g p j.val t.val)))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrH a o (saltCol a.st.c g p j.val t.val)) := by
  have R := a.rowRanges
  rw [bytesAt_siteH a z o 128 (bcCol a.st.c g p) (fun j hj t ht => Hm.bcCol_lt R hg hp hj ht),
    bytesAt_siteH a z o 192 (saltCol a.st.c g p) (fun j hj t ht => Hm.saltCol_lt R hg hp hj ht),
    ofFn_siteH a z o _ (Typed.msgCol a.st.c g p)
      (fun j _ t ht => Layout.msgCol_lt_of_parse a.parse hg hp j t ht)]
  exact a.row_bc hupv hrf hz o hg hp

/-- `bc_site_colH` at the address the regions pin (`Layout.bcAddr`). -/
theorem bc_site_commitH (a : AcceptedH T) (hupv : 0 < a.st.c.upv)
    (hrf : Hidden.Any.RowsFitS a.st.c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString (Hidden.Any.rowMsg a.st.c.ports[p]!
        (List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (siteAddrH a o (Typed.msgCol a.st.c g p j.val t.val)))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrH a o (saltCol a.st.c g p j.val t.val)) := by
  have hbc : (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      fun j t => siteAddrH a o (bcCol a.st.c g p j.val t.val) := by
    funext j t
    rw [Layout.bcAddr_eq_slotCol _ _ _ _ _ (Hm.perVu_hm_pow2 a.rowRanges)]
    rfl
  rw [hbc]
  exact bc_site_colH a hupv hrf hz o hg hp

end FlockSoundness.Discharge.HiddenExec
