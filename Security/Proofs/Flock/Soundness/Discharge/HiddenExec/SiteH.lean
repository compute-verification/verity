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
  proved unit at rank `o · c.g + g`, for every row, input or output (`Hidden.bc_drawnS_unpin`);
  `BcSiteH` bundles what it reads;
* `copy_siteH`: a stacked unit input whose Δ key is message bit `bit` of byte `byte` of row `q` holds the witness's bit
  at that message column (`copyPos_blockH`, `Aliased.copyPos_key` at the flat twin);
* `out_copyH`: copy `u'`'s output `j`, bit `t` holds the witness's bit at output row `nIn + jo`'s message column for its
  bit `16 k + t`, where the copy's output word is word `k` of output port `jo`, an output port with a row (`Hidden.Any.msg_copyP`, `outCol_leaf`);
* `bc_site_colH`, `bc_site_commitH`: the `b ‖ c` at row `p`'s `hm96` output is the commit string of the bytes the row
  hashes (`Hidden.Any.rowMsg`: its own prefix and the bytes at its message columns, or `RowSeg.enc` of them), under the
  salt at `saltCol` (`Hidden.Any.sat_flatP`, `Hidden.Any.row_bc_tgtS`), at rows each fit or `sha512/row-seg/v1`
  (`Hidden.Any.RowsFitP`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model
open FlockSoundness.Discharge.Hm (blockBit blockBit_of_lt bytesOf bcCol saltCol RowRanges)

theorem bit_inj : ∀ x y : ZMod 2, FlockSoundness.bit x = FlockSoundness.bit y → x = y := by decide

variable {T : Model.Statement}

namespace AcceptedH

variable (a : AcceptedH T)

/-- A flat accepted statement's circuit file parses (`HmRow.parse`) to its circuit (`parse_untyped`). -/
theorem parse (hi : a.tags.isTmpl = false) : HmRow.parse a.tags a.cf a.tables = .ok a.st.c :=
  (Hidden.parse_untyped a.ht (a.h.parse_flat hi)).1

theorem checkRows : HmOut.checkRows a.st.c = .ok () :=
  (Hidden.Committed.check a.ht a.committed).2.2

/-- **The statement circuit `c` is the committed one with other nets and no template**
(`Hidden.Committed.unrolled_eq`): its ports, leaves, ranges and net indices are the committed circuit's. -/
theorem stmt_eq {c : Flock.Circuit} (hce : HmOut.unrolled a.st.c = c) :
    ∃ nets, c = { a.st.c with nets := nets, tmpl := none } := by
  obtain ⟨nets, he⟩ := Hidden.Committed.unrolled_eq a.ht a.committed
  exact ⟨nets, hce.symm.trans he⟩

theorem ports_eqU {c : Flock.Circuit} (hce : HmOut.unrolled a.st.c = c) : a.st.c.ports = c.ports := by
  obtain ⟨nets, rfl⟩ := a.stmt_eq hce
  rfl

theorem outNetU {c : Flock.Circuit} (hce : HmOut.unrolled a.st.c = c) (ho : c.outNet = c.unitNet) :
    a.st.c.outNet = a.st.c.unitNet := by
  obtain ⟨nets, rfl⟩ := a.stmt_eq hce
  exact ho

theorem rowsFitU {c : Flock.Circuit} (hce : HmOut.unrolled a.st.c = c) (hrf : Hidden.Any.RowsFitS c) :
    Hidden.Any.RowsFitS a.st.c := by
  obtain ⟨nets, rfl⟩ := a.stmt_eq hce
  exact hrf

/-- The statement circuit's output rows pass `HmOut.checkRows`' checks of them (`Hidden.RowsOut`), which read no net
(`checkPinned` does, and the statement circuit's nets are not the committed one's). -/
theorem checkRowsU {c : Flock.Circuit} (hce : HmOut.unrolled a.st.c = c) : Hidden.RowsOut c := by
  obtain ⟨nets, rfl⟩ := a.stmt_eq hce
  exact fun j hj => Hidden.checkRows_out a.checkRows hj

/-- **Every returned output is an output word** of the statement circuit (`Layout.check_out`). -/
theorem leavesOut_ltU {c : Flock.Circuit} (hce : HmOut.unrolled a.st.c = c) (ho : c.outNet = c.unitNet) :
    ∀ u < c.upv, ∀ o ∈ (c.leavesOut[u]!).toList, o < c.outWords := by
  obtain ⟨nets, rfl⟩ := a.stmt_eq hce
  obtain ⟨hty, ⟨ls, hls⟩, -⟩ := Hidden.Committed.check a.ht a.committed
  exact fun u hu => ((Layout.check_out hty ho hls).2.2 u hu).2

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

/-- **The `b ‖ c` at a drawn unit's site is the population's** (`Hidden.bc_drawnS_unpin`), at a `BcSiteH`, for every row
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
  exact Hidden.bc_drawnS_unpin h.self hpop hD hlt L regs hregs hz o hg hg2 hp hr

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
  have F := h.unitFacts ht
  have Lf := h.flatLayout ht
  have hb := copyPos_blockH L Lf regs F hR hg x hx hz o
  have hx' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    rw [← L.unitRows_nIn]; exact hx
  rcases Aliased.copyPos_key Lf regs F hg hx' (by simp [hk]) (fun k off => by simp [hk]) with he | ⟨hn, -⟩
  · simp only [AcceptedH.col, siteAddrH, Fin.cast_eq_self]
    refine hb.trans ?_
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
statement whose statement circuit (`HmOut.unrolled` of its committed one) is `c`, whose unit returns the outputs and
whose rows are covered (`hrf`), at block `o` and template instance `g`, copy `u'`'s logical computed column
`outCol j t` of `c` (its output `j`, bit `t`) holds the witness's bit at output row `nIn + jo`'s message column for bit
`16 k + t` (`Typed.msgCol`, byte `(16 k + t) / 8`, bit `(16 k + t) % 8`), where the copy's output word `lo[j]` is word
`k` of output port `jo`, an output port with a row (`hjr`). The committed column is `upos` of the logical one
(`Hidden.outCol_commit16`). -/
theorem out_copyH (a : AcceptedH T) {c : Flock.Circuit} (hce : HmOut.unrolled a.st.c = c)
    (ho : c.outNet = c.unitNet) (hrf : Hidden.Any.RowsFitP c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' j t : ℕ} (hu' : u' < c.upv) (hj : j < (c.leavesOut.getD u' #[]).size) (ht : t < 16)
    (h1 : (unitRows a.st a.L.unit_built).nIn ≤ Layout.outCol c j t)
    (h2 : Layout.outCol c j t < (unitRows a.st a.L.unit_built).nIn + (unitRows a.st a.L.unit_built).nComp)
    (hx : x.val = c.upv * (unitRows a.st a.L.unit_built).nIn +
      (u' * ((unitRows a.st a.L.unit_built).nComp + 1) + (1 + (Layout.outCol c j t -
        (unitRows a.st a.L.unit_built).nIn))))
    {jo k : ℕ} (hjo : jo < c.outPorts.size) (hjr : jo < c.outRowPorts.size) (hk : k < (c.outPorts.getD jo default).2)
    (hw : Hidden.outAt c jo + k = (c.leavesOut.getD u' #[])[j]) :
    T.block z o (a.col hg x) =
      z (siteAddrH a o (Typed.msgCol a.st.c g (HmOut.nIn c + jo) ((16 * k + t) / 8) ((16 * k + t) % 8))) := by
  obtain ⟨nets, he⟩ := Hidden.Committed.unrolled_eq a.ht a.committed
  have hc : c = { a.st.c with nets := nets, tmpl := none } := hce.symm.trans he
  subst hc
  have hu0 : u' < a.st.c.upv := hu'
  have hj0 : j < (a.st.c.leavesOut.getD u' #[]).size := hj
  have ho0 : a.st.c.outNet = a.st.c.unitNet := ho
  have hrf0 : Hidden.Any.RowsFitP a.st.c := hrf
  have hjo0 : jo < a.st.c.outPorts.size := hjo
  have hjr0 : jo < a.st.c.outRowPorts.size := hjr
  have hk0 : k < (a.st.c.outPorts.getD jo default).2 := hk
  have hw0 : Hidden.outAt a.st.c jo + k = (a.st.c.leavesOut.getD u' #[])[j]'hj0 := hw
  have hnIn : HmOut.nIn { a.st.c with nets := nets, tmpl := none } = HmOut.nIn a.st.c := rfl
  rw [hnIn]
  obtain ⟨hty, ⟨ls, hls⟩, hcr⟩ := Hidden.Committed.check a.ht a.committed
  obtain ⟨h16, -, hlo⟩ := Layout.check_out hty ho0 hls
  have hget : ∀ u < a.st.c.upv, a.st.c.leavesOut.getD u #[] = a.st.c.leavesOut[u]! := fun u hu => by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hj16 : j < (a.st.c.unit.outGroups.getD 0 default).bits.size := by
    rw [← (hlo u' hu0).1, ← hget u' hu0]; exact hj0
  have hoc := Hidden.outCol_commit16 a.ht a.committed h16 hj16 ht
  rw [he] at hoc
  have hcol := col_val_compH a hg x hu0 h1 h2 hx
  rw [← hoc] at hcol
  have hin := Hidden.Committed.msgCol_lt a.ht a.committed hg (hrf0.1.2 jo hjr0).1 (2 * k + t / 8) (t % 8)
    (Nat.mod_lt _ (by norm_num))
  have hs := a.h.self
  have L0 := a.flatLayout
  have F0 := a.unitFacts
  clear hw hk hjo hjr hrf ho hj hu' hx h2 h1 hoc hce
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hg x hu0 hj0 ho0 hrf0 hjo0 hjr0 hk0 hw0 hcol hty hls hcr h16 hlo hget hin hs L0 F0 ⊢
  have C := check_facts hty hls
  have hda : (Hidden.unpin st).da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := rfl
  have hrows := hrf0.1
  have hupv : 0 < st.c.upv := Hm.upv_pos C ho0
  have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count (by omega)
  have hleaf := outCol_leaf hty ho0 (check_returned hty ho0 hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g hu0 hj0
  have hkw : k < (st.c.ports.getD (HmOut.nIn st.c + jo) default).words :=
    lt_of_lt_of_le hk0 (checkRows_wordsN hcr hrows.1 hrf0.bits16 hjr0 hjo0)
  have hsrc : Hidden.outSrc st.c g jo k t = st.c.unitRange.slot (g * st.c.upv + u') + Layout.outCol st.c j t := by
    unfold Hidden.outSrc
    rw [Hidden.outGroup_singleN hrows.1 hjr0 hjo0]
    dsimp only
    simp only [hk0, ↓reduceIte]
    rw [hw0, hleaf]
    unfold Layout.outCol
    omega
  have hmc := Hidden.Any.msg_copyP L L0 F0 hda hrows hz o hg hjr0 hkw ht
  have hb : (16 * k + t) / 8 = 2 * k + t / 8 := by omega
  have hb8 : (16 * k + t) % 8 = t % 8 := by omega
  refine bit_inj _ _ ?_
  rw [hb, hb8, bit_siteAddrH ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, rfl⟩
    z o hin, Hidden.msgCol_delta st.c g (HmOut.nIn st.c + jo) (2 * k + t / 8) (t % 8), hmc, hsrc, ← hcol, blockBit_of_lt _ z o (Fin.isLt _)]

/-! ## The rows' `b ‖ c` at a site -/

/-- `Hidden.Any.row_bc_tgtS` in an accepted statement `T`, through its flat twin (`Hidden.Any.sat_flatP`). -/
theorem AcceptedH.row_bc (a : AcceptedH T) (hupv : 0 < a.st.c.upv)
    (hrf : Hidden.Any.RowsFitP a.st.c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      (Hidden.Any.rowMsg a.st.c.ports[p]!
        (bytesOf (blockBit T z o) (2 * a.st.c.ports[p]!.words) (Typed.msgCol a.st.c g p)))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hupv hrf hg hp ⊢
  have L0 := h.flatLayout ht
  have F0 := h.unitFacts ht
  have hda : (Hidden.unpin st).da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := rfl
  have hz0 := Hidden.Any.sat_flatP L L0 F0 hda hrf.1 hz
  exact Hidden.Any.row_bc_tgtS L0 F0 rfl hupv (hz0.ex _) o hg hp (hrf.kind hp)

/-- **A row's `b ‖ c` at a drawn unit's site**: in a satisfying witness of an accepted statement with a unit per VU
(`hupv`) and whose rows are each fit or `sha512/row-seg/v1` (`hrf`), at block `o`, template instance `g` and row `p`, the
128 bytes at the `hm96` slot's output (`bcCol`) are the commit string of the bytes the row hashes for the `2 · words`
bytes at its message columns (`Typed.msgCol`, `Hidden.Any.rowMsg`), under the 192 bytes at `saltCol`. -/
theorem bc_site_colH (a : AcceptedH T) (hupv : 0 < a.st.c.upv)
    (hrf : Hidden.Any.RowsFitP a.st.c)
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
      (fun j _ t ht => Hidden.Committed.msgCol_lt a.ht a.committed hg hp j t ht)]
  exact a.row_bc hupv hrf hz o hg hp

/-- `bc_site_colH` at the address the regions pin (`Layout.bcAddr`). -/
theorem bc_site_commitH (a : AcceptedH T) (hupv : 0 < a.st.c.upv)
    (hrf : Hidden.Any.RowsFitP a.st.c)
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
