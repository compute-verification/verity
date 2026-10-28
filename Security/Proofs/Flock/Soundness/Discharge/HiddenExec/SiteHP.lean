import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
import Proofs.Flock.Soundness.Discharge.HiddenExec.PlacedP
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegFitP
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SharedP

set_option Elab.async false

/-!
# A drawn unit's site in its accepted statement at public inputs

`SiteH.lean` at `AcceptedHP T`, whose Δ is `HmOut.delta`'s without the pairs into a public leaf's bits. A satisfying
witness satisfies the flat twin off the unit input rows (`Hidden.Any.sat_flatFP`), which is all the hash rows' facts read
(`Hidden.Any.row_bc_tgtS`), so at template instance `g` of block `o`:

* `siteAddrHP a o x`: in-block column `x` of block `o`, as an address of `T`;
* `bc_site_ofHP`: in a satisfying witness, the 128 bytes at row `p`'s `hm96` output are the population's `b ‖ c` of the
  proved unit at rank `o · c.g + g` (`Hidden.bc_drawnWP`); `BcSiteHP` bundles what it reads;
* `copy_siteHP`: a stacked unit input off a public leaf whose Δ key is message bit `bit` of byte `byte` of row `q` holds
  the witness's bit at that message column (`copyPos_blockHP`, `Aliased.copyPos_key` at the flat twin);
* `out_copyHP`: copy `u'`'s output `j`, bit `t` of the statement circuit `c` holds the witness's bit at output row
  `nIn + jo`'s message column for its bit `16 k + t` (`Hidden.Any.msg_copyFP`, `outCol_leaf`, `Hidden.outCol_commit16`);
* `bc_site_colHP`, `bc_site_commitHP`: the `b ‖ c` at row `p`'s `hm96` output is the commit string of the bytes the row
  hashes under the salt at `saltCol` (`AcceptedHP.row_bc`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model
open FlockSoundness.Discharge.Hm (blockBit blockBit_of_lt bytesOf bcCol saltCol RowRanges)

variable {T : Model.Statement}

namespace AcceptedHP

variable (a : AcceptedHP T)

theorem checkRows : HmOut.checkRows a.st.c = .ok () :=
  (Hidden.Committed.check a.ht a.committed).2.2

/-- **The statement circuit `c` is the committed one with other nets and no template**, as `AcceptedH.stmt_eq`. -/
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

end AcceptedHP

/-! ## Site addresses -/

/-- **In-block column `x` of block `o`** of an accepted statement at public inputs, as an address of `T` (column 0 past
the block). -/
def siteAddrHP (a : AcceptedHP T) (o : Fin (2 ^ T.nbl)) (x : ℕ) : Fin (2 ^ T.m) :=
  T.addr o (Fin.cast a.kLog_eq (Layout.inBlockOr a.st.c.kLog x))

theorem bit_siteAddrHP (a : AcceptedHP T) (z : Witness T.m) (o : Fin (2 ^ T.nbl)) {y : ℕ}
    (hy : y < 2 ^ a.st.c.kLog) : FlockSoundness.bit (z (siteAddrHP a o y)) = blockBit T z o y := by
  have hy' : y < 2 ^ T.kLog := by rw [← a.kLog_eq]; exact hy
  have hin : Layout.inBlockOr a.st.c.kLog y = ⟨y, hy⟩ := by
    simp only [Layout.inBlockOr, hy, ↓reduceDIte]
  unfold blockBit siteAddrHP
  simp only [hy', ↓reduceDIte]
  rw [hin]
  rfl

theorem ofFn_siteHP (a : AcceptedHP T) (z : Witness T.m) (o : Fin (2 ^ T.nbl)) (N : ℕ) (col : ℕ → ℕ → ℕ)
    (hcol : ∀ j < N, ∀ t < 8, col j t < 2 ^ a.st.c.kLog) :
    (List.ofFn fun j : Fin N => Binding.byteOf fun t => FlockSoundness.bit (z (siteAddrHP a o (col j.val t.val)))) =
      bytesOf (blockBit T z o) N col := by
  unfold bytesOf
  congr 1
  funext j
  congr 1
  funext t
  exact bit_siteAddrHP a z o (hcol j j.isLt t t.isLt)

theorem bytesAt_siteHP (a : AcceptedHP T) (z : Witness T.m) (o : Fin (2 ^ T.nbl)) (N : ℕ) (col : ℕ → ℕ → ℕ)
    (hcol : ∀ j < N, ∀ t < 8, col j t < 2 ^ a.st.c.kLog) :
    Binding.bytesAt z (fun (j : Fin N) (t : Fin 8) => siteAddrHP a o (col j.val t.val)) =
      bytesOf (blockBit T z o) N col :=
  ofFn_siteHP a z o N col hcol

/-! ## The registered `b ‖ c` at a site -/

/-- **What `bc_site_ofHP` reads at a site** (`BcSiteH`): block `o` and template instance `g` of an accepted statement at
public inputs whose draw `d` decodes to `D`, whose proved units are in its population; whose public file
`HmIn.loadPublic` loads to `pop`; with `g` below `2^⌊log₂ c.g⌋`; and whose proved unit at rank `o · c.g + g` is `u`. -/
def BcSiteHP (a : AcceptedHP T) (pop : Public) (o : Fin (2 ^ T.nbl)) (g u : ℕ) : Prop :=
  ∃ d D, a.draw = some d ∧ Draw.UnitDraw.ofJson d = .ok D ∧ (∀ v ∈ D.proved, v < D.population) ∧
    HmIn.loadPublic a.tags a.st.c a.pi a.pf = .ok pop ∧ g < 2 ^ a.st.c.g.log2 ∧
    D.proved[o.val * a.st.c.g + g]? = some u

/-- **The `b ‖ c` at a drawn unit's site is the population's** (`Hidden.bc_drawnWP`), at a `BcSiteHP`, for every row
`p`. -/
theorem bc_site_ofHP {a : AcceptedHP T} {pop : Public} {o : Fin (2 ^ T.nbl)} {g u : ℕ} (hs : BcSiteHP a pop o g u)
    {z : Witness T.m} (hz : T.Satisfies z) (hg : g < a.st.c.g) {p : ℕ} (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrHP a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      (pop.digest u p a.st.c.ports.size).data.toList := by
  obtain ⟨d, D, hd, hD, hlt, hpop, hg2, hu⟩ := hs
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, h, pi, s, L, hR, regs, hregs, hT⟩ := a
  subst hT
  dsimp only at hd hpop hu hg hg2 hp ⊢
  subst hd
  obtain ⟨hr, rfl⟩ := Array.getElem?_eq_some_iff.1 hu
  exact Hidden.bc_drawnWP s hpop hD hlt L regs hregs hz o hg hg2 hp hr

/-! ## Copies at a site -/

/-- **A drawn unit's message-keyed input off a public leaf holds its row's bit** (`copy_siteH`): a public leaf's input
is the in regions' word instead (`pub_block`). -/
theorem copy_siteHP (a : AcceptedHP T) {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ}
    (hg : g < a.st.c.g) (x : Fin a.rows.w) (hx : x.val < a.rows.nIn) (hneg : Aliased.pubLeaf a.st.c a.pi x.val < 0)
    {q byte bit : ℕ} (hk : Aliased.srcKey a.st.c x.val = .msg q byte bit)
    (hin : Typed.msgCol a.st.c g q byte bit < 2 ^ a.st.c.kLog) :
    T.block z o (a.col hg x) = z (siteAddrHP a o (Typed.msgCol a.st.c g q byte bit)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, h, pi, s, L, hR, regs, hregs, hT⟩ := a
  subst hT
  dsimp only at hg x hx hneg hk hin ⊢
  have F := flat_unitFactsP ht s
  have Lf := flat_layoutP ht s
  have hb := copyPos_blockHP L Lf regs F s.ports F.check hR hg x hx hneg hz o
  have hx' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    rw [← L.unitRows_nIn]; exact hx
  rcases Aliased.copyPos_key Lf regs F hg hx' (by simp [hk]) (fun k off => by simp [hk]) with he | ⟨hn, -⟩
  · simp only [AcceptedHP.col, siteAddrHP, Fin.cast_eq_self]
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

theorem col_val_compHP (a : AcceptedHP T) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w) {u' c0 : ℕ}
    (hu' : u' < a.st.c.upv) (h1 : (unitRows a.st a.L.unit_built).nIn ≤ c0)
    (h2 : c0 < (unitRows a.st a.L.unit_built).nIn + (unitRows a.st a.L.unit_built).nComp)
    (hx : x.val = a.st.c.upv * (unitRows a.st a.L.unit_built).nIn +
      (u' * ((unitRows a.st a.L.unit_built).nComp + 1) + (1 + (c0 - (unitRows a.st a.L.unit_built).nIn)))) :
    (a.col hg x).val = a.st.c.unitRange.slot (g * a.st.c.upv + u') + upos a.st.c c0 := by
  unfold AcceptedHP.col
  rw [Fin.val_cast]
  exact a.L.col_val_comp a.regs hg hu' h1 h2 x hx

/-- **A drawn unit's returned output holds its output row's message bit**, at public inputs (`out_copyH`), at the
statement circuit `c` (`HmOut.unrolled` of the committed one). -/
theorem out_copyHP (a : AcceptedHP T) {c : Flock.Circuit} (hce : HmOut.unrolled a.st.c = c)
    (ho : c.outNet = c.unitNet) (hrf : Hidden.Any.RowsFitS c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' j t : ℕ} (hu' : u' < c.upv) (hj : j < (c.leavesOut.getD u' #[]).size) (ht : t < 16)
    (h1 : (unitRows a.st a.L.unit_built).nIn ≤ Layout.outCol c j t)
    (h2 : Layout.outCol c j t < (unitRows a.st a.L.unit_built).nIn + (unitRows a.st a.L.unit_built).nComp)
    (hx : x.val = c.upv * (unitRows a.st a.L.unit_built).nIn +
      (u' * ((unitRows a.st a.L.unit_built).nComp + 1) + (1 + (Layout.outCol c j t -
        (unitRows a.st a.L.unit_built).nIn))))
    {jo k : ℕ} (hjo : jo < c.outPorts.size) (hk : k < (c.outPorts.getD jo default).2)
    (hw : Hidden.outAt c jo + k = (c.leavesOut.getD u' #[])[j]) :
    T.block z o (a.col hg x) =
      z (siteAddrHP a o (Typed.msgCol a.st.c g (HmOut.nIn c + jo) ((16 * k + t) / 8) ((16 * k + t) % 8))) := by
  obtain ⟨nets, he⟩ := Hidden.Committed.unrolled_eq a.ht a.committed
  have hc : c = { a.st.c with nets := nets, tmpl := none } := hce.symm.trans he
  subst hc
  have hu0 : u' < a.st.c.upv := hu'
  have hj0 : j < (a.st.c.leavesOut.getD u' #[]).size := hj
  have ho0 : a.st.c.outNet = a.st.c.unitNet := ho
  have hrf0 : Hidden.Any.RowsFitS a.st.c := hrf
  have hjo0 : jo < a.st.c.outPorts.size := hjo
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
  have hcol := col_val_compHP a hg x hu0 h1 h2 hx
  rw [← hoc] at hcol
  have hin := Hidden.Committed.msgCol_lt a.ht a.committed hg (hrf0.1.2 jo hjo0).1 (2 * k + t / 8) (t % 8)
    (Nat.mod_lt _ (by norm_num))
  have hpin := a.committed.unpinned_of_rep a.ht hrf0.1.1
  clear hw hk hjo hrf ho hj hu' hx h2 h1 hoc hce
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht', h, pi, s, L, hR, regs, hregs, hT⟩ := a
  subst hT
  dsimp only at hg x hu0 hj0 ho0 hrf0 hjo0 hk0 hw0 hcol hty hls hcr h16 hlo hget hin hpin ⊢
  have C := check_facts hty hls
  have L0 := flat_layoutP ht' s
  have F0 := flat_unitFactsP ht' s
  have hrows := hrf0.1
  have hupv : 0 < st.c.upv := Hm.upv_pos C ho0
  have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count (by omega)
  have hleaf := outCol_leaf hty ho0 (check_returned hty ho0 hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g hu0 hj0
  have hkw : k < (st.c.ports.getD (HmOut.nIn st.c + jo) default).words :=
    lt_of_lt_of_le hk0 (checkRows_words hcr hrows.1 hrf0.bits16 hjo0)
  have hsrc : Hidden.outSrc st.c g jo k t = st.c.unitRange.slot (g * st.c.upv + u') + Layout.outCol st.c j t := by
    unfold Hidden.outSrc
    rw [Hidden.outGroup_single hrows.1 hjo0]
    dsimp only
    simp only [hk0, ↓reduceIte]
    rw [hw0, hleaf]
    unfold Layout.outCol
    omega
  have hmc := Hidden.Any.msg_copyFP L L0 F0 (Hidden.da_offUnit L0 F0 s.ports s.da hpin) hrows (hz.ex _) o hg hjo0
    hkw ht
  have hb : (16 * k + t) / 8 = 2 * k + t / 8 := by omega
  have hb8 : (16 * k + t) % 8 = t % 8 := by omega
  refine bit_inj _ _ ?_
  rw [hb, hb8, bit_siteAddrHP ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht', h, pi, s, L, hR,
    regs, hregs, rfl⟩ z o hin, Hidden.msgCol_delta st.c g (HmOut.nIn st.c + jo) (2 * k + t / 8) (t % 8), hmc, hsrc,
    ← hcol, blockBit_of_lt _ z o (Fin.isLt _)]

/-! ## The rows' `b ‖ c` at a site -/

/-- `Hidden.Any.row_bc_tgtS` in an accepted statement at public inputs, through its flat twin off the unit input rows
(`Hidden.Any.sat_flatFP`). -/
theorem AcceptedHP.row_bc (a : AcceptedHP T) (hupv : 0 < a.st.c.upv)
    (hrf : Hidden.Any.RowsFitS a.st.c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      (Hidden.Any.rowMsg a.st.c.ports[p]!
        (bytesOf (blockBit T z o) (2 * a.st.c.ports[p]!.words) (Typed.msgCol a.st.c g p)))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, h, pi, s, L, hR, regs, hregs, hT⟩ := a
  subst hT
  dsimp only at hupv hrf hg hp ⊢
  have L0 := flat_layoutP ht s
  have F0 := flat_unitFactsP ht s
  have hz0 := Hidden.Any.sat_flatFP L L0 F0 (Hidden.da_offUnit L0 F0 s.ports s.da
    ((Hidden.committed_of_commit s.parse).unpinned_of_rep ht hrf.1.1)) hrf.1 hz
  exact Hidden.Any.row_bc_tgtS L0 F0 rfl hupv hz0 o hg hp (hrf.kind hp)

/-- **A row's `b ‖ c` at a drawn unit's site**, at public inputs (`bc_site_colH`). -/
theorem bc_site_colHP (a : AcceptedHP T) (hupv : 0 < a.st.c.upv)
    (hrf : Hidden.Any.RowsFitS a.st.c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrHP a o (bcCol a.st.c g p j.val t.val)) =
      Binding.commitString (Hidden.Any.rowMsg a.st.c.ports[p]!
        (List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (siteAddrHP a o (Typed.msgCol a.st.c g p j.val t.val)))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrHP a o (saltCol a.st.c g p j.val t.val)) := by
  have R := a.rowRanges
  rw [bytesAt_siteHP a z o 128 (bcCol a.st.c g p) (fun j hj t ht => Hm.bcCol_lt R hg hp hj ht),
    bytesAt_siteHP a z o 192 (saltCol a.st.c g p) (fun j hj t ht => Hm.saltCol_lt R hg hp hj ht),
    ofFn_siteHP a z o _ (Typed.msgCol a.st.c g p)
      (fun j _ t ht => Hidden.Committed.msgCol_lt a.ht a.committed hg hp j t ht)]
  exact a.row_bc hupv hrf hz o hg hp

/-- `bc_site_colHP` at the address the regions pin (`Layout.bcAddr`). -/
theorem bc_site_commitHP (a : AcceptedHP T) (hupv : 0 < a.st.c.upv)
    (hrf : Hidden.Any.RowsFitS a.st.c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrHP a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString (Hidden.Any.rowMsg a.st.c.ports[p]!
        (List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (siteAddrHP a o (Typed.msgCol a.st.c g p j.val t.val)))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrHP a o (saltCol a.st.c g p j.val t.val)) := by
  have hbc : (fun (j : Fin 128) (t : Fin 8) => siteAddrHP a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      fun j t => siteAddrHP a o (bcCol a.st.c g p j.val t.val) := by
    funext j t
    rw [Layout.bcAddr_eq_slotCol _ _ _ _ _ (Hm.perVu_hm_pow2 a.rowRanges)]
    rfl
  rw [hbc]
  exact bc_site_colHP a hupv hrf hz o hg hp

end FlockSoundness.Discharge.HiddenExec
