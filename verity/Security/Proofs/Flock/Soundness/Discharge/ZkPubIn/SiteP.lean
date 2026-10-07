import Proofs.Flock.Soundness.Discharge.ZkPubIn.Sites
import Proofs.Flock.Soundness.Discharge.ZkPubIn.HmP

set_option Elab.async false

/-!
# A drawn unit's site in a statement with the verifier's public inputs

`HiddenExec/SiteH.lean` at `AcceptedHP T`. At VU `g` of block `o`:

* `siteAddrHP a o x`: in-block column `x` of block `o`, as an address of `T`;
* `bc_site_ofHP`: in a satisfying witness, the 128 bytes at row `p`'s `hm96` output are the population's `b ‖ c` of the
  proved unit at rank `o · c.g + g`, the population the public file `HmIn.loadPublic` loads (`bc_drawnWP`, through
  `HmRow.loadPublic` at the circuit with its public input words as its output port, `loadPublic_nilP`); `BcSiteHP`
  bundles what it reads;
* `copy_siteHP`: a stacked unit input that is no public leaf's bit and whose Δ key is message bit `bit` of byte `byte`
  of row `q` holds the witness's bit at that message column (`copyPos_blockHP`);
* `out_copyHP`: copy `u'`'s output `j`, bit `t` holds the witness's bit at its output row's message column
  (`msg_copyP`);
* `bc_site_colHP`, `bc_site_commitHP`: the `b ‖ c` at row `p`'s `hm96` output is the commit string of the row prefix
  and the bytes at its message columns, under the salt at `saltCol` (`row_bc_flatP`).
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Model HiddenExec
open FlockSoundness.Discharge.Hm (blockBit blockBit_of_lt bytesOf bcCol saltCol RowRanges)

/-- **`HmIn.loadPublic` loads `HmRow.loadPublic`'s file** at the circuit with its public input words as its output port,
then only checks it. -/
theorem loadPublic_nilP {tags : Tags} {c : Circuit} {pi : HmIn.Ports} {pf : ByteArray} {pop : Public}
    (h : HmIn.loadPublic tags c pi pf = .ok pop) :
    HmRow.loadPublic tags { c with outPorts := if pi.ports.isEmpty then #[] else #[("public inputs", pi.words)] } pf =
      .ok pop := by
  unfold HmIn.loadPublic at h
  dsimp only at h
  simp only [Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    assumption

variable {T : Model.Statement}

namespace AcceptedHP

variable (a : AcceptedHP T)

/-- An accepted statement's circuit file parses (`HmRow.parse`) to its circuit. -/
theorem parse : HmRow.parse a.tags a.cf a.tables = .ok a.st.c :=
  (Hidden.parse_untyped a.ht (setupHidden_specP a.h).choose_spec.parse).1

theorem checkRows : HmOut.checkRows a.st.c = .ok () :=
  (Hidden.parse_untyped a.ht (setupHidden_specP a.h).choose_spec.parse).2

theorem rowRanges : RowRanges a.st.c :=
  RowRanges.ofL (st := Hidden.flatTwin a.st) a.flatLayout a.unitFacts.parse a.unitFacts.check

end AcceptedHP

/-! ## Site addresses -/

/-- **Column `x` of block `o` of an accepted statement's witness** (`siteAddrH`). -/
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

/-- **`Hidden.bc_drawnW` with the verifier's public inputs**: the `b ‖ c` a satisfying witness of a drawn statement
holds at a proved unit's `hm96` slot is the unit's `b ‖ c` in the population the public file loads, at every row. -/
theorem bc_drawnWP {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {d : Lean.Json} {partition : Option Lean.Json} {program : Option ByteArray} {own : ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program (some own) = .ok st) {pi : HmIn.Ports}
    (hpi : HmIn.parse st.c = .ok pi) {pop : Public} (hpop : HmIn.loadPublic tags st.c pi pf = .ok pop)
    {D : Draw.UnitDraw} (hD : Draw.UnitDraw.ofJson d = .ok D) (hlt : ∀ u ∈ D.proved, u < D.population)
    (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hg2 : g < 2 ^ st.c.g.log2) (hp : p < st.c.ports.size)
    (hr : o.val * st.c.g + g < D.proved.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) =>
        (L.model regs).addr o (Layout.inBlockOr st.c.kLog (Layout.bcAddr st.c g p j.val t.val))) =
      (pop.digest D.proved[o.val * st.c.g + g] p st.c.ports.size).data.toList := by
  obtain ⟨pi₀, s⟩ := setupHidden_specP h
  obtain rfl : pi₀ = pi := Except.ok.inj (s.inParse.symm.trans hpi)
  obtain ⟨pop', hpop', -, -, -, hdr⟩ := s.pub
  rw [hpop] at hpop'
  cases hpop'
  have hdrawn : HmRow.drawn { st.c with outPorts := #[("public inputs", pi₀.words)] } pop d = .ok st.pub := by
    have := hdr d rfl
    unfold HmIn.drawn at this
    simpa only [s.ports, Bool.false_eq_true, ↓reduceIte] using this
  have hload : HmRow.loadPublic tags { st.c with outPorts := #[("public inputs", pi₀.words)] } pf = .ok pop := by
    simpa only [s.ports, Bool.false_eq_true, ↓reduceIte] using loadPublic_nilP hpop
  have hpopn : D.population = pop.n := by
    obtain ⟨D', hD', hpn, -⟩ := Layout.drawn_spec hdrawn
    rw [hD] at hD'
    cases hD'
    exact hpn
  have hchunk : ∀ g' (hr' : o.val * st.c.g + g' < D.proved.size),
      Layout.bcChunk st.c st.pub o.val g' p = pop.digest D.proved[o.val * st.c.g + g'] p st.c.ports.size :=
    fun g' hr' => Hidden.bcChunk_drawnW hload hdrawn hD hlt hr' hp
  have hsz128 : ∀ g' ≤ g, (Layout.bcChunk st.c st.pub o.val g' p).size = 128 := fun g' hg' => by
    have hr' : o.val * st.c.g + g' < D.proved.size := by omega
    rw [hchunk g' hr']
    exact Hidden.pop_digest_sizeW hload (by rw [← hpopn]; exact hlt _ (Array.getElem_mem hr')) hp
  obtain ⟨ro, ri, hro, -, hreg⟩ := s.regions
  obtain ⟨hrsz, hall⟩ := Hidden.regions_get hro
  have hpr : p < ro.size := hrsz ▸ hp
  have hmem : ro[p] ∈ st.regions.toList := by
    rw [hreg, Array.toList_append]
    exact List.mem_append_left _ (Array.getElem_mem_toList hpr)
  rw [← hchunk g hr]
  exact Hidden.bc_pinned' L regs hregs hz o hg hg2 hmem (hall _ hpr) hsz128

/-- **What the registered `b ‖ c` and the public words read at a site** (`BcSiteH` with `HmIn.loadPublic`): block
`o` and VU `g` of an accepted statement whose draw `d` decodes to `D`, whose proved units are in its population;
whose block R1CS's regions contain the executable's; whose public file `HmIn.loadPublic` loads to `pop` at the
circuit's public input ports `pi`; with `g` below `2^⌊log₂ c.g⌋`; and whose proved unit at rank `o · c.g + g` is `u`. -/
def BcSiteHP (a : AcceptedHP T) (pi : HmIn.Ports) (pop : Public) (o : Fin (2 ^ T.nbl)) (g u : ℕ) : Prop :=
  ∃ d D, a.draw = some d ∧ Draw.UnitDraw.ofJson d = .ok D ∧ (∀ v ∈ D.proved, v < D.population) ∧
    (∀ r ∈ a.st.regions.toList, Refine.regionOf a.st.c.kLog (a.st.m - a.st.c.kLog) r ∈ a.regs) ∧
    HmIn.parse a.st.c = .ok pi ∧ HmIn.loadPublic a.tags a.st.c pi a.pf = .ok pop ∧ g < 2 ^ a.st.c.g.log2 ∧
    D.proved[o.val * a.st.c.g + g]? = some u

/-- **The `b ‖ c` at a drawn unit's site is the population's** (`bc_drawnWP`), at a `BcSiteHP`, for every row `p`. -/
theorem bc_site_ofHP {a : AcceptedHP T} {pi : HmIn.Ports} {pop : Public} {o : Fin (2 ^ T.nbl)} {g u : ℕ}
    (hs : BcSiteHP a pi pop o g u) {z : Witness T.m} (hz : T.Satisfies z) (hg : g < a.st.c.g) {p : ℕ}
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrHP a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      (pop.digest u p a.st.c.ports.size).data.toList := by
  obtain ⟨d, D, hd, hD, hlt, hregs, hpi, hpop, hg2, hu⟩ := hs
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hd hregs hpi hpop hu hg hg2 hp ⊢
  subst hd
  obtain ⟨hr, rfl⟩ := Array.getElem?_eq_some_iff.1 hu
  exact bc_drawnWP h hpi hpop hD hlt L regs hregs hz o hg hg2 hp hr

/-! ## Copies at a site -/

/-- **A drawn unit's message-keyed input holds its row's bit** (`copy_siteH`), at an input that is no public leaf's
bit. -/
theorem copy_siteHP (a : AcceptedHP T) {pi : HmIn.Ports} (hpi : HmIn.parse a.st.c = .ok pi) {z : Witness T.m}
    (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    (hx : x.val < a.rows.nIn) (hpub : pubLeaf a.st.c pi x.val = none) {q byte bit : ℕ}
    (hk : Aliased.srcKey a.st.c x.val = .msg q byte bit) (hin : Typed.msgCol a.st.c g q byte bit < 2 ^ a.st.c.kLog) :
    T.block z o (a.col hg x) = z (siteAddrHP a o (Typed.msgCol a.st.c g q byte bit)) := by
  have lp := a.leavesPublic hpi
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hpi lp hg x hx hpub hk hin ⊢
  have F := flat_unitFactsP ht h
  have Lf := flat_layoutP ht h
  have hb := copyPos_blockHP L Lf regs F hR hpi lp hg x hx hpub hz o
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

/-- **VU `g`'s column of copy `u'`'s computed column `c0`** (`col_val_compH`). -/
theorem col_val_compHP (a : AcceptedHP T) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w) {u' c0 : ℕ}
    (hu' : u' < a.st.c.upv) (h1 : (unitRows a.st a.L.unit_built).nIn ≤ c0)
    (h2 : c0 < (unitRows a.st a.L.unit_built).nIn + (unitRows a.st a.L.unit_built).nComp)
    (hx : x.val = a.st.c.upv * (unitRows a.st a.L.unit_built).nIn +
      (u' * ((unitRows a.st a.L.unit_built).nComp + 1) + (1 + (c0 - (unitRows a.st a.L.unit_built).nIn)))) :
    (a.col hg x).val = a.st.c.unitRange.slot (g * a.st.c.upv + u') + c0 := by
  unfold AcceptedHP.col
  rw [Fin.val_cast]
  exact Layout.stackPlace_val_comp _ _ _ _ _ hu' h1 h2 x hx

/-- **A drawn unit's returned output holds its output row's message bit** (`out_copyH`, through `msg_copyP`). -/
theorem out_copyHP (a : AcceptedHP T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv1 : ∀ p ∈ a.st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
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
      z (siteAddrHP a o (Typed.msgCol a.st.c g (HmOut.nIn a.st.c + jo) ((16 * k + t) / 8) ((16 * k + t) % 8))) := by
  have hcol := col_val_compHP a hg x hu' h1 h2 hx
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht', h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at ho hv1 hg x hu' hj h1 h2 hx hjo hk hw hcol ⊢
  obtain ⟨pi, s⟩ := setupHidden_specP h
  obtain ⟨hpa, hcr⟩ := Hidden.parse_untyped ht' s.parse
  have P := parse_facts hpa
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have L0 := flat_layoutP ht' h
  have F0 := flat_unitFactsP ht' h
  have hda := (delta_filterP ht' s).1
  have hrows : Hidden.OutRows st.c := Hidden.out_rows_v1 hpa hcr fun p hp => (hv1 p hp).1
  obtain ⟨h16, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : ∀ u < st.c.upv, st.c.leavesOut.getD u #[] = st.c.leavesOut[u]! := fun u hu => by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hupv : 0 < st.c.upv := Hm.upv_pos C ho
  have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count (by omega)
  have hleaf := outCol_leaf P.typed ho (check_returned P.typed ho hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g hu' hj
  have hkw : k < (st.c.ports.getD (HmOut.nIn st.c + jo) default).words :=
    lt_of_lt_of_le hk (checkRows_words hcr hrows.2.1 hrows.1 hjo)
  have hsrc : Hidden.outSrc st.c g jo k t = st.c.unitRange.slot (g * st.c.upv + u') + Layout.outCol st.c j t := by
    unfold Hidden.outSrc
    rw [Hidden.outGroup_single hrows.2.1 hjo]
    dsimp only
    simp only [hk, ↓reduceIte]
    rw [hw, hleaf]
    unfold Layout.outCol
    omega
  have hmc := msg_copyP L F0 L0 s.leavesPublic hrows hda hv1 hz o hg hjo hkw ht
  obtain ⟨hp, hw2⟩ := hrows.2.2 jo hjo
  have hb : (16 * k + t) / 8 = 2 * k + t / 8 := by omega
  have hb8 : (16 * k + t) % 8 = t % 8 := by omega
  have hks : (HmRow.byteAt (st.c.ports.getD (HmOut.nIn st.c + jo) default) (2 * k + t / 8)).1 <
      ((HmRow.rowSlots st.c.ports).2[HmOut.nIn st.c + jo]!).size := by
    rw [Typed.byteAt_none (Hidden.segs_getD hrows.1 hp), Hm.rowSlots_size hp, Hidden.getElem!_eq_getD']
    omega
  have hin := Layout.msgCol_lt_of_parse hpa hg hp (2 * k + t / 8) (t % 8) (Nat.mod_lt _ (by norm_num))
  refine bit_inj _ _ ?_
  rw [hb, hb8, bit_siteAddrHP ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht', h, L, hR, regs, rfl⟩
    z o hin, Hidden.msgCol_delta hks, hmc, hsrc, ← hcol, blockBit_of_lt _ z o (Fin.isLt _)]

/-! ## The rows' `b ‖ c` at a site -/

/-- `row_bc_flatP` in an accepted statement `T`. -/
theorem AcceptedHP.row_bc (a : AcceptedHP T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv1 : ∀ p ∈ a.st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      ((HmRow.rowPrefix a.st.c.ports[p]!.words).data.toList ++
        bytesOf (blockBit T z o) (2 * a.st.c.ports[p]!.words) (Typed.msgCol a.st.c g p))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at ho hv1 hg hp ⊢
  obtain ⟨pi, s⟩ := setupHidden_specP h
  obtain ⟨hpa, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L0 := flat_layoutP ht h
  have F0 := flat_unitFactsP ht h
  have hda := (delta_filterP ht s).1
  have hrows : Hidden.OutRows st.c := Hidden.out_rows_v1 hpa hcr fun p hp => (hv1 p hp).1
  exact row_bc_flatP L F0 L0 s.leavesPublic hrows hda (Hm.upv_pos F0.check ho) hv1 hz o hg hp

/-- **A row's `b ‖ c` at a drawn unit's site** (`bc_site_colH`). -/
theorem bc_site_colHP (a : AcceptedHP T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv1 : ∀ p ∈ a.st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrHP a o (bcCol a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.rowPrefix a.st.c.ports[p]!.words).data.toList ++
        List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (siteAddrHP a o (Typed.msgCol a.st.c g p j.val t.val))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrHP a o (saltCol a.st.c g p j.val t.val)) := by
  have R := a.rowRanges
  rw [bytesAt_siteHP a z o 128 (bcCol a.st.c g p) (fun j hj t ht => Hm.bcCol_lt R hg hp hj ht),
    bytesAt_siteHP a z o 192 (saltCol a.st.c g p) (fun j hj t ht => Hm.saltCol_lt R hg hp hj ht),
    ofFn_siteHP a z o _ (Typed.msgCol a.st.c g p)
      (fun j _ t ht => Layout.msgCol_lt_of_parse a.parse hg hp j t ht)]
  exact a.row_bc ho hv1 hz o hg hp

/-- `bc_site_colHP` at the address the regions pin (`Layout.bcAddr`). -/
theorem bc_site_commitHP (a : AcceptedHP T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv1 : ∀ p ∈ a.st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrHP a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.rowPrefix a.st.c.ports[p]!.words).data.toList ++
        List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (siteAddrHP a o (Typed.msgCol a.st.c g p j.val t.val))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrHP a o (saltCol a.st.c g p j.val t.val)) := by
  have hbc : (fun (j : Fin 128) (t : Fin 8) => siteAddrHP a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      fun j t => siteAddrHP a o (bcCol a.st.c g p j.val t.val) := by
    funext j t
    rw [Layout.bcAddr_eq_slotCol _ _ _ _ _ (Hm.perVu_hm_pow2 a.rowRanges)]
    rfl
  rw [hbc]
  exact bc_site_colHP a ho hv1 hz o hg hp

end FlockSoundness.Discharge.ZkPubIn
