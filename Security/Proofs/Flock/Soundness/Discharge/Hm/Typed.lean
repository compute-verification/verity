/-
A row's `b ‖ c` in a satisfying witness of a typed template's statement (`row_bc_typed`, and at the statement
`Stmt.setupH` accepted, `row_bc_setupH_typed`): the commit string of the row and salt its slots read. `row_bc_of_sat`'s
four copies (`row_copies_of_sat`) are Δ's one copy at each `cv` and `pad` input bit (`inBit_copy`, with `RowRanges.ofT`'s
filters); the constant bits read the pin (`pin_bitT`) and the root's first slot's forced-zero row (`zero_bitT`, there
because `HmRow.checkTyped` refuses a root that fills its slot). The rows are `sha512/row/v1`'s because `HmRow.rowPorts`
refuses a typed statement's bit rows (`ParseFactsT.v1`).
-/
import Proofs.Flock.Soundness.Discharge.Hm.Copy

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll Types.Dag

section Typed

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

/-- The pin carries 1, in every block of a satisfying witness of a typed template's statement. -/
theorem pin_bitT {u : D} (T : TemplateLayout st tb words done u)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) :
    blockBit (T.model regs) z o st.pin = true := by
  rw [blockBit_of_lt _ z o T.pin_lt]
  have hpin : (⟨st.pin, T.pin_lt⟩ : Fin (2 ^ (T.model regs).kLog)) = (T.model regs).pin := Fin.ext rfl
  rw [hpin, hz.2.1 o]
  rfl

/-- **The root's first slot's forced-zero row carries 0**, in every block of a satisfying witness of a typed template's
statement with a template instance: the root's net has no row past its useful rows (`checkTyped` refuses a root that
fills its slot), and no slot constant, flat Δ pair (those are in the rows' circuits' slots) or typed tail entry (those
are at unit columns) is there. -/
theorem zero_bitT (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) (C : CheckFactsT st.c tb) (hg : 0 < st.c.g)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) :
    blockBit (T.model regs) z o (zeroCol st.c) = false := by
  obtain ⟨hm, hk, hc, hsl⟩ := T.range T.unitNet
  obtain ⟨nm, hr⟩ := T.root
  obtain ⟨-, -, -, huse, hcp, -, hcr, -⟩ := Typed.netOfD_ok hr
  have hat : AtSlot st.c st.c.unitNet 0 st.c.unit.useful (zeroCol st.c) :=
    ⟨by rw [hc, T.root_perVu, Nat.mul_one]; exact hg, by rw [hsl]; exact C.root_zero, rfl⟩
  have h32 : st.c.unit.unitLog ≤ 32 := by
    have := hat.slotLog_le T.bound hm
    rw [hsl] at this
    exact le_trans this T.kLog
  have hsz : (Typed.rowsA (Typed.rootOf 𝒖)).size = st.c.unit.useful := by rw [huse]; simp [Typed.rowsA]
  have hszB : (Typed.rowsB (Typed.rootOf 𝒖)).size = st.c.unit.useful := by rw [huse]; simp [Typed.rowsB]
  have past : ∀ (a : Array (Array ℕ)) (i : ℕ), a.size ≤ i → a.getD i #[] = #[] := fun a i h => by
    simp [Array.getD, Nat.not_lt.2 h]
  have hra := past _ _ hsz.le
  have hrb := past _ _ hszB.le
  have hcu : st.c.unit.constPos < st.c.unit.useful := by
    rw [hcp, ← hsz]
    by_contra hge
    rw [past _ _ (by omega)] at hcr
    exact absurd hcr (by simp)
  have hlk : st.c.unit.lookup = none := by
    have := C.unit_lookup
    revert this
    cases st.c.unit.lookup <;> simp
  have hy : ∀ g' < st.c.g, ∀ x, InUnit done 𝒖 x → NotConst 𝒖 x → Typed.col st.c tb g' x ≠ zeroCol st.c := by
    intro g' hg' x hx _ he
    rcases T.at_inUnit hd hg' hx with ⟨hxs, hat'⟩ | ⟨i, hi, hu, -, -, -, hat'⟩
    · obtain ⟨-, -, h3⟩ := hat'.inj T.disj hm hk hm hk (he ▸ hat)
      have : st.c.unit.useful = 𝒖.size := huse
      omega
    · obtain ⟨-, hpn, hpu, -⟩ := T.part i hi hu
      exact hpu (hat'.inj T.disj (T.range hpn).1 (T.range hpn).2.1 hm hk (he ▸ hat)).1
  have hrow : ∀ j, rangesEntry st (slotEntryA st) (zeroCol st.c) j + deltaEntry st.da (zeroCol st.c) j = 0 ∧
      rangesEntry st (slotEntryB st) (zeroCol st.c) j + deltaEntry st.db (zeroCol st.c) j = 0 := by
    intro j
    obtain ⟨hda, hdb⟩ := T.delta_count (T.deltaIn_ne T.unitNet T.shaNet.2.symm T.hmNet.2.symm hat) j
    have hcp0 : (constPairs st.c st.pin).count (zeroCol st.c, j) = 0 := by
      rw [constPairs_count T.disj T.slotLog T.fit hm hk hat j,
        ite_eq_right (fun h => absurd h.1 (Nat.ne_of_gt hcu))]
    have hre := fun e => rangesEntry_at T.disj hm hk hat e j
    refine ⟨?_, ?_⟩
    · rw [hre, hda, hcp0, T.tail_zero hd hul U hy true j]
      show (if _ then sparseEntry st.c.unit.a st.c.unit.useful _ else 0) + _ = _
      rw [(netOfD_sparse hr h32 _ _).1, hra]
      simp
    · rw [hre, hdb, hcp0, T.tail_zero hd hul U hy false j]
      show (if _ then sparseEntry st.c.unit.b st.c.unit.useful _ + lookupEntry st.c.unit.lookup st.c.unit.useful _
        else 0) + _ = _
      rw [hlk]
      show (if _ then sparseEntry st.c.unit.b st.c.unit.useful _ + 0 else 0) + _ = _
      rw [add_zero, (netOfD_sparse hr h32 _ _).2, hrb]
      simp
  have hx : zeroCol st.c < 2 ^ (T.model regs).kLog := hat.lt T.bound hm
  have hZ : ZeroRow (T.model regs) ⟨zeroCol st.c, hx⟩ := ⟨fun j => (hrow j.val).1, fun j => (hrow j.val).2⟩
  rw [blockBit_of_lt _ z o hx, hZ.block_eq hz o]
  rfl

/-- **The four copies into template instance `g`'s row `p`**, in every block of a satisfying witness of a typed
template's statement of `sha512/row/v1` rows (`hv1`): the first compression's `cv` is the row prefix's midstate, each
later compression's `cv` the previous one's output, the `hm96` slot's `cv` the last compression's output, and its `pad`
the padding block. These are `row_bc_of_sat`'s hypotheses. -/
theorem row_copies_of_sat {tags : Tags} {t : Typed.Template} (hd : deriveChecked types ls info words unit = .ok done)
    (hul : unit < ls.size) (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖)
    (P : ParseFactsT tags t st.c) (C : CheckFactsT st.c tb)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hda : st.da = (HmRow.delta st.c st.pin).1) (hdb : st.db = (HmRow.delta st.c st.pin).2)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) :
    (∀ t < 512, blockBit (T.model regs) z o (shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
      wordsBit (HmRow.midstate st.c.ports[p]!.words) t) ∧
    (∀ k, k + 1 < st.c.ports[p]!.chunks → ∀ t < 512,
      blockBit (T.model regs) z o (shaBase st.c g p (k + 1) + shaIn st.c (2 * shaSub st.c p (k + 1)) + t) =
        blockBit (T.model regs) z o (shaBase st.c g p k + shaOut st.c (shaSub st.c p k) + t)) ∧
    (∀ t < 512, blockBit (T.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      blockBit (T.model regs) z o
        (shaBase st.c g p (st.c.ports[p]!.chunks - 1) + shaOut st.c (shaSub st.c p (st.c.ports[p]!.chunks - 1)) + t)) ∧
    (∀ t < 1024, blockBit (T.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + t) =
      wordsBit (HmRow.padWords st.c.ports[p]!.words) t) := by
  have R := RowRanges.ofT T C P
  have hpos : ∀ q < st.c.ports.size, 0 < st.c.ports[q]!.chunks := fun q hq => (hv1 _ (getElem_mem_toList' hq)).2.2
  have hseg : ∀ q < st.c.ports.size, st.c.ports[q]!.segs = none := fun q hq =>
    P.v1Segs _ (getElem_mem_toList' hq) (hv1 _ (getElem_mem_toList' hq)).1
  have hwc : wireCopies st.c = wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList := by
    rw [wireCopies_eq P.wires hpos hseg, wireCopiesOf_rows _ hpos hseg]
  have h0 := hpos p hp
  have hpin := pin_bitT T hz o
  have hzero := zero_bitT hd hul T U C (by omega) hz o
  refine ⟨fun t ht => ?_, fun k hk t ht => ?_, fun t ht => ?_, fun t ht => ?_⟩
  · have hs : shaSub st.c p 0 < 3 := (rowSlot_lt hp h0).2
    have hf : (rowCopies st.c st.pin ++ wireCopies st.c).filter
          (fun e => e.1 = shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
        [(shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t,
          constSrc st.c st.pin (HmRow.midstate st.c.ports[p]!.words) t)] := by
      rw [hwc]; exact R.cv0_filter hpos hseg hg hp ht
    rw [inBit_copy hd hul T U P C hv1 hda hdb hz o (Or.inl rfl) (R.shaIn_at hg hp h0 ht) (shaNet_ofRows R.rowNets)
      (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf]
    exact blockBit_constSrc _ _ hpin hzero
  · have hs : shaSub st.c p (k + 1) < 3 := (rowSlot_lt hp hk).2
    have hf := R.chain_filter hpos hseg (pin := st.pin) (k := k + 1) hg hp (by omega) hk ht
    rw [chainDst_eq R.rowNets hs, ← hwc] at hf
    rw [inBit_copy hd hul T U P C hv1 hda hdb hz o (Or.inl rfl) (R.shaIn_at hg hp hk ht) (shaNet_ofRows R.rowNets)
      (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf]
    rfl
  · rw [(hmIn_eq R.rowNets).1, Nat.add_zero]
    have hf := R.hmcv_filter hpos hseg (pin := st.pin) hg hp ht
    have hx : Flat.wireDst st.c g (hmWire st.c.ports st.c.shaNet st.c.hmNet p) = st.c.slotCol st.c.hmNet g p :=
      hmWireDst_eq R.rowNets g p
    rw [hx, ← hwc] at hf
    rw [inBit_copy hd hul T U P C hv1 hda hdb hz o (Or.inr rfl) (R.hmIn_at hg hp (off := t) (by omega))
      (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    unfold Flat.wireSrc hmWire shaBase shaOut shaSub
    simp only [rowSlots_size hp]
  · rw [(hmIn_eq R.rowNets).2.1, Nat.add_assoc]
    have hf := R.pad_filter hpos hseg (pin := st.pin) hg hp ht
    have hx : padCol st.c g p + t = st.c.slotCol st.c.hmNet g p + (512 + t) := by
      rw [padCol_eq R.rowNets, Nat.add_assoc]
    rw [hx, ← hwc] at hf
    rw [inBit_copy hd hul T U P C hv1 hda hdb hz o (Or.inr rfl) (R.hmIn_at hg hp (off := 512 + t) (by omega))
      (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    exact blockBit_constSrc _ _ hpin hzero

/-- **A row's `b ‖ c`**, in every block of a satisfying witness of a typed template's statement of `sha512/row/v1` rows
(`hv1`): the commit string of the row prefix and message its `sha512x3` slots read, and the salt its `hm96` slot reads.
-/
theorem row_bc_typed {tags : Tags} {t : Typed.Template} (hd : deriveChecked types ls info words unit = .ok done)
    (hul : unit < ls.size) (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖)
    (P : ParseFactsT tags t st.c) (C : CheckFactsT st.c tb)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hda : st.da = (HmRow.delta st.c st.pin).1) (hdb : st.db = (HmRow.delta st.c st.pin).2)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) :
    bytesOf (blockBit (T.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.rowPrefix st.c.ports[p]!.words).data.toList ++
        bytesOf (blockBit (T.model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p))
      (bytesOf (blockBit (T.model regs) z o) 192 (saltCol st.c g p)) := by
  obtain ⟨hcv0, hchain, hcvh, hpad⟩ := row_copies_of_sat hd hul T U P C hv1 hda hdb hz o hg hp
  obtain ⟨hv, -, hn⟩ := hv1 _ (getElem_mem_toList' hp)
  exact row_bc_of_sat hd hul T U P C hz o hg hp (P.v1Segs _ (getElem_mem_toList' hp) hv) hn
    (P.portWords _ (getElem_mem_toList' hp) hv) hcv0 hchain hcvh hpad

end Typed

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **A row's `b ‖ c` at the typed template statement `Stmt.setupH` accepted**: in every block of a satisfying witness,
template instance `g`'s row `p`'s `hm96` slot outputs the commit string of the row prefix and message its `sha512x3` slots
read, under the salt its `hm96` slot reads. Unlike `row_bc_setupH`, it takes no hypothesis on the rows: they are
`sha512/row/v1` rows because `HmRow.rowPorts` refuses a typed statement's others (`ParseFactsT.v1`). -/
theorem row_bc_setupH_typed (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t))
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness ((setupH_templateLayout ht h hr).2.2.model regs).m}
    (hz : ((setupH_templateLayout ht h hr).2.2.model regs).Satisfies z)
    (o : Fin (2 ^ ((setupH_templateLayout ht h hr).2.2.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) :
    bytesOf (blockBit ((setupH_templateLayout ht h hr).2.2.model regs) z o) 128 (bcCol st.c g p) =
      Binding.commitString ((HmRow.rowPrefix st.c.ports[p]!.words).data.toList ++
        bytesOf (blockBit ((setupH_templateLayout ht h hr).2.2.model regs) z o) (2 * st.c.ports[p]!.words)
          (Typed.msgCol st.c g p))
      (bytesOf (blockBit ((setupH_templateLayout ht h hr).2.2.model regs) z o) 192 (saltCol st.c g p)) := by
  obtain ⟨-, P, C⟩ := setupH_factsT ht h hr
  obtain ⟨-, -, hda, hdb, -, -⟩ := setupH_spec_typed ht h
  exact row_bc_typed k.hd (unit_lt k.hd) (setupH_templateLayout ht h hr).2.2 (unitShape_of_class k) P C (P.v1 ht)
    hda hdb hz o hg hp

end FlockSoundness.Discharge.Hm
