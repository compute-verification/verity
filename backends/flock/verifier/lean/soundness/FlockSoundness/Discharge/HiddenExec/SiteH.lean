import FlockSoundness.Discharge.HiddenExec.Placed
import FlockSoundness.Discharge.HiddenExec.Ret
import FlockSoundness.Discharge.Hidden.Committed
import FlockSoundness.Discharge.Hidden.Wide.Shared
import FlockSoundness.Discharge.Layout.OutPin
import FlockSoundness.Protocol.Discharge.HiddenExec.SiteH

set_option Elab.async false

/-!
# A drawn unit's site in its hidden-output table's accepted statement

`Layout/Site.lean`, `Hm/Site.lean` and `Layout.col_val_comp` at `AcceptedH T`. At VU `g` of block `o`:

* `siteAddrH a o x`: in-block column `x` of block `o`, as an address of `T`;
* `bc_site_ofH`: in a satisfying witness, the 128 bytes at row `p`'s `hm96` output are the population's `b ‖ c` of the
  proved unit at rank `o · c.g + g`, for every row, input or output (`Hidden.bc_drawnW`);
  `BcSiteH` bundles what it reads;
* `copy_siteH`: a stacked unit input whose Δ key is message bit `bit` of byte `byte` of row `q` holds the witness's bit
  at that message column (`copyPos_blockH`, `Aliased.copyPos_key` at the flat twin);
* `out_copyH`: copy `u'`'s output `j`, bit `t` holds the witness's bit at output row `nIn + jo`'s message column for its
  bit `16 k + t`, where the copy's output word is word `k` of output port `jo` (`Hidden.msg_copy`, `outCol_leaf`);
* `bc_site_colH`, `bc_site_commitH`: the `b ‖ c` at row `p`'s `hm96` output is the commit string of the row prefix and
  the bytes at its message columns, under the salt at `saltCol` (`Hidden.sat_flat`, `Hm.row_bc_flat`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model
open FlockSoundness.Discharge.Hm (blockBit blockBit_of_lt bytesOf bcCol saltCol RowRanges)

variable {T : Model.Statement}

namespace AcceptedH

variable (a : AcceptedH T)

theorem rowRanges : RowRanges a.st.c :=
  RowRanges.ofL (st := Hidden.flatTwin a.st) a.flatLayout a.unitFacts.parse a.unitFacts.check

end AcceptedH

/-! ## Site addresses -/

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

/-! ## Copies at a site -/

/-! ## The rows' `b ‖ c` at a site -/

/-- `Hm.row_bc_flat` in an accepted statement `T`, through its flat twin (`Hidden.sat_flat`). -/
theorem AcceptedH.row_bc (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv1 : ∀ p ∈ a.st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      ((HmRow.rowPrefix a.st.c.ports[p]!.words).data.toList ++
        bytesOf (blockBit T z o) (2 * a.st.c.ports[p]!.words) (Typed.msgCol a.st.c g p))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at ho hv1 hg hp ⊢
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hpa, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L0 := flat_layout ht h
  have F0 := flat_unitFacts ht h
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta']
  have hrows : Hidden.OutRows st.c := Hidden.out_rows_v1 hpa hcr fun p hp => (hv1 p hp).1
  have hz0 := Hidden.sat_flat L L0 F0 hv1 hda hrows hz
  exact Hm.row_bc_flat L0 F0 rfl (Hm.upv_pos F0.check ho) hv1 hz0 o hg hp

/-- **A row's `b ‖ c` at a drawn unit's site**: in a satisfying witness of an accepted statement whose unit returns the
outputs and whose rows are `sha512/row/v1` rows (`hv1`), at block `o`, VU `g` and row `p`, the 128 bytes at the `hm96`
slot's output (`bcCol`) are the commit string of the row prefix and the `2 · words` bytes at the message columns
(`Typed.msgCol`), under the 192 bytes at `saltCol`. -/
theorem bc_site_colH (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv1 : ∀ p ∈ a.st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (bcCol a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.rowPrefix a.st.c.ports[p]!.words).data.toList ++
        List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (siteAddrH a o (Typed.msgCol a.st.c g p j.val t.val))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrH a o (saltCol a.st.c g p j.val t.val)) := by
  have R := a.rowRanges
  rw [bytesAt_siteH a z o 128 (bcCol a.st.c g p) (fun j hj t ht => Hm.bcCol_lt R hg hp hj ht),
    bytesAt_siteH a z o 192 (saltCol a.st.c g p) (fun j hj t ht => Hm.saltCol_lt R hg hp hj ht),
    ofFn_siteH a z o _ (Typed.msgCol a.st.c g p)
      (fun j _ t ht => Layout.msgCol_lt_of_parse a.parse hg hp j t ht)]
  exact a.row_bc ho hv1 hz o hg hp

/-- `bc_site_colH` at the address the regions pin (`Layout.bcAddr`). -/
theorem bc_site_commitH (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv1 : ∀ p ∈ a.st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.rowPrefix a.st.c.ports[p]!.words).data.toList ++
        List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (siteAddrH a o (Typed.msgCol a.st.c g p j.val t.val))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrH a o (saltCol a.st.c g p j.val t.val)) := by
  have hbc : (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      fun j t => siteAddrH a o (bcCol a.st.c g p j.val t.val) := by
    funext j t
    rw [Layout.bcAddr_eq_slotCol _ _ _ _ _ (Hm.perVu_hm_pow2 a.rowRanges)]
    rfl
  rw [hbc]
  exact bc_site_colH a ho hv1 hz o hg hp

end FlockSoundness.Discharge.HiddenExec
