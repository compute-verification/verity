import FlockSoundness.Protocol.Discharge.Hidden.Flat
import FlockSoundness.Protocol.Discharge.Hidden.Public
import FlockSoundness.Protocol.Discharge.Layout.Keyed
import FlockSoundness.Discharge.Layout.Scope

/-!
# An output row's registered `b ‖ c` (`OutputsCommitted`)

In a satisfying witness of an untyped hidden-output statement, at block `o`, VU `g` and output row `j`:
* the 128 bytes at the row's `hm96` output are the population's `b ‖ c` of the drawn unit there: `HmOut.regions`' region
  `p` pins them (`bc_pinned'`, `Layout.bc_pinned` at the region `regions_get` gives), and `HmOut.drawn` is `HmRow.drawn`
  on the circuit without output ports (`Layout.bcChunk_drawn`): `bc_drawnH`, at every row;
* they are the commit string of the row prefix and the row's message bytes under its salt: the witness satisfies the flat
  twin (`sat_flat`), where `Hm.row_bc_flat` holds;
* the row's message bytes are the output words the VU computes (`msg_bytes`): each message bit is a copy of `outSrc`
  (`msg_copy`), the output word's bit below the port's words and the forced-zero row past them (`Hm.zero_bit`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

section Pinned

open Model Refine Binding FlockSoundness.Discharge.Layout

/-- **The `b ‖ c` a satisfying witness holds at an `hm96` slot's output** is the one a region of the statement that
`mkRegion` built as "digest p" pins (`Layout.bc_pinned`, for any such region rather than `HmRow.regions`'). -/
theorem bc_pinned' {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    {z : Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hg2 : g < 2 ^ st.c.g.log2) {name : String} {r : Flock.Region}
    (hr : r ∈ st.regions.toList)
    (hmk : mkRegion st.c.kLog name (hmFree st.c) (hmFixed st.c p) (hmBytes st.c st.pub p) = .ok r)
    (hsz : ∀ g' ≤ g, (bcChunk st.c st.pub o.val g' p).size = 128) :
    bytesAt z (fun (j : Fin 128) (t : Fin 8) =>
        (L.model regs).addr o (inBlockOr st.c.kLog (bcAddr st.c g p j.val t.val))) =
      (bcChunk st.c st.pub o.val g p).data.toList := by
  have hmem : Refine.regionOf (L.model regs).kLog ((L.model regs).m - (L.model regs).kLog) r ∈
      (L.model regs).regions := hregs r hr
  obtain ⟨hrf, -, -, hnd, -, hlt⟩ := mkRegion_spec hmk
  have hfree : r.free = hmFree st.c := by rw [hrf]
  have hbytes : r.bytes = hmBytes st.c st.pub p := by rw [hrf]
  refine bytesAt_eq z _ _ (by rw [Array.length_toList, ByteArray.size_data, hsz g le_rfl]) fun j t => ?_
  have hlen : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.length = 10 + st.c.g.log2 := by
    rw [← List.length_map (f := Fin.val), regionOf_free_map _ _ _ (fun b hb => hlt b (by
      rw [hfree] at hb; exact Array.mem_toList_iff.1 hb)), hfree, hmFree_toList]
    simp
  have hk : g * 1024 + (8 * j.val + t.val) < 2 ^ (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.length := by
    rw [hlen, pow_add, show (2 : ℕ) ^ 10 = 1024 by norm_num]
    have := j.isLt; have := t.isLt
    nlinarith
  have hib := inBlock_digest (st.m - st.c.kLog) hmk hg2 j.isLt t.isLt
  have hnd' : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.Nodup := by
    unfold Refine.regionOf
    rw [hfree]
    exact filterMap_fin_nodup _ _ hnd
  have ha : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).inBlock (g * 1024 + (8 * j.val + t.val)) <
      2 ^ st.c.kLog :=
    inBlock_lt _ hnd' _
  have hread := region_read hz hmem o _ hk ha
  have hfin : inBlockOr st.c.kLog (bcAddr st.c g p j.val t.val) = ⟨_, ha⟩ := by
    simp only [inBlockOr, ← hib, ha, ↓reduceDIte]
  show bit ((L.model regs).block z o (inBlockOr st.c.kLog (bcAddr st.c g p j.val t.val))) = _
  rw [hfin]
  refine hread.trans ?_
  rw [hbytes, show (g * 1024 + (8 * j.val + t.val)) / 8 = g * 128 + j.val by omega,
    show (g * 1024 + (8 * j.val + t.val)) % 8 = t.val by omega, hmBytes_get hg j.isLt hsz, get!_eq_getD]
  have hjl : j.val < (bcChunk st.c st.pub o.val g p).data.toList.length := by
    rw [Array.length_toList, ByteArray.size_data, hsz g le_rfl]; exact j.isLt
  rw [List.getD_eq_getElem _ _ hjl]
  rfl

end Pinned

open Hm Compose Flock.DeriveAll

/-- A row's message column (`Typed.msgCol`, `HmRow.delta`'s) is `HmOut.delta`'s (`deltaMsgCol`). -/
theorem msgCol_delta (c : Circuit) (g p b t : ℕ) : Typed.msgCol c g p b t = deltaMsgCol c g p b t := by
  simp only [Typed.msgCol, deltaMsgCol, getElem!_eq_getD']

section Msg

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (L0 : FlockSoundness.Layout (flatTwin st))
  (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
  (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) (hrows : OutRows st.c)
include L L0 F0 hv1 hda hrows

end Msg

end FlockSoundness.Discharge.Hidden
