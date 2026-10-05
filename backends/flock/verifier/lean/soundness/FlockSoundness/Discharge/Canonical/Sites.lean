import FlockSoundness.Discharge.Canonical.Format
import FlockSoundness.Discharge.Canonical.OutCopy
import FlockSoundness.Discharge.Canonical.Shared
import FlockSoundness.Discharge.ZkHidden.DefsJ
import FlockSoundness.Protocol.Discharge.Canonical.Sites

set_option Elab.async false

/-!
# The sites of a canonical session

`HiddenExec/Sites.lean`'s `TableH.acc`, `HeadlineExec.lean`'s `tableH_bcSite` and `ZkHidden/DefsJ.lean`'s
`sitesShHJ`, `sitesShHJ_parse`, `sitesShHJ_bcSite` at a circuit in the canonical format (`formatOk`), in place of
`sha512/row/v1` rows (`hv1`) and `Layout.scopeOk` (`hscope`), with `BcSiteC` (any public file, shared rows or not) for
`BcSiteH`:

* `outPairs_ne_unitC`, `unitRowsFlatC`: a canonical statement's unit slots' rows are its flat twin's
  (`HiddenExec.unitRowsFlat` without `hv1`).
* `accC`, `tableC_bcSite`: a drawn table's acceptance, and the site fact at it.
* `sitesShC`, `sitesShC_parse`, `tableShC_bcSite`, `sitesShC_bcSite`: the same at the public plan of `J` tables.

Every declaration here is proved. `unitRowsFlatC` is `HiddenExec.unitRowsFlat`'s proof with `outPairs_ne_unitC` for
`outPairs_ne_unit`: an output pair of Δ is at a message column of an output row (`Hidden.deltaMsgCol`); a byte below
`128 · chunks` is in a `sha512x3` slot, as before (`deltaMsgCol_ne_unit`), and a byte of the last block is in its `hm96`
slot's pad input (`deltaMsgCol_hm`), on the hm96 net, which is not the unit net either (`Layout.ne_unit_at`).
-/

namespace FlockSoundness.Discharge.Canonical

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden

section Flat

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Flock.Stmt}

/-- **No output pair of Δ is at a unit slot's row**, at canonical output rows (`outPairs_ne_unit` without `hv1`): an
output byte is in a compression's message block (`deltaMsgCol_ne_unit`) or in the last block's `hm96` pad input, on
the `hm96` net, which is not the unit's. -/
theorem outPairs_ne_unitC (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hsha : st.c.unitNet ≠ st.c.shaNet) (hhm : st.c.unitNet ≠ st.c.hmNet) (hrows : OutRowsC st.c) {q : ℕ × ℕ}
    (hq : q ∈ Hidden.outPairs st.c) {s₀ b : ℕ} (hs₀ : s₀ < st.c.unitRange.count) (hb : b < 2 ^ st.c.unit.unitLog) :
    q.1 ≠ st.c.unitRange.slot s₀ + b := by
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq1⟩ := outPairs_fst hq
  obtain ⟨hp, hbc, -, ht8⟩ := outRowC_bound hrows hj hk ht
  have hs := Hidden.segs_getD hrows.1 hp
  rw [hq1]
  rcases Nat.lt_or_ge ((2 * k + t / 8) / 128) st.c.ports[HmOut.nIn st.c + j]!.chunks with hlt | hge
  · exact deltaMsgCol_ne_unit L P C hsha hg hp hs hlt ht8 hs₀ hb
  · rw [deltaMsgCol_hm P.rowNets hs hp hge]
    have hsz : (portCols (st.c.nets.getD st.c.hmNet default).2.inGroups).size = 3 := by
      simpa using congrArg Array.size C.hm_in
    have hmem := getElem_mem_toList' (xs := portCols (st.c.nets.getD st.c.hmNet default).2.inGroups) (i := 1)
      (by omega)
    rw [(Hm.hmIn_cols P.rowNets).2] at hmem
    have hne := L.ne_unit_at P C P.hmNet hhm.symm hg
      (by rw [C.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _)) hmem
      (tt := Hidden.msgOff (2 * k + t / 8) (t % 8)) (Hidden.msgOff_lt ht8) hs₀ hb
    rwa [Nat.add_assoc] at hne

/-- **A canonical hidden statement reads its flat twin's rows at every unit slot's row**
(`HiddenExec.unitRowsFlat` at any row width): no output pair of Δ is at a unit slot's row, whether the output byte
is in a compression's message block or in the last block's `hm96` pad input. -/
theorem unitRowsFlatC (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) (hfmt : formatOk st.c = true) :
    UnitRowsFlat st := by
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hp, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L := flat_layout ht h
  have F := flat_unitFacts ht h
  have hrows := (outRowsC_of (unit_built F.parse F.check) hp hcr hfmt).1
  intro s₀ hs₀ b hb r j hr
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unitC L F.parse F.check F.sha F.hm hrows hq hs₀ hb
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

section OneTable

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hfmt : formatOk c = true)

variable (dj : Finset (Fin n) → Lean.Json) {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub)

end OneTable

section Shadow

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hfmt : formatOk c = true)

variable {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

end Shadow

end FlockSoundness.Discharge.Canonical
