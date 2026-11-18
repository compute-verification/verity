module

public import Proofs.Flock.SpotFlock.InstanceStmt.TraceRows
import Flock.HmRow
import Flock.HmNets
import Flock.Canon
import Flock.JsonParse
import Flock.Net
import Flock.ZkServer
import Flock.Norm
import Flock.Public
import Flock.Statement
import Flock.Comp
import Flock.Bytes
import Flock.FV3
import Flock.Hm96
import Flock.Hash
import Flock.RowLeaf
import Flock.Circuit
import Flock.Tags
import Flock.Zk
import Flock.CoinTree
import Flock.Merkle
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.Array.Attach
import all Init.WF


/-!
# A row net's rows that hold, over a layout and the circuit's row ranges

`TraceRows`' lemmas for a row net's computed and constant rows read `Aliased.UnitFacts` only for `RowRanges`; the
hidden-output statement has a layout of its own but unit facts only for its flat twin, so these take `RowRanges st.c`:
`rowNet_entryR` and `holds_netRowR` (`Hm.rowNet_entryL` and `holds_netRow`), then `holds_hmNet_atR`, `holds_shaNet_atR`,
`copyRow_netConst_atR`, `holds_hmConst_atR` and `holds_shaConst_atR`.
-/

@[expose] public section

open Flock FlockLevel3 FlockSoundness FlockSoundness.Discharge.Hm
open Flock.HmNets hiding and
open FlockDeploy.Concrete.Trace (copyRow_holds netRow_holds xorRow_congr getD_past
  shaLaid_constPos hmLaid_constPos shaLaid_const)
open SpotFlock.InstanceStmt.TraceRows

set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt.TraceRows

section netR

variable {st : Stmt}

/-- **The block rows at a row net's row past its input rows**, in a statement with a layout whose circuit has row
ranges: the net's row, placed in the slot, and the slot constants' pairs there (`Hm.rowNet_entryL` over `RowRanges`). -/
theorem rowNet_entryR (L : FlockSoundness.Layout st) (R : RowRanges st.c) {n s off x : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x)
    (hlo : (st.c.nets.getD n default).2.inWords * Flock.WORD ≤ off) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (j : ℕ) :
    rangesEntry st (slotEntryA st) x j + deltaEntry st.da x j =
        ((((l.ra.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·)).count j +
          (constPairs st.c st.pin).count (x, j) : ℕ) : ZMod 2) ∧
      rangesEntry st (slotEntryB st) x j + deltaEntry st.db x j =
        ((((l.rb.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·)).count j +
          (constPairs st.c st.pin).count (x, j) : ℕ) : ZMod 2) := by
  have hnn := R.rowNet_lt hn
  obtain ⟨hm, hk, -, hsl⟩ := R.range hnn
  have h32 : (st.c.nets.getD n default).2.unitLog ≤ 32 := by
    rw [← hsl]; exact le_trans (hat.slotLog_le R.bound hm) R.kLog
  have hw : 2 ^ (st.c.nets.getD n default).2.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) h32
  obtain ⟨-, ha, hb, -, -, -, -, -, -, -, -, -, hlk, -⟩ := Net.ofRows_ok hl
  have hd : deltaEntry st.da x j = (((constPairs st.c st.pin).count (x, j) : ℕ) : ZMod 2) := by
    obtain ⟨rest, hsplit, hrest⟩ := L.delta
    have h0 : rest.count (x, j) = 0 := List.count_eq_zero.2 fun hmem => by
      have hin := hrest _ hmem
      simp only [hat.2.2] at hin
      exact L.not_inBit hm hk hat.1 hat.2.1 hlo hin
    rw [deltaEntry_eq_count, hsplit, List.count_append, h0, Nat.add_zero]
  have hre := fun e => rangesEntry_at R.disj hm hk hat e j
  have hla := ofRows_cols_lt ha off
  have hlb := ofRows_cols_lt hb off
  refine ⟨?_, ?_⟩
  · rw [hre, hd]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.a off _ else 0) + _ = _
    rw [sparseEntry_ofRows ha, ofRows_row ha hw, count_slot _ (by rw [hsl]; exact hla) j]
    push_cast; ring
  · rw [hre, L.delta_b, hd]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.b off _ +
      lookupEntry (st.c.nets.getD n default).2.lookup off _ else 0) + _ = _
    rw [hlk]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.b off _ + 0 else 0) + _ = _
    rw [add_zero, sparseEntry_ofRows hb, ofRows_row hb hw, count_slot _ (by rw [hsl]; exact hlb) j]
    push_cast; ring

end netR

section place

variable {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
  {zI : ℕ → Bool}

/-- **A row net's computed row holds** where the slot constants' pairs skip it and the XORs of the block over its laid
rows multiply to its bit (`holds_netRow` over `RowRanges`). -/
theorem holds_netRowR (R : RowRanges st.c) {n s off : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) {r : Fin (2 ^ (L.model regs).kLog)} (hat : AtSlot st.c n s off r.val)
    (hlo : (st.c.nets.getD n default).2.inWords * Flock.WORD ≤ off) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hc : ∀ j, (constPairs st.c st.pin).count (r.val, j) = 0) {z : Model.Witness (L.model regs).m}
    (o : Fin (2 ^ (L.model regs).nbl))
    (h : (xorRow (fun j => chunkBit (L.model regs) z o ((st.c.rangeOf n).slot s + j)) (l.ra.getD off #[]) &&
      xorRow (fun j => chunkBit (L.model regs) z o ((st.c.rangeOf n).slot s + j)) (l.rb.getD off #[])) =
        chunkBit (L.model regs) z o r.val) :
    ((L.model regs).A₀.mulVec ((L.model regs).block z o)) r * ((L.model regs).B₀.mulVec ((L.model regs).block z o)) r =
      (L.model regs).block z o r := by
  obtain ⟨hm, -, -, hsl⟩ := R.range (R.rowNet_lt hn)
  obtain ⟨-, ha, hb, -⟩ := Net.ofRows_ok hl
  have hcl : ∀ rr : Array ℕ, (∀ y ∈ rr.toList, y < 2 ^ (st.c.nets.getD n default).2.unitLog) →
      ∀ y ∈ rr.toList.map ((st.c.rangeOf n).slot s + ·), y < 2 ^ (L.model regs).kLog := fun rr hrr y hy => by
    obtain ⟨y', hy', rfl⟩ := List.mem_map.1 hy
    exact AtSlot.lt R.bound hm ⟨hat.1, by rw [hsl]; exact hrr y' hy', rfl⟩
  refine netRow_holds (b := (st.c.rangeOf n).slot s) (ra := l.ra.getD off #[]) (rb := l.rb.getD off #[])
    (fun j => ?_) (fun j => ?_) (hcl _ (ofRows_cols_lt ha off)) (hcl _ (ofRows_cols_lt hb off)) h
  · have e := (rowNet_entryR L R hn hat hlo hl j.val).1
    rw [hc, Nat.add_zero] at e
    exact e
  · have e := (rowNet_entryR L R hn hat hlo hl j.val).2
    rw [hc, Nat.add_zero] at e
    exact e


/-- **An `hm96` slot's computed row holds** where the trace on the slot is the net's forward evaluation from some
inputs (`Total.hm_total`) and the slot constants' pairs skip the row. -/
theorem holds_hmNet_atR (R : RowRanges st.c)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) {s b : ℕ} (hsl : SlotAt st.c st.c.hmNet s b)
    (hb : b + 262144 ≤ 4194304) (hiw : (st.c.nets.getD st.c.hmNet default).2.inWords = 32) {x : ℕ → Bool}
    (hz : ∀ j < 262144, zI (b + j) = vHof x j) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} {off : ℕ} (ho : 4096 ≤ off ∧ off < 246784) (hr : r.val = b + off)
    (hc : ∀ j, (constPairs st.c st.pin).count (r.val, j) = 0) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  obtain ⟨-, -, -, hsl'⟩ := R.range (R.rowNet_lt (Or.inr rfl : st.c.hmNet = st.c.shaNet ∨ st.c.hmNet = st.c.hmNet))
  have hwid : (st.c.nets.getD st.c.hmNet default).2.unitLog = 18 := hsl'.symm.trans hsl.2.1
  obtain ⟨-, ha, hb', -⟩ := Net.ofRows_ok (hmNet_ofRows R.rowNets)
  have hat : AtSlot st.c st.c.hmNet s off r.val := by rw [hr]; exact hsl.atSlot (by omega)
  have hv : ∀ cs : Array ℕ, (∀ y ∈ cs.toList, y < 2 ^ (st.c.nets.getD st.c.hmNet default).2.unitLog) →
      xorRow (fun j => chunkBit (L.model regs) (zOf zI (L.model regs).m) o ((st.c.rangeOf st.c.hmNet).slot s + j)) cs =
        xorRow (vHof x) cs := by
    intro cs hcs
    refine xorRow_congr fun j hj => ?_
    have hj : j < 2 ^ (st.c.nets.getD st.c.hmNet default).2.unitLog := hcs j hj
    rw [hwid] at hj
    have hj' : j < 262144 := lt_of_lt_of_eq hj (by norm_num)
    show chunkBit (L.model regs) (zOf zI (L.model regs).m) o ((st.c.rangeOf st.c.hmNet).slot s + j) = vHof x j
    rw [hsl.2.2, chunkBit_zOf (S := L.model regs) hk o (lt_of_lt_of_le (show b + j < 4194304 by omega) (by norm_num))]
    exact hz j hj'
  refine holds_netRowR L regs R (Or.inr rfl) hat (by rw [hiw]; show 32 * 128 ≤ _; omega) (hmNet_ofRows R.rowNets)
    hc o ?_
  rw [hv _ (ofRows_cols_lt ha _), hv _ (ofRows_cols_lt hb' _),
    chunkBit_zOf (S := L.model regs) hk o (lt_of_lt_of_le (show r.val < 4194304 by omega) (by norm_num)), hr,
    hz off (by omega)]
  exact (FlockDeploy.Concrete.Total.hm_total x).2 off (by rw [inRows_HM_IN]; omega)
    (by rw [hmLaid_constPos hn]; show off < 246784; omega)

/-- **A `sha512x3` slot's computed row holds** where the trace on the slot is the net's forward evaluation from some
inputs (`Total.sha_total`) and the slot constants' pairs skip the row. -/
theorem holds_shaNet_atR (R : RowRanges st.c)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) {s b : ℕ} (hsl : SlotAt st.c st.c.shaNet s b)
    (hb : b + 262144 ≤ 4194304) (hiw : (st.c.nets.getD st.c.shaNet default).2.inWords = 64) {x : ℕ → Bool}
    (hz : ∀ j < 262144, zI (b + j) = vSof x j) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} {off : ℕ} (ho : 8192 ≤ off ∧ off < 256000) (hr : r.val = b + off)
    (hc : ∀ j, (constPairs st.c st.pin).count (r.val, j) = 0) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  obtain ⟨-, -, -, hsl'⟩ := R.range (R.rowNet_lt (Or.inl rfl : st.c.shaNet = st.c.shaNet ∨ _))
  have hwid : (st.c.nets.getD st.c.shaNet default).2.unitLog = 18 := hsl'.symm.trans hsl.2.1
  obtain ⟨-, ha, hb', -⟩ := Net.ofRows_ok (shaNet_ofRows R.rowNets)
  have hat : AtSlot st.c st.c.shaNet s off r.val := by rw [hr]; exact hsl.atSlot (by omega)
  have hv : ∀ cs : Array ℕ, (∀ y ∈ cs.toList, y < 2 ^ (st.c.nets.getD st.c.shaNet default).2.unitLog) →
      xorRow (fun j => chunkBit (L.model regs) (zOf zI (L.model regs).m) o ((st.c.rangeOf st.c.shaNet).slot s + j)) cs =
        xorRow (vSof x) cs := by
    intro cs hcs
    refine xorRow_congr fun j hj => ?_
    have hj : j < 2 ^ (st.c.nets.getD st.c.shaNet default).2.unitLog := hcs j hj
    rw [hwid] at hj
    have hj' : j < 262144 := lt_of_lt_of_eq hj (by norm_num)
    show chunkBit (L.model regs) (zOf zI (L.model regs).m) o ((st.c.rangeOf st.c.shaNet).slot s + j) = vSof x j
    rw [hsl.2.2, chunkBit_zOf (S := L.model regs) hk o (lt_of_lt_of_le (show b + j < 4194304 by omega) (by norm_num))]
    exact hz j hj'
  refine holds_netRowR L regs R (Or.inl rfl) hat (by rw [hiw]; show 64 * 128 ≤ _; omega) (shaNet_ofRows R.rowNets)
    hc o ?_
  rw [hv _ (ofRows_cols_lt ha _), hv _ (ofRows_cols_lt hb' _),
    chunkBit_zOf (S := L.model regs) hk o (lt_of_lt_of_le (show r.val < 4194304 by omega) (by norm_num)), hr,
    hz off (by omega)]
  exact (FlockDeploy.Concrete.Total.sha_total x).2 off (by rw [inRows_SHA_IN]; omega)
    (by rw [shaLaid_constPos hn]; omega)

/-- **A row net's constant row copies the pin** where its laid row reads itself and the slot constants' pairs at it,
with that row, read the pin once. -/
theorem copyRow_netConst_atR (R : RowRanges st.c) {n s b cp : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hsl : SlotAt st.c n s b)
    (hlo : (st.c.nets.getD n default).2.inWords * Flock.WORD ≤ cp) (hcp : cp < 262144) {l : Flock.HmNets.Laid}
    {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hc : l.ra.getD cp #[] = #[cp] ∧ l.rb.getD cp #[] = #[cp]) {r : Fin (2 ^ (L.model regs).kLog)}
    (hr : r.val = b + cp)
    (hcnt : ∀ j, ((((#[cp] : Array ℕ).toList.map (b + ·)).count j + (constPairs st.c st.pin).count (r.val, j) : ℕ) :
      ZMod 2) = if j = st.pin then 1 else 0) :
    CopyRow (L.model regs) r (L.model regs).pin := by
  have hat : AtSlot st.c n s cp r.val := by rw [hr]; exact hsl.atSlot hcp
  have e := fun j => rowNet_entryR L R hn hat hlo hl j
  refine ⟨fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
    rw [(e j.val).1, hc.1, hsl.2.2, hcnt]
    exact if_congr ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩ rfl rfl
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
    rw [(e j.val).2, hc.2, hsl.2.2, hcnt]
    exact if_congr ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩ rfl rfl

/-- **An `hm96` slot's constant row holds** where it copies the pin and the trace is 1 at both. -/
theorem holds_hmConst_atR (R : RowRanges st.c)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) {s b : ℕ} (hsl : SlotAt st.c st.c.hmNet s b)
    (hiw : (st.c.nets.getD st.c.hmNet default).2.inWords = 32) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = b + 246784)
    (hcnt : ∀ j, ((((#[246784] : Array ℕ).toList.map (b + ·)).count j +
      (constPairs st.c st.pin).count (r.val, j) : ℕ) : ZMod 2) = if j = st.pin then 1 else 0)
    (hz : zI r.val = true) (hp : zI st.pin = true) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  refine copyRow_holds (copyRow_netConst_atR L regs R (Or.inr rfl) hsl (by rw [hiw]; show 32 * 128 ≤ 246784; omega)
    (by omega) (hmNet_ofRows R.rowNets) (hmLaid_const hn) hr hcnt) ?_
  rw [block_pin_of L regs hk hp o, block_zOf (S := L.model regs) hk, hz]
  rfl

/-- **A `sha512x3` slot's constant row holds** where it copies the pin and the trace is 1 at both. -/
theorem holds_shaConst_atR (R : RowRanges st.c)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) {s b : ℕ} (hsl : SlotAt st.c st.c.shaNet s b)
    (hiw : (st.c.nets.getD st.c.shaNet default).2.inWords = 64) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = b + 256000)
    (hcnt : ∀ j, ((((#[256000] : Array ℕ).toList.map (b + ·)).count j +
      (constPairs st.c st.pin).count (r.val, j) : ℕ) : ZMod 2) = if j = st.pin then 1 else 0)
    (hz : zI r.val = true) (hp : zI st.pin = true) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  refine copyRow_holds (copyRow_netConst_atR L regs R (Or.inl rfl) hsl (by rw [hiw]; show 64 * 128 ≤ 256000; omega)
    (by omega) (shaNet_ofRows R.rowNets) (shaLaid_const hn) hr hcnt) ?_
  rw [block_pin_of L regs hk hp o, block_zOf (S := L.model regs) hk, hz]
  rfl


end place

end SpotFlock.InstanceStmt.TraceRows
