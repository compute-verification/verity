module

public import Proofs.Flock.SessionLoop.Concrete.Trace
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
# Rows that hold, at any slot

`FlockDeploy.Concrete.Trace`'s rows by place, restated at any slot of the `hm96` and `sha512x3` nets' ranges and of the
unit's, over any trace in one block `zI` (`zOf zI` in every block), so a table whose slots sit elsewhere than the
concrete table's reads them at each. Each lemma takes the slot's base (`SlotAt`), Δ's entries at the row and the
trace's values it reads as hypotheses:

* `holds_copy`, `holds_zero`: a copy or a zero row holds where the trace agrees;
* `copyRow_netIn_at`, `holds_hmNet_at`, `holds_shaNet_at`, `holds_hmGap_at`, `holds_shaGap_at`, `holds_hmPast_at`,
  `holds_shaPast_at`, `copyRow_netConst_at`: a row net's input, computed, padding, past and constant rows;
* `copyRow_unitLeaf_at`, `holds_unitLeaf_at`, `holds_unitPad_at`, `holds_unitComp_at`, `holds_unitPast_at`,
  `holds_unitConst_at`: the unit's, where the trace is 0 on its slot but at its constant;
* `copyRow_mask_at`, `holds_outside`: the mask's rows and the rows no range covers.
-/

@[expose] public section

open Flock FlockLevel3 FlockSoundness FlockSoundness.Discharge.Hm
open Flock.HmNets hiding and
open FlockDeploy.Concrete.Trace (copyRow_holds zeroRow_holds xorRow_congr zeroRow_netEmpty holds_netRow getD_past
  shaLaid_constPos hmLaid_constPos laidVal_past sha_outs hm_outs sha_end hm_end shaLaid_gap hmLaid_gap shaLaid_const
  count_one ite_cancel mod_block copyRow_unitConst copyRow_pin holds_unitComp zeroRow_unitEmpty zeroRow_outside
  UN_a_lt UN_empty UN_in UN_past)
open FlockDeploy.Concrete (UN)

set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt.TraceRows

/-- The `hm96` net's column values from inputs `x`: its forward evaluation. -/
def vHof (x : ℕ → Bool) : ℕ → Bool :=
  FlockDeploy.Concrete.Total.laidVal ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT
    ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.toArray x

/-- The `sha512x3` net's column values from inputs `x`: its forward evaluation. -/
def vSof (x : ℕ → Bool) : ℕ → Bool :=
  FlockDeploy.Concrete.Total.laidVal (compression3.run St.empty).2 SHA_IN SHA_OUT
    (compression3.run St.empty).1.2.flatten.toArray x

/-- A trace in every block: `zI` at each in-block address. -/
def zOf (zI : ℕ → Bool) (m : ℕ) : Model.Witness m := fun a => if zI (a.val % 2 ^ 22) then 1 else 0

theorem block_zOf {zI : ℕ → Bool} {S : Model.Statement} (hk : S.kLog = 22) (o : Fin (2 ^ S.nbl))
    (i : Fin (2 ^ S.kLog)) : S.block (zOf zI S.m) o i = if zI i.val then 1 else 0 := by
  have hi : i.val < 2 ^ 22 := lt_of_lt_of_eq i.isLt (by rw [hk])
  show (if zI ((o.val * 2 ^ S.kLog + i.val) % 2 ^ 22) then 1 else 0) = _
  rw [mod_block _ _ _ hk hi]

theorem chunkBit_zOf {zI : ℕ → Bool} {S : Model.Statement} (hk : S.kLog = 22) (o : Fin (2 ^ S.nbl)) {y : ℕ}
    (hy : y < 2 ^ 22) : chunkBit S (zOf zI S.m) o y = zI y := by
  have hy' : y < 2 ^ S.kLog := lt_of_lt_of_eq hy (by rw [hk])
  rw [chunkBit_of_lt _ _ o hy', block_zOf hk]
  show FlockSoundness.bit (if zI y then 1 else 0) = zI y
  cases zI y <;> rfl

theorem vHof_input {x : ℕ → Bool} {j : ℕ} (h : j < 4096) : vHof x j = x j :=
  FlockDeploy.Concrete.Total.laidVal_input (show j < inRowsOf HM_IN by rw [inRows_HM_IN]; exact h)

theorem vSof_input {x : ℕ → Bool} {j : ℕ} (h : j < 8192) : vSof x j = x j :=
  FlockDeploy.Concrete.Total.laidVal_input (show j < inRowsOf SHA_IN by rw [inRows_SHA_IN]; exact h)

theorem vHof_const (hn : Properties.Flock.Nets.FixedNetsBuild) (x : ℕ → Bool) : vHof x 246784 = true := by
  have hcp : (hmLaid ()).constPos = 246784 := hmLaid_constPos hn
  rw [← hcp]
  exact (FlockDeploy.Concrete.Total.hm_total x).1

theorem vSof_const (hn : Properties.Flock.Nets.FixedNetsBuild) (x : ℕ → Bool) : vSof x 256000 = true := by
  have hcp : (shaLaid ()).constPos = 256000 := shaLaid_constPos hn
  rw [← hcp]
  exact (FlockDeploy.Concrete.Total.sha_total x).1

theorem vHof_past (hn : Properties.Flock.Nets.FixedNetsBuild) (x : ℕ → Bool) {j : ℕ} (h : 246784 < j) :
    vHof x j = false :=
  laidVal_past hm_outs (by rw [hm_end hn]; exact h)

theorem vSof_past (hn : Properties.Flock.Nets.FixedNetsBuild) (x : ℕ → Bool) {j : ℕ} (h : 256000 < j) :
    vSof x j = false :=
  laidVal_past sha_outs (by rw [sha_end hn]; exact h)

theorem hmLaid_const (hn : Properties.Flock.Nets.FixedNetsBuild) :
    (HmNets.hmLaid ()).ra.getD 246784 #[] = #[246784] ∧ (HmNets.hmLaid ()).rb.getD 246784 #[] = #[246784] := by
  obtain ⟨-, -, -, -, -, -, -, -, ha, hb, -⟩ := Net.ofRows_ok hn.2.2.2.1
  rw [hmLaid_constPos hn] at ha hb
  exact ⟨ha, hb⟩

/-- **Slot `s` of net `n`'s range starts at column `b`**, its slots `2 ^ 18` columns wide. -/
abbrev SlotAt (c : Flock.Circuit) (n s b : ℕ) : Prop :=
  s < (c.rangeOf n).count ∧ (c.rangeOf n).slotLog = 18 ∧ (c.rangeOf n).slot s = b

theorem SlotAt.atSlot {c : Flock.Circuit} {n s b : ℕ} (h : SlotAt c n s b) {off : ℕ} (ho : off < 262144) :
    AtSlot c n s off (b + off) :=
  ⟨h.1, by rw [h.2.1]; exact lt_of_lt_of_le ho (by norm_num), by rw [h.2.2]⟩

section place

variable {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
  {zI : ℕ → Bool}

/-- **A copy row holds** where the trace agrees at the row and its source. -/
theorem holds_copy (hk : st.c.kLog = 22) (o : Fin (2 ^ (L.model regs).nbl)) {r src : Fin (2 ^ (L.model regs).kLog)}
    (h : CopyRow (L.model regs) r src) (he : zI r.val = zI src.val) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  copyRow_holds h (by rw [block_zOf (S := L.model regs) hk, block_zOf (S := L.model regs) hk, he])

/-- **A zero row holds** where the trace is 0. -/
theorem holds_zero (hk : st.c.kLog = 22) (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)}
    (h : ZeroRow (L.model regs) r) (he : zI r.val = false) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  zeroRow_holds h (by rw [block_zOf (S := L.model regs) hk, he]; rfl)

/-- **The trace is 1 at the pin** in every block, where `zI` is 1 there. -/
theorem block_pin_of (hk : st.c.kLog = 22) (hp : zI st.pin = true) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block (zOf zI (L.model regs).m) o (L.model regs).pin = 1 := by
  rw [block_zOf (S := L.model regs) hk]
  show (if zI st.pin then 1 else 0) = 1
  rw [hp]
  rfl

/-- **The pin's row holds.** -/
theorem holds_pin (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = st.pin) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  have he : r = (L.model regs).pin := Fin.ext hr
  subst he
  exact copyRow_holds (copyRow_pin L regs) rfl

/-- **A row net's input row copies its source** where Δ's pairs at it are its self pair and the source's. -/
theorem copyRow_netIn_at (R : RowRanges st.c) {n s b : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet)
    (hsl : SlotAt st.c n s b) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    {r : Fin (2 ^ (L.model regs).kLog)} {off : ℕ} (ho : off < 262144) (hr : r.val = b + off)
    (hself : l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off]) (src : Fin (2 ^ (L.model regs).kLog))
    (hd : ∀ j, deltaEntry st.da r.val j = (((if j = r.val then 1 else 0) + (if j = src.val then 1 else 0) : ℕ) : ZMod 2)) :
    CopyRow (L.model regs) r src := by
  have hat : AtSlot st.c n s off r.val := by rw [hr]; exact hsl.atSlot ho
  have h := inBit_rows R hn hat hl hself r.isLt (fun j => ⟨hd j, by rw [L.delta_b]; exact hd j⟩)
  exact ⟨fun j => (h j).1.trans (if_congr Fin.ext_iff.symm rfl rfl),
    fun j => (h j).2.trans (if_congr Fin.ext_iff.symm rfl rfl)⟩

/-- **An `hm96` input row holds** where it copies its source and the trace agrees there. -/
theorem holds_hmIn_at (hk : st.c.kLog = 22) (R : RowRanges st.c) {s b : ℕ} (hsl : SlotAt st.c st.c.hmNet s b)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} {off : ℕ} (ho : off < 3072)
    (hr : r.val = b + off) (src : Fin (2 ^ (L.model regs).kLog))
    (hd : ∀ j, deltaEntry st.da r.val j = (((if j = r.val then 1 else 0) + (if j = src.val then 1 else 0) : ℕ) : ZMod 2))
    (he : zI r.val = zI src.val) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  holds_copy L regs hk o (copyRow_netIn_at L regs R (Or.inr rfl) hsl (hmNet_ofRows R.rowNets) (by omega) hr
    (hmLaid_self ho) src hd) he

/-- **A `sha512x3` input row holds** where it copies its source and the trace agrees there. -/
theorem holds_shaIn_at (hk : st.c.kLog = 22) (R : RowRanges st.c) {s b : ℕ} (hsl : SlotAt st.c st.c.shaNet s b)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} {off : ℕ} (ho : off < 4608)
    (hr : r.val = b + off) (src : Fin (2 ^ (L.model regs).kLog))
    (hd : ∀ j, deltaEntry st.da r.val j = (((if j = r.val then 1 else 0) + (if j = src.val then 1 else 0) : ℕ) : ZMod 2))
    (he : zI r.val = zI src.val) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  holds_copy L regs hk o (copyRow_netIn_at L regs R (Or.inl rfl) hsl (shaNet_ofRows R.rowNets) (by omega) hr
    (shaLaid_self ho) src hd) he

/-- **An `hm96` slot's input padding row holds**: it is zero where Δ skips it, and so is the trace there. -/
theorem holds_hmGap_at (hk : st.c.kLog = 22) (R : RowRanges st.c) {s b : ℕ} (hsl : SlotAt st.c st.c.hmNet s b)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} {off : ℕ} (ho : 3072 ≤ off ∧ off < 4096)
    (hr : r.val = b + off) (hd : ∀ j, deltaEntry st.da r.val j = 0) (hz : zI r.val = false) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  holds_zero L regs hk o (zeroRow_netEmpty L regs R (Or.inr rfl) (by rw [hr]; exact hsl.atSlot (by omega))
    (hmNet_ofRows R.rowNets) (hmLaid_gap ho.1 ho.2) hd) hz

/-- **A `sha512x3` slot's input padding row holds**: it is zero where Δ skips it, and so is the trace there. -/
theorem holds_shaGap_at (hk : st.c.kLog = 22) (R : RowRanges st.c) {s b : ℕ} (hsl : SlotAt st.c st.c.shaNet s b)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} {off : ℕ} (ho : 4608 ≤ off ∧ off < 8192)
    (hr : r.val = b + off) (hd : ∀ j, deltaEntry st.da r.val j = 0) (hz : zI r.val = false) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  holds_zero L regs hk o (zeroRow_netEmpty L regs R (Or.inl rfl) (by rw [hr]; exact hsl.atSlot (by omega))
    (shaNet_ofRows R.rowNets) (shaLaid_gap ho.1 ho.2) hd) hz

/-- **An `hm96` slot's row past its constant holds**: it is zero where Δ skips it, and so is the trace there. -/
theorem holds_hmPast_at (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) (R : RowRanges st.c)
    {s b : ℕ} (hsl : SlotAt st.c st.c.hmNet s b) (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)}
    {off : ℕ} (ho : 246784 < off ∧ off < 262144) (hr : r.val = b + off) (hd : ∀ j, deltaEntry st.da r.val j = 0)
    (hz : zI r.val = false) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  holds_zero L regs hk o (zeroRow_netEmpty L regs R (Or.inr rfl) (by rw [hr]; exact hsl.atSlot ho.2)
    (hmNet_ofRows R.rowNets) ⟨getD_past (by rw [hn.2.1]; omega), getD_past (by rw [hn.2.2.1]; omega)⟩ hd) hz

/-- **A `sha512x3` slot's row past its constant holds**: it is zero where Δ skips it, and so is the trace there. -/
theorem holds_shaPast_at (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) (R : RowRanges st.c)
    {s b : ℕ} (hsl : SlotAt st.c st.c.shaNet s b) (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)}
    {off : ℕ} (ho : 256000 < off ∧ off < 262144) (hr : r.val = b + off) (hd : ∀ j, deltaEntry st.da r.val j = 0)
    (hz : zI r.val = false) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  holds_zero L regs hk o (zeroRow_netEmpty L regs R (Or.inl rfl) (by rw [hr]; exact hsl.atSlot ho.2)
    (shaNet_ofRows R.rowNets) ⟨getD_past (by rw [hn.1.1]; omega), getD_past (by rw [hn.1.2.1]; omega)⟩ hd) hz

/-- **An `hm96` slot's computed row holds** where the trace on the slot is the net's forward evaluation from some
inputs (`Total.hm_total`) and the slot constants' pairs skip the row. -/
theorem holds_hmNet_at {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) {s b : ℕ} (hsl : SlotAt st.c st.c.hmNet s b)
    (hb : b + 262144 ≤ 4194304) (hiw : (st.c.nets.getD st.c.hmNet default).2.inWords = 32) {x : ℕ → Bool}
    (hz : ∀ j < 262144, zI (b + j) = vHof x j) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} {off : ℕ} (ho : 4096 ≤ off ∧ off < 246784) (hr : r.val = b + off)
    (hc : ∀ j, (constPairs st.c st.pin).count (r.val, j) = 0) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  have R := RowRanges.ofL L F.parse F.check
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
  refine holds_netRow L regs F (Or.inr rfl) hat (by rw [hiw]; show 32 * 128 ≤ _; omega) (hmNet_ofRows R.rowNets)
    hc o ?_
  rw [hv _ (ofRows_cols_lt ha _), hv _ (ofRows_cols_lt hb' _),
    chunkBit_zOf (S := L.model regs) hk o (lt_of_lt_of_le (show r.val < 4194304 by omega) (by norm_num)), hr,
    hz off (by omega)]
  exact (FlockDeploy.Concrete.Total.hm_total x).2 off (by rw [inRows_HM_IN]; omega)
    (by rw [hmLaid_constPos hn]; show off < 246784; omega)

/-- **A `sha512x3` slot's computed row holds** where the trace on the slot is the net's forward evaluation from some
inputs (`Total.sha_total`) and the slot constants' pairs skip the row. -/
theorem holds_shaNet_at {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) {s b : ℕ} (hsl : SlotAt st.c st.c.shaNet s b)
    (hb : b + 262144 ≤ 4194304) (hiw : (st.c.nets.getD st.c.shaNet default).2.inWords = 64) {x : ℕ → Bool}
    (hz : ∀ j < 262144, zI (b + j) = vSof x j) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} {off : ℕ} (ho : 8192 ≤ off ∧ off < 256000) (hr : r.val = b + off)
    (hc : ∀ j, (constPairs st.c st.pin).count (r.val, j) = 0) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  have R := RowRanges.ofL L F.parse F.check
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
  refine holds_netRow L regs F (Or.inl rfl) hat (by rw [hiw]; show 64 * 128 ≤ _; omega) (shaNet_ofRows R.rowNets)
    hc o ?_
  rw [hv _ (ofRows_cols_lt ha _), hv _ (ofRows_cols_lt hb' _),
    chunkBit_zOf (S := L.model regs) hk o (lt_of_lt_of_le (show r.val < 4194304 by omega) (by norm_num)), hr,
    hz off (by omega)]
  exact (FlockDeploy.Concrete.Total.sha_total x).2 off (by rw [inRows_SHA_IN]; omega)
    (by rw [shaLaid_constPos hn]; omega)

/-- **A row net's constant row copies the pin** where its laid row reads itself and the slot constants' pairs at it,
with that row, read the pin once. -/
theorem copyRow_netConst_at {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st) {n s b cp : ℕ}
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
  have e := fun j => rowNet_entryL L F hn hat hlo hl j
  refine ⟨fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
    rw [(e j.val).1, hc.1, hsl.2.2, hcnt]
    exact if_congr ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩ rfl rfl
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
    rw [(e j.val).2, hc.2, hsl.2.2, hcnt]
    exact if_congr ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩ rfl rfl

/-- **An `hm96` slot's constant row holds** where it copies the pin and the trace is 1 at both. -/
theorem holds_hmConst_at {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) {s b : ℕ} (hsl : SlotAt st.c st.c.hmNet s b)
    (hiw : (st.c.nets.getD st.c.hmNet default).2.inWords = 32) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = b + 246784)
    (hcnt : ∀ j, ((((#[246784] : Array ℕ).toList.map (b + ·)).count j +
      (constPairs st.c st.pin).count (r.val, j) : ℕ) : ZMod 2) = if j = st.pin then 1 else 0)
    (hz : zI r.val = true) (hp : zI st.pin = true) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  have R := RowRanges.ofL L F.parse F.check
  refine copyRow_holds (copyRow_netConst_at L regs F (Or.inr rfl) hsl (by rw [hiw]; show 32 * 128 ≤ 246784; omega)
    (by omega) (hmNet_ofRows R.rowNets) (hmLaid_const hn) hr hcnt) ?_
  rw [block_pin_of L regs hk hp o, block_zOf (S := L.model regs) hk, hz]
  rfl

/-- **A `sha512x3` slot's constant row holds** where it copies the pin and the trace is 1 at both. -/
theorem holds_shaConst_at {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) {s b : ℕ} (hsl : SlotAt st.c st.c.shaNet s b)
    (hiw : (st.c.nets.getD st.c.shaNet default).2.inWords = 64) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = b + 256000)
    (hcnt : ∀ j, ((((#[256000] : Array ℕ).toList.map (b + ·)).count j +
      (constPairs st.c st.pin).count (r.val, j) : ℕ) : ZMod 2) = if j = st.pin then 1 else 0)
    (hz : zI r.val = true) (hp : zI st.pin = true) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  have R := RowRanges.ofL L F.parse F.check
  refine copyRow_holds (copyRow_netConst_at L regs F (Or.inl rfl) hsl (by rw [hiw]; show 64 * 128 ≤ 256000; omega)
    (by omega) (shaNet_ofRows R.rowNets) (shaLaid_const hn) hr hcnt) ?_
  rw [block_pin_of L regs hk hp o, block_zOf (S := L.model regs) hk, hz]
  rfl

/-- **A unit leaf row copies its source**: its slot entry reads the row itself, and Δ's pairs at it cancel that and
read the source. -/
theorem copyRow_unitLeaf_at (htm : st.c.tmpl = none) (hU : st.c.unit = UN) (hs1 : 0 < st.c.unitRange.count)
    {u : ℕ} (hsl : st.c.unitRange.slot 0 = u) {r : Fin (2 ^ (L.model regs).kLog)} (hr : u ≤ r.val ∧ r.val < u + 16)
    (src : Fin (2 ^ (L.model regs).kLog))
    (hd : ∀ j, deltaEntry st.da r.val j = (((if r.val = j then 1 else 0) + (if src.val = j then 1 else 0) : ℕ) : ZMod 2)) :
    CopyRow (L.model regs) r src := by
  have hB := L.built_flat htm
  have hW : 0 < 2 ^ st.c.unit.unitLog := Nat.two_pow_pos _
  have hc : r.val - u < 2 ^ st.c.unit.unitLog := by rw [hU]; show _ < 512; omega
  have hr' : r.val = st.c.unitRange.slot 0 + (r.val - u) := by rw [hsl]; omega
  have hin := UN_in (r.val - u) (by omega)
  rw [← hU] at hin
  have hlk : st.c.unit.lookup = none := L.unit_lookup
  have key : ∀ (e : Kind → ℕ → ℕ → ZMod 2),
      (∀ j', e (.net st.c.unitNet) (r.val - u) j' = if j' = r.val - u then 1 else 0) →
      ∀ j : ℕ, rangesEntry st e r.val j = if j = r.val then 1 else 0 := by
    intro e he j
    rw [hr', L.rangesEntry_unit hs1 hc e j,
      ite_slot_eq (v := e (.net st.c.unitNet) (r.val - u)) hW hc he, ← L.slot_eq]
  have hA : ∀ j', slotEntryA st (.net st.c.unitNet) (r.val - u) j' = if j' = r.val - u then 1 else 0 := by
    intro j'
    show sparseEntry st.c.unit.a (r.val - u) j' = _
    rw [sparseEntry_built_a hB, hin.1, count_one]
  have hBe : ∀ j', slotEntryB st (.net st.c.unitNet) (r.val - u) j' = if j' = r.val - u then 1 else 0 := by
    intro j'
    show sparseEntry st.c.unit.b (r.val - u) j' + lookupEntry st.c.unit.lookup (r.val - u) j' = _
    rw [sparseEntry_built_b hB, hin.2, count_one, hlk]
    simp only [lookupEntry, add_zero]
  refine ⟨fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
    rw [key (slotEntryA st) hA, hd, ite_cancel]
    exact if_congr Fin.ext_iff.symm rfl rfl
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
    rw [L.delta_b, key (slotEntryB st) hBe, hd, ite_cancel]
    exact if_congr Fin.ext_iff.symm rfl rfl

/-- **A unit leaf row holds** where it copies its source and the trace is 0 at both. -/
theorem holds_unitLeaf_at (hk : st.c.kLog = 22) (htm : st.c.tmpl = none) (hU : st.c.unit = UN)
    (hs1 : 0 < st.c.unitRange.count) {u : ℕ} (hsl : st.c.unitRange.slot 0 = u) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : u ≤ r.val ∧ r.val < u + 16) (src : Fin (2 ^ (L.model regs).kLog))
    (hd : ∀ j, deltaEntry st.da r.val j = (((if r.val = j then 1 else 0) + (if src.val = j then 1 else 0) : ℕ) : ZMod 2))
    (he : zI r.val = zI src.val) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  holds_copy L regs hk o (copyRow_unitLeaf_at L regs htm hU hs1 hsl hr src hd) he

/-- **A unit row past its 16 inputs and before its computed rows holds**: it is zero where Δ skips it, and so is the
trace there. -/
theorem holds_unitPad_at (hk : st.c.kLog = 22) (htm : st.c.tmpl = none) (hU : st.c.unit = UN)
    (hs1 : 0 < st.c.unitRange.count) {u : ℕ} (hsl : st.c.unitRange.slot 0 = u) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : u + 16 ≤ r.val ∧ r.val < u + 128)
    (hd : ∀ j, deltaEntry st.da r.val j = 0) (hz : zI r.val = false) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  have he := UN_empty (r.val - (u + 16)) (by omega)
  rw [show 16 + (r.val - (u + 16)) = r.val - u by omega, ← hU] at he
  refine holds_zero L regs hk o (zeroRow_unitEmpty L regs (L.built_flat htm) (s₀ := 0) hs1 ?_ he.1 he.2 ?_ hd) hz
  · rw [hU]; show _ < 512; omega
  · rw [hsl]; omega

/-- **A unit row past its constant holds**: it is zero where Δ skips it, and so is the trace there. -/
theorem holds_unitPast_at (hk : st.c.kLog = 22) (htm : st.c.tmpl = none) (hU : st.c.unit = UN)
    (hs1 : 0 < st.c.unitRange.count) {u : ℕ} (hsl : st.c.unitRange.slot 0 = u) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : u + 256 < r.val ∧ r.val < u + 512)
    (hd : ∀ j, deltaEntry st.da r.val j = 0) (hz : zI r.val = false) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  have he := UN_past (r.val - (u + 257)) (by omega)
  rw [show 257 + (r.val - (u + 257)) = r.val - u by omega, ← hU] at he
  refine holds_zero L regs hk o (zeroRow_unitEmpty L regs (L.built_flat htm) (s₀ := 0) hs1 ?_ he.1 he.2 ?_ hd) hz
  · rw [hU]; show _ < 512; omega
  · rw [hsl]; omega

/-- **The unit's computed rows hold** where the trace is 0 at its inputs and at the row: they read only its inputs. -/
theorem holds_unitComp_at (hk : st.c.kLog = 22) (htm : st.c.tmpl = none) (hU : st.c.unit = UN)
    (hs1 : 0 < st.c.unitRange.count) {u : ℕ} (hsl : st.c.unitRange.slot 0 = u) (hu : u + 512 ≤ 4194304)
    (hzi : ∀ y < 16, zI (u + y) = false) (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)}
    (hr : u + 128 ≤ r.val ∧ r.val < u + 256) (hz : zI r.val = false) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  have hc : r.val = st.c.unitRange.slot 0 + (r.val - u) := by rw [hsl]; omega
  have hlo : st.c.unit.inWords * WORD ≤ r.val - u := by rw [hU]; show 1 * 128 ≤ _; omega
  have hhi : r.val - u < st.c.unit.constPos := by rw [hU]; show _ < 256; omega
  refine holds_unitComp L regs (L.built_flat htm) hs1 hlo hhi o hc (fun y hy => ?_) ?_
  · clear hc hlo hhi
    rw [hU] at hy
    have hy' : y < 16 := UN_a_lt (r.val - (u + 128)) (by omega) y
      (by rwa [show 128 + (r.val - (u + 128)) = r.val - u by omega])
    rw [hsl, chunkBit_zOf (S := L.model regs) hk o (lt_of_lt_of_le (show u + y < 4194304 by omega) (by norm_num))]
    exact hzi y hy'
  · clear hc hlo hhi
    rw [chunkBit_zOf (S := L.model regs) hk o (lt_of_lt_of_le (show r.val < 4194304 by omega) (by norm_num))]
    exact hz

/-- **The unit's constant row holds** where the trace is 1 at it and at the pin. -/
theorem holds_unitConst_at (hk : st.c.kLog = 22) (hs1 : 0 < st.c.unitRange.count) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = st.c.unitRange.slot 0 + st.c.unit.constPos)
    (hz : zI r.val = true) (hp : zI st.pin = true) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r := by
  refine copyRow_holds (copyRow_unitConst L regs hs1 hr) ?_
  rw [block_pin_of L regs hk hp o, block_zOf (S := L.model regs) hk, hz]
  rfl

/-- **A mask row reads itself** where the mask's range is one slot of `2 ^ 14` at `64 · 2 ^ 14` and Δ skips it. -/
theorem copyRow_mask_at
    (hm : (Kind.mask, 14, 64, 1) ∈ st.c.ranges.toList.map (fun r => (r.kind, r.slotLog, r.pos0, r.count)))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : 1048576 ≤ r.val ∧ r.val < 1064960)
    (hd : ∀ j, deltaEntry st.da r.val j = 0) :
    CopyRow (L.model regs) r r := by
  obtain ⟨R, hR, hf⟩ := List.mem_map.1 hm
  simp only [Prod.mk.injEq] at hf
  obtain ⟨hk, hs, hp, hc⟩ := hf
  have hcov : FlockLevel3.covers R r.val := by
    unfold FlockLevel3.covers
    rw [hs, hp, hc]
    norm_num
    omega
  have key : ∀ e : Kind → ℕ → ℕ → ZMod 2, (∀ i i', e .mask i i' = if i = i' then 1 else 0) →
      ∀ j : Fin (2 ^ (L.model regs).kLog), rangesEntry st e r.val j.val = if j = r then 1 else 0 := by
    intro e he j
    rw [rangesEntry_of_covers L.disj hR hcov e j.val, hk, he, hs]
    by_cases hj : j = r
    · subst hj; simp
    · have hv : j.val ≠ r.val := fun h => hj (Fin.ext h)
      rw [ite_eq_right hj]
      by_cases hq : j.val / 2 ^ 14 = r.val / 2 ^ 14
      · rw [ite_eq_left hq, ite_eq_right]
        intro hm'
        exact hv (by rw [← Nat.div_add_mod j.val (2 ^ 14), ← Nat.div_add_mod r.val (2 ^ 14), hq, hm'])
      · rw [ite_eq_right hq]
  refine ⟨fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
    rw [hd, add_zero]
    exact key _ (fun _ _ => rfl) j
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
    rw [L.delta_b, hd, add_zero]
    exact key _ (fun _ _ => rfl) j

/-- **A mask row holds.** -/
theorem holds_mask_at
    (hm : (Kind.mask, 14, 64, 1) ∈ st.c.ranges.toList.map (fun r => (r.kind, r.slotLog, r.pos0, r.count)))
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : 1048576 ≤ r.val ∧ r.val < 1064960)
    (hd : ∀ j, deltaEntry st.da r.val j = 0) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  copyRow_holds (copyRow_mask_at L regs hm hr hd) rfl

/-- **A row no range covers holds** where Δ skips it and the trace is 0 there. -/
theorem holds_outside (hk : st.c.kLog = 22) (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)}
    (hc : ∀ R ∈ st.c.ranges.toList, ¬ FlockLevel3.covers R r.val) (hd : ∀ j, deltaEntry st.da r.val j = 0)
    (hz : zI r.val = false) :
    RowHolds (L.model regs) (zOf zI (L.model regs).m) o r :=
  holds_zero L regs hk o (zeroRow_outside L regs hc hd) hz

end place

end SpotFlock.InstanceStmt.TraceRows
