module

public import Proofs.Flock.SpotFlock.InstanceStmt.TraceNetRows
public import Proofs.Flock.SpotFlock.InstanceStmt.TraceDelta
public import Proofs.Flock.SpotFlock.InstanceStmt.Trace
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
# The trace satisfies the two-instance table's rows

`holds_all`: every row of the block holds at the trace `TraceVal.zIX`, by place. It reads the statement through `Facts`: its
slots and constants (`TraceFacts`), its Δ as `EX ++ OX` (`TraceDa.da_stX`) and its ranges. The first `hm96` slot's rows are
its `cv` (copies of the `sha512x3` slot's first output), `pad` (copies of the padding block's bits, from the pin or the
forced-zero row), `y`, padding, computed rows, constant (the pin) and past rows; the second's likewise but for its constant
row (a copy of the pin) and the 16 message bits of its `pad`, copies of the unit's outputs; the `sha512x3` slot's, with `cv0`
copies of the midstate's bits; the unit's leaf, padding, computed, constant and past rows; and the mask's and the uncovered rows.
`satisfies_rows` is the model's rows and pin at no public region.
-/

@[expose] public section

open Flock FlockLevel3 FlockSoundness FlockSoundness.Discharge.Hm
open Flock.HmNets hiding and
open FlockDeploy.Concrete (UN)
open SpotFlock.InstanceStmt.TraceRows
open SpotFlock.InstanceStmt.TraceDelta (PIN ZERO consts EX OX msI msO padI padO cs csU srcB consts_count_zero delta_zero
  delta_R1 delta_R2 delta_R3a delta_R3b delta_R4 delta_R5a delta_R5b delta_R6 delta_O1 delta_O2 delta_selfForm form_flip)
open SpotFlock.InstanceStmt.TraceVal

set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt.TraceHolds

/-- The table's ranges, as `TraceFacts.ranges_cO` lists them. -/
abbrev RS : List (Kind × ℕ × ℕ × ℕ) :=
  [(Kind.net 2, 18, 0, 2), (Kind.net 1, 18, 2, 1), (Kind.net 0, 9, 1536, 1), (Kind.mask, 14, 64, 1)]

/-- **No range covers a column of the gaps**: past the unit's slot and before the mask's, and past the mask's. -/
theorem not_covers_of {rs : List Range} (hrs : rs.map (fun r => (r.kind, r.slotLog, r.pos0, r.count)) = RS) {x : ℕ}
    (hx : (786944 ≤ x ∧ x < 1048576) ∨ 1064960 ≤ x) : ∀ R ∈ rs, ¬ FlockLevel3.covers R x := by
  intro R hR hc
  have hm := List.mem_map_of_mem (f := fun r : Range => (r.kind, r.slotLog, r.pos0, r.count)) hR
  rw [hrs] at hm
  unfold FlockLevel3.covers at hc
  simp only [RS, List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
  rcases hm with ⟨-, h1, h2, h3⟩ | ⟨-, h1, h2, h3⟩ | ⟨-, h1, h2, h3⟩ | ⟨-, h1, h2, h3⟩ <;>
    rw [h1, h2, h3] at hc <;> norm_num at hc <;> omega

/-- At the second `hm96` slot's constant row, its laid row's self entry and the slot constant's self pair cancel, leaving
the pin. -/
theorem constRow_count_h1 (j : ℕ) :
    ((((#[246784] : Array ℕ).toList.map (262144 + ·)).count j + consts.count (508928, j) : ℕ) : ZMod 2) =
      if j = PIN then 1 else 0 := by
  by_cases h1 : j = 508928
  · subst h1; decide +kernel
  · by_cases h2 : j = PIN
    · subst h2; decide +kernel
    · rw [ite_eq_right h2]
      have e1 : ((#[246784] : Array ℕ).toList.map (262144 + ·)).count j = 0 :=
        List.count_eq_zero.2 fun hm => by simp at hm; omega
      have e2 : consts.count (508928, j) = 0 :=
        List.count_eq_zero.2 fun hm => by
          simp only [consts, List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
          omega
      rw [e1, e2, Nat.add_zero, Nat.cast_zero]

/-- The same at the `sha512x3` slot's constant row. -/
theorem constRow_count_sha (j : ℕ) :
    ((((#[256000] : Array ℕ).toList.map (524288 + ·)).count j + consts.count (780288, j) : ℕ) : ZMod 2) =
      if j = PIN then 1 else 0 := by
  by_cases h1 : j = 780288
  · subst h1; decide +kernel
  · by_cases h2 : j = PIN
    · subst h2; decide +kernel
    · rw [ite_eq_right h2]
      have e1 : ((#[256000] : Array ℕ).toList.map (524288 + ·)).count j = 0 :=
        List.count_eq_zero.2 fun hm => by simp at hm; omega
      have e2 : consts.count (780288, j) = 0 :=
        List.count_eq_zero.2 fun hm => by
          simp only [consts, List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
          omega
      rw [e1, e2, Nat.add_zero, Nat.cast_zero]

/-- **What the rows' proof reads of the statement**: its layout's numbers (`TraceFacts`), its Δ (`TraceDa.da_stX`) and
its ranges. -/
structure Facts (st : Stmt) : Prop where
  kLog : st.c.kLog = 22
  pin : st.pin = 246784
  tmpl : st.c.tmpl = none
  unit : st.c.unit = UN
  unitCount : st.c.unitRange.count = 1
  unitSlot : st.c.unitRange.slot 0 = 786432
  sha : SlotAt st.c st.c.shaNet 0 524288
  hm0 : SlotAt st.c st.c.hmNet 0 0
  hm1 : SlotAt st.c st.c.hmNet 1 262144
  iwSha : (st.c.nets.getD st.c.shaNet default).2.inWords = 64
  iwHm : (st.c.nets.getD st.c.hmNet default).2.inWords = 32
  constPairs : constPairs st.c st.pin = consts
  da : st.da.toList = EX ++ OX
  ranges : st.c.ranges.toList.map (fun r => (r.kind, r.slotLog, r.pos0, r.count)) = RS

theorem hold_pin_true {st : Stmt} (F : Facts st) (hn : Properties.Flock.Nets.FixedNetsBuild) : zIX st.pin = true := by
  rw [F.pin]
  exact zIX_pin hn

section place

variable {st : Stmt} (L : FlockSoundness.Layout st) (R : RowRanges st.c) (F : Facts st)
  (hn : Properties.Flock.Nets.FixedNetsBuild)
include R F hn

/-- **Every row of the block holds** at the trace, by its place. -/
theorem holds_all (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (o : Fin (2 ^ (L.model regs).nbl)) (r : Fin (2 ^ (L.model regs).kLog)) :
    RowHolds (L.model regs) (zOf zIX (L.model regs).m) o r := by
  have hk := F.kLog
  have hK : (L.model regs).kLog = 22 := hk
  have mk : ∀ y : ℕ, y < 4194304 → y < 2 ^ (L.model regs).kLog := fun y hy =>
    lt_of_lt_of_le hy (by rw [hK]; norm_num)
  have hp : zIX st.pin = true := hold_pin_true F hn
  have hz0 : ∀ x : ℕ, (1536 ≤ x ∧ (x < 262144 ∨ 263680 ≤ x) ∧ (x < 524288 ∨ 524800 ≤ x) ∧ (x < 786432 ∨ 786448 ≤ x) ∧
      x ≠ 508928 ∧ x ≠ 780288 ∧ x ≠ 786688) → ∀ j, deltaEntry st.da x j = 0 :=
    fun x hx j => delta_zero F.da hx j
  have hc0 : ∀ x : ℕ, (x ≠ 508928 ∧ x ≠ 780288 ∧ x ≠ 786688) →
      ∀ j, (constPairs st.c st.pin).count (x, j) = 0 := fun x hx j => by
    rw [F.constPairs]
    exact consts_count_zero hx j
  have hs1 : 0 < st.c.unitRange.count := by rw [F.unitCount]; exact Nat.one_pos
  have hrv : r.val < 4194304 := lt_of_lt_of_eq r.isLt (by rw [hK]; norm_num)
  -- hm slot 0
  by_cases h1 : r.val < 512
  · exact holds_hmIn_at L regs (zI := zIX) hk R F.hm0 o (off := r.val) (by omega) (by omega)
      ⟨778240 + r.val, mk _ (by omega)⟩ (fun j => (delta_R6 F.da h1 j).trans (form_flip _ _ j))
      (zIX_cv h1)
  by_cases h2 : r.val < 1536
  · exact holds_hmIn_at L regs (zI := zIX) hk R F.hm0 o (off := r.val) (by omega) (by omega)
      ⟨cs padI (r.val - 512), mk _ (cs_lt4 _ _)⟩
      (fun j => (delta_R2 F.da ⟨by omega, h2⟩ j).trans (form_flip _ _ j)) (zIX_pad0 hn (by omega) h2)
  by_cases h3 : r.val < 3072
  · exact holds_hmIn_at L regs (zI := zIX) hk R F.hm0 o (off := r.val) (by omega) (by omega) r
      (fun j => delta_selfForm (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩) j) rfl
  by_cases h4 : r.val < 4096
  · exact holds_hmGap_at L regs (zI := zIX) hk R F.hm0 o (off := r.val) ⟨by omega, by omega⟩ (by omega)
      (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
      (zIX_hm0_zero (by omega) h4)
  by_cases h5 : r.val < 246784
  · exact holds_hmNet_atR L regs (zI := zIX) R hn hk F.hm0 (by norm_num) F.iwHm (x := xH0)
      (fun j hj => by rw [Nat.zero_add]; exact zIX_h0 hj) o (off := r.val) ⟨by omega, h5⟩ (by omega)
      (hc0 r.val ⟨by omega, by omega, by omega⟩)
  by_cases h6 : r.val = 246784
  · exact holds_pin L regs (zI := zIX) o (by rw [F.pin]; exact h6)
  by_cases h7 : r.val < 262144
  · exact holds_hmPast_at L regs (zI := zIX) hn hk R F.hm0 o (off := r.val) ⟨by omega, h7⟩ (by omega)
      (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
      (zIX_h0_past hn (by omega) h7)
  -- hm slot 1
  by_cases h8a : r.val < 262656
  · exact holds_hmIn_at L regs (zI := zIX) hk R F.hm1 o (off := r.val - 262144) (by omega) (by omega)
      ⟨csU msO (r.val - 262144), mk _ (csU_lt4 _ _)⟩
      (fun j => (delta_R3a F.da ⟨by omega, h8a⟩ j).trans (form_flip _ _ j)) (zIX_R3a hn (by omega) h8a)
  by_cases h8 : r.val < 262704
  · exact holds_hmIn_at L regs (zI := zIX) hk R F.hm1 o (off := r.val - 262144) (by omega) (by omega)
      ⟨cs padO (r.val - 262656), mk _ (cs_lt4 _ _)⟩
      (fun j => (delta_R3b F.da ⟨by omega, h8⟩ j).trans (form_flip _ _ j)) (zIX_R3b hn (by omega) h8)
  by_cases h9 : r.val < 262712
  · exact holds_hmIn_at L regs (zI := zIX) hk R F.hm1 o (off := r.val - 262144) (by omega) (by omega)
      ⟨786568 + (r.val - 262704), mk _ (by omega)⟩
      (fun j => (delta_O2 F.da ⟨by omega, h9⟩ j).trans (form_flip _ _ j))
      (by rw [zIX_msg (by omega) (by omega), zIX_unit_false (i := 786568 + (r.val - 262704)) (by omega) (by omega)])
  by_cases h10 : r.val < 262720
  · exact holds_hmIn_at L regs (zI := zIX) hk R F.hm1 o (off := r.val - 262144) (by omega) (by omega)
      ⟨786560 + (r.val - 262712), mk _ (by omega)⟩
      (fun j => (delta_O1 F.da ⟨by omega, h10⟩ j).trans (form_flip _ _ j))
      (by rw [zIX_msg (by omega) (by omega), zIX_unit_false (i := 786560 + (r.val - 262712)) (by omega) (by omega)])
  by_cases h11 : r.val < 263680
  · exact holds_hmIn_at L regs (zI := zIX) hk R F.hm1 o (off := r.val - 262144) (by omega) (by omega)
      ⟨srcB (r.val - 262720), mk _ (srcB_lt4 _)⟩
      (fun j => (delta_R4 F.da ⟨by omega, h11⟩ j).trans (form_flip _ _ j)) (zIX_srcB hn (by omega) h11)
  by_cases h12 : r.val < 265216
  · exact holds_hmIn_at L regs (zI := zIX) hk R F.hm1 o (off := r.val - 262144) (by omega) (by omega) r
      (fun j => delta_selfForm (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩) j) rfl
  by_cases h13 : r.val < 266240
  · exact holds_hmGap_at L regs (zI := zIX) hk R F.hm1 o (off := r.val - 262144) ⟨by omega, by omega⟩ (by omega)
      (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
      (zIX_hm1_zero (by omega) h13)
  by_cases h14 : r.val < 508928
  · exact holds_hmNet_atR L regs (zI := zIX) R hn hk F.hm1 (by norm_num) F.iwHm (x := xH1)
      (fun j hj => zIX_h1 hj) o (off := r.val - 262144) ⟨by omega, by omega⟩ (by omega)
      (hc0 r.val ⟨by omega, by omega, by omega⟩)
  by_cases h15 : r.val = 508928
  · exact holds_hmConst_atR L regs (zI := zIX) R hn hk F.hm1 F.iwHm o (by omega)
      (fun j => by rw [F.constPairs, F.pin, show r.val = 508928 from h15]; exact constRow_count_h1 j)
      (by rw [h15]; exact zIX_h1Const hn) hp
  by_cases h16 : r.val < 524288
  · exact holds_hmPast_at L regs (zI := zIX) hn hk R F.hm1 o (off := r.val - 262144) ⟨by omega, by omega⟩ (by omega)
      (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
      (zIX_h1_past hn (by omega) h16)
  -- sha slot
  by_cases h17 : r.val < 524800
  · exact holds_shaIn_at L regs (zI := zIX) hk R F.sha o (off := r.val - 524288) (by omega) (by omega)
      ⟨csU msI (r.val - 524288), mk _ (csU_lt4 _ _)⟩
      (fun j => (delta_R1 F.da ⟨by omega, h17⟩ j).trans (form_flip _ _ j)) (zIX_sha_cv0 hn (by omega) h17)
  by_cases h18 : r.val < 528896
  · exact holds_shaIn_at L regs (zI := zIX) hk R F.sha o (off := r.val - 524288) (by omega) (by omega) r
      (fun j => delta_selfForm (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩) j) rfl
  by_cases h19 : r.val < 532480
  · exact holds_shaGap_at L regs (zI := zIX) hk R F.sha o (off := r.val - 524288) ⟨by omega, by omega⟩ (by omega)
      (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
      (zIX_sha_gap (by omega) h19)
  by_cases h20 : r.val < 780288
  · exact holds_shaNet_atR L regs (zI := zIX) R hn hk F.sha (by norm_num) F.iwSha (x := xS)
      (fun j hj => zIX_s hj) o (off := r.val - 524288) ⟨by omega, by omega⟩ (by omega)
      (hc0 r.val ⟨by omega, by omega, by omega⟩)
  by_cases h21 : r.val = 780288
  · exact holds_shaConst_atR L regs (zI := zIX) R hn hk F.sha F.iwSha o (by omega)
      (fun j => by rw [F.constPairs, F.pin, show r.val = 780288 from h21]; exact constRow_count_sha j)
      (by rw [h21]; exact zIX_shaConst hn) hp
  by_cases h22 : r.val < 786432
  · exact holds_shaPast_at L regs (zI := zIX) hn hk R F.sha o (off := r.val - 524288) ⟨by omega, by omega⟩ (by omega)
      (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
      (zIX_sha_past hn (by omega) h22)
  -- the unit slot
  by_cases h23a : r.val < 786440
  · exact holds_unitLeaf_at L regs (zI := zIX) hk F.tmpl F.unit hs1 F.unitSlot o ⟨by omega, by omega⟩
      ⟨524856 + (r.val - 786432), mk _ (by omega)⟩
      (fun j => delta_R5a F.da (x := r.val) ⟨by omega, h23a⟩ j)
      (by rw [zIX_unit_false (i := r.val) (by omega) (by omega)]
          exact (zIX_sha_in (x := 524856 + (r.val - 786432)) (by omega) (by omega)).symm)
  by_cases h23 : r.val < 786448
  · exact holds_unitLeaf_at L regs (zI := zIX) hk F.tmpl F.unit hs1 F.unitSlot o ⟨by omega, by omega⟩
      ⟨524848 + (r.val - 786440), mk _ (by omega)⟩
      (fun j => delta_R5b F.da (x := r.val) ⟨by omega, h23⟩ j)
      (by rw [zIX_unit_false (i := r.val) (by omega) (by omega)]
          exact (zIX_sha_in (x := 524848 + (r.val - 786440)) (by omega) (by omega)).symm)
  by_cases h24 : r.val < 786560
  · exact holds_unitPad_at L regs (zI := zIX) hk F.tmpl F.unit hs1 F.unitSlot o ⟨by omega, by omega⟩
      (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
      (zIX_unit_false (by omega) (by omega))
  by_cases h25 : r.val < 786688
  · exact holds_unitComp_at L regs (zI := zIX) hk F.tmpl F.unit hs1 F.unitSlot (by norm_num)
      (fun y hy => zIX_unit_false (i := 786432 + y) (by omega) (by omega)) o ⟨by omega, by omega⟩
      (zIX_unit_false (by omega) (by omega))
  by_cases h26 : r.val = 786688
  · exact holds_unitConst_at L regs (zI := zIX) hk hs1 o
      (by rw [F.unitSlot, F.unit]; show r.val = 786432 + 256; omega) (by rw [h26]; exact zIX_unitConst) hp
  by_cases h27 : r.val < 786944
  · exact holds_unitPast_at L regs (zI := zIX) hk F.tmpl F.unit hs1 F.unitSlot o ⟨by omega, by omega⟩
      (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
      (zIX_unit_false (by omega) (by omega))
  by_cases h28 : r.val < 1048576
  · exact holds_outside L regs (zI := zIX) hk o (not_covers_of (x := r.val) F.ranges (Or.inl ⟨by omega, h28⟩))
      (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
      (zIX_unit_false (by omega) (by omega))
  by_cases h29 : r.val < 1064960
  · exact holds_mask_at L regs (zI := zIX) (by rw [F.ranges]; simp [RS]) o ⟨by omega, h29⟩
      (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
  exact holds_outside L regs (zI := zIX) hk o (not_covers_of (x := r.val) F.ranges (Or.inr (by omega)))
    (hz0 r.val ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩)
    (zIX_unit_false (by omega) (by omega))

/-- **The trace satisfies the table's rows and pin**, at no public region. -/
theorem satisfies_rows : (L.model []).Satisfies (zOf zIX (L.model []).m) :=
  ⟨fun o r => holds_all L R F hn [] o r, fun o => block_pin_of L [] F.kLog (hold_pin_true F hn) o,
    fun R' hR' => absurd hR' List.not_mem_nil⟩

end place

end SpotFlock.InstanceStmt.TraceHolds
