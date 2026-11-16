module

public import Proofs.Flock.SpotFlock.InstanceStmt.TraceDelta
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
# The two-instance table's trace

The values that satisfy every row of `x r`'s table, its fixed bits aside: every hidden bit and salt is 0. The `sha512x3`
slot holds its net's forward evaluation (`TraceRows.vSof`) from `xS`: `cv0` the first row's midstate and every other input
0. The first `hm96` slot holds its net's from `xH0`: `cv` the `sha512x3` slot's first output, `pad` the first row's padding
block and `y` 0. The second `hm96` slot holds its net's from `xH1`: `cv` the second row's midstate, `pad` its padding
block (message 0) and `y` 0. The unit slot holds its constant and nothing else, and every other column is 0 (`zIX`).

`zIX_cs` reads the trace at a constant bit's source, `zIX_leaf` and `zIX_msg` at the unit's leaf sources and outputs.
-/

@[expose] public section

open Flock FlockSoundness
open FlockDeploy.Concrete.Trace (wbit)
open SpotFlock.InstanceStmt.TraceRows
open SpotFlock.InstanceStmt.TraceDelta (PIN ZERO msI msO padI padO cs csU ubit srcB)

set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt.TraceVal

/-- The `sha512x3` slot's inputs: `cv0` the first midstate's bits, every other input 0. -/
def xS (j : ℕ) : Bool := if j < 512 then ubit msI j else false

/-- The `sha512x3` slot's column values. -/
def vS : ℕ → Bool := vSof xS

/-- The first `hm96` slot's inputs: `cv` the `sha512x3` slot's first output, `pad` the padding block's bits, `y` 0. -/
def xH0 (j : ℕ) : Bool := if j < 512 then vS (253952 + j) else if j < 1536 then wbit padI (j - 512) else false

/-- The first `hm96` slot's column values. -/
def vH0 : ℕ → Bool := vHof xH0

/-- The second `hm96` slot's inputs: `cv` the second midstate's bits, `pad` the second padding block's bits, `y` 0. -/
def xH1 (j : ℕ) : Bool := if j < 512 then ubit msO j else if j < 1536 then wbit padO (j - 512) else false

/-- The second `hm96` slot's column values. -/
def vH1 : ℕ → Bool := vHof xH1

/-- **The table's trace in one block**: the slots' values, 1 at the unit's constant and 0 elsewhere. -/
def zIX (i : ℕ) : Bool :=
  if i < 262144 then vH0 i else if i < 524288 then vH1 (i - 262144) else if i < 786432 then vS (i - 524288)
  else decide (i = 786688)

theorem zIX_h0 {i : ℕ} (h : i < 262144) : zIX i = vH0 i := by
  unfold zIX
  rw [ite_eq_left h]

theorem zIX_h1 {j : ℕ} (h : j < 262144) : zIX (262144 + j) = vH1 j := by
  unfold zIX
  rw [ite_eq_right (show ¬ 262144 + j < 262144 by omega), ite_eq_left (show 262144 + j < 524288 by omega),
    Nat.add_sub_cancel_left]

theorem zIX_s {j : ℕ} (h : j < 262144) : zIX (524288 + j) = vS j := by
  unfold zIX
  rw [ite_eq_right (show ¬ 524288 + j < 262144 by omega), ite_eq_right (show ¬ 524288 + j < 524288 by omega),
    ite_eq_left (show 524288 + j < 786432 by omega), Nat.add_sub_cancel_left]

theorem zIX_u {i : ℕ} (h : 786432 ≤ i) : zIX i = decide (i = 786688) := by
  unfold zIX
  rw [ite_eq_right (show ¬ i < 262144 by omega), ite_eq_right (show ¬ i < 524288 by omega),
    ite_eq_right (show ¬ i < 786432 by omega)]

theorem zIX_unit_false {i : ℕ} (h1 : 786432 ≤ i) (h2 : i ≠ 786688) : zIX i = false := by
  rw [zIX_u h1]
  exact decide_eq_false h2

theorem zIX_unitConst : zIX 786688 = true := by
  rw [zIX_u (i := 786688) (by norm_num)]
  exact decide_eq_true rfl

theorem zIX_pin (hn : Properties.Flock.Nets.FixedNetsBuild) : zIX 246784 = true := by
  rw [zIX_h0 (i := 246784) (by norm_num)]
  exact vHof_const hn _

theorem zIX_h1Const (hn : Properties.Flock.Nets.FixedNetsBuild) : zIX 508928 = true := by
  have h := zIX_h1 (j := 246784) (by norm_num)
  rw [show 262144 + 246784 = 508928 from rfl] at h
  rw [h]
  exact vHof_const hn _

theorem zIX_shaConst (hn : Properties.Flock.Nets.FixedNetsBuild) : zIX 780288 = true := by
  have h := zIX_s (j := 256000) (by norm_num)
  rw [show 524288 + 256000 = 780288 from rfl] at h
  rw [h]
  exact vSof_const hn _

/-- **The trace at a constant bit's source** is the bit: 1 at the pin, 0 at the forced-zero row. -/
theorem zIX_cs (hn : Properties.Flock.Nets.FixedNetsBuild) (ws : Array ℕ) (t : ℕ) : zIX (cs ws t) = wbit ws t := by
  unfold cs
  by_cases h : wbit ws t = true
  · rw [ite_eq_left h, zIX_pin hn, h]
  · rw [ite_eq_right h, zIX_unit_false (i := ZERO) (by decide) (by decide)]
    exact (Bool.eq_false_iff.2 h).symm

/-- **The trace at a midstate bit's source** is the bit. -/
theorem zIX_csU (hn : Properties.Flock.Nets.FixedNetsBuild) (ms : Array UInt64) (t : ℕ) : zIX (csU ms t) = ubit ms t := by
  unfold csU
  by_cases h : ubit ms t = true
  · rw [ite_eq_left h, zIX_pin hn, h]
  · rw [ite_eq_right h, zIX_unit_false (i := ZERO) (by decide) (by decide)]
    exact (Bool.eq_false_iff.2 h).symm

theorem cs_lt (ws : Array ℕ) (t : ℕ) : cs ws t < 2 ^ 22 := by
  unfold cs
  split <;> decide

theorem cs_lt4 (ws : Array ℕ) (t : ℕ) : cs ws t < 4194304 := lt_of_lt_of_le (cs_lt ws t) (by norm_num)

theorem csU_lt4 (ms : Array UInt64) (t : ℕ) : csU ms t < 4194304 := by
  unfold csU
  split <;> decide

theorem srcB_lt4 (t : ℕ) : srcB t < 4194304 := cs_lt4 _ _

theorem xS_input {j : ℕ} (h : j < 8192) : vS j = xS j := vSof_input h

theorem xH0_input {j : ℕ} (h : j < 4096) : vH0 j = xH0 j := vHof_input h

theorem xH1_input {j : ℕ} (h : j < 4096) : vH1 j = xH1 j := vHof_input h

/-- The second row's padding block has its 16 message bits 0. -/
theorem padO_msg : ∀ k < 16, wbit padO (48 + k) = false := by decide +kernel

/-- Rows `0 … 511` copy the `sha512x3` slot's first output. -/
theorem zIX_cv {x : ℕ} (hx : x < 512) : zIX x = zIX (778240 + x) := by
  have h2 := zIX_s (j := 253952 + x) (by omega)
  rw [show 524288 + (253952 + x) = 778240 + x by omega] at h2
  rw [h2, zIX_h0 (i := x) (by omega), xH0_input (j := x) (by omega)]
  unfold xH0
  rw [ite_eq_left hx]

theorem zIX_pad0 (hn : Properties.Flock.Nets.FixedNetsBuild) {x : ℕ} (h1 : 512 ≤ x) (h2 : x < 1536) :
    zIX x = zIX (cs padI (x - 512)) := by
  rw [zIX_cs hn, zIX_h0 (i := x) (by omega), xH0_input (j := x) (by omega)]
  unfold xH0
  rw [ite_eq_right (show ¬ x < 512 by omega), ite_eq_left h2]

theorem zIX_hm0_zero {x : ℕ} (h1 : 1536 ≤ x) (h2 : x < 4096) : zIX x = false := by
  rw [zIX_h0 (i := x) (by omega), xH0_input (j := x) h2]
  unfold xH0
  rw [ite_eq_right (show ¬ x < 512 by omega), ite_eq_right (show ¬ x < 1536 by omega)]

theorem zIX_h0_past (hn : Properties.Flock.Nets.FixedNetsBuild) {x : ℕ} (h1 : 246784 < x) (h2 : x < 262144) :
    zIX x = false := by
  rw [zIX_h0 (i := x) h2]
  exact vHof_past hn xH0 h1

theorem zIX_R3a (hn : Properties.Flock.Nets.FixedNetsBuild) {x : ℕ} (h1 : 262144 ≤ x) (h2 : x < 262656) :
    zIX x = zIX (csU msO (x - 262144)) := by
  have h := zIX_h1 (j := x - 262144) (by omega)
  rw [show 262144 + (x - 262144) = x by omega] at h
  rw [zIX_csU hn, h, xH1_input (j := x - 262144) (by omega)]
  unfold xH1
  rw [ite_eq_left (show x - 262144 < 512 by omega)]

theorem zIX_R3b (hn : Properties.Flock.Nets.FixedNetsBuild) {x : ℕ} (h1 : 262656 ≤ x) (h2 : x < 262704) :
    zIX x = zIX (cs padO (x - 262656)) := by
  have h := zIX_h1 (j := x - 262144) (by omega)
  rw [show 262144 + (x - 262144) = x by omega] at h
  rw [zIX_cs hn, h, xH1_input (j := x - 262144) (by omega)]
  unfold xH1
  rw [ite_eq_right (show ¬ x - 262144 < 512 by omega), ite_eq_left (show x - 262144 < 1536 by omega),
    show x - 262144 - 512 = x - 262656 by omega]

theorem zIX_msg {x : ℕ} (h1 : 262704 ≤ x) (h2 : x < 262720) : zIX x = false := by
  have h := zIX_h1 (j := x - 262144) (by omega)
  rw [show 262144 + (x - 262144) = x by omega] at h
  rw [h, xH1_input (j := x - 262144) (by omega)]
  unfold xH1
  rw [ite_eq_right (show ¬ x - 262144 < 512 by omega), ite_eq_left (show x - 262144 < 1536 by omega)]
  have := padO_msg (x - 262144 - 512 - 48) (by omega)
  rwa [show 48 + (x - 262144 - 512 - 48) = x - 262144 - 512 by omega] at this

theorem zIX_srcB (hn : Properties.Flock.Nets.FixedNetsBuild) {x : ℕ} (h1 : 262720 ≤ x) (h2 : x < 263680) :
    zIX x = zIX (srcB (x - 262720)) := by
  have h := zIX_h1 (j := x - 262144) (by omega)
  rw [show 262144 + (x - 262144) = x by omega] at h
  rw [h, xH1_input (j := x - 262144) (by omega)]
  unfold srcB xH1
  rw [zIX_cs hn, ite_eq_right (show ¬ x - 262144 < 512 by omega), ite_eq_left (show x - 262144 < 1536 by omega),
    show x - 262720 + 64 = x - 262144 - 512 by omega]

theorem zIX_hm1_zero {x : ℕ} (h1 : 263680 ≤ x) (h2 : x < 266240) : zIX x = false := by
  have h := zIX_h1 (j := x - 262144) (by omega)
  rw [show 262144 + (x - 262144) = x by omega] at h
  rw [h, xH1_input (j := x - 262144) (by omega)]
  unfold xH1
  rw [ite_eq_right (show ¬ x - 262144 < 512 by omega), ite_eq_right (show ¬ x - 262144 < 1536 by omega)]

theorem zIX_h1_past (hn : Properties.Flock.Nets.FixedNetsBuild) {x : ℕ} (h1 : 508928 < x) (h2 : x < 524288) :
    zIX x = false := by
  have h := zIX_h1 (j := x - 262144) (by omega)
  rw [show 262144 + (x - 262144) = x by omega] at h
  rw [h]
  exact vHof_past hn xH1 (by omega)

theorem zIX_sha_cv0 (hn : Properties.Flock.Nets.FixedNetsBuild) {x : ℕ} (h1 : 524288 ≤ x) (h2 : x < 524800) :
    zIX x = zIX (csU msI (x - 524288)) := by
  have h := zIX_s (j := x - 524288) (by omega)
  rw [show 524288 + (x - 524288) = x by omega] at h
  rw [zIX_csU hn, h, xS_input (j := x - 524288) (by omega)]
  unfold xS
  rw [ite_eq_left (show x - 524288 < 512 by omega)]

theorem zIX_sha_in {x : ℕ} (h1 : 524800 ≤ x) (h2 : x < 528896) : zIX x = false := by
  have h := zIX_s (j := x - 524288) (by omega)
  rw [show 524288 + (x - 524288) = x by omega] at h
  rw [h, xS_input (j := x - 524288) (by omega)]
  unfold xS
  rw [ite_eq_right (show ¬ x - 524288 < 512 by omega)]

theorem zIX_sha_gap {x : ℕ} (h1 : 528896 ≤ x) (h2 : x < 532480) : zIX x = false := by
  have h := zIX_s (j := x - 524288) (by omega)
  rw [show 524288 + (x - 524288) = x by omega] at h
  rw [h, xS_input (j := x - 524288) (by omega)]
  unfold xS
  rw [ite_eq_right (show ¬ x - 524288 < 512 by omega)]

theorem zIX_sha_past (hn : Properties.Flock.Nets.FixedNetsBuild) {x : ℕ} (h1 : 780288 < x) (h2 : x < 786432) :
    zIX x = false := by
  have h := zIX_s (j := x - 524288) (by omega)
  rw [show 524288 + (x - 524288) = x by omega] at h
  rw [h]
  exact vSof_past hn xS (by omega)

end SpotFlock.InstanceStmt.TraceVal
