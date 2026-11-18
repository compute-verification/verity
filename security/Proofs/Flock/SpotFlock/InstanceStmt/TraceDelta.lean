module

public import Proofs.Flock.SessionLoop.Concrete.Trace
public import Proofs.Flock.SpotFlock.InstanceStmt.Bytes
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
# Δ of the two-instance table, as runs

`EX ++ OX` is the list of Δ's pairs of `x r`'s table (`HmOut.delta`: `HmRow.delta`'s, then the output rows'): the three
slot constants' self pairs and pins, then runs of copies `(b + t, b + t), (b + t, f t)`: the `sha512x3` slot's `cv0` from
the first row's midstate, the first `hm96` slot's `pad` (the padding block) and `cv` (the `sha512x3` slot's first output),
the second `hm96` slot's `cv` and `pad` from the second row's midstate and padding block (but for its 16 message bits,
which are `OX`'s, copies of the unit's output), and the unit's 16 leaf bits (the first row's first two bytes).

The pairs of a row are read off the list: one lemma per run (`delta_R1` … `delta_O2`), each for a row inside it, and
`delta_zero` for a row no run or constant touches. They read Δ only through `da.toList = EX ++ OX`.
-/

@[expose] public section

open Flock FlockLevel3 FlockSoundness
open FlockDeploy.Concrete.Trace (run count_run wbit)

set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt.TraceDelta

/-- The pinned column: the first `hm96` slot's constant. -/
abbrev PIN : ℕ := 246784
/-- The unit slot's forced-zero row. -/
abbrev ZERO : ℕ := 786689

/-- The first row's prefix midstate, as `HmRow.delta` reads it: the kernel never computes its words, since `HmRow.delta`
names the same term. -/
abbrev msI : Array UInt64 := HmRow.midstateOf i0

/-- The second row's prefix midstate. -/
abbrev msO : Array UInt64 := HmRow.midstateOf o0

/-- Bit `t` of a midstate, as `HmRow.delta` tests it. -/
def ubit (ms : Array UInt64) (t : ℕ) : Bool := (ms[t / 64]! >>> (t % 64).toUInt64) &&& 1 == 1

/-- The first row's padding block (a 128-byte message). -/
def padI : Array ℕ := #[9223372036854775808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2048]

/-- The second row's padding block (a 2-byte message), its message bits 0. -/
def padO : Array ℕ := #[140737488355328, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1040]

/-- A constant bit's source: the pin for 1, the forced-zero row for 0. -/
def cs (ws : Array ℕ) (t : ℕ) : ℕ := if wbit ws t then PIN else ZERO

/-- A midstate bit's source. -/
def csU (ms : Array UInt64) (t : ℕ) : ℕ := if ubit ms t then PIN else ZERO

/-- The second `hm96` slot's `pad` past its 16 message bits. -/
def srcB (t : ℕ) : ℕ := cs padO (t + 64)

/-- The slot constants' pairs. -/
def consts : List (ℕ × ℕ) :=
  [(508928, 508928), (508928, PIN), (780288, 780288), (780288, PIN), (786688, 786688), (786688, PIN)]

/-- `HmRow.delta`'s pairs at `x r`'s table. -/
def EX : List (ℕ × ℕ) :=
  consts ++ run 524288 512 (csU msI) ++ run 512 1024 (cs padI) ++ run 262144 512 (csU msO) ++
    run 262656 48 (cs padO) ++ run 262720 960 srcB ++ run 786432 8 (524856 + ·) ++ run 786440 8 (524848 + ·) ++
    run 0 512 (778240 + ·)

/-- The output rows' pairs: the second `hm96` slot's message bits copy the unit's outputs. -/
def OX : List (ℕ × ℕ) := run 262712 8 (786560 + ·) ++ run 262704 8 (786568 + ·)

theorem count_T (x j : ℕ) : (EX ++ OX).count (x, j) =
    consts.count (x, j) + (run 524288 512 (csU msI)).count (x, j) + (run 512 1024 (cs padI)).count (x, j) +
      (run 262144 512 (csU msO)).count (x, j) + (run 262656 48 (cs padO)).count (x, j) +
      (run 262720 960 srcB).count (x, j) + (run 786432 8 (524856 + ·)).count (x, j) +
      (run 786440 8 (524848 + ·)).count (x, j) + (run 0 512 (778240 + ·)).count (x, j) +
      (run 262712 8 (786560 + ·)).count (x, j) + (run 262704 8 (786568 + ·)).count (x, j) := by
  simp only [EX, OX, List.count_append]
  omega

theorem consts_count_zero {x : ℕ} (hx : x ≠ 508928 ∧ x ≠ 780288 ∧ x ≠ 786688) (j : ℕ) :
    consts.count (x, j) = 0 :=
  List.count_eq_zero.2 fun hm => by
    simp only [consts, List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
    omega

theorem ite_flip (x j : ℕ) : (if x = j then 1 else 0 : ℕ) = if j = x then 1 else 0 := if_congr eq_comm rfl rfl

section rows

variable {da : Array (ℕ × ℕ)} (hda : da.toList = EX ++ OX)
include hda

/-- **Δ adds nothing at a row no run or constant touches.** -/
theorem delta_zero {x : ℕ}
    (hx : 1536 ≤ x ∧ (x < 262144 ∨ 263680 ≤ x) ∧ (x < 524288 ∨ 524800 ≤ x) ∧ (x < 786432 ∨ 786448 ≤ x) ∧
      x ≠ 508928 ∧ x ≠ 780288 ∧ x ≠ 786688) (j : ℕ) : deltaEntry da x j = 0 := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.cast_zero]

/-- **Δ at a `cv0` row of the `sha512x3` slot**: its self pair and its copy of the first midstate's bit. -/
theorem delta_R1 {x : ℕ} (hx : 524288 ≤ x ∧ x < 524800) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if csU msI (x - 524288) = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.sub_zero]

/-- **Δ at a `pad` row of the first `hm96` slot**: its self pair and its copy of the first padding block's bit. -/
theorem delta_R2 {x : ℕ} (hx : 512 ≤ x ∧ x < 1536) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if cs padI (x - 512) = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.sub_zero]

/-- **Δ at a `cv` row of the second `hm96` slot**: its self pair and its copy of the second midstate's bit. -/
theorem delta_R3a {x : ℕ} (hx : 262144 ≤ x ∧ x < 262656) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if csU msO (x - 262144) = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.sub_zero]

/-- **Δ at a `pad` row of the second `hm96` slot before its message**. -/
theorem delta_R3b {x : ℕ} (hx : 262656 ≤ x ∧ x < 262704) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if cs padO (x - 262656) = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.sub_zero]

/-- **Δ at a `pad` row of the second `hm96` slot past its message**. -/
theorem delta_R4 {x : ℕ} (hx : 262720 ≤ x ∧ x < 263680) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if srcB (x - 262720) = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.sub_zero]

/-- **Δ at the unit's leaf rows `0 … 7`**: the self pair and the copy of the `sha512x3` slot's message bit. -/
theorem delta_R5a {x : ℕ} (hx : 786432 ≤ x ∧ x < 786440) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if 524856 + (x - 786432) = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.sub_zero]

/-- **Δ at the unit's leaf rows `8 … 15`**. -/
theorem delta_R5b {x : ℕ} (hx : 786440 ≤ x ∧ x < 786448) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if 524848 + (x - 786440) = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.sub_zero]

/-- **Δ at a `cv` row of the first `hm96` slot**: its self pair and its copy of the `sha512x3` slot's first output. -/
theorem delta_R6 {x : ℕ} (hx : x < 512) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if 778240 + x = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.sub_zero]

/-- **Δ at the second `hm96` slot's message bits** `262712 …`: the self pair and the copy of the unit's output. -/
theorem delta_O1 {x : ℕ} (hx : 262712 ≤ x ∧ x < 262720) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if 786560 + (x - 262712) = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.sub_zero]

/-- **Δ at the second `hm96` slot's message bits** `262704 …`. -/
theorem delta_O2 {x : ℕ} (hx : 262704 ≤ x ∧ x < 262712) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if 786568 + (x - 262704) = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_T, consts_count_zero (x := x) (by omega) j]
  simp (disch := omega) only [count_run, if_pos, if_neg, Nat.zero_add, Nat.add_zero, Nat.sub_zero]

end rows

/-- A row Δ skips, in the form of a self pair and a copy from itself. -/
theorem delta_selfForm {d : Array (ℕ × ℕ)} {x : ℕ} (h : ∀ j, deltaEntry d x j = 0) (j : ℕ) :
    deltaEntry d x j = (((if j = x then 1 else 0) + (if j = x then 1 else 0) : ℕ) : ZMod 2) := by
  rw [h]
  split_ifs <;> decide

/-- A pair of copies read in the other direction. -/
theorem form_flip (x s j : ℕ) :
    ((((if x = j then 1 else 0) + (if s = j then 1 else 0) : ℕ) : ZMod 2)) =
      (((if j = x then 1 else 0) + (if j = s then 1 else 0) : ℕ) : ZMod 2) := by
  rw [ite_flip x j, ite_flip s j]

end SpotFlock.InstanceStmt.TraceDelta
