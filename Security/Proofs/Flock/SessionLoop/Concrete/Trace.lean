module

public import Proofs.Flock.SessionLoop.Concrete.Setup
public import Proofs.Flock.SessionLoop.Concrete.Total
public import Proofs.Flock.Soundness.Discharge.Hm.Flat
import all Flock.HmRow
import all Flock.HmNets
import all Flock.Canon
import all Flock.JsonParse
import all Flock.Net
import all Flock.Firewall
import all Flock.Norm
import all Flock.Public
import all Flock.Statement
import all Flock.Comp
import all Flock.Bytes
import all Flock.FV3
import all Flock.Hm96
import all Flock.Hash
import all Flock.RowLeaf
import all Flock.Circuit
import all Flock.Tags
import all Flock.Zk
import all Flock.CoinTree
import all Flock.Merkle
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

/-! # The concrete table's trace

The values that satisfy every row of the concrete statement's one table (`stW`), its fixed bits aside: the row's hidden
bits and the salt are 0, the `sha512x3` slot holds its net's forward evaluation (`Total.laidVal`) from its inputs, `cv0`
the row prefix's midstate and the rest 0, the `hm96` slot holds its net's from `cv` the `sha512x3` slot's first output,
`pad` the padding block and `y` 0, the unit slot holds its constant and nothing else, and every other column is 0.

Δ is read once, by the kernel: `HmRow.deltaL` at the concrete circuit is the list `E` (`deltaL_cW`), the two slot
constants' pairs and five runs of copies (`run`), so the trace's proof counts pairs in `E` (`count_run`) and never
evaluates Δ. A row holds by its kind: a copy row carries its source (`copyRow_holds`), a zero row 0 (`zeroRow_holds`),
and a net's row the product of its XORs (`netRow_holds`).
-/

@[expose] public section

open Flock FlockLevel3 FlockSoundness FlockSoundness.Discharge.Hm
open Flock.HmNets hiding and

set_option maxRecDepth 200000

namespace FlockDeploy.Concrete.Trace

/-! ## Rows that hold -/

section rows

variable {S : Model.Statement} {z : Model.Witness S.m} {o : Fin (2 ^ S.nbl)}

theorem bit_inj' : ∀ x y : ZMod 2, FlockSoundness.bit x = FlockSoundness.bit y → x = y := by decide

/-- **A copy row holds** in a block where it carries its source. -/
theorem copyRow_holds {r src : Fin (2 ^ S.kLog)} (h : CopyRow S r src) (he : S.block z o r = S.block z o src) :
    (S.A₀.mulVec (S.block z o)) r * (S.B₀.mulVec (S.block z o)) r = S.block z o r := by
  rw [FlockSoundness.mulVec_single S.A₀ h.1, FlockSoundness.mulVec_single S.B₀ h.2, he]
  have key : ∀ x : ZMod 2, x * x = x := by decide
  exact key _

/-- **A zero row holds** in a block where it carries 0. -/
theorem zeroRow_holds {r : Fin (2 ^ S.kLog)} (h : ZeroRow S r) (he : S.block z o r = 0) :
    (S.A₀.mulVec (S.block z o)) r * (S.B₀.mulVec (S.block z o)) r = S.block z o r := by
  rw [FlockSoundness.mulVec_zero_row S.A₀ h.1, FlockSoundness.mulVec_zero_row S.B₀ h.2, zero_mul, he]

/-- **A net's row holds** where its A and B rows are a net's rows `ra` and `rb` shifted by `b`, and the XORs of the
block's bits over them multiply to the row's bit. -/
theorem netRow_holds {r : Fin (2 ^ S.kLog)} {b : ℕ} {ra rb : Array ℕ}
    (hA : ∀ j : Fin (2 ^ S.kLog), S.A₀ r j = (((ra.toList.map (b + ·)).count j.val : ℕ) : ZMod 2))
    (hB : ∀ j : Fin (2 ^ S.kLog), S.B₀ r j = (((rb.toList.map (b + ·)).count j.val : ℕ) : ZMod 2))
    (hla : ∀ y ∈ ra.toList.map (b + ·), y < 2 ^ S.kLog) (hlb : ∀ y ∈ rb.toList.map (b + ·), y < 2 ^ S.kLog)
    (h : (xorRow (fun j => blockBit S z o (b + j)) ra && xorRow (fun j => blockBit S z o (b + j)) rb) =
      blockBit S z o r.val) :
    (S.A₀.mulVec (S.block z o)) r * (S.B₀.mulVec (S.block z o)) r = S.block z o r := by
  have hmv : ∀ (rr : Array ℕ) (M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)),
      (∀ y ∈ rr.toList.map (b + ·), y < 2 ^ S.kLog) →
      (∀ j : Fin (2 ^ S.kLog), M r j = (((rr.toList.map (b + ·)).count j.val : ℕ) : ZMod 2)) →
      FlockSoundness.bit (M.mulVec (S.block z o) r) = xorRow (fun j => blockBit S z o (b + j)) rr := by
    intro rr M hlt hM
    simp only [Matrix.mulVec, dotProduct, hM]
    rw [sum_count_nat _ hlt]
    exact bit_sum_row _ _ _
  apply bit_inj'
  rw [bit_mul, hmv ra S.A₀ hla hA, hmv rb S.B₀ hlb hB, h]
  exact blockBit_of_lt _ z o r.isLt

end rows

/-! ## Δ at the concrete circuit -/

/-- A run of `n` copies into columns `b, b + 1, …`: `(b + t, b + t)`, cancelling the self row, and `(b + t, f t)`. -/
def run (b n : ℕ) (f : ℕ → ℕ) : List (ℕ × ℕ) := (List.range n).flatMap fun t => [(b + t, b + t), (b + t, f t)]

theorem count_pair (a b c x j : ℕ) :
    [(a, b), (a, c)].count (x, j) = (if a = x ∧ b = j then 1 else 0) + (if a = x ∧ c = j then 1 else 0) := by
  by_cases h1 : a = x <;> by_cases h2 : b = j <;> by_cases h3 : c = j <;> simp [h1, h2, h3] <;> omega

/-- **The pairs of a run at column `x`**: the self pair and the copy's, when `x` is in the run. -/
theorem count_run (b n : ℕ) (f : ℕ → ℕ) (x j : ℕ) :
    (run b n f).count (x, j) =
      if b ≤ x ∧ x < b + n then (if x = j then 1 else 0) + (if f (x - b) = j then 1 else 0) else 0 := by
  unfold run
  by_cases h : b ≤ x ∧ x < b + n
  · rw [ite_eq_left h, count_flatMap_range _ _ (s₀ := x - b) (by omega) (fun s _ hne => by
      rw [count_pair, ite_eq_right (fun h' => hne (by have := h'.1; omega)),
        ite_eq_right (fun h' => hne (by have := h'.1; omega))]), count_pair]
    have hx : b + (x - b) = x := by omega
    simp only [hx, true_and]
  · rw [ite_eq_right h]
    refine count_flatMap_zero fun t ht => ?_
    rw [List.mem_range] at ht
    rw [count_pair, ite_eq_right (fun h' => h ⟨by have := h'.1; omega, by have := h'.1; omega⟩),
      ite_eq_right (fun h' => h ⟨by have := h'.1; omega, by have := h'.1; omega⟩)]

/-- The pinned constant column: the `hm96` slot's constant. -/
abbrev PIN : ℕ := 246784
/-- The unit slot's forced-zero row. -/
abbrev ZERO : ℕ := 524545

/-- The row prefix's midstate, the words whose bits `cv0` copies (bit `t` is bit `t % 64` of word `t / 64`). -/
def msW : Array ℕ := #[4727051306283312623, 12109514399784713723, 16297059344280646113, 6100433240194671855,
  606600717250638612, 18367179903051992734, 11735301908592016143, 1437849705087211466]

/-- The padding block's words, whose bits `pad` copies. -/
def padW : Array ℕ := #[9223372036854775808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2048]

def wbit (ws : Array ℕ) (t : ℕ) : Bool := (ws.getD (t / 64) 0).testBit (t % 64)

/-- A constant bit's source: the pin for 1, the forced-zero row for 0. -/
def constSrcW (ws : Array ℕ) (t : ℕ) : ℕ := if wbit ws t then PIN else ZERO

/-- The unit's leaf bits' sources: the row's first word's bits in `msg0`, high byte first. -/
def leafSrc (t : ℕ) : ℕ := if t < 8 then 262712 + t else 262696 + t

/-- **Δ at the concrete circuit**: the `sha512x3` and unit slots' constants from the pin, `cv0` the midstate, the
`hm96` slot's `pad` the padding block, the unit's leaf bits the row's first word, and the `hm96` slot's `cv` the
`sha512x3` slot's first output. -/
def E : List (ℕ × ℕ) :=
  [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)] ++ run 262144 512 (constSrcW msW) ++
    run 512 1024 (constSrcW padW) ++ run 524288 16 leafSrc ++ run 0 512 (516096 + ·)

section cW

variable (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
  (hello : ByteArray → Nat → String)

theorem pin_stW : (stW ls lh H A B A' B' s1 s3 s4 s6 hello).pin = PIN := by kernel_rfl'

theorem deltaL_cW : (HmRow.deltaL (cW ls lh H A B A' B' s1) PIN).1 = E.reverse := by kernel_rfl'

theorem dA_cW : (dA (cW ls lh H A B A' B' s1) PIN).toList = E := by
  have h := congrArg Prod.fst (deltaL_eq (cW ls lh H A B A' B' s1) PIN)
  rw [deltaL_cW] at h
  simp only [Φ, φ, List.reverse_inj] at h
  exact h

/-- **The concrete table's Δ_A is `E`.** -/
theorem da_stW : (stW ls lh H A B A' B' s1 s3 s4 s6 hello).da.toList = E := by
  have hda : (stW ls lh H A B A' B' s1 s3 s4 s6 hello).da =
      dA (cW ls lh H A B A' B' s1) (stW ls lh H A B A' B' s1 s3 s4 s6 hello).pin := rfl
  rw [hda, pin_stW]
  exact dA_cW ls lh H A B A' B' s1

end cW

/-! ## The concrete circuit's slots

One slot per range: `hm96` (net 2) at 0 and `sha512x3` (net 1) at `2^18`, each `2^18` rows; the unit (net 0) at
`2^19`, `2^9` rows; the mask's `2^14` rows at `2^20`. The pin is `hm96`'s constant, so Δ's slot-constant pairs are
`sha512x3`'s and the unit's (`constPairs_cW`). -/

section facts

variable (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
  (hello : ByteArray → Nat → String)

theorem c_stW : (stW ls lh H A B A' B' s1 s3 s4 s6 hello).c = cW ls lh H A B A' B' s1 := by kernel_rfl'

theorem kLog_cW : (cW ls lh H A B A' B' s1).kLog = 22 := by kernel_rfl'

theorem nets_cW : (cW ls lh H A B A' B' s1).unitNet = 0 ∧ (cW ls lh H A B A' B' s1).shaNet = 1 ∧
    (cW ls lh H A B A' B' s1).hmNet = 2 :=
  ⟨by kernel_rfl', by kernel_rfl', by kernel_rfl'⟩

theorem ranges_cW : (cW ls lh H A B A' B' s1).ranges.toList.map (fun r => (r.kind, r.slotLog, r.pos0, r.count)) =
    [(Kind.net 2, 18, 0, 1), (Kind.net 1, 18, 1, 1), (Kind.net 0, 9, 1024, 1), (Kind.mask, 14, 64, 1)] := by
  kernel_rfl'

theorem constPos_cW : ((cW ls lh H A B A' B' s1).nets.getD 0 default).2.constPos = 256 ∧
    ((cW ls lh H A B A' B' s1).nets.getD 1 default).2.constPos = 256000 ∧
    ((cW ls lh H A B A' B' s1).nets.getD 2 default).2.constPos = 246784 :=
  ⟨by kernel_rfl', by kernel_rfl', by kernel_rfl'⟩

theorem inWords_cW : ((cW ls lh H A B A' B' s1).nets.getD 0 default).2.inWords = 1 ∧
    ((cW ls lh H A B A' B' s1).nets.getD 1 default).2.inWords = 64 ∧
    ((cW ls lh H A B A' B' s1).nets.getD 2 default).2.inWords = 32 :=
  ⟨by kernel_rfl', by kernel_rfl', by kernel_rfl'⟩

theorem constPairs_cW :
    constPairs (cW ls lh H A B A' B' s1) PIN = [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)] := by
  kernel_rfl'

end facts

/-! ## The trace -/

/-- The `sha512x3` slot's inputs: `cv0` the midstate's bits, every other input 0 (the row's bits and the unused
compressions'). -/
def xS (j : ℕ) : Bool := if j < 512 then wbit msW j else false

/-- The `sha512x3` slot's column values: its net's forward evaluation from `xS`. -/
def vS : ℕ → Bool :=
  Total.laidVal (compression3.run St.empty).2 SHA_IN SHA_OUT (compression3.run St.empty).1.2.flatten.toArray xS

/-- The `hm96` slot's inputs: `cv` the `sha512x3` slot's first output, `pad` the padding block's bits, `y` 0. -/
def xH (j : ℕ) : Bool := if j < 512 then vS (253952 + j) else if j < 1536 then wbit padW (j - 512) else false

/-- The `hm96` slot's column values: its net's forward evaluation from `xH`. -/
def vH : ℕ → Bool :=
  Total.laidVal ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT
    ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.toArray xH

/-- **The concrete table's trace in one block**: the `hm96` slot's values, the `sha512x3` slot's, 1 at the unit's
constant and 0 elsewhere. -/
def zIn (i : ℕ) : Bool := if i < 262144 then vH i else if i < 524288 then vS (i - 262144) else decide (i = 524544)

/-- The trace in every block: `zIn` at each in-block address. -/
def zC (m : ℕ) : Model.Witness m := fun a => if zIn (a.val % 2 ^ 22) then 1 else 0

theorem mod_block (a k iv : ℕ) (hk : k = 22) (hiv : iv < 2 ^ 22) : (a * 2 ^ k + iv) % 2 ^ 22 = iv := by
  subst hk
  rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hiv]

theorem block_zC {S : Model.Statement} (hk : S.kLog = 22) (o : Fin (2 ^ S.nbl)) (i : Fin (2 ^ S.kLog)) :
    S.block (zC S.m) o i = if zIn i.val then 1 else 0 := by
  have hi : i.val < 2 ^ 22 := lt_of_lt_of_eq i.isLt (by rw [hk])
  show (if zIn ((o.val * 2 ^ S.kLog + i.val) % 2 ^ 22) then 1 else 0) = _
  rw [mod_block _ _ _ hk hi]

theorem blockBit_zC {S : Model.Statement} (hk : S.kLog = 22) (o : Fin (2 ^ S.nbl)) {y : ℕ} (hy : y < 2 ^ 22) :
    blockBit S (zC S.m) o y = zIn y := by
  have hy' : y < 2 ^ S.kLog := lt_of_lt_of_eq hy (by rw [hk])
  rw [blockBit_of_lt _ _ o hy', block_zC hk]
  show FlockSoundness.bit (if zIn y then 1 else 0) = zIn y
  cases zIn y <;> rfl

end FlockDeploy.Concrete.Trace
