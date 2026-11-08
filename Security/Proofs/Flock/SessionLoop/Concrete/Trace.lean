module

public import Proofs.Flock.SessionLoop.Concrete.Setup
public import Proofs.Flock.SessionLoop.Concrete.Total
public import Proofs.Flock.Soundness.Discharge.Hm.Flat
import Flock.HmRow
import Flock.HmNets
import Flock.Canon
import Flock.JsonParse
import Flock.Net
import Flock.Firewall
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
  by_cases h1 : a = x <;> by_cases h2 : b = j <;> by_cases h3 : c = j <;> simp [h1, h2, h3]

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

set_option maxHeartbeats 0 -- `deltaL_cW` evaluates the concrete circuit's Δ in the kernel

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

set_option maxHeartbeats 2000000 -- the kernel evaluates the parsed circuit's fields and the unit net `UN`

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

theorem unit_cW : (cW ls lh H A B A' B' s1).unit = UN := by kernel_rfl'

theorem tmpl_cW : (cW ls lh H A B A' B' s1).tmpl = none := by kernel_rfl'

theorem unitRange_cW : (cW ls lh H A B A' B' s1).unitRange.count = 1 ∧
    (cW ls lh H A B A' B' s1).unitRange.slot 0 = 524288 :=
  ⟨by kernel_rfl', by kernel_rfl'⟩

theorem shaRange_cW : ((cW ls lh H A B A' B' s1).rangeOf (cW ls lh H A B A' B' s1).shaNet).count = 1 ∧
    ((cW ls lh H A B A' B' s1).rangeOf (cW ls lh H A B A' B' s1).shaNet).slotLog = 18 ∧
    ((cW ls lh H A B A' B' s1).rangeOf (cW ls lh H A B A' B' s1).shaNet).slot 0 = 262144 :=
  ⟨by kernel_rfl', by kernel_rfl', by kernel_rfl'⟩

theorem hmRange_cW : ((cW ls lh H A B A' B' s1).rangeOf (cW ls lh H A B A' B' s1).hmNet).count = 1 ∧
    ((cW ls lh H A B A' B' s1).rangeOf (cW ls lh H A B A' B' s1).hmNet).slotLog = 18 ∧
    ((cW ls lh H A B A' B' s1).rangeOf (cW ls lh H A B A' B' s1).hmNet).slot 0 = 0 :=
  ⟨by kernel_rfl', by kernel_rfl', by kernel_rfl'⟩

/-- The unit's computed rows read only its 16 input rows: rows `128 … 143` copy one input each, the rest are empty. -/
theorem UN_a_lt : ∀ c < 128, ∀ y ∈ (UN.a.row (128 + c)).toList, y < 16 := by decide +kernel

/-- The unit's input rows past its 16 inputs are empty. -/
theorem UN_empty : ∀ c < 112, UN.a.row (16 + c) = #[] ∧ UN.b.row (16 + c) = #[] := by decide +kernel

/-- The unit's 16 input rows read themselves. -/
theorem UN_in : ∀ t < 16, UN.a.row t = #[t] ∧ UN.b.row t = #[t] := by decide +kernel

/-- The unit's rows past its constant are empty. -/
theorem UN_past : ∀ c < 255, UN.a.row (257 + c) = #[] ∧ UN.b.row (257 + c) = #[] := by decide +kernel

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

/-! ## Rows by place

Each row of the block by where it sits: in no range, in the mask's slot, or in a slot of a net. The lemmas take the
statement's layout facts (`Layout`), Δ as `E` and the ranges as the concrete circuit's, so they read nothing else of
the concrete statement. -/

section place

/-- **No range's entry at a row no range covers**: a covered column in the row's slot would cover the row. -/
theorem rangesEntry_outside {st : Stmt} {x : ℕ} (hx : ∀ R ∈ st.c.ranges.toList, ¬ FlockLevel3.covers R x)
    (e : Kind → ℕ → ℕ → ZMod 2) (j : ℕ) : rangesEntry st e x j = 0 := by
  unfold rangesEntry
  split
  next rg hrg =>
    obtain ⟨hm, hc⟩ := lastCover_mem hrg
    by_cases hd : x / 2 ^ rg.slotLog = j / 2 ^ rg.slotLog
    · exact absurd (show FlockLevel3.covers rg x by unfold FlockLevel3.covers; rw [hd]; exact hc) (hx rg hm)
    · rw [ite_eq_right hd]
  next => rfl

/-- **A row no range covers and Δ skips is zero.** -/
theorem zeroRow_outside {st : Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {r : Fin (2 ^ (L.model regs).kLog)}
    (hc : ∀ R ∈ st.c.ranges.toList, ¬ FlockLevel3.covers R r.val) (hd : ∀ j, deltaEntry st.da r.val j = 0) :
    ZeroRow (L.model regs) r := by
  refine ⟨fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = 0
    rw [rangesEntry_outside hc, hd, add_zero]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = 0
    rw [rangesEntry_outside hc, L.delta_b, hd, add_zero]

/-- **A unit slot's constant row copies the pin.** -/
theorem copyRow_unitConst {st : Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {s₀ : ℕ} (hs : s₀ < st.c.unitRange.count)
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = st.c.unitRange.slot s₀ + st.c.unit.constPos) :
    CopyRow (L.model regs) r (L.model regs).pin := by
  refine ⟨fun j => ?_, fun j => ?_⟩
  · rw [show (L.model regs).A₀ r j = placedA st r j from rfl, L.placedA_const hs r j hr]
    exact if_congr ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩ rfl rfl
  · rw [show (L.model regs).B₀ r j = placedB st r j from rfl, L.placedB_const hs r j hr]
    exact if_congr ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩ rfl rfl

/-- **The pin's row copies itself.** -/
theorem copyRow_pin {st : Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    CopyRow (L.model regs) (L.model regs).pin (L.model regs).pin := by
  refine ⟨fun j => ?_, fun j => ?_⟩
  · rw [show (L.model regs).A₀ (L.model regs).pin j = placedA st ⟨st.pin, L.pin_lt⟩ j from rfl, (L.placed_pin j).1]
    exact if_congr ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩ rfl rfl
  · rw [show (L.model regs).B₀ (L.model regs).pin j = placedB st ⟨st.pin, L.pin_lt⟩ j from rfl, (L.placed_pin j).2]
    exact if_congr ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩ rfl rfl

/-- **The trace is 1 at the pin**, at the verifier's `hm96` net: the pin is that net's constant column. -/
theorem vH_pin (hn : Properties.Flock.Nets.FixedNetsBuild) : vH PIN = true := by
  have hcp : (hmLaid ()).constPos = PIN := (Net.ofRows_ok hn.2.2.2.1).2.2.2.2.1.symm.trans rfl
  rw [← hcp]
  exact (Total.hm_total xH).1

theorem zIn_pin (hn : Properties.Flock.Nets.FixedNetsBuild) : zIn PIN = true := by
  unfold zIn
  rw [ite_eq_left (show PIN < 262144 by decide)]
  exact vH_pin hn

theorem zIn_unitConst : zIn 524544 = true := by
  unfold zIn
  rw [ite_eq_right (show ¬ 524544 < 262144 by decide), ite_eq_right (show ¬ 524544 < 524288 by decide)]
  exact decide_eq_true rfl

/-- **The trace is 1 at the pin** in every block. -/
theorem block_pin {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) (hpin : st.pin = PIN)
    (o : Fin (2 ^ (L.model regs).nbl)) : (L.model regs).block (zC (L.model regs).m) o (L.model regs).pin = 1 := by
  rw [block_zC (S := L.model regs) hk]
  show (if zIn st.pin then 1 else 0) = 1
  rw [hpin, zIn_pin hn]
  rfl

/-- **The pin's row and the unit's constant row hold** at the trace: both copy the pin, and the trace is 1 at both. -/
theorem holds_pins {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) (hpin : st.pin = PIN)
    (hs : 0 < st.c.unitRange.count) (hu : st.c.unitRange.slot 0 + st.c.unit.constPos = 524544)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = PIN ∨ r.val = 524544) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  rcases hr with hr | hr
  · have he : r = (L.model regs).pin := Fin.ext (hr.trans hpin.symm)
    subst he
    exact copyRow_holds (copyRow_pin L regs) rfl
  · refine copyRow_holds (copyRow_unitConst L regs hs (hr.trans hu.symm)) ?_
    rw [block_pin L regs hn hk hpin o, block_zC (S := L.model regs) hk, hr, zIn_unitConst]
    rfl

theorem xorRow_false {v : ℕ → Bool} {cs : Array ℕ} (h : ∀ j ∈ cs.toList, v j = false) : xorRow v cs = false := by
  unfold xorRow
  generalize cs.toList = l at h
  induction l with
  | nil => rfl
  | cons a l ih =>
    have ha : v a = false := h a (by simp)
    have hl := ih fun j hj => h j (by simp [hj])
    simp [List.foldr_cons, ha, hl]

/-- **A unit slot's computed row holds** where the block is 0 at the row and at every column its A row reads. -/
theorem holds_unitComp {st : Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) (hU : Built st.c.unit) {s₀ c : ℕ}
    (hs : s₀ < st.c.unitRange.count) (hlo : st.c.unit.inWords * WORD ≤ c) (hhi : c < st.c.unit.constPos)
    {z : Model.Witness (L.model regs).m} (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)}
    (hr : r.val = st.c.unitRange.slot s₀ + c)
    (hz : ∀ y ∈ (st.c.unit.a.row c).toList, blockBit (L.model regs) z o (st.c.unitRange.slot s₀ + y) = false)
    (hr0 : blockBit (L.model regs) z o r.val = false) :
    ((L.model regs).A₀.mulVec ((L.model regs).block z o)) r * ((L.model regs).B₀.mulVec ((L.model regs).block z o)) r =
      (L.model regs).block z o r := by
  have hlt := L.unit_row_lt hU hlo hhi (L.unitLog_le hs)
  have hcl : ∀ rr : Array ℕ, (∀ x ∈ rr.toList, x < 2 ^ st.c.unit.unitLog) →
      ∀ y ∈ rr.toList.map (st.c.unitRange.slot s₀ + ·), y < 2 ^ (L.model regs).kLog := fun rr h y hy => by
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hy
    exact L.unit_slot_lt hs (h x hx)
  refine netRow_holds (b := st.c.unitRange.slot s₀) (ra := st.c.unit.a.row c) (rb := st.c.unit.b.row c)
    (fun j => L.placedA_comp hU hs hlo hhi r j hr) (fun j => L.placedB_comp hU hs hlo hhi r j hr)
    (hcl _ hlt.1) (hcl _ hlt.2) ?_
  rw [xorRow_false (v := fun j => blockBit (L.model regs) z o (st.c.unitRange.slot s₀ + j)) hz, Bool.false_and, hr0]

/-- **A unit row whose rows are empty and that Δ skips is zero.** -/
theorem zeroRow_unitEmpty {st : Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) (hU : Built st.c.unit) {s₀ c : ℕ}
    (hs : s₀ < st.c.unitRange.count) (hc : c < 2 ^ st.c.unit.unitLog) (ha : st.c.unit.a.row c = #[])
    (hb : st.c.unit.b.row c = #[]) {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = st.c.unitRange.slot s₀ + c)
    (hd : ∀ j, deltaEntry st.da r.val j = 0) : ZeroRow (L.model regs) r := by
  refine ⟨fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = 0
    rw [hd, add_zero, hr, L.rangesEntry_unit hs hc]
    refine ite_eq_right_iff.2 fun _ => ?_
    show sparseEntry st.c.unit.a c _ = 0
    rw [sparseEntry_built_a hU, ha]
    simp
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = 0
    rw [L.delta_b, hd, add_zero, hr, L.rangesEntry_unit hs hc]
    refine ite_eq_right_iff.2 fun _ => ?_
    show sparseEntry st.c.unit.b c _ + lookupEntry st.c.unit.lookup c _ = 0
    rw [sparseEntry_built_b hU, hb, L.unit_lookup]
    simp [lookupEntry]

theorem zIn_unit {i : ℕ} (h1 : 524288 ≤ i) (h2 : i ≠ 524544) : zIn i = false := by
  unfold zIn
  rw [ite_eq_right (show ¬ i < 262144 by omega), ite_eq_right (show ¬ i < 524288 by omega)]
  exact decide_eq_false h2

/-- **The unit's computed rows hold** at the trace: they read only its inputs, which are 0. -/
theorem holds_unitComp_cW {st : Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) (hk : st.c.kLog = 22) (htm : st.c.tmpl = none)
    (hU : st.c.unit = UN) (hs1 : st.c.unitRange.count = 1) (hsl : st.c.unitRange.slot 0 = 524288)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : 524416 ≤ r.val ∧ r.val < 524544) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have hc : r.val = st.c.unitRange.slot 0 + (r.val - 524288) := by rw [hsl]; omega
  have hlo : st.c.unit.inWords * WORD ≤ r.val - 524288 := by rw [hU]; show 1 * 128 ≤ _; omega
  have hhi : r.val - 524288 < st.c.unit.constPos := by rw [hU]; show _ < 256; omega
  refine holds_unitComp L regs (L.built_flat htm) (by omega) hlo hhi o hc (fun y hy => ?_) ?_
  · clear hc hlo hhi
    rw [hU] at hy
    have hy' : y < 16 := UN_a_lt (r.val - 524416) (Nat.sub_lt_left_of_lt_add hr.1 hr.2) y
      (by rwa [show 128 + (r.val - 524416) = r.val - 524288 by omega])
    rw [hsl, blockBit_zC (S := L.model regs) hk o (lt_of_lt_of_le (show 524288 + y < 524304 by omega) (by norm_num))]
    exact zIn_unit (by omega) (by omega)
  · clear hc hlo hhi
    rw [blockBit_zC (S := L.model regs) hk o (lt_of_lt_of_le (show r.val < 524544 by omega) (by norm_num))]
    exact zIn_unit (by omega) (by omega)

theorem count_E (x j : ℕ) : E.count (x, j) =
    [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)].count (x, j) +
      (run 262144 512 (constSrcW msW)).count (x, j) + (run 512 1024 (constSrcW padW)).count (x, j) +
      (run 524288 16 leafSrc).count (x, j) + (run 0 512 (516096 + ·)).count (x, j) := by
  simp only [E, List.count_append]

/-- **Δ adds nothing at a column outside its runs and the slot constants**, at a statement whose Δ is `E`. -/
theorem delta_free {da : Array (ℕ × ℕ)} (hda : da.toList = E) {x : ℕ}
    (hx : 1536 ≤ x ∧ (x < 262144 ∨ 262656 ≤ x) ∧ x ≠ 518144 ∧ x ≠ 524544 ∧ (x < 524288 ∨ 524304 ≤ x)) (j : ℕ) :
    deltaEntry da x j = 0 := by
  rw [deltaEntry_eq_count, hda, count_E, count_run, count_run, count_run, count_run,
    ite_eq_right (show ¬ (262144 ≤ x ∧ x < 262144 + 512) by omega),
    ite_eq_right (show ¬ (512 ≤ x ∧ x < 512 + 1024) by omega),
    ite_eq_right (show ¬ (524288 ≤ x ∧ x < 524288 + 16) by omega),
    ite_eq_right (show ¬ (0 ≤ x ∧ x < 0 + 512) by omega)]
  have h4 : [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)].count (x, j) = 0 :=
    List.count_eq_zero.2 fun hm => by
      simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
      omega
  rw [h4]
  simp

/-- The concrete circuit's ranges, as `ranges_cW` lists them. -/
abbrev RS : List (Kind × ℕ × ℕ × ℕ) :=
  [(Kind.net 2, 18, 0, 1), (Kind.net 1, 18, 1, 1), (Kind.net 0, 9, 1024, 1), (Kind.mask, 14, 64, 1)]

/-- **The concrete circuit's ranges cover no column of the gaps**: past the unit's slot and before the mask's, and past
the mask's. -/
theorem not_covers_of {rs : List Range} (hrs : rs.map (fun r => (r.kind, r.slotLog, r.pos0, r.count)) = RS) {x : ℕ}
    (hx : (524800 ≤ x ∧ x < 1048576) ∨ 1064960 ≤ x) : ∀ R ∈ rs, ¬ FlockLevel3.covers R x := by
  intro R hR hc
  have hm := List.mem_map_of_mem (f := fun r : Range => (r.kind, r.slotLog, r.pos0, r.count)) hR
  rw [hrs] at hm
  unfold FlockLevel3.covers at hc
  simp only [RS, List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
  rcases hm with ⟨-, h1, h2, h3⟩ | ⟨-, h1, h2, h3⟩ | ⟨-, h1, h2, h3⟩ | ⟨-, h1, h2, h3⟩ <;>
    rw [h1, h2, h3] at hc <;> norm_num at hc <;> omega

theorem zIn_gap {i : ℕ} (h : 524800 ≤ i) : zIn i = false := by
  unfold zIn
  rw [ite_eq_right (show ¬ i < 262144 by omega), ite_eq_right (show ¬ i < 524288 by omega)]
  exact decide_eq_false (by omega)

/-- **Δ at a unit leaf row**, at a statement whose Δ is `E`: the leaf run's self pair and its source's. -/
theorem delta_leaf {da : Array (ℕ × ℕ)} (hda : da.toList = E) {x : ℕ} (hx : 524288 ≤ x ∧ x < 524304) (j : ℕ) :
    deltaEntry da x j = (((if x = j then 1 else 0) + (if leafSrc (x - 524288) = j then 1 else 0) : ℕ) : ZMod 2) := by
  rw [deltaEntry_eq_count, hda, count_E, count_run, count_run, count_run, count_run,
    ite_eq_right (show ¬ (262144 ≤ x ∧ x < 262144 + 512) by omega),
    ite_eq_right (show ¬ (512 ≤ x ∧ x < 512 + 1024) by omega),
    ite_eq_left (show 524288 ≤ x ∧ x < 524288 + 16 by omega),
    ite_eq_right (show ¬ (0 ≤ x ∧ x < 0 + 512) by omega)]
  have h4 : [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)].count (x, j) = 0 :=
    List.count_eq_zero.2 fun hm => by
      simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
      omega
  rw [h4]
  simp only [Nat.zero_add, Nat.add_zero]

theorem count_one (t j : ℕ) : (((#[t] : Array ℕ).toList.count j : ℕ) : ZMod 2) = if j = t then 1 else 0 := by
  show ((([t] : List ℕ).count j : ℕ) : ZMod 2) = _
  by_cases h : j = t
  · subst h; simp
  · rw [ite_eq_right h]; simp [Ne.symm h]

/-- A row's own entry, cancelled by Δ's self pair, leaves Δ's copy of its source. -/
theorem ite_cancel (j x src : ℕ) : ((if j = x then (1 : ZMod 2) else 0) +
    (((if x = j then 1 else 0) + if src = j then 1 else 0 : ℕ) : ZMod 2)) = if j = src then 1 else 0 := by
  by_cases h1 : j = x
  · rw [ite_eq_left h1, ite_eq_left h1.symm]
    by_cases h2 : src = j
    · rw [ite_eq_left h2, ite_eq_left h2.symm]; decide
    · rw [ite_eq_right h2, ite_eq_right (Ne.symm h2)]; decide
  · rw [ite_eq_right h1, ite_eq_right (Ne.symm h1), zero_add, Nat.zero_add]
    by_cases h2 : src = j
    · rw [ite_eq_left h2, ite_eq_left h2.symm]; simp
    · rw [ite_eq_right h2, ite_eq_right (Ne.symm h2)]; simp

/-- **The trace is 0 at the unit's leaf sources**: they are inputs of the `sha512x3` slot past `cv0`. -/
theorem zIn_leaf {t : ℕ} (ht : t < 16) : zIn (leafSrc t) = false := by
  have h1 : 262704 ≤ leafSrc t ∧ leafSrc t < 262720 := by unfold leafSrc; split <;> omega
  unfold zIn vS
  rw [ite_eq_right (show ¬ leafSrc t < 262144 by omega), ite_eq_left (show leafSrc t < 524288 by omega),
    Total.laidVal_input (show leafSrc t - 262144 < inRowsOf SHA_IN by rw [inRows_SHA_IN]; omega)]
  unfold xS
  exact ite_eq_right (show ¬ leafSrc t - 262144 < 512 by omega)

/-- **The layout's input padding rows are empty**: those of the input group past the input bits, in A and in B. -/
theorem layout_inGap (st : HmNets.St) (inBits outBits : Array ℕ) (outs : Array HmNets.Lin) {i : ℕ}
    (h1 : inBits.foldl (· + ·) 0 ≤ i) (h2 : i < groupWords (inBits.foldl (· + ·) 0) * WORD) :
    (layout st inBits outBits outs).ra.getD i #[] = #[] ∧ (layout st inBits outBits outs).rb.getD i #[] = #[] := by
  let nIn := inBits.foldl (· + ·) 0
  let inRows := groupWords nIn * WORD
  let span := groupWords (outBits.foldl (· + ·) 0) * WORD
  let start := (inRows + st.rows.length + span - 1) / span * span
  let rows := st.rows.reverse.toArray
  let e := fun (n : ℕ) => Array.replicate n (#[] : Array ℕ)
  let ins := (Array.range nIn).map fun i => #[i]
  have h1' : nIn ≤ i := h1
  have h2' : i < inRows := h2
  have hra : (layout st inBits outBits outs).ra = ins ++ e (inRows - nIn) ++
      rows.map (HmNets.cols nIn inRows (start + span) ·.a) ++ e (start - inRows - st.rows.length) ++
      outs.map (HmNets.cols nIn inRows (start + span)) ++ e (span - outs.size) ++ #[#[start + span]] := by
    simp only [layout, rows, e, ins, start, span, inRows, nIn, List.size_toArray, List.length_reverse]
  have hrb : (layout st inBits outBits outs).rb = ins ++ e (inRows - nIn) ++
      rows.map (HmNets.cols nIn inRows (start + span) ·.b) ++ e (start - inRows - st.rows.length) ++
      outs.map (fun _ => #[start + span]) ++ e (span - outs.size) ++ #[#[start + span]] := by
    simp only [layout, rows, e, ins, start, span, inRows, nIn, List.size_toArray, List.length_reverse]
  have hs : ∀ n, (e n).size = n := fun n => Array.size_replicate
  have hr : rows.size = st.rows.length := by simp [rows]
  have hi : ins.size = nIn := by simp [ins]
  constructor
  · rw [hra]
    simp (disch := (simp only [Array.size_append, Array.size_map, hs, hr, hi, List.size_toArray, Array.size_singleton]; omega))
      only [getD_append_left, getD_append_right]
    simp only [hi]
    simp only [e, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
    split <;> rfl
  · rw [hrb]
    simp (disch := (simp only [Array.size_append, Array.size_map, hs, hr, hi, List.size_toArray, Array.size_singleton]; omega))
      only [getD_append_left, getD_append_right]
    simp only [hi]
    simp only [e, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
    split <;> rfl

theorem shaLaid_gap {off : ℕ} (h1 : 4608 ≤ off) (h2 : off < 8192) :
    (HmNets.shaLaid ()).ra.getD off #[] = #[] ∧ (HmNets.shaLaid ()).rb.getD off #[] = #[] := by
  rw [HmNets.shaLaid_eq]
  exact layout_inGap _ _ _ _ (by rw [sum_SHA_IN]; exact h1)
    (show off < inRowsOf SHA_IN by rw [inRows_SHA_IN]; exact h2)

theorem hmLaid_gap {off : ℕ} (h1 : 3072 ≤ off) (h2 : off < 4096) :
    (HmNets.hmLaid ()).ra.getD off #[] = #[] ∧ (HmNets.hmLaid ()).rb.getD off #[] = #[] := by
  rw [HmNets.hmLaid_eq]
  exact layout_inGap _ _ _ _ (by rw [sum_HM_IN]; exact h1)
    (show off < inRowsOf HM_IN by rw [inRows_HM_IN]; exact h2)

theorem getD_past {xs : Array (Array ℕ)} {i : ℕ} (h : xs.size ≤ i) : xs.getD i #[] = #[] := by
  simp [Array.getElem?_eq_none h]

theorem shaLaid_constPos (hn : Properties.Flock.Nets.FixedNetsBuild) : (HmNets.shaLaid ()).constPos = 256000 :=
  (Net.ofRows_ok hn.1.2.2.1).2.2.2.2.1.symm.trans rfl

theorem hmLaid_constPos (hn : Properties.Flock.Nets.FixedNetsBuild) : (HmNets.hmLaid ()).constPos = PIN :=
  (Net.ofRows_ok hn.2.2.2.1).2.2.2.2.1.symm.trans rfl

/-- **A laid-out net's values past its constant are 0**, when its outputs fit their group. -/
theorem laidVal_past {s : HmNets.St} {inBits outBits : Array ℕ} {outs : Array HmNets.Lin} {x : ℕ → Bool} {i : ℕ}
    (hsz : outs.size ≤ outBits.foldl (· + ·) 0) (h : outStart s inBits outBits + spanOf outBits < i) :
    Total.laidVal s inBits outBits outs x i = false := by
  have hspan : 0 < spanOf outBits := groupWords_pos _
  have hst : inRowsOf inBits + s.rows.length ≤ outStart s inBits outBits := le_roundUp _ _ hspan
  have hsz' : outs.size ≤ spanOf outBits := Nat.le_trans hsz (le_groupWords _)
  unfold Total.laidVal
  rw [ite_eq_right (show ¬ i < inRowsOf inBits by omega),
    ite_eq_right (show ¬ i < inRowsOf inBits + s.rows.length by omega),
    ite_eq_right (show ¬ (outStart s inBits outBits ≤ i ∧ i < outStart s inBits outBits + outs.size) by omega)]
  exact decide_eq_false (by omega)

theorem sha_outs : (compression3.run HmNets.St.empty).1.2.flatten.toArray.size ≤ SHA_OUT.foldl (· + ·) 0 := by
  have h : Total.Tot [] compression3 _ := Total.tot_compression 3
  have h := h HmNets.St.empty Total.scoped_empty (Total.belowL_nil _)
  have hz := Total.stHolds_fwd (fun _ => false) h.1
  obtain ⟨-, hl3, -⟩ := compression3_spec _ hz
  have hlens := compression3_lens _ hz
  have hflat : (compression3.run HmNets.St.empty).1.2.flatten.length = 1536 := by
    rw [List.length_flatten, List.map_congr_left (g := fun _ => 512) hlens, List.map_const', List.sum_replicate, hl3]
    rfl
  rw [List.size_toArray, hflat]
  decide

theorem hm_outs :
    ((hm96Row Hm96.Default512.key).run HmNets.St.empty).1.2.2.2.toArray.size ≤ HM_OUT.foldl (· + ·) 0 := by
  have h := Total.tot_hm96Row Hm96.Default512.key HmNets.St.empty Total.scoped_empty (Total.belowL_nil _)
  obtain ⟨-, -, -, hl, -⟩ := hm96Row_spec _ _ (Total.stHolds_fwd (fun _ => false) h.1)
  rw [List.size_toArray, hl]
  decide

/-- **A layout's output group ends at its constant.** Stated over any builder state, so no proof at a concrete net
compares its unfolded row count. -/
theorem outStart_span (s : HmNets.St) (inBits outBits : Array ℕ) (outs : Array HmNets.Lin) :
    outStart s inBits outBits + spanOf outBits = (layout s inBits outBits outs).constPos :=
  (HmNets.layout_rows s inBits outBits outs).1.symm

/-- The `sha512x3` net's output group ends at its constant. -/
theorem sha_end (hn : Properties.Flock.Nets.FixedNetsBuild) :
    outStart (compression3.run HmNets.St.empty).2 SHA_IN SHA_OUT + spanOf SHA_OUT = 256000 := by
  rw [outStart_span _ _ _ (compression3.run HmNets.St.empty).1.2.flatten.toArray, ← HmNets.shaLaid_eq]
  exact shaLaid_constPos hn

/-- The `hm96` net's output group ends at its constant. -/
theorem hm_end (hn : Properties.Flock.Nets.FixedNetsBuild) :
    outStart ((hm96Row Hm96.Default512.key).run HmNets.St.empty).2 HM_IN HM_OUT + spanOf HM_OUT = PIN := by
  rw [outStart_span _ _ _ ((hm96Row Hm96.Default512.key).run HmNets.St.empty).1.2.2.2.toArray, ← HmNets.hmLaid_eq]
  exact hmLaid_constPos hn

theorem xorRow_congr {v w : ℕ → Bool} {cs : Array ℕ} (h : ∀ j ∈ cs.toList, v j = w j) :
    xorRow v cs = xorRow w cs := by
  unfold xorRow
  generalize cs.toList = l at h
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.foldr_cons, h a (by simp), ih fun j hj => h j (by simp [hj])]

/-- **The `sha512x3` net's constant row reads itself**, in A and in B: `Net.ofRows` checks it. -/
theorem shaLaid_const (hn : Properties.Flock.Nets.FixedNetsBuild) :
    (HmNets.shaLaid ()).ra.getD 256000 #[] = #[256000] ∧ (HmNets.shaLaid ()).rb.getD 256000 #[] = #[256000] := by
  obtain ⟨-, -, -, -, -, -, -, -, ha, hb, -⟩ := Net.ofRows_ok hn.1.2.2.1
  rw [shaLaid_constPos hn] at ha hb
  exact ⟨ha, hb⟩

/-- At the `sha512x3` slot's constant row, its laid row's self entry and the slot constant's self pair cancel, leaving
the pin. -/
theorem constRow_count (j : ℕ) :
    ((((#[256000] : Array ℕ).toList.map (262144 + ·)).count j +
      [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)].count (518144, j) : ℕ) : ZMod 2) =
        if j = PIN then 1 else 0 := by
  by_cases h1 : j = 518144
  · subst h1; decide +kernel
  · by_cases h2 : j = PIN
    · subst h2; decide +kernel
    · rw [ite_eq_right h2]
      have e1 : ((#[256000] : Array ℕ).toList.map (262144 + ·)).count j = 0 :=
        List.count_eq_zero.2 fun hm => by simp at hm; omega
      have e2 : [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)].count (518144, j) = 0 :=
        List.count_eq_zero.2 fun hm => by
          simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
          omega
      rw [e1, e2, Nat.add_zero, Nat.cast_zero]

theorem zIn_shaConst (hn : Properties.Flock.Nets.FixedNetsBuild) : zIn 518144 = true := by
  unfold zIn
  rw [ite_eq_right (show ¬ 518144 < 262144 by decide), ite_eq_left (show 518144 < 524288 by decide)]
  show vS 256000 = true
  rw [← shaLaid_constPos hn]
  exact (Total.sha_total xS).1

theorem ite_flip (x j : ℕ) : (if x = j then 1 else 0 : ℕ) = if j = x then 1 else 0 := if_congr eq_comm rfl rfl

/-- **Δ at a `cv0` row**, at a statement whose Δ is `E`: its self pair and its copy of the midstate's bit. -/
theorem delta_cv0 {da : Array (ℕ × ℕ)} (hda : da.toList = E) {off : ℕ} (ho : off < 512) (j : ℕ) :
    deltaEntry da (262144 + off) j =
      (((if j = 262144 + off then 1 else 0) + (if j = constSrcW msW off then 1 else 0) : ℕ) : ZMod 2) := by
  have h4 : [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)].count (262144 + off, j) = 0 :=
    List.count_eq_zero.2 fun hm => by
      simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
      omega
  rw [deltaEntry_eq_count, hda, count_E, count_run, count_run, count_run, count_run,
    ite_eq_left (show 262144 ≤ 262144 + off ∧ 262144 + off < 262144 + 512 by omega),
    ite_eq_right (show ¬ (512 ≤ 262144 + off ∧ 262144 + off < 512 + 1024) by omega),
    ite_eq_right (show ¬ (524288 ≤ 262144 + off ∧ 262144 + off < 524288 + 16) by omega),
    ite_eq_right (show ¬ (0 ≤ 262144 + off ∧ 262144 + off < 0 + 512) by omega), h4, Nat.add_sub_cancel_left,
    ite_flip (262144 + off) j, ite_flip (constSrcW msW off) j]
  simp only [Nat.zero_add, Nat.add_zero]

/-- **Δ at a `pad` row** of the `hm96` slot: its self pair and its copy of the padding block's bit. -/
theorem delta_pad {da : Array (ℕ × ℕ)} (hda : da.toList = E) {off : ℕ} (ho : off < 1024) (j : ℕ) :
    deltaEntry da (512 + off) j =
      (((if j = 512 + off then 1 else 0) + (if j = constSrcW padW off then 1 else 0) : ℕ) : ZMod 2) := by
  have h4 : [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)].count (512 + off, j) = 0 :=
    List.count_eq_zero.2 fun hm => by
      simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
      omega
  rw [deltaEntry_eq_count, hda, count_E, count_run, count_run, count_run, count_run,
    ite_eq_right (show ¬ (262144 ≤ 512 + off ∧ 512 + off < 262144 + 512) by omega),
    ite_eq_left (show 512 ≤ 512 + off ∧ 512 + off < 512 + 1024 by omega),
    ite_eq_right (show ¬ (524288 ≤ 512 + off ∧ 512 + off < 524288 + 16) by omega),
    ite_eq_right (show ¬ (0 ≤ 512 + off ∧ 512 + off < 0 + 512) by omega), h4, Nat.add_sub_cancel_left,
    ite_flip (512 + off) j, ite_flip (constSrcW padW off) j]
  simp only [Nat.zero_add, Nat.add_zero]

/-- **Δ at a `cv` row** of the `hm96` slot: its self pair and its copy of the `sha512x3` slot's first output. -/
theorem delta_cv {da : Array (ℕ × ℕ)} (hda : da.toList = E) {off : ℕ} (ho : off < 512) (j : ℕ) :
    deltaEntry da off j = (((if j = off then 1 else 0) + (if j = 516096 + off then 1 else 0) : ℕ) : ZMod 2) := by
  have h4 : [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)].count (off, j) = 0 :=
    List.count_eq_zero.2 fun hm => by
      simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
      omega
  rw [deltaEntry_eq_count, hda, count_E, count_run, count_run, count_run, count_run,
    ite_eq_right (show ¬ (262144 ≤ off ∧ off < 262144 + 512) by omega),
    ite_eq_right (show ¬ (512 ≤ off ∧ off < 512 + 1024) by omega),
    ite_eq_right (show ¬ (524288 ≤ off ∧ off < 524288 + 16) by omega),
    ite_eq_left (show 0 ≤ off ∧ off < 0 + 512 by omega), h4, Nat.sub_zero, ite_flip off j, ite_flip (516096 + off) j]
  simp only [Nat.zero_add, Nat.add_zero]

/-- A row Δ skips, in the form of a self pair and a copy from itself. -/
theorem delta_selfForm {d : Array (ℕ × ℕ)} {x : ℕ} (h : ∀ j, deltaEntry d x j = 0) (j : ℕ) :
    deltaEntry d x j = (((if j = x then 1 else 0) + (if j = x then 1 else 0) : ℕ) : ZMod 2) := by
  rw [h]
  split_ifs <;> decide

theorem zIn_sha {x : ℕ} (h1 : 262144 ≤ x) (h2 : x < 270336) : zIn x = xS (x - 262144) := by
  unfold zIn vS
  rw [ite_eq_right (show ¬ x < 262144 by omega), ite_eq_left (show x < 524288 by omega),
    Total.laidVal_input (show x - 262144 < inRowsOf SHA_IN by rw [inRows_SHA_IN]; omega)]

theorem zIn_hm {x : ℕ} (h : x < 4096) : zIn x = xH x := by
  unfold zIn vH
  rw [ite_eq_left (show x < 262144 by omega),
    Total.laidVal_input (show x < inRowsOf HM_IN by rw [inRows_HM_IN]; exact h)]

/-- **The trace at a constant bit's source** is the bit: 1 at the pin, 0 at the forced-zero row. -/
theorem zIn_constSrc (hn : Properties.Flock.Nets.FixedNetsBuild) (ws : Array ℕ) (t : ℕ) :
    zIn (constSrcW ws t) = wbit ws t := by
  unfold constSrcW
  by_cases h : wbit ws t = true
  · rw [ite_eq_left h, zIn_pin hn, h]
  · rw [ite_eq_right h, zIn_unit (i := ZERO) (by decide) (by decide)]
    exact (Bool.eq_false_iff.2 h).symm

theorem constSrcW_lt (ws : Array ℕ) (t : ℕ) : constSrcW ws t < 2 ^ 22 := by
  unfold constSrcW
  split <;> decide

variable {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
  (hda : st.da.toList = E) (hrs : st.c.ranges.toList.map (fun r => (r.kind, r.slotLog, r.pos0, r.count)) = RS)
include hda hrs

/-- **A row of the gaps is zero.** -/
theorem zeroRow_gap {r : Fin (2 ^ (L.model regs).kLog)} (hr : (524800 ≤ r.val ∧ r.val < 1048576) ∨ 1064960 ≤ r.val) :
    ZeroRow (L.model regs) r :=
  zeroRow_outside L regs (not_covers_of hrs hr) (delta_free hda (x := r.val) (by omega))

/-- **A mask row reads itself**: the mask's slot entry is the identity, and Δ skips it. -/
theorem copyRow_mask {r : Fin (2 ^ (L.model regs).kLog)} (hr : 1048576 ≤ r.val ∧ r.val < 1064960) :
    CopyRow (L.model regs) r r := by
  have hm : (Kind.mask, 14, 64, 1) ∈ st.c.ranges.toList.map (fun r => (r.kind, r.slotLog, r.pos0, r.count)) := by
    rw [hrs]; simp [RS]
  obtain ⟨R, hR, hf⟩ := List.mem_map.1 hm
  simp only [Prod.mk.injEq] at hf
  obtain ⟨hk, hs, hp, hc⟩ := hf
  have hcov : FlockLevel3.covers R r.val := by
    unfold FlockLevel3.covers
    rw [hs, hp, hc]
    norm_num
    omega
  have hd := delta_free hda (x := r.val) (by omega)
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

/-- **The gaps' and the mask's rows hold** at the trace. -/
theorem holds_gapMask (hk : st.c.kLog = 22) (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)}
    (hr : 524800 ≤ r.val) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  by_cases hm : 1048576 ≤ r.val ∧ r.val < 1064960
  · exact copyRow_holds (copyRow_mask L regs hda hrs hm) rfl
  · exact zeroRow_holds (zeroRow_gap L regs hda hrs (by omega)) (by rw [block_zC (S := L.model regs) hk, zIn_gap hr]; rfl)

omit hrs in
/-- **The unit's input rows past its 16 inputs hold** at the trace: they are zero, and so is the trace there. -/
theorem holds_unitPad (hk : st.c.kLog = 22) (htm : st.c.tmpl = none) (hU : st.c.unit = UN)
    (hs1 : st.c.unitRange.count = 1) (hsl : st.c.unitRange.slot 0 = 524288) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : 524304 ≤ r.val ∧ r.val < 524416) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have he := UN_empty (r.val - 524304) (by omega)
  rw [show 16 + (r.val - 524304) = r.val - 524288 by omega, ← hU] at he
  refine zeroRow_holds (zeroRow_unitEmpty L regs (L.built_flat htm) (s₀ := 0) (by omega) ?_ he.1 he.2 ?_
    (delta_free hda (x := r.val) (by omega))) ?_
  · rw [hU]; show _ < 512; omega
  · rw [hsl]; omega
  · rw [block_zC (S := L.model regs) hk, zIn_unit (i := r.val) (by omega) (by omega)]
    rfl

omit hrs in
/-- **A unit leaf row copies its source**: its slot entry reads the row itself, and Δ's leaf run cancels that and reads
the source. -/
theorem copyRow_unitLeaf (htm : st.c.tmpl = none) (hU : st.c.unit = UN) (hs1 : st.c.unitRange.count = 1)
    (hsl : st.c.unitRange.slot 0 = 524288) {r : Fin (2 ^ (L.model regs).kLog)} (hr : 524288 ≤ r.val ∧ r.val < 524304)
    (src : Fin (2 ^ (L.model regs).kLog)) (hsrc : src.val = leafSrc (r.val - 524288)) :
    CopyRow (L.model regs) r src := by
  have hB := L.built_flat htm
  have hs : 0 < st.c.unitRange.count := by omega
  have hW : 0 < 2 ^ st.c.unit.unitLog := Nat.two_pow_pos _
  have hc : r.val - 524288 < 2 ^ st.c.unit.unitLog := by rw [hU]; show _ < 512; omega
  have hr' : r.val = st.c.unitRange.slot 0 + (r.val - 524288) := by rw [hsl]; omega
  have hin := UN_in (r.val - 524288) (by omega)
  rw [← hU] at hin
  have hlk : st.c.unit.lookup = none := L.unit_lookup
  have hd := delta_leaf hda (x := r.val) hr
  have key : ∀ (e : Kind → ℕ → ℕ → ZMod 2),
      (∀ j', e (.net st.c.unitNet) (r.val - 524288) j' = if j' = r.val - 524288 then 1 else 0) →
      ∀ j : ℕ, rangesEntry st e r.val j = if j = r.val then 1 else 0 := by
    intro e he j
    rw [hr', L.rangesEntry_unit hs hc e j,
      ite_slot_eq (v := e (.net st.c.unitNet) (r.val - 524288)) hW hc he, ← L.slot_eq]
  have hA : ∀ j', slotEntryA st (.net st.c.unitNet) (r.val - 524288) j' = if j' = r.val - 524288 then 1 else 0 := by
    intro j'
    show sparseEntry st.c.unit.a (r.val - 524288) j' = _
    rw [sparseEntry_built_a hB, hin.1, count_one]
  have hBe : ∀ j', slotEntryB st (.net st.c.unitNet) (r.val - 524288) j' = if j' = r.val - 524288 then 1 else 0 := by
    intro j'
    show sparseEntry st.c.unit.b (r.val - 524288) j' + lookupEntry st.c.unit.lookup (r.val - 524288) j' = _
    rw [sparseEntry_built_b hB, hin.2, count_one, hlk]
    simp only [lookupEntry, add_zero]
  refine ⟨fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
    rw [key (slotEntryA st) hA, hd, ite_cancel, ← hsrc]
    exact if_congr Fin.ext_iff.symm rfl rfl
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
    rw [L.delta_b, key (slotEntryB st) hBe, hd, ite_cancel, ← hsrc]
    exact if_congr Fin.ext_iff.symm rfl rfl

omit hrs in
/-- **The unit's leaf rows hold** at the trace: each copies its source, and the trace is 0 at both. -/
theorem holds_unitLeaf (hk : st.c.kLog = 22) (htm : st.c.tmpl = none) (hU : st.c.unit = UN)
    (hs1 : st.c.unitRange.count = 1) (hsl : st.c.unitRange.slot 0 = 524288) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : 524288 ≤ r.val ∧ r.val < 524304) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have hsrc : leafSrc (r.val - 524288) < 2 ^ (L.model regs).kLog := by
    have : leafSrc (r.val - 524288) < 262720 := by unfold leafSrc; split <;> omega
    exact lt_of_lt_of_le this (by rw [show (L.model regs).kLog = 22 from hk]; norm_num)
  refine copyRow_holds (copyRow_unitLeaf L regs hda htm hU hs1 hsl hr ⟨leafSrc (r.val - 524288), hsrc⟩
    (Fin.val_mk hsrc)) ?_
  rw [block_zC (S := L.model regs) hk, block_zC (S := L.model regs) hk,
    zIn_unit (i := r.val) hr.1 (Nat.ne_of_lt (lt_of_lt_of_le hr.2 (by norm_num))), Fin.val_mk hsrc,
    zIn_leaf (t := r.val - 524288) (Nat.sub_lt_left_of_lt_add hr.1 hr.2)]

omit hda hrs in
/-- **A `sha512x3` input row copies its source** where Δ's pairs at it are its self pair and the source's: its slot
entry reads itself (`shaLaid_self`). -/
theorem copyRow_shaIn (R : RowRanges st.c)
    (hR : (st.c.rangeOf st.c.shaNet).count = 1 ∧ (st.c.rangeOf st.c.shaNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.shaNet).slot 0 = 262144)
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : 262144 ≤ r.val ∧ r.val < 266752) (src : Fin (2 ^ (L.model regs).kLog))
    (hd : ∀ j, deltaEntry st.da r.val j = (((if j = r.val then 1 else 0) + (if j = src.val then 1 else 0) : ℕ) : ZMod 2)) :
    CopyRow (L.model regs) r src := by
  have hat : AtSlot st.c st.c.shaNet 0 (r.val - 262144) r.val :=
    ⟨lt_of_lt_of_eq Nat.one_pos hR.1.symm,
      by rw [hR.2.1]; exact lt_of_lt_of_le (show r.val - 262144 < 262144 by omega) (by norm_num),
      by rw [hR.2.2]; omega⟩
  have h := inBit_rows R (Or.inl rfl) hat (shaNet_ofRows R.rowNets) (shaLaid_self (off := r.val - 262144) (by omega))
    r.isLt (fun j => ⟨hd j, by rw [L.delta_b]; exact hd j⟩)
  exact ⟨fun j => (h j).1.trans (if_congr Fin.ext_iff.symm rfl rfl),
    fun j => (h j).2.trans (if_congr Fin.ext_iff.symm rfl rfl)⟩

omit hda hrs in
/-- **An `hm96` input row copies its source** where Δ's pairs at it are its self pair and the source's. -/
theorem copyRow_hmIn (R : RowRanges st.c)
    (hR : (st.c.rangeOf st.c.hmNet).count = 1 ∧ (st.c.rangeOf st.c.hmNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.hmNet).slot 0 = 0)
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val < 3072) (src : Fin (2 ^ (L.model regs).kLog))
    (hd : ∀ j, deltaEntry st.da r.val j = (((if j = r.val then 1 else 0) + (if j = src.val then 1 else 0) : ℕ) : ZMod 2)) :
    CopyRow (L.model regs) r src := by
  have hat : AtSlot st.c st.c.hmNet 0 r.val r.val :=
    ⟨lt_of_lt_of_eq Nat.one_pos hR.1.symm,
      by rw [hR.2.1]; exact lt_of_lt_of_le (show r.val < 262144 by omega) (by norm_num),
      by rw [hR.2.2]; omega⟩
  have h := inBit_rows R (Or.inr rfl) hat (hmNet_ofRows R.rowNets) (hmLaid_self (off := r.val) hr)
    r.isLt (fun j => ⟨hd j, by rw [L.delta_b]; exact hd j⟩)
  exact ⟨fun j => (h j).1.trans (if_congr Fin.ext_iff.symm rfl rfl),
    fun j => (h j).2.trans (if_congr Fin.ext_iff.symm rfl rfl)⟩

omit hrs in
/-- **The `sha512x3` slot's input rows hold** at the trace: `cv0` copies the midstate's bits from the pin or the forced
zero, and the other inputs copy themselves. -/
theorem holds_shaIn (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) (R : RowRanges st.c)
    (hR : (st.c.rangeOf st.c.shaNet).count = 1 ∧ (st.c.rangeOf st.c.shaNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.shaNet).slot 0 = 262144)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : 262144 ≤ r.val ∧ r.val < 266752) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  by_cases hc : r.val < 262656
  · have hsrc : constSrcW msW (r.val - 262144) < 2 ^ (L.model regs).kLog :=
      lt_of_lt_of_eq (constSrcW_lt _ _) (by rw [show (L.model regs).kLog = 22 from hk])
    have hd := delta_cv0 hda (off := r.val - 262144) (by omega)
    rw [show 262144 + (r.val - 262144) = r.val by omega] at hd
    refine copyRow_holds (copyRow_shaIn L regs R hR hr ⟨constSrcW msW (r.val - 262144), hsrc⟩ hd) ?_
    rw [block_zC (S := L.model regs) hk, block_zC (S := L.model regs) hk, Fin.val_mk hsrc, zIn_constSrc hn,
      zIn_sha (x := r.val) hr.1 (lt_of_lt_of_le hr.2 (by norm_num))]
    unfold xS
    rw [ite_eq_left (show r.val - 262144 < 512 from Nat.sub_lt_left_of_lt_add hr.1 hc)]
  · exact copyRow_holds (copyRow_shaIn L regs R hR hr r (delta_selfForm (delta_free hda (x := r.val) (by omega))))
      rfl

omit hrs in
/-- **The `hm96` slot's input rows hold** at the trace: `cv` copies the `sha512x3` slot's first output, `pad` the
padding block's bits from the pin or the forced zero, and `y` copies itself. -/
theorem holds_hmIn (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) (R : RowRanges st.c)
    (hR : (st.c.rangeOf st.c.hmNet).count = 1 ∧ (st.c.rangeOf st.c.hmNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.hmNet).slot 0 = 0)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val < 3072) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have hk' : (L.model regs).kLog = 22 := hk
  by_cases h1 : r.val < 512
  · have hsrc : 516096 + r.val < 2 ^ (L.model regs).kLog := lt_of_lt_of_eq
      (show 516096 + r.val < 2 ^ 22 from lt_of_lt_of_le (show 516096 + r.val < 524288 by omega) (by norm_num))
      (by rw [hk'])
    refine copyRow_holds (copyRow_hmIn L regs R hR hr ⟨516096 + r.val, hsrc⟩ (delta_cv hda h1)) ?_
    rw [block_zC (S := L.model regs) hk, block_zC (S := L.model regs) hk, Fin.val_mk hsrc,
      zIn_hm (x := r.val) (lt_of_lt_of_le hr (by norm_num))]
    unfold xH zIn
    rw [ite_eq_left h1, ite_eq_right (show ¬ 516096 + r.val < 262144 by omega),
      ite_eq_left (show 516096 + r.val < 524288 by omega), show 516096 + r.val - 262144 = 253952 + r.val by omega]
  by_cases h2 : r.val < 1536
  · have hsrc : constSrcW padW (r.val - 512) < 2 ^ (L.model regs).kLog :=
      lt_of_lt_of_eq (constSrcW_lt _ _) (by rw [hk'])
    have hd := delta_pad hda (off := r.val - 512) (by omega)
    rw [show 512 + (r.val - 512) = r.val by omega] at hd
    refine copyRow_holds (copyRow_hmIn L regs R hR hr ⟨constSrcW padW (r.val - 512), hsrc⟩ hd) ?_
    rw [block_zC (S := L.model regs) hk, block_zC (S := L.model regs) hk, Fin.val_mk hsrc, zIn_constSrc hn,
      zIn_hm (x := r.val) (lt_of_lt_of_le hr (by norm_num))]
    unfold xH
    rw [ite_eq_right h1, ite_eq_left h2]
  · exact copyRow_holds (copyRow_hmIn L regs R hR hr r (delta_selfForm (delta_free hda (x := r.val) (by omega)))) rfl

omit hda hrs in
/-- **A row net's row whose laid rows are empty and that Δ skips is zero.** -/
theorem zeroRow_netEmpty (R : RowRanges st.c) {n s off : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet)
    {r : Fin (2 ^ (L.model regs).kLog)} (hat : AtSlot st.c n s off r.val) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (he : l.ra.getD off #[] = #[] ∧ l.rb.getD off #[] = #[]) (hd : ∀ j, deltaEntry st.da r.val j = 0) :
    ZeroRow (L.model regs) r := by
  obtain ⟨hm, hk, -, hsl⟩ := R.range (R.rowNet_lt hn)
  have h32 : (st.c.nets.getD n default).2.unitLog ≤ 32 := by
    rw [← hsl]; exact le_trans (hat.slotLog_le R.bound hm) R.kLog
  have hw : 2 ^ (st.c.nets.getD n default).2.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) h32
  obtain ⟨-, ha, hb, -, -, -, -, -, -, -, -, -, hlk, -⟩ := Net.ofRows_ok hl
  refine ⟨fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = 0
    rw [hd, add_zero, rangesEntry_at R.disj hm hk hat]
    refine ite_eq_right_iff.2 fun _ => ?_
    show sparseEntry (st.c.nets.getD n default).2.a off _ = 0
    rw [sparseEntry_ofRows ha, ofRows_row ha hw, he.1]
    simp
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = 0
    rw [L.delta_b, hd, add_zero, rangesEntry_at R.disj hm hk hat]
    refine ite_eq_right_iff.2 fun _ => ?_
    show sparseEntry (st.c.nets.getD n default).2.b off _ + lookupEntry (st.c.nets.getD n default).2.lookup off _ = 0
    rw [hlk, sparseEntry_ofRows hb, ofRows_row hb hw, he.2]
    simp [lookupEntry]

omit hrs in
/-- **The `sha512x3` slot's input padding rows hold** at the trace: they are zero, and so is the trace there. -/
theorem holds_shaGap (hk : st.c.kLog = 22) (R : RowRanges st.c)
    (hR : (st.c.rangeOf st.c.shaNet).count = 1 ∧ (st.c.rangeOf st.c.shaNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.shaNet).slot 0 = 262144)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : 266752 ≤ r.val ∧ r.val < 270336) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have hat : AtSlot st.c st.c.shaNet 0 (r.val - 262144) r.val :=
    ⟨lt_of_lt_of_eq Nat.one_pos hR.1.symm,
      by rw [hR.2.1]; exact lt_of_lt_of_le (show r.val - 262144 < 262144 by omega) (by norm_num),
      by rw [hR.2.2]; omega⟩
  refine zeroRow_holds (zeroRow_netEmpty L regs R (Or.inl rfl) hat (shaNet_ofRows R.rowNets)
    (shaLaid_gap (off := r.val - 262144) (by omega) (by omega)) (delta_free hda (x := r.val) (by omega))) ?_
  rw [block_zC (S := L.model regs) hk, zIn_sha (x := r.val) (by omega) hr.2]
  unfold xS
  rw [ite_eq_right (show ¬ r.val - 262144 < 512 by omega)]
  rfl

omit hrs in
/-- **The `hm96` slot's input padding rows hold** at the trace: they are zero, and so is the trace there. -/
theorem holds_hmGap (hk : st.c.kLog = 22) (R : RowRanges st.c)
    (hR : (st.c.rangeOf st.c.hmNet).count = 1 ∧ (st.c.rangeOf st.c.hmNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.hmNet).slot 0 = 0)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : 3072 ≤ r.val ∧ r.val < 4096) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have hat : AtSlot st.c st.c.hmNet 0 r.val r.val :=
    ⟨lt_of_lt_of_eq Nat.one_pos hR.1.symm,
      by rw [hR.2.1]; exact lt_of_lt_of_le (show r.val < 262144 by omega) (by norm_num),
      by rw [hR.2.2]; omega⟩
  refine zeroRow_holds (zeroRow_netEmpty L regs R (Or.inr rfl) hat (hmNet_ofRows R.rowNets)
    (hmLaid_gap (off := r.val) hr.1 hr.2) (delta_free hda (x := r.val) (by omega))) ?_
  rw [block_zC (S := L.model regs) hk, zIn_hm (x := r.val) hr.2]
  unfold xH
  rw [ite_eq_right (show ¬ r.val < 512 by omega), ite_eq_right (show ¬ r.val < 1536 by omega)]
  rfl

omit hda hrs in
/-- **A row net's computed row holds** where the slot constants' pairs skip it and the XORs of the block over its laid
rows multiply to its bit. -/
theorem holds_netRow {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st) {n s off : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) {r : Fin (2 ^ (L.model regs).kLog)} (hat : AtSlot st.c n s off r.val)
    (hlo : (st.c.nets.getD n default).2.inWords * Flock.WORD ≤ off) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hc : ∀ j, (constPairs st.c st.pin).count (r.val, j) = 0) {z : Model.Witness (L.model regs).m}
    (o : Fin (2 ^ (L.model regs).nbl))
    (h : (xorRow (fun j => blockBit (L.model regs) z o ((st.c.rangeOf n).slot s + j)) (l.ra.getD off #[]) &&
      xorRow (fun j => blockBit (L.model regs) z o ((st.c.rangeOf n).slot s + j)) (l.rb.getD off #[])) =
        blockBit (L.model regs) z o r.val) :
    ((L.model regs).A₀.mulVec ((L.model regs).block z o)) r * ((L.model regs).B₀.mulVec ((L.model regs).block z o)) r =
      (L.model regs).block z o r := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨hm, -, -, hsl⟩ := R.range (R.rowNet_lt hn)
  obtain ⟨-, ha, hb, -⟩ := Net.ofRows_ok hl
  have hcl : ∀ rr : Array ℕ, (∀ y ∈ rr.toList, y < 2 ^ (st.c.nets.getD n default).2.unitLog) →
      ∀ y ∈ rr.toList.map ((st.c.rangeOf n).slot s + ·), y < 2 ^ (L.model regs).kLog := fun rr hrr y hy => by
    obtain ⟨y', hy', rfl⟩ := List.mem_map.1 hy
    exact AtSlot.lt R.bound hm ⟨hat.1, by rw [hsl]; exact hrr y' hy', rfl⟩
  refine netRow_holds (b := (st.c.rangeOf n).slot s) (ra := l.ra.getD off #[]) (rb := l.rb.getD off #[])
    (fun j => ?_) (fun j => ?_) (hcl _ (ofRows_cols_lt ha off)) (hcl _ (ofRows_cols_lt hb off)) h
  · have e := (rowNet_entryL L F hn hat hlo hl j.val).1
    rw [hc, Nat.add_zero] at e
    exact e
  · have e := (rowNet_entryL L F hn hat hlo hl j.val).2
    rw [hc, Nat.add_zero] at e
    exact e

omit hda hrs in
/-- **The `sha512x3` slot's computed rows hold** at the trace: its laid rows hold at its forward evaluation
(`Total.sha_total`). -/
theorem holds_shaNet {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st) (hn : Properties.Flock.Nets.FixedNetsBuild)
    (hk : st.c.kLog = 22)
    (hR : (st.c.rangeOf st.c.shaNet).count = 1 ∧ (st.c.rangeOf st.c.shaNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.shaNet).slot 0 = 262144)
    (hiw : (st.c.nets.getD st.c.shaNet default).2.inWords = 64)
    (hcp : constPairs st.c st.pin = [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)])
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : 270336 ≤ r.val ∧ r.val < 518144) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨-, -, -, hsl⟩ := R.range (R.rowNet_lt (Or.inl rfl : st.c.shaNet = st.c.shaNet ∨ _))
  have hwid : (st.c.nets.getD st.c.shaNet default).2.unitLog = 18 := hsl.symm.trans hR.2.1
  obtain ⟨-, ha, hb, -⟩ := Net.ofRows_ok (shaNet_ofRows R.rowNets)
  have hat : AtSlot st.c st.c.shaNet 0 (r.val - 262144) r.val :=
    ⟨lt_of_lt_of_eq Nat.one_pos hR.1.symm,
      by rw [hR.2.1]; exact lt_of_lt_of_le (show r.val - 262144 < 262144 by omega) (by norm_num),
      by rw [hR.2.2]; omega⟩
  have hv : ∀ cs : Array ℕ, (∀ y ∈ cs.toList, y < 2 ^ (st.c.nets.getD st.c.shaNet default).2.unitLog) →
      xorRow (fun j => blockBit (L.model regs) (zC (L.model regs).m) o ((st.c.rangeOf st.c.shaNet).slot 0 + j)) cs =
        xorRow vS cs := by
    intro cs hcs
    refine xorRow_congr fun j hj => ?_
    have hj : j < 2 ^ (st.c.nets.getD st.c.shaNet default).2.unitLog := hcs j hj
    rw [hwid] at hj
    have hj' : j < 262144 := lt_of_lt_of_eq hj (by norm_num)
    show blockBit (L.model regs) (zC (L.model regs).m) o ((st.c.rangeOf st.c.shaNet).slot 0 + j) = vS j
    rw [hR.2.2, blockBit_zC (S := L.model regs) hk o (lt_of_lt_of_le (show 262144 + j < 524288 by omega) (by norm_num))]
    unfold zIn
    rw [ite_eq_right (show ¬ 262144 + j < 262144 by omega), ite_eq_left (show 262144 + j < 524288 by omega),
      Nat.add_sub_cancel_left]
  refine holds_netRow L regs F (Or.inl rfl) hat (by rw [hiw]; show 64 * 128 ≤ _; omega) (shaNet_ofRows R.rowNets)
    (fun j => ?_) o ?_
  · rw [hcp]
    exact List.count_eq_zero.2 fun hm => by
      simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
      omega
  · rw [hv _ (ofRows_cols_lt ha _), hv _ (ofRows_cols_lt hb _),
      blockBit_zC (S := L.model regs) hk o (lt_of_lt_of_le (show r.val < 518144 by omega) (by norm_num))]
    have hz : zIn r.val = vS (r.val - 262144) := by
      unfold zIn
      rw [ite_eq_right (show ¬ r.val < 262144 by omega), ite_eq_left (show r.val < 524288 by omega)]
    rw [hz]
    exact (Total.sha_total xS).2 (r.val - 262144) (by rw [inRows_SHA_IN]; omega)
      (by rw [shaLaid_constPos hn]; omega)

omit hda hrs in
/-- **The `hm96` slot's computed rows hold** at the trace: its laid rows hold at its forward evaluation
(`Total.hm_total`). -/
theorem holds_hmNet {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st) (hn : Properties.Flock.Nets.FixedNetsBuild)
    (hk : st.c.kLog = 22)
    (hR : (st.c.rangeOf st.c.hmNet).count = 1 ∧ (st.c.rangeOf st.c.hmNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.hmNet).slot 0 = 0)
    (hiw : (st.c.nets.getD st.c.hmNet default).2.inWords = 32)
    (hcp : constPairs st.c st.pin = [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)])
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : 4096 ≤ r.val ∧ r.val < 246784) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨-, -, -, hsl⟩ := R.range (R.rowNet_lt (Or.inr rfl : st.c.hmNet = st.c.shaNet ∨ st.c.hmNet = st.c.hmNet))
  have hwid : (st.c.nets.getD st.c.hmNet default).2.unitLog = 18 := hsl.symm.trans hR.2.1
  obtain ⟨-, ha, hb, -⟩ := Net.ofRows_ok (hmNet_ofRows R.rowNets)
  have hat : AtSlot st.c st.c.hmNet 0 r.val r.val :=
    ⟨lt_of_lt_of_eq Nat.one_pos hR.1.symm,
      by rw [hR.2.1]; exact lt_of_lt_of_le (show r.val < 262144 by omega) (by norm_num),
      by rw [hR.2.2]; omega⟩
  have hv : ∀ cs : Array ℕ, (∀ y ∈ cs.toList, y < 2 ^ (st.c.nets.getD st.c.hmNet default).2.unitLog) →
      xorRow (fun j => blockBit (L.model regs) (zC (L.model regs).m) o ((st.c.rangeOf st.c.hmNet).slot 0 + j)) cs =
        xorRow vH cs := by
    intro cs hcs
    refine xorRow_congr fun j hj => ?_
    have hj : j < 2 ^ (st.c.nets.getD st.c.hmNet default).2.unitLog := hcs j hj
    rw [hwid] at hj
    have hj' : j < 262144 := lt_of_lt_of_eq hj (by norm_num)
    show blockBit (L.model regs) (zC (L.model regs).m) o ((st.c.rangeOf st.c.hmNet).slot 0 + j) = vH j
    rw [hR.2.2, Nat.zero_add,
      blockBit_zC (S := L.model regs) hk o (lt_of_lt_of_le (show j < 262144 by omega) (by norm_num))]
    unfold zIn
    rw [ite_eq_left hj']
  refine holds_netRow L regs F (Or.inr rfl) hat (by rw [hiw]; show 32 * 128 ≤ _; omega) (hmNet_ofRows R.rowNets)
    (fun j => ?_) o ?_
  · rw [hcp]
    exact List.count_eq_zero.2 fun hm => by
      simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hm
      omega
  · rw [hv _ (ofRows_cols_lt ha _), hv _ (ofRows_cols_lt hb _),
      blockBit_zC (S := L.model regs) hk o (lt_of_lt_of_le (show r.val < 262144 by omega) (by norm_num))]
    have hz : zIn r.val = vH r.val := by
      unfold zIn
      rw [ite_eq_left (show r.val < 262144 by omega)]
    rw [hz]
    exact (Total.hm_total xH).2 r.val (by rw [inRows_HM_IN]; omega)
      (by rw [hmLaid_constPos hn]; show r.val < 246784; omega)

omit hrs in
/-- **The `sha512x3` slot's rows past its constant hold** at the trace: they are zero, and so is the trace there. -/
theorem holds_shaPast (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) (R : RowRanges st.c)
    (hR : (st.c.rangeOf st.c.shaNet).count = 1 ∧ (st.c.rangeOf st.c.shaNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.shaNet).slot 0 = 262144)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : 518144 < r.val ∧ r.val < 524288) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have hat : AtSlot st.c st.c.shaNet 0 (r.val - 262144) r.val :=
    ⟨lt_of_lt_of_eq Nat.one_pos hR.1.symm,
      by rw [hR.2.1]; exact lt_of_lt_of_le (show r.val - 262144 < 262144 by omega) (by norm_num),
      by rw [hR.2.2]; omega⟩
  have he : (HmNets.shaLaid ()).ra.getD (r.val - 262144) #[] = #[] ∧
      (HmNets.shaLaid ()).rb.getD (r.val - 262144) #[] = #[] :=
    ⟨getD_past (by rw [hn.1.1]; omega), getD_past (by rw [hn.1.2.1]; omega)⟩
  refine zeroRow_holds (zeroRow_netEmpty L regs R (Or.inl rfl) hat (shaNet_ofRows R.rowNets) he
    (delta_free hda (x := r.val) (by omega))) ?_
  have hz : zIn r.val = false := by
    unfold zIn vS
    rw [ite_eq_right (show ¬ r.val < 262144 by omega), ite_eq_left (show r.val < 524288 by omega)]
    exact laidVal_past sha_outs (by rw [sha_end hn]; omega)
  rw [block_zC (S := L.model regs) hk, hz]
  rfl

omit hrs in
/-- **The `hm96` slot's rows past its constant hold** at the trace: they are zero, and so is the trace there. -/
theorem holds_hmPast (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) (R : RowRanges st.c)
    (hR : (st.c.rangeOf st.c.hmNet).count = 1 ∧ (st.c.rangeOf st.c.hmNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.hmNet).slot 0 = 0)
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : 246784 < r.val ∧ r.val < 262144) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have hat : AtSlot st.c st.c.hmNet 0 r.val r.val :=
    ⟨lt_of_lt_of_eq Nat.one_pos hR.1.symm,
      by rw [hR.2.1]; exact lt_of_lt_of_le (show r.val < 262144 by omega) (by norm_num),
      by rw [hR.2.2]; omega⟩
  have he : (HmNets.hmLaid ()).ra.getD r.val #[] = #[] ∧ (HmNets.hmLaid ()).rb.getD r.val #[] = #[] :=
    ⟨getD_past (by rw [hn.2.1]; omega), getD_past (by rw [hn.2.2.1]; omega)⟩
  refine zeroRow_holds (zeroRow_netEmpty L regs R (Or.inr rfl) hat (hmNet_ofRows R.rowNets) he
    (delta_free hda (x := r.val) (by omega))) ?_
  have hz : zIn r.val = false := by
    unfold zIn vH
    rw [ite_eq_left (show r.val < 262144 by omega)]
    exact laidVal_past hm_outs (by rw [hm_end hn]; show 246784 < r.val; omega)
  rw [block_zC (S := L.model regs) hk, hz]
  rfl

omit hda hrs in
/-- **The `sha512x3` slot's constant row copies the pin**: its laid row reads itself, and the slot constant's pairs
cancel that and read the pin. -/
theorem copyRow_shaConst {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st)
    (hn : Properties.Flock.Nets.FixedNetsBuild)
    (hR : (st.c.rangeOf st.c.shaNet).count = 1 ∧ (st.c.rangeOf st.c.shaNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.shaNet).slot 0 = 262144)
    (hiw : (st.c.nets.getD st.c.shaNet default).2.inWords = 64) (hpin : st.pin = PIN)
    (hcp : constPairs st.c st.pin = [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)])
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = 518144) :
    CopyRow (L.model regs) r (L.model regs).pin := by
  have R := RowRanges.ofL L F.parse F.check
  have hat : AtSlot st.c st.c.shaNet 0 256000 r.val :=
    ⟨lt_of_lt_of_eq Nat.one_pos hR.1.symm,
      by rw [hR.2.1]; exact lt_of_lt_of_le (show 256000 < 262144 by omega) (by norm_num),
      by rw [hR.2.2]; omega⟩
  have hlo : (st.c.nets.getD st.c.shaNet default).2.inWords * Flock.WORD ≤ 256000 := by
    rw [hiw]; show 64 * 128 ≤ 256000; omega
  have e := fun j => rowNet_entryL L F (Or.inl rfl) hat hlo (shaNet_ofRows R.rowNets) j
  have hc := shaLaid_const hn
  refine ⟨fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
    rw [(e j.val).1, hc.1, hR.2.2, hcp, hr, constRow_count, ← hpin]
    exact if_congr ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩ rfl rfl
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
    rw [(e j.val).2, hc.2, hR.2.2, hcp, hr, constRow_count, ← hpin]
    exact if_congr ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩ rfl rfl

omit hda hrs in
/-- **The `sha512x3` slot's constant row holds** at the trace: it copies the pin, and the trace is 1 at both. -/
theorem holds_shaConst {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22)
    (hR : (st.c.rangeOf st.c.shaNet).count = 1 ∧ (st.c.rangeOf st.c.shaNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.shaNet).slot 0 = 262144)
    (hiw : (st.c.nets.getD st.c.shaNet default).2.inWords = 64) (hpin : st.pin = PIN)
    (hcp : constPairs st.c st.pin = [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)])
    (o : Fin (2 ^ (L.model regs).nbl)) {r : Fin (2 ^ (L.model regs).kLog)} (hr : r.val = 518144) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  refine copyRow_holds (copyRow_shaConst L regs F hn hR hiw hpin hcp hr) ?_
  rw [block_pin L regs hn hk hpin o, block_zC (S := L.model regs) hk, hr, zIn_shaConst hn]
  rfl

omit hrs in
/-- **The unit's rows past its constant hold** at the trace: they are zero, and so is the trace there. -/
theorem holds_unitPast (hk : st.c.kLog = 22) (htm : st.c.tmpl = none) (hU : st.c.unit = UN)
    (hs1 : st.c.unitRange.count = 1) (hsl : st.c.unitRange.slot 0 = 524288) (o : Fin (2 ^ (L.model regs).nbl))
    {r : Fin (2 ^ (L.model regs).kLog)} (hr : 524544 < r.val ∧ r.val < 524800) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have he := UN_past (r.val - 524545) (by omega)
  rw [show 257 + (r.val - 524545) = r.val - 524288 by omega, ← hU] at he
  refine zeroRow_holds (zeroRow_unitEmpty L regs (L.built_flat htm) (s₀ := 0) (by omega) ?_ he.1 he.2 ?_
    (delta_free hda (x := r.val) (by omega))) ?_
  · rw [hU]; show _ < 512; omega
  · rw [hsl]; omega
  · rw [block_zC (S := L.model regs) hk, zIn_unit (i := r.val) (by omega) (by omega)]
    rfl

/-- **Every row of the block holds** at the trace, by its place: the `hm96` slot's input, padding, computed and past
rows and its constant, the pin; the `sha512x3` slot's likewise; the unit's leaf, padding, computed and past rows and
its constant; and the gaps' and the mask's rows. -/
theorem holds_all {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st) (hn : Properties.Flock.Nets.FixedNetsBuild)
    (hk : st.c.kLog = 22) (hpin : st.pin = PIN) (htm : st.c.tmpl = none) (hU : st.c.unit = UN)
    (hs1 : st.c.unitRange.count = 1) (hsl : st.c.unitRange.slot 0 = 524288)
    (hRs : (st.c.rangeOf st.c.shaNet).count = 1 ∧ (st.c.rangeOf st.c.shaNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.shaNet).slot 0 = 262144)
    (hRh : (st.c.rangeOf st.c.hmNet).count = 1 ∧ (st.c.rangeOf st.c.hmNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.hmNet).slot 0 = 0)
    (hiws : (st.c.nets.getD st.c.shaNet default).2.inWords = 64)
    (hiwh : (st.c.nets.getD st.c.hmNet default).2.inWords = 32)
    (hcp : constPairs st.c st.pin = [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)])
    (o : Fin (2 ^ (L.model regs).nbl)) (r : Fin (2 ^ (L.model regs).kLog)) :
    ((L.model regs).A₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r *
        ((L.model regs).B₀.mulVec ((L.model regs).block (zC (L.model regs).m) o)) r =
      (L.model regs).block (zC (L.model regs).m) o r := by
  have R := RowRanges.ofL L F.parse F.check
  have hu : st.c.unitRange.slot 0 + st.c.unit.constPos = 524544 := by
    rw [hsl, hU]; show 524288 + 256 = 524544; rfl
  by_cases h1 : r.val < 3072
  · exact holds_hmIn L regs hda hn hk R hRh o h1
  by_cases h2 : r.val < 4096
  · exact holds_hmGap L regs hda hk R hRh o ⟨by omega, h2⟩
  by_cases h3 : r.val < 246784
  · exact holds_hmNet L regs F hn hk hRh hiwh hcp o ⟨by omega, h3⟩
  by_cases h4 : r.val = 246784
  · exact holds_pins L regs hn hk hpin (by omega) hu o (Or.inl h4)
  by_cases h5 : r.val < 262144
  · exact holds_hmPast L regs hda hn hk R hRh o ⟨by omega, h5⟩
  by_cases h6 : r.val < 266752
  · exact holds_shaIn L regs hda hn hk R hRs o ⟨by omega, h6⟩
  by_cases h7 : r.val < 270336
  · exact holds_shaGap L regs hda hk R hRs o ⟨by omega, h7⟩
  by_cases h8 : r.val < 518144
  · exact holds_shaNet L regs F hn hk hRs hiws hcp o ⟨by omega, h8⟩
  by_cases h9 : r.val = 518144
  · exact holds_shaConst L regs F hn hk hRs hiws hpin hcp o h9
  by_cases h10 : r.val < 524288
  · exact holds_shaPast L regs hda hn hk R hRs o ⟨by omega, h10⟩
  by_cases h11 : r.val < 524304
  · exact holds_unitLeaf L regs hda hk htm hU hs1 hsl o ⟨by omega, h11⟩
  by_cases h12 : r.val < 524416
  · exact holds_unitPad L regs hda hk htm hU hs1 hsl o ⟨by omega, h12⟩
  by_cases h13 : r.val < 524544
  · exact holds_unitComp_cW L regs hk htm hU hs1 hsl o ⟨by omega, h13⟩
  by_cases h14 : r.val = 524544
  · exact holds_pins L regs hn hk hpin (by omega) hu o (Or.inr h14)
  by_cases h15 : r.val < 524800
  · exact holds_unitPast L regs hda hk htm hU hs1 hsl o ⟨by omega, h15⟩
  exact holds_gapMask L regs hda hrs hk o (by omega)

/-- **The trace satisfies the concrete table's statement** at no public region: every row holds in every block, and
the trace is 1 at the pin. -/
theorem satisfies_zC {tags : Tags} (F : Discharge.Aliased.UnitFacts tags st)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (hk : st.c.kLog = 22) (hpin : st.pin = PIN) (htm : st.c.tmpl = none)
    (hU : st.c.unit = UN) (hs1 : st.c.unitRange.count = 1) (hsl : st.c.unitRange.slot 0 = 524288)
    (hRs : (st.c.rangeOf st.c.shaNet).count = 1 ∧ (st.c.rangeOf st.c.shaNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.shaNet).slot 0 = 262144)
    (hRh : (st.c.rangeOf st.c.hmNet).count = 1 ∧ (st.c.rangeOf st.c.hmNet).slotLog = 18 ∧
      (st.c.rangeOf st.c.hmNet).slot 0 = 0)
    (hiws : (st.c.nets.getD st.c.shaNet default).2.inWords = 64)
    (hiwh : (st.c.nets.getD st.c.hmNet default).2.inWords = 32)
    (hcp : constPairs st.c st.pin = [(518144, 518144), (518144, PIN), (524544, 524544), (524544, PIN)]) :
    (L.model []).Satisfies (zC (L.model []).m) := by
  refine ⟨fun o r => holds_all L [] hda hrs F hn hk hpin htm hU hs1 hsl hRs hRh hiws hiwh hcp o r,
    fun o => block_pin L [] hn hk hpin o, fun R hR => ?_⟩
  exact absurd hR List.not_mem_nil

end place

/-- **The trace satisfies the concrete table's statement** (`stW`) at no public region, at any layout and unit facts
of it: the statement's slots, constants and Δ are the concrete circuit's (`c_stW` and the slot facts). -/
theorem satisfies_stW (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
    (hello : ByteArray → Nat → String) (L : FlockSoundness.Layout (stW ls lh H A B A' B' s1 s3 s4 s6 hello))
    {tags : Tags} (F : Discharge.Aliased.UnitFacts tags (stW ls lh H A B A' B' s1 s3 s4 s6 hello))
    (hn : Properties.Flock.Nets.FixedNetsBuild) :
    (L.model []).Satisfies (zC (L.model []).m) := by
  have hc := c_stW ls lh H A B A' B' s1 s3 s4 s6 hello
  have hnets := nets_cW ls lh H A B A' B' s1
  have hiw := inWords_cW ls lh H A B A' B' s1
  exact satisfies_zC L (da_stW ls lh H A B A' B' s1 s3 s4 s6 hello)
    (by rw [hc]; exact ranges_cW ls lh H A B A' B' s1) F hn
    (by rw [hc]; exact kLog_cW ls lh H A B A' B' s1) (pin_stW ls lh H A B A' B' s1 s3 s4 s6 hello)
    (by rw [hc]; exact tmpl_cW ls lh H A B A' B' s1) (by rw [hc]; exact unit_cW ls lh H A B A' B' s1)
    (by rw [hc]; exact (unitRange_cW ls lh H A B A' B' s1).1) (by rw [hc]; exact (unitRange_cW ls lh H A B A' B' s1).2)
    (by rw [hc]; exact shaRange_cW ls lh H A B A' B' s1) (by rw [hc]; exact hmRange_cW ls lh H A B A' B' s1)
    (by rw [hc, hnets.2.1]; exact hiw.2.1) (by rw [hc, hnets.2.2]; exact hiw.2.2)
    (by rw [hc, pin_stW]; exact constPairs_cW ls lh H A B A' B' s1)

end FlockDeploy.Concrete.Trace
