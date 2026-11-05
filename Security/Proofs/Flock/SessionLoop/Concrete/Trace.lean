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

set_option maxHeartbeats 0 -- `deltaL_cW` and the slot facts evaluate the concrete circuit's Δ and ranges in the kernel
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
    exact if_congr Fin.ext_iff.symm rfl rfl
  · rw [show (L.model regs).B₀ r j = placedB st r j from rfl, L.placedB_const hs r j hr]
    exact if_congr Fin.ext_iff.symm rfl rfl

/-- **The pin's row copies itself.** -/
theorem copyRow_pin {st : Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    CopyRow (L.model regs) (L.model regs).pin (L.model regs).pin := by
  refine ⟨fun j => ?_, fun j => ?_⟩
  · rw [show (L.model regs).A₀ (L.model regs).pin j = placedA st ⟨st.pin, L.pin_lt⟩ j from rfl, (L.placed_pin j).1]
    exact if_congr Fin.ext_iff.symm rfl rfl
  · rw [show (L.model regs).B₀ (L.model regs).pin j = placedB st ⟨st.pin, L.pin_lt⟩ j from rfl, (L.placed_pin j).2]
    exact if_congr Fin.ext_iff.symm rfl rfl

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

end place

end FlockDeploy.Concrete.Trace
