/-
The in-circuit row leaf as a parameter of the circuit statement (PROTOCOL.md §16.5). The leaf is the part of the block that
hashes each instance's input rows into the digests the frame-v3 roots are built from: its slot circuit, its Δ, its regions
and the public check of its digests. The statement composes it with the units, the mask slot and the output region.
Today's leaf is the keyed-BLAKE3 row (`Blake3Row`); `hm96-sha256/v1` (Halevi–Micali on SHA-256, PR #88) is the next.
-/
import Flock.Public

namespace Flock

/-- A region (§16.2): `free` in-block bit positions (LSB first), the other bits fixed to `fixed` where `mask` is set, and
the region's known bytes in each block, in free-assignment order (bit `8i + t` is bit `t` of byte `i`). -/
structure Region where
  name : String
  free : Array Nat
  mask : Nat
  fixed : Nat
  bytes : Nat → ByteArray
  deriving Inhabited

def PT_LOCAL : Nat := 24

/-- A region of a block of `2^kl` bits: at most `PT_LOCAL` distinct free bits below `kl`, `fixed` inside the non-free
bits. Distinct free bits are part of the pinned bound's hypothesis (`Statement.LinkLayout`), so a repeat is refused. -/
def mkRegion (kl : Nat) (name : String) (free : Array Nat) (fixed : Nat) (bytes : Nat → ByteArray) : V Region := do
  let freeMask := free.foldl (fun acc b => acc ||| 2 ^ b) 0
  let mask := (2 ^ kl - 1) &&& ((2 ^ kl - 1) ^^^ freeMask)
  if fixed ≥ 2 ^ kl || fixed &&& freeMask != 0 || free.size > PT_LOCAL || free.any (· ≥ kl) then
    throw s!"{name}: not an aligned region of at most {PT_LOCAL} free bits"
  if !decide free.toList.Nodup then throw s!"{name}: a free bit repeats; a region's free bits are distinct"
  return ⟨name, free, mask, fixed, bytes⟩

/-- A region `mkRegion` builds has the free bits the soundness bound's `LinkLayout` asks for: the ones given, distinct, at
most `PT_LOCAL` of them, each below `kl`. -/
theorem mkRegion_ok {kl : Nat} {name : String} {free : Array Nat} {fixed : Nat} {bytes : Nat → ByteArray} {r : Region}
    (h : mkRegion kl name free fixed bytes = .ok r) :
    r.free = free ∧ free.toList.Nodup ∧ free.size ≤ PT_LOCAL ∧ ∀ b ∈ free, b < kl := by
  unfold mkRegion at h
  simp only at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  rename_i h1
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  rename_i h2
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  simp only [Bool.or_eq_true, not_or] at h1
  obtain ⟨⟨⟨-, -⟩, hsize⟩, hany⟩ := h1
  refine ⟨rfl, by simpa using h2, by simpa using hsize, fun b hb => ?_⟩
  have := Array.any_eq_false.mp (Bool.eq_false_iff.mpr hany)
  rw [Array.mem_iff_getElem] at hb
  obtain ⟨i, hi, rfl⟩ := hb
  simpa using this i hi

def Circuit.instOf (c : Circuit) (n b g : Nat) : Option Nat :=
  let i := b * c.g + g
  if i < n then some i else none

structure RowLeaf where
  /-- the frame-v3 schema of an input port's row digests -/
  schema : String
  /-- the leaf's slot circuit: the `comp` ranges -/
  comp : Comp
  /-- the leaf's part of the layout: its ranges' roles and slot size, the input ports' chunking -/
  checkLayout : Circuit → V Unit
  /-- Δ of `comp` slot `s` of range `r` beyond its constant wiring (chaining, parent messages), as (A, B) entries -/
  slotDelta : Circuit → Range → Nat → V (Array (Nat × Nat) × Array (Nat × Nat))
  /-- the block column where template instance `g`'s byte `byte` of input port `p` is hashed (the unit's leaf bits copy it): `c g p byte` -/
  messageCol : Circuit → Nat → Nat → Nat → V Nat
  /-- the leaf's regions for `n` instances, with their known bytes -/
  regions : Circuit → Public → V (Array Region)
  /-- the input ports' frame-v3 roots over the public digests, and the pinned padding digests -/
  checkPublic : Circuit → Public → V Unit
  /-- the key the statement digest binds after the circuit's SHA-256 -/
  key : Circuit → ByteArray
  deriving Inhabited

end Flock
