import Flock.Hash
import RunModel.Prog

/-!
# A toy sampled audit, as the auditor's program

The developer claims a table of `2^d` rows, row `i` holding `want i`, and registers a SHA-512 Merkle root over it. The
auditor draws `k` distinct units after the root is held and asks for each one's opening; it accepts when every opening
climbs to the root and holds the claimed row. Checking the opened row stands in for verifying the unit's proof, which is
what sampled proofs does with C-Flock.

The library functions (`draw`, `checkOpening`) are the same ones a stateless executable would expose to a Python
driver; `auditor` is the auditor's program over the run, which calls them in the protocol's order.
-/

namespace RunModel.Audit

open Flock RunModel.Frame

def leaf (row : ByteArray) : ByteArray := Sha512.hash (ByteArray.mk #[0] ++ row)

def node (l r : ByteArray) : ByteArray := Sha512.hash (ByteArray.mk #[1] ++ l ++ r)

/-- The claimed row `i`: `i² + 7` modulo `2^64`, as 8 big-endian bytes. -/
def want (i : Nat) : ByteArray := u64be ((i * i + 7) % 2 ^ 64)

/-- `k` distinct units of `[0, 2^d)` from a 32-byte seed: the low `d` bits of SHA-512(seed ‖ counter), repeats skipped. -/
def draw (seed : ByteArray) (d k : Nat) : Array Nat := Id.run do
  let n := 2 ^ d
  let mut out : Array Nat := #[]
  let mut ctr := 0
  while out.size < min k n do
    let u := ofU64be (Sha512.hash (seed ++ u64be ctr)) 0 % n
    if !out.contains u then out := out.push u
    ctr := ctr + 1
  return out

/-- An opening of unit `u` at depth `d`: the row's 8 bytes, then `d` siblings of 64 bytes from the leaf up. -/
def checkOpening (root : ByteArray) (d u : Nat) (o : ByteArray) : Except String Unit := do
  if o.size != 8 + 64 * d then throw s!"the opening is {o.size} bytes, not {8 + 64 * d}"
  let row := o.extract 0 8
  let mut h := leaf row
  for j in [0:d] do
    let sib := o.extract (8 + 64 * j) (8 + 64 * (j + 1))
    h := if (u >>> j) % 2 == 0 then node h sib else node sib h
  if h != root then throw "the opening does not climb to the registered root"
  if row != want u then throw "the opened row is not the claimed one"

def hex (b : ByteArray) : String :=
  b.foldl (fun s x => s ++ (if x < 16 then "0" else "") ++ String.ofList (Nat.toDigits 16 x.toNat)) ""

structure Verdict where
  accepted : Bool
  why : String
  root : ByteArray
  units : Array Nat

def Verdict.json (v : Verdict) : String :=
  s!"\{\"accepted\": {v.accepted}, \"why\": \"{v.why}\", \"root\": \"{hex v.root}\", \"units\": {v.units.toList}}"

/-- The auditor's program: take the registration, then draw, then open each drawn unit. The draw happens once, after
the root is held, and its coins never leave the program before the openings are asked for. -/
def auditor (d k : Nat) : Prog Verdict := do
  let root ← Prog.req "developer" "register".toUTF8 64
  if root.size != 64 then return ⟨false, "registration: the root is not 64 bytes", root, #[]⟩
  let seed ← Prog.coins 32
  let units := draw seed d k
  for u in units do
    let o ← Prog.req "developer" ("open".toUTF8 ++ u64be u) (8 + 64 * d)
    if let .error e := checkOpening root d u o then
      return ⟨false, s!"unit {u}: {e}", root, units⟩
  return ⟨true, "", root, units⟩

end RunModel.Audit
