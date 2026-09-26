/-
PROTOCOL.md §16.5: an expanded circuit in `flock-ir-unit/v2` form (a unit, a tail stage, or a lookup slot built from its
table), and the text helpers its parser uses.
-/
import Lean.Data.Json
import Flock.Comp

namespace Flock

open Lean (Json)

def WORD : Nat := 128

def log2ceil (n : Nat) : Nat := if n ≤ 1 then 0 else Nat.log2 (n - 1) + 1
def isPow2 (n : Nat) : Bool := n > 0 && 2 ^ Nat.log2 n == n
def nextPow2 (n : Nat) : Nat := 2 ^ log2ceil n

def tokens (s : String) : Array String := Id.run do
  let mut out := #[]
  let mut cur := ""
  for c in s.toList do
    if c.isWhitespace then
      if cur != "" then
        out := out.push cur
        cur := ""
    else cur := cur.push c
  if cur != "" then out := out.push cur
  return out

/-- A port group of an expanded circuit: its first 128-bit word column, its word count (an aligned power of two) and its
ports' widths. -/
structure Group where
  col : Nat
  words : Nat
  bits : Array Nat
  deriving Inhabited, Repr

/-- Every port of a group list: its first column in the slot and its width, groups in order. -/
def portCols (gs : Array Group) : Array (Nat × Nat) := Id.run do
  let mut out := #[]
  for g in gs do
    let mut at_ := g.col * WORD
    for w in g.bits do
      out := out.push (at_, w)
      at_ := at_ + w
  return out

/-- A lookup slot's table and where its product rows sit (`Flock.Lookup`). -/
structure LookupTable where
  table : ByteArray
  n : Nat
  lo : Nat
  bits : Nat
  prod0 : Nat
  /-- the low decoder's top block: low minterm `l` is row `loTop + l` -/
  loTop : Nat
  deriving Inhabited

/-- An expanded circuit in `flock-ir-unit/v2` form, `2^unitLog` rows (rows past `useful` are empty). -/
structure Net where
  name : String
  useful : Nat
  constPos : Nat
  inGroups : Array Group
  outGroups : Array Group
  inWords : Nat
  a : Sparse
  b : Sparse
  unitLog : Nat
  /-- a lookup slot: what the fold reads from its table beyond the explicit rows -/
  lookup : Option LookupTable := none
  deriving Inhabited

namespace Net

private def num (h : Array String) (i : Nat) : V Nat :=
  match h[i]?.bind String.toNat? with
  | some n => pure n
  | none => throw s!"flock-ir-unit/v2: header field {i} is not a number"

private def groups (h : Array String) (at0 : Nat) : V (Array Group × Nat) := do
  let count ← num h at0
  let mut at_ := at0 + 1
  let mut out := #[]
  for _ in [0:count] do
    let col ← num h at_
    let words ← num h (at_ + 1)
    let np ← num h (at_ + 2)
    let mut bits := #[]
    for i in [0:np] do bits := bits.push (← num h (at_ + 3 + i))
    if !(isPow2 words && col % words == 0) then throw "flock-ir-unit/v2: a port group is not an aligned power of two of words"
    out := out.push ⟨col, words, bits⟩
    at_ := at_ + 3 + np
  return (out, at_)

/-- Parse one expanded circuit (the lines of a `NET` block). -/
def parse (lines : Array String) : V Net := do
  let h := tokens (lines.getD 0 "")
  if h[0]? != some "flock-ir-unit/v2" then throw "not a flock-ir-unit/v2 circuit"
  let name := h.getD 1 ""
  let useful ← num h 2
  let constPos ← num h 3
  let (inGroups, at1) ← groups h 4
  let (outGroups, at2) ← groups h at1
  if at2 != h.size then throw "flock-ir-unit/v2: trailing header fields"
  if outGroups.isEmpty then throw "flock-ir-unit/v2: no output group"
  if constPos + 1 != useful then throw "flock-ir-unit/v2: the constant is not the last useful column"
  let inWords := (inGroups.back?.map fun g => g.col + g.words).getD 0
  let unitLog := max 7 (log2ceil useful)
  let mut ra := #[]
  let mut rb := #[]
  for i in [0:useful] do
    let v ← (tokens (lines.getD (1 + i) "")).mapM fun t => match t.toNat? with
      | some n => pure n
      | none => throw s!"flock-ir-unit/v2: row {i} is not numbers"
    let na := v.getD 0 0
    let nb := v.getD (1 + na) 0
    if v.size == 0 || v.size != 2 + na + nb then throw s!"flock-ir-unit/v2: row {i} is malformed"
    ra := ra.push (v.extract 1 (1 + na))
    rb := rb.push (v.extract (2 + na) (2 + na + nb))
  for i in [0:inWords * WORD] do
    let ai := ra.getD i #[]
    let bi := rb.getD i #[]
    if !((ai == #[i] && bi == #[i]) || (ai.isEmpty && bi.isEmpty)) then throw s!"flock-ir-unit/v2: row {i} is not an input row"
  if ra.getD constPos #[] != #[constPos] || rb.getD constPos #[] != #[constPos] then
    throw "flock-ir-unit/v2: the constant row is not A = B = [const]"
  -- after the rows: an optional `LEAVES <json>` line (bound by the pin, not read), and nothing else
  let mut at_ := 1 + useful
  if (lines.getD at_ "").startsWith "LEAVES " then
    match Json.parse ((lines.getD at_ "").drop 7).toString with
    | .ok _ => at_ := at_ + 1
    | .error e => throw s!"flock-ir-unit/v2: the LEAVES line is not JSON: {e}"
  if (lines.getD at_ "").startsWith "CUT " then
    throw s!"{name}: a CUT line (a tail the verifier evaluates); the verifier evaluates no part of the computation"
  if at_ != lines.size then throw "flock-ir-unit/v2: text after the rows that is not its LEAVES line"
  let a ← Sparse.ofRows ra (2 ^ unitLog)
  let b ← Sparse.ofRows rb (2 ^ unitLog)
  return { name, useful, constPos, inGroups, outGroups, inWords, a, b, unitLog }

end Net

end Flock
