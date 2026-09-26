/-
PROTOCOL.md §16.5: a table-lookup slot, `value = table[index]` for a pinned MUFU table, built from the table itself the same
way on both sides (PR #83's `lookup.rs`), as an expanded circuit. Rows: the index bits (one 128-bit input word); the one-hot
decoders of the low `lo` and high `n − lo` index bits (recursively, each minterm the AND of two half minterms); per output
bit `j` and high minterm `h`, one product row `hi_h · (XOR of the low minterms l with bit j of table[h·2^lo + l] set)`;
padding to a word; the output word, bit `j` the XOR of its products; the constant.

Every row is explicit except the product rows' B side (about `bits · 2^(n−1)` entries for a table of `2^n` words), which the
fold reads from the table directly (`Lookup.foldB`).
-/
import Flock.Net

namespace Flock

/-- The pinned MUFU tables: name, SHA-256 of the raw `u32` LE file, word count. -/
def MUFU_TABLES : List (String × String × Nat) :=
  [("rsq", "218fc04d271773269367dedf4cff9ff719081b1218c3d67d248d8d6126040246", 2 ^ 24),
   ("sqrt", "4e017678de306e0be8bed66193c2ebf09e9c60a9600cad56cd05bb7b6c3f266f", 2 ^ 24),
   ("rcp", "c408381446132e07fa5581e930dc33fb790bea7ec4276ef3621e322df2492a55", 2 ^ 23),
   ("ex2", "b2a42c4a626469c986e33e43f16f41bde9d84de94347cd9ceaa1bd68d36bcdf0", 2 ^ 23)]

namespace Lookup

/-- A linear form: columns, and whether the constant is added. -/
abbrev Form := Array Nat × Bool

structure Builder where
  a : Array (Array Nat) := #[]
  b : Array (Array Nat) := #[]

/-- The constant's column is only known at the end: forms carry `none` for it until then. -/
def KONST : Nat := 0xFFFFFFFFFFFF

def cols (f : Form) : Array Nat := if f.2 then f.1.push KONST else f.1

def row (bd : Builder) (x y : Form) : Builder × Nat :=
  ({ a := bd.a.push (cols x), b := bd.b.push (cols y) }, bd.a.size)

/-- Minterms of `bits` (index `i` = the assignment of value `i`), committing the blocks of two or more bits; returns the
builder, the forms, and the first row of the top block (for two or more bits). -/
def decode (bd0 : Builder) (bits : List Nat) (fuel : Nat) : Builder × Array Form × Nat :=
  match fuel, bits with
  | _, [b] => (bd0, #[(#[b], true), (#[b], false)], 0)
  | 0, _ => (bd0, #[], 0)
  | fuel + 1, _ =>
    let h := bits.length / 2
    let (bd1, lo, _) := decode bd0 (bits.take h) fuel
    let (bd2, hi, _) := decode bd1 (bits.drop h) fuel
    let first := bd2.a.size
    let (bd3, out) := (List.range (2 ^ bits.length)).foldl (fun (acc : Builder × Array Form) i =>
      let (bd, r) := row acc.1 (lo.getD (i % 2 ^ h) default) (hi.getD (i / 2 ^ h) default)
      (bd, acc.2.push (#[r], false))) (bd2, #[])
    (bd3, out, first)

/-- The slot's expanded circuit, and what the fold reads from the table. -/
def build (name : String) (table : ByteArray) (n lo bits : Nat) : V Net := do
  if table.size != 4 * 2 ^ n || !(2 ≤ lo && lo < n) || bits > 32 || n > 128 then throw s!"lookup {name}: not a table of 2^{n} words"
  let mut bd : Builder := {}
  for i in [0:128] do
    bd := if i < n then { a := bd.a.push #[i], b := bd.b.push #[i] } else { a := bd.a.push #[], b := bd.b.push #[] }
  let (bd1, loM, loTop) := decode bd ((List.range lo)) lo
  let (bd2, hiM, _) := decode bd1 ((List.range (n - lo)).map (lo + ·)) (n - lo)
  bd := bd2
  let nlo := 2 ^ lo
  let nh := hiM.size
  let prod0 := bd.a.size
  for _ in [0:bits] do
    for h in [0:nh] do
      bd := { a := bd.a.push (cols (hiM.getD h default)), b := bd.b.push #[] }
  while bd.a.size % 128 != 0 do bd := { a := bd.a.push #[], b := bd.b.push #[] }
  let out0 := bd.a.size
  for j in [0:128] do
    bd := if j < bits then { a := bd.a.push ((Array.range nh).map (prod0 + j * nh + ·)), b := bd.b.push #[KONST] }
          else { a := bd.a.push #[], b := bd.b.push #[] }
  let konst := bd.a.size
  bd := { a := bd.a.push #[konst], b := bd.b.push #[konst] }
  let fix := fun (rs : Array (Array Nat)) => rs.map fun r => r.map fun c => if c == KONST then konst else c
  let useful := bd.a.size
  let unitLog := max 7 (log2ceil useful)
  if loM.size != nlo || nh != 2 ^ (n - lo) then throw s!"lookup {name}: decoder shape"
  let a ← Sparse.ofRows (fix bd.a) (2 ^ unitLog)
  let b ← Sparse.ofRows (fix bd.b) (2 ^ unitLog)
  return { name, useful, constPos := konst, inGroups := #[⟨0, 1, #[n]⟩], outGroups := #[⟨out0 / 128, 1, #[bits]⟩],
           inWords := 1, a, b, unitLog, lookup := some { table, n, lo, bits, prod0, loTop } }

/-- The product rows' B side of `Bᵀe`: `out[loTop + l] += Σ_{j, h : bit j of table[h·2^lo + l]} e[prod0 + j·nh + h]`,
with one table of subset sums per byte of the word, per high minterm. -/
def foldB (lk : LookupTable) (e : Array F128) (acc0 : Array F128) : Array F128 := Id.run do
  let nlo := 2 ^ lk.lo
  let nh := 2 ^ (lk.n - lk.lo)
  let mut acc := acc0
  let wmask := 2 ^ lk.bits - 1
  for h in [0:nh] do
    let ej := fun (j : Nat) => if j < lk.bits then e.getD (lk.prod0 + j * nh + h) 0 else 0
    let tabs := (Array.range 4).map fun k => Id.run do
      let mut t : Array F128 := Array.replicate 256 0
      for x in [1:256] do
        let low := x &&& (256 - x)
        t := t.set! x (t[x ^^^ low]! + ej (8 * k + Nat.log2 low))
      return t
    for l in [0:nlo] do
      let w := rd32 lk.table (4 * (h * nlo + l)) &&& wmask
      if w == 0 then continue
      let v := tabs[0]![w % 256]! + tabs[1]![(w >>> 8) % 256]! + tabs[2]![(w >>> 16) % 256]! + tabs[3]![(w >>> 24) % 256]!
      acc := acc.modify (lk.loTop + l) (· + v)
  return acc

end Lookup

end Flock
