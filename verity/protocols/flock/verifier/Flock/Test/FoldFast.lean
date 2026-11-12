module

public import Flock.Statement
meta import Flock.Statement

/-!
# The lincheck fold's compiled loops against the functions they replace

Compiled code runs `Sparse.foldTFast` for `Sparse.foldT`, `Stmt.tensorFast` for `Stmt.tensor` and `Stmt.projectFast`
for `Stmt.project` (their `@[csimp]` equations), so a call of one of these functions here runs its twin. Each function's
body is restated here under another name, which no equation rewrites, with `rfl` that the restatement is the function;
the twin's result is compared with the restatement's on inputs that reach each of its cases. The walk: a row twice in a
column, an empty row, a zero coefficient, more and fewer coefficients than rows. The tensor: `eq(ρ)` below the 1024
entries its tables start at, at them, twice to sixteen times them, and no `lw`. The projection: a base of many blocks,
of one, of none, and one longer than its blocks. No case reads past an array's end, which would print a panic for every
such read.
-/

@[expose] public section

namespace Flock.Test.FoldFast

open Flock

def f (i : Nat) : F128 := ⟨(0x9e3779b97f4a7c15 * (i.toUInt64 + 1)) ^^^ 0x5bd1e995, 0xc2b2ae3d27d4eb4f * (i.toUInt64 + 7)⟩

/-- `Sparse.foldT`'s body. -/
def foldTRef (m : Sparse) (e : Array F128) : Array F128 :=
  m.foldRows e (min m.rows e.size) 0 (Array.replicate m.width 0)

example : @foldTRef = @Sparse.foldT := rfl

/-- 40 rows of 0 to 6 entries in 23 columns: some rows empty, and a row of 5 or 6 has its first column again as its
fifth. -/
def rows : Array (Array Nat) :=
  (Array.range 40).map fun i => (Array.range (i % 7)).map fun t => (i * 31 + t % 4 * 17) % 23

/-- The twin and the restatement agree on `rows` in `width` columns, on `n` coefficients with a zero at `zeroAt`. -/
def foldAgree (width n zeroAt : Nat) : Bool :=
  match Sparse.ofRows rows width with
  | .ok m =>
    let e := (Array.range n).map fun i => if i == zeroAt then 0 else f (900 + i)
    let t := m.foldT e
    t == foldTRef m e && t.size == width && t.any (!·.isZero)
  | .error _ => false

#guard rows.any (·.isEmpty) && rows.any fun r => r.size ≥ 5 && r[0]! == r[4]!
#guard foldAgree 23 40 5
#guard foldAgree 23 12 3
#guard foldAgree 23 80 0
#guard foldAgree 64 40 40

/-- `Stmt.tensor`'s body. -/
def tensorRef (lw : Array F128) (rho : Array F128) : Array F128 :=
  let eIn := eqTable rho
  Array.ofFn (n := lw.size * eIn.size) fun k => lw[k.val % lw.size]! * eIn[k.val / lw.size]!

example : @tensorRef = @Stmt.tensor := rfl

/-- The twin and the restatement agree on `n` weights and `m` coins, `eq(ρ)` of `2^m` entries. -/
def tensorAgree (n m : Nat) : Bool :=
  let lw := (Array.range n).map fun i => f (100 + i)
  let rho := (Array.range m).map fun i => f (200 + i)
  let t := Stmt.tensor lw rho
  t == tensorRef lw rho && t.size == n * 2 ^ m

#guard tensorAgree 64 3
#guard tensorAgree 3 9
#guard tensorAgree 3 10
#guard tensorAgree 2 11
#guard tensorAgree 2 12
#guard tensorAgree 1 13
#guard tensorAgree 1 14
#guard tensorAgree 0 10
#guard (Stmt.tensor ((Array.range 2).map fun i => f (100 + i)) ((Array.range 12).map fun i => f (200 + i))).all (!·.isZero)

/-- `Stmt.project`'s body. -/
def projectRef (base eLo : Array F128) : Array F128 :=
  Array.ofFn (n := 64) fun s => Stmt.sumUpTo eLo.size fun u => base[s.val + 64 * u]! * eLo[u]!

example : @projectRef = @Stmt.project := rfl

/-- The twin and the restatement agree on a base of `n` entries and `m` blocks' weights, `64 * m ≤ n`. -/
def projectAgree (n m : Nat) : Bool :=
  let base := (Array.range n).map fun i => f (300 + i)
  let eLo := (Array.range m).map fun i => f (7000 + i)
  let p := Stmt.project base eLo
  p == projectRef base eLo && p.size == 64

#guard projectAgree (64 * 37) 37
#guard projectAgree (64 * 37 + 100) 37
#guard projectAgree 64 1
#guard projectAgree 640 0
#guard projectAgree (64 * 200) 200
#guard (Stmt.project ((Array.range (64 * 37)).map fun i => f (300 + i)) ((Array.range 37).map fun i => f (7000 + i))).all (!·.isZero)

end Flock.Test.FoldFast
