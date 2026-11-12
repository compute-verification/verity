module

public import Flock.Statement
meta import Flock.Statement

/-!
# The lincheck fold's compiled loops against the functions they replace

Compiled code runs `Sparse.foldTFast` for `Sparse.foldT` and `Stmt.projectFast` for `Stmt.project` (their `@[csimp]`
equations), so a call of either function here runs its twin. Each function's body is restated here under another name,
which no equation rewrites, with `rfl` that the restatement is the function; the twin's result is compared with the
restatement's on matrices and bases that reach each of its cases: a row twice in a column, an empty row, a zero
coefficient, more and fewer coefficients than rows, a base that ends inside a block, past its last block, and none.
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
    m.foldT e == foldTRef m e && (m.foldT e).size == width && (m.foldT e).any (!·.isZero)
  | .error _ => false

#guard rows.any (·.isEmpty) && rows.any fun r => r.size ≥ 5 && r[0]! == r[4]!
#guard foldAgree 23 40 5
#guard foldAgree 23 12 3
#guard foldAgree 23 80 0
#guard foldAgree 64 40 40

/-- `Stmt.project`'s body. -/
def projectRef (base eLo : Array F128) : Array F128 :=
  Array.ofFn (n := 64) fun s => Stmt.sumUpTo eLo.size fun u => base[s.val + 64 * u]! * eLo[u]!

example : @projectRef = @Stmt.project := rfl

/-- The twin and the restatement agree on a base of `n` entries and `m` blocks' weights. -/
def projectAgree (n m : Nat) : Bool :=
  let base := (Array.range n).map fun i => f (300 + i)
  let eLo := (Array.range m).map fun i => f (7000 + i)
  Stmt.project base eLo == projectRef base eLo && (Stmt.project base eLo).size == 64

#guard projectAgree (64 * 37) 37
#guard projectAgree (64 * 37 - 5) 37
#guard projectAgree (64 * 37 + 100) 37
#guard projectAgree 64 1
#guard projectAgree 0 5
#guard projectAgree 640 0
#guard projectAgree (64 * 200) 200
#guard (Stmt.project ((Array.range (64 * 37)).map fun i => f (300 + i)) ((Array.range 37).map fun i => f (7000 + i))).all (!·.isZero)

end Flock.Test.FoldFast
