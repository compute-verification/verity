module

public import Definitions.Flock.Sparse
public import Flock.RecAlgebra
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! `rec_sparse.py` restated value for value over F: the verifier's row `verifierRow`, the products' structure
`sparseS` and the output `out`. -/

@[expose] public section

namespace FlockVBridge.Sparse

open Flock Flock.RecAlgebra FlockSoundness.Model

variable (sh : SparseShape)

def nR : ℕ := sh.nu + sh.sLog

def deg (j : ℕ) : ℕ := if j < sh.nu then 2 * sh.ell + 1 else 2 * sh.ell + 2

def degSum (j : ℕ) : ℕ := j * (2 * sh.ell + 1) + (j - sh.nu)

def tw : ℕ := if sh.tLog = 0 then 0 else 2 ^ sh.tLog

def wPre (j : ℕ) : ℕ := 64 + 2 * j
def wV2 : ℕ := 76
def wSc (j : ℕ) : ℕ := 77 + degSum sh j
def wU : ℕ := 77 + degSum sh (nR sh)
def wVt : ℕ := wU sh + 2 ^ sh.pLog
def wClaim : ℕ := wVt sh + tw sh
def nw : ℕ := wClaim sh + 1

def vPre (j : ℕ) : ℕ := 64 + 2 * j
def vWhat : ℕ := 76
def vSc (j : ℕ) : ℕ := 77 + degSum sh j
def vAlpha : ℕ := 77 + degSum sh (nR sh)
def vAx : ℕ := vAlpha sh + 1
def vAy : ℕ := vAx sh + sh.ell
def vUr : ℕ := vAy sh + sh.ell
def vEqp : ℕ := vUr sh + 1
def vEqs : ℕ := vEqp sh + 2 ^ sh.pLog
def vEqt : ℕ := vEqs sh + 2 ^ sh.sLog
def nv : ℕ := vEqp sh + 2 ^ sh.pLog + (if sh.tLog = 0 then 0 else 2 ^ sh.sLog + 2 ^ sh.tLog)

inductive RName where
  | h1
  | pre (j : ℕ)
  | wv
  | sc (j : ℕ)
  | f0
  | fx (j : ℕ)
  | fy (j : ℕ)
  | bp
  | claim
  deriving DecidableEq

def names : List RName :=
  [RName.h1] ++ (List.range 6).map RName.pre ++ [RName.wv] ++ (List.range (nR sh)).map RName.sc ++ [RName.f0] ++
    (List.range sh.ell).flatMap (fun j => [RName.fx j, RName.fy j]) ++ [RName.bp] ++
    (if sh.tLog = 0 then [] else [RName.claim])

def spec : RName → List (Term × Term) × List Term
  | .h1 => ((List.range 64).map fun s => (Term.v s, Term.w s), [])
  | .pre j => ([(Term.w (wPre j + 1), Term.v (vPre j + 1))], [Term.w (wPre j)])
  | .wv => ([(Term.v vWhat, Term.w wV2)], [])
  | .sc j => ((List.range (deg sh j - 1)).map fun i => (Term.w (wSc sh j + (i + 1)), Term.v (vSc sh j + (i + 1))),
      [Term.w (wSc sh j)])
  | .f0 => ([(Term.v (vAlpha sh), Term.w (wU sh + 2 * sh.ell))], [Term.w (wU sh + (2 * sh.ell + 1))])
  | .fx j => ([], [Term.v (vAx sh + j), Term.w (wU sh + j)])
  | .fy j => ([], [Term.v (vAy sh + j), Term.w (wU sh + (sh.ell + j))])
  | .bp => ((List.range (2 ^ sh.pLog)).map fun p => (Term.v (vEqp sh + p), Term.w (wU sh + p)),
      if sh.tLog = 0 then [Term.w (wClaim sh)] else [])
  | .claim => ((List.range (2 ^ sh.tLog)).map fun t => (Term.v (vEqt sh + t), Term.w (wVt sh + t)),
      [Term.w (wClaim sh)])

def opTerms : List Term := (names sh).flatMap fun nm => (spec sh nm).1.flatMap fun ab => [ab.1, ab.2]

def opList : List Term := (opTerms sh).eraseDups

def opIx (t : Term) : ℕ := (opList sh).idxOf t

def resOf (nm : RName) : Res :=
  ⟨((spec sh nm).1.map fun ab => (opIx sh ab.1, opIx sh ab.2)).toArray, (spec sh nm).2.toArray⟩

def sparseS : GfS := ⟨nw sh, nv sh, ((opList sh).map fun t => #[t]).toArray, ((names sh).map (resOf sh)).toArray⟩

structure SCoins where
  alpha : F128
  z : F128
  xRest : Array F128
  yRest : Array F128
  U : Array F128
  cs : Array F128

def nc : ℕ := 12 + nR sh + sh.pLog + sh.tLog

def powN (r : F128) : ℕ → F128
  | 0 => 1
  | n + 1 => powN r n * r

def xLo (c : SCoins) (j : ℕ) : F128 := if j < 6 then c.cs[6 + j]! else c.xRest[j - 6]!

def yLo (c : SCoins) (j : ℕ) : F128 := if j < 6 then c.cs[j]! else c.yRest[j - 6]!

def roundOf (i : ℕ) : ℕ :=
  if i < sh.nu * (2 * sh.ell + 1) then i / (2 * sh.ell + 1)
  else sh.nu + (i - sh.nu * (2 * sh.ell + 1)) / (2 * sh.ell + 2)

def slice (c : SCoins) (a n : ℕ) : Array F128 := c.cs.extract a (a + n)

def vAt (c : SCoins) (i : ℕ) : F128 :=
  if i < 64 then (eqTable (slice c 0 6))[i]!
  else if i < 76 then
    if (i - 64) % 2 = 0 then c.cs[6 + (i - 64) / 2]!
    else c.cs[6 + (i - 64) / 2]! * c.cs[6 + (i - 64) / 2]! + c.cs[6 + (i - 64) / 2]!
  else if i = 76 then dot (eqTable (slice c 6 6)) (skipWeights c.z)
  else if i < vAlpha sh then
    if i - 77 - degSum sh (roundOf sh (i - 77)) = 0 then c.cs[12 + roundOf sh (i - 77)]!
    else powN c.cs[12 + roundOf sh (i - 77)]! (i - 77 - degSum sh (roundOf sh (i - 77)) + 1) +
      c.cs[12 + roundOf sh (i - 77)]!
  else if i = vAlpha sh then c.alpha
  else if i < vAy sh then xLo c (i - vAx sh) + 1
  else if i < vUr sh then yLo c (i - vAy sh) + 1
  else if i = vUr sh then dot (eqTable (slice c (12 + sh.nu) sh.sLog)) c.U
  else if i < vEqs sh then (eqTable (slice c (12 + nR sh) sh.pLog))[i - vEqp sh]!
  else if i < vEqt sh then (eqTable (slice c (12 + sh.nu) sh.sLog))[i - vEqs sh]!
  else (eqTable (slice c (12 + nR sh + sh.pLog) sh.tLog))[i - vEqt sh]!

def verifierRow (c : SCoins) : Array F128 := Array.ofFn (n := nv sh) fun i => vAt sh c i.val

def rget (rs : Array F128) (nm : RName) : F128 := rs[(names sh).idxOf nm]!

def outOf (rs w v : Array F128) (τ : Array ℕ) : Array F128 × Array ℕ :=
  let R := rget sh rs
  let c₂ := (List.range 6).foldl (fun c j => R (.pre j) + c * v[vPre j]!) (R .h1)
  let c₄ := (List.range (nR sh)).foldl (fun c j => R (.sc j) + c * v[vSc sh j]!) w[wV2]!
  let φ := (List.range sh.ell).foldl (fun x j => x * (R (.fx j) * R (.fy j))) (R .f0)
  let h2 := c₂ + R .wv
  let h3 := c₄ + φ * v[vUr sh]!
  if sh.tLog = 0 then (#[h2, h3, R .bp], #[])
  else
    let wt (t : ℕ) : F128 := (List.range (2 ^ sh.sLog)).foldl
      (fun x s => x + if τ[s]! % 2 ^ sh.tLog = t then v[vEqs sh + s]! else 0) 0
    let h4 := (List.range (2 ^ sh.tLog)).foldl (fun x t => x + wt t * w[wVt sh + t]!) (R .bp)
    (#[h2, h3, h4, R .claim], ((List.range (2 ^ sh.sLog)).map fun s => τ[s]! - τ[s]! % 2 ^ sh.tLog).toArray)

/-- Zero exactly when the rep's sparse phase accepts at a τ inside the bucket (`sparse_refines`). -/
def out (w v : Array F128) (τ : Array ℕ) : Except String (Array F128 × Array ℕ) := do
  let rs ← residuals (sparseS sh) w v
  pure (outOf sh rs w v τ)

end FlockVBridge.Sparse
