import Definitions.Flock.Sparse
import Flock.RecAlgebra

/-!
# `RecSparse_v1` over F

`rec_sparse.py` restated value for value over F: the verifier row `verifierRow`
(`verifier_row`: every value the coins and the public layout decide), the `GfResiduals_v1` structure of its products
`sparseS` (`products_structure`, with `Flock.RecAlgebra`'s semantics), and its output `out` (`_steps`): H2's and H3's
claim chains on the residuals, Φ's product, with `T > 1` the plane batch against the type-selected sum and the
registered claim's residual, then the slots' types past their low `T_LOG` bits. The circuit takes Φ's product as a tree
of `Gf128Mul`, the selects as a demultiplexer of AND gates and its sums as trees of XOR; over F those are the product,
the selected entries and the sums `out` takes, which is what a forms bridge for its unit proves.

w is the message ports' elements in order (`message_ports`): `h` (64), `pre0 … pre5` (`c₀, c₂`), `v2`, `sc<j>`
(`c₀, c₂ … c_deg`), `u` (`P'`), with `T > 1` `vt` (`T`), and `claim`. `SCoins` is `rec_sparse.Coins`: the inner rep's
`α`, `z`, `x_rest`, `y_rest`, the layout factors `U`, and the phase's own coins `cs` in the order drawn: `ζ_c` (6),
`ζ_r` (6), `r_k` (`ν`), `r_s` (`S_LOG`), `ζ_p` (`P_LOG`), `ρ_t` (`T_LOG`).
-/

namespace FlockVBridge.Sparse

open Flock Flock.RecAlgebra FlockSoundness.Model

variable (sh : SparseShape)

/-! ## Statics and offsets -/

/-- H3's rounds. -/
def nR : ℕ := sh.nu + sh.sLog

/-- Round `j`'s degree (`degrees`): `d = 2ℓ + 1` over the entries, `d + 1` over the slots. -/
def deg (j : ℕ) : ℕ := if j < sh.nu then 2 * sh.ell + 1 else 2 * sh.ell + 2

/-- The degrees of the rounds before `j`, `Σ_{i<j} deg i`. -/
def degSum (j : ℕ) : ℕ := j * (2 * sh.ell + 1) + (j - sh.nu)

/-- The stacked types' values' elements: `T` with `T > 1`, else none. -/
def tw : ℕ := if sh.tLog = 0 then 0 else 2 ^ sh.tLog

def wPre (j : ℕ) : ℕ := 64 + 2 * j
def wV2 : ℕ := 76
def wSc (j : ℕ) : ℕ := 77 + degSum sh j
def wU : ℕ := 77 + degSum sh (nR sh)
def wVt : ℕ := wU sh + 2 ^ sh.pLog
def wClaim : ℕ := wVt sh + tw sh
/-- w's elements. -/
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
/-- v's elements (`v_layout`): `eqc`, `pre`, `what`, `sc`, `alpha`, `ax`, `ay`, `ur`, `eqp`, with `T > 1` `eqs`, `eqt`. -/
def nv : ℕ := vEqp sh + 2 ^ sh.pLog + (if sh.tLog = 0 then 0 else 2 ^ sh.sLog + 2 ^ sh.tLog)

/-! ## The products' structure -/

/-- `products_structure`'s residuals by name. -/
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

/-- The residuals in order. -/
def names : List RName :=
  [RName.h1] ++ (List.range 6).map RName.pre ++ [RName.wv] ++ (List.range (nR sh)).map RName.sc ++ [RName.f0] ++
    (List.range sh.ell).flatMap (fun j => [RName.fx j, RName.fy j]) ++ [RName.bp] ++
    (if sh.tLog = 0 then [] else [RName.claim])

/-- A residual's products (pairs of terms) and added terms: `h1` `Σ_s eq(ζ_c, s)·h_s`, `pre<j>` `c₀ + c₂·(r + r²)`,
`wv` `ŵ·v₂`, `sc<j>` `c₀ + Σ_{i≥2} c_i·(r + r^i)`, `f0` `α·u_{2ℓ} + u_{2ℓ+1}`, `fx<j>`, `fy<j>` Φ's factors, `bp`
`Σ_p eq(ζ_p, p)·u_p` (plus `claim` with `T = 1`), and with `T > 1` `claim` `claim + Σ_t eq(ρ_t, t)·vt_t`. -/
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

/-- The products' terms in the order `products_structure` meets them. -/
def opTerms : List Term := (names sh).flatMap fun nm => (spec sh nm).1.flatMap fun ab => [ab.1, ab.2]

/-- The operands: each term once, at its first use (`opix`). -/
def opList : List Term := (opTerms sh).eraseDups

/-- A term's operand. -/
def opIx (t : Term) : ℕ := (opList sh).idxOf t

def resOf (nm : RName) : Res :=
  ⟨((spec sh nm).1.map fun ab => (opIx sh ab.1, opIx sh ab.2)).toArray, (spec sh nm).2.toArray⟩

/-- **`RecSparse`'s `GfResiduals_v1{S}`** (`products_structure`). -/
def sparseS : GfS := ⟨nw sh, nv sh, ((opList sh).map fun t => #[t]).toArray, ((names sh).map (resOf sh)).toArray⟩

/-! ## The verifier's row -/

/-- `rec_sparse.Coins`, with the phase's own coins `cs` in the order drawn. -/
structure SCoins where
  alpha : F128
  z : F128
  xRest : Array F128
  yRest : Array F128
  U : Array F128
  cs : Array F128

/-- The phase's coins: `ζ_c`, `ζ_r`, then H3's at 12, `ζ_p` at `12 + nR`, `ρ_t` after it. -/
def nc : ℕ := 12 + nR sh + sh.pLog + sh.tLog

/-- `r^n`, as `verifier_row`'s `pows` multiplies. -/
def powN (r : F128) : ℕ → F128
  | 0 => 1
  | n + 1 => powN r n * r

/-- The row point's coordinate `j < ℓ` (`x_lo`): `ζ_r`, then `x_rest`. -/
def xLo (c : SCoins) (j : ℕ) : F128 := if j < 6 then c.cs[6 + j]! else c.xRest[j - 6]!

/-- The column point's coordinate `j < ℓ` (`y_lo`): `ζ_c`, then `y_rest`. -/
def yLo (c : SCoins) (j : ℕ) : F128 := if j < 6 then c.cs[j]! else c.yRest[j - 6]!

/-- The H3 round whose coefficients hold `sc`'s slot `i`. -/
def roundOf (i : ℕ) : ℕ :=
  if i < sh.nu * (2 * sh.ell + 1) then i / (2 * sh.ell + 1)
  else sh.nu + (i - sh.nu * (2 * sh.ell + 1)) / (2 * sh.ell + 2)

/-- `cs`'s coins `a … a + n − 1`. -/
def slice (c : SCoins) (a n : ℕ) : Array F128 := c.cs.extract a (a + n)

/-- v's element `i` (`verifier_row`, by `v_layout`'s parts). -/
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

/-- **The verifier's row** (`verifier_row`). -/
def verifierRow (c : SCoins) : Array F128 := Array.ofFn (n := nv sh) fun i => vAt sh c i.val

/-! ## The output -/

/-- `GfResiduals`' residual `nm` among `rs`. -/
def rget (rs : Array F128) (nm : RName) : F128 := rs[(names sh).idxOf nm]!

/-- `_steps` over F on `GfResiduals`' residuals `rs`, w, v and the slots' types `τ`: the output's residuals (`h2`,
`h3`, with `T > 1` `h4` and `claim`, else `bp`) and its tail (with `T > 1`, each type with its low `T_LOG` bits
cleared; a type is a u16 word). -/
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

/-- **`RecSparse_v1`'s output over F**: zero exactly when the rep's sparse phase accepts at a τ inside the bucket
(`Refine.lean`). -/
def out (w v : Array F128) (τ : Array ℕ) : Except String (Array F128 × Array ℕ) := do
  let rs ← residuals (sparseS sh) w v
  pure (outOf sh rs w v τ)

end FlockVBridge.Sparse
