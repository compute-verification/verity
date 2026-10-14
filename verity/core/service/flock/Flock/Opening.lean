/-
PROTOCOL.md §12: claim batching and ring switching; the batched basis `b̂` that Ligerito consumes.
-/
import Flock.Piop

namespace Flock

open Transcript

/-- One switched claim: its batching coin, its 7 coins `r''`, and its suffix point `z = x[1..]`. -/
structure Switched where
  gamma : F128
  rpp : Array F128
  z : Array F128
  deriving Inhabited

/-- `M_g`: the images `g·x^u` for `u < 128` (the F2-linear map "multiply by g" as 128 columns). -/
def mulColumns (g : F128) : Array F128 := Id.run do
  let mut out : Array F128 := Array.mkEmpty 128
  let mut v := g
  let x := F128.monomial 1
  for _ in [0:128] do
    out := out.push v
    v := v * x
  return out

/-- `S' = M S` for the map with columns `cols`: `S'_l = Σ_u bit_l(cols[u]) · S_u`. -/
def applyMap (cols : Array F128) (s : Array F256) : Array F256 := Id.run do
  let mut out : Array F256 := Array.replicate 128 0
  for u in [0:128] do
    let su := s[u]!
    if su.c0.isZero && su.c1.isZero then continue
    let cu := cols[u]!
    for l in [0:128] do
      if cu.bit l then out := out.set! l (out[l]! + su)
  return out

/-- The width-128 branching program of §12.4 over the prefix `q ∈ K^{ℓ−e}`, then every Boolean suffix. -/
def basisForClaim (sw : Switched) (q : Array F256) (e : Nat) : Array F256 := Id.run do
  let eqR := eqTable sw.rpp
  let mut s : Array F256 := (Array.replicate 128 (0 : F256)).set! 0 1
  for j in [0:q.size] do
    let zj := sw.z[j]!
    let m := applyMap (mulColumns (1 + zj)) s
    let qj := q[j]!
    s := (Array.range 128).map fun l => m[l]! + qj * s[l]!
  let pre := s
  let mut out : Array F256 := Array.mkEmpty (2 ^ e)
  let colsOne := (Array.range e).map fun i => mulColumns (sw.z[q.size + i]!)
  let colsZero := (Array.range e).map fun i => mulColumns (1 + sw.z[q.size + i]!)
  for y in [0:2 ^ e] do
    let mut t := pre
    for i in [0:e] do
      t := applyMap (if (y >>> i) % 2 == 1 then colsOne[i]! else colsZero[i]!) t
    let v := (Array.range 128).foldl (fun acc u => acc + (t[u]!).smul (eqR[u]!)) (0 : F256)
    out := out.push v
  return out

/-- `b̂(q ‖ bits(y))` for `y < 2^e`: `Σ_k γ_k · B̂_k`. -/
def basis (sws : Array Switched) (q : Array F256) (e : Nat) : Array F256 :=
  sws.foldl (fun acc sw =>
    let b := basisForClaim sw q e
    (Array.range (2 ^ e)).map fun y => acc[y]! + (b[y]!).smul sw.gamma) (Array.replicate (2 ^ e) 0)

/-- §12.2–§12.3: returns `T` and the switched claims. -/
def ringSwitch (m : Nat) (claims : Array Claim) (rs : Array (Array F128)) : TM (F128 × Array Switched) := do
  let n := claims.size
  label "flock-pcs-open-batch-v0"
  let mut betas : Array F128 := #[]
  let mut perClaim : Array (Array F128 × Array F128) := #[]
  for k in [0:n] do
    let c := claims[k]!
    let s := rs[k]!
    label "flock-ring-switch-v0"
    absorbSlice s
    let x0 := c.x[0]!
    let w := (Array.range 128).map fun v => c.skip[v % 64]! * (if v / 64 == 1 then x0 else 1 + x0)
    check (dot w s == c.value) s!"opening: ring-switch claim {k} mismatch"
    let rpp ← sampleVec 7
    -- t[u]: bit v of t[u] is bit u of s[v]
    let t := (Array.range 128).map fun u =>
      (Array.range 128).foldl (fun acc v => if (s[v]!).bit u then acc + F128.monomial v else acc) (0 : F128)
    betas := betas.push (dot (eqTable rpp) t)
    perClaim := perClaim.push (rpp, c.x.extract 1 c.x.size)
  let gamma ← sampleVec n
  let T := dot gamma betas
  let sws := (Array.range n).map fun k => { gamma := gamma[k]!, rpp := (perClaim[k]!).1, z := (perClaim[k]!).2 : Switched }
  return (T, sws)

end Flock
