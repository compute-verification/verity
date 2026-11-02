import Proofs.Flock.Claims.Complete.Prover.Transcript

/-!
# The honest prover's PIOP: zerocheck, lincheck and ring switch

`Flock.Piop`, `Flock.Opening` from the prover's side, on a table's trace (`Trace`): the bits `z` of the `2^m` addresses
(`a = s + 64·x`, `s` the skip index), and the products `Az`, `Bz` of the block R1CS. Each step absorbs and squeezes
exactly as the verifier's (`bindAndZerocheckWith`, `lincheckWith`, `ringSwitchWith`) and returns the proof's part and
what the verifier derives from it (`ZcOut`, `LcOut`, the switched claims).

* **Zerocheck** (`Flock.Piop`). With `Â(Y, x) = Σ_s L_s(Y)·Az[s + 64x]` over the skip nodes `S`, `B̂` and `Ĉ` (from `z`) alike,
  and `r` the pinned coordinates then the outer coins: `round1_ab[i] = Σ_x eq(r, x)·Â(λ_i, x)·B̂(λ_i, x)` and
  `round1_c[i] = Σ_x eq(r, x)·Ĉ(λ_i, x)` at the coset nodes `λ_i = φ8(64 + i)`. At `ζ`, the residual sumcheck runs on
  the tables `a(x) = Â(ζ, x)`, `b(x) = B̂(ζ, x)`, binding the low variable first: round `i` sends `g(1)` and `g`'s
  leading coefficient for `g(X) = Σ_{x'} eq(r_{>i}, x')·a(X, x')·b(X, x')`. `final_c` is the verifier's `P_c(ζ)`.
* **Lincheck** (§10). The columns' fold `α·A₀ᵀe + B₀ᵀe` (the statement's `CircuitFold.fold`) with `β` at the pin, against
  `Zρ[c] = Σ_o eq(ρ_out, o)·z[c + 2^kLog·o]`; each round sends `q(1)` and `q`'s leading coefficient, then halves both
  (`halve`, the top bit first); `z_partial` is `Zρ` once halved to 64 entries.
* **Ring switch** (§12). Claim `k`'s `s_k[v] = Σ_y eq(x_k[1..], y)·z[128y + v]`.
-/

namespace FlockComplete

open Flock Transcript

/-- **A table's trace**: the bits of its `2^m` addresses and of the block R1CS's products `A·z` and `B·z`. -/
structure Trace where
  z : Array Bool
  az : Array Bool
  bz : Array Bool
  deriving Inhabited

/-- `Σ_{i < n} f(i)`. -/
def sumF (n : Nat) (f : Nat → F128) : F128 := (List.range n).foldl (fun acc i => acc + f i) 0

/-- A bit table's skip extension at weights `w` on block-row `x`: `Σ_s w[s]·bits[s + 64x]`. -/
def skipHat (bits : Array Bool) (w : Array F128) (x : Nat) : F128 :=
  sumF 64 fun s => if bits[s + 64 * x]! then w[s]! else 0

/-- One sumcheck fold of a table at `r`, the low variable: `a'[k] = a[2k] + r·(a[2k+1] + a[2k])`. -/
def foldLow (r : F128) (a : Array F128) : Array F128 :=
  (Array.range (a.size / 2)).map fun k => a[2 * k]! + r * (a[2 * k + 1]! + a[2 * k]!)

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The zerocheck prover** (`Flock.Piop`): the proof's zerocheck and the verifier's `ZcOut`. -/
def zerocheckP (io : Asks M) (m : Nat) (digest cap : ByteArray) (tr : Trace) : TMW M (ZerocheckProof × ZcOut) := do
  liftTM (label "flock-r1cs-v0")
  liftTM (absorbBytes digest)
  liftTM (absorbBytes cap)
  liftTM (label "flock-zerocheck-v0")
  let _rSkip ← sampleVecP io 6
  let rOuter ← sampleVecP io (m - 13)
  let r := pinned ++ rOuter
  let nx := 2 ^ (m - 6)
  let eqR := eqTable r
  let r1ab := (Array.range 64).map fun i =>
    let w := skipWeights (phi8 (64 + i))
    sumF nx fun x => eqR[x]! * (skipHat tr.az w x * skipHat tr.bz w x)
  let r1c := (Array.range 64).map fun i =>
    let w := skipWeights (phi8 (64 + i))
    sumF nx fun x => eqR[x]! * skipHat tr.z w x
  liftTM (absorbSlice r1ab)
  liftTM (absorbSlice r1c)
  let z ← sampleP io
  let pc := dot (lagrange nodesL den6Inv z) r1c
  let wz := skipWeights z
  let mut a := (Array.range nx).map (skipHat tr.az wz)
  let mut b := (Array.range nx).map (skipHat tr.bz wz)
  let mut rounds : Array (F128 × F128) := #[]
  let mut rho : Array F128 := #[]
  for i in [0:m - 6] do
    let h := a.size / 2
    let eqT := eqTable (r.extract (i + 1) (m - 6))
    let g1 := sumF h fun k => eqT[k]! * (a[2 * k + 1]! * b[2 * k + 1]!)
    let ginf := sumF h fun k => eqT[k]! * ((a[2 * k]! + a[2 * k + 1]!) * (b[2 * k]! + b[2 * k + 1]!))
    liftTM (absorbF g1)
    liftTM (absorbF ginf)
    let rr ← sampleP io
    rho := rho.push rr
    rounds := rounds.push (g1, ginf)
    a := foldLow rr a
    b := foldLow rr b
  let fa := a[0]!
  let fb := b[0]!
  liftTM (absorbF fa)
  liftTM (absorbF fb)
  return (⟨r1ab, r1c, rounds, fa, fb, pc⟩, { z, rho, rRest := r, vA := fa, vB := fb, vC := pc })

/-- **The lincheck prover** (§10): the proof's lincheck and the verifier's `LcOut`. -/
def lincheckP (io : Asks M) (c : CircuitFold) (m : Nat) (tr : Trace) (zc : ZcOut) : TMW M (LincheckProof × LcOut) := do
  let kLog := c.kLog
  let rhoIn := zc.rho.extract 0 (kLog - 6)
  let eOut := eqTable (zc.rho.extract (kLog - 6) (m - 6))
  liftTM (label "flock-lincheck-v0")
  let alpha ← sampleP io
  let beta ← sampleP io
  let full ← match c.fold alpha (skipWeights zc.z) rhoIn with
    | .ok v => pure v
    | .error err => throw err
  let nb := 2 ^ kLog
  let mut comb := full.modify c.pin (· + beta)
  let mut zr := (Array.range nb).map fun col => sumF eOut.size fun o => if tr.z[col + nb * o]! then eOut[o]! else 0
  let mut rounds : Array (F128 × F128) := #[]
  let mut ts : Array F128 := #[]
  for _ in [0:kLog - 6] do
    let h := comb.size / 2
    let q1 := sumF h fun i => comb[i + h]! * zr[i + h]!
    let qinf := sumF h fun i => (comb[i]! + comb[i + h]!) * (zr[i]! + zr[i + h]!)
    liftTM (absorbF q1)
    liftTM (absorbF qinf)
    let t ← sampleP io
    ts := ts.push t
    rounds := rounds.push (q1, qinf)
    comb := halve t comb
    zr := halve t zr
  liftTM (absorbSlice zr)
  let rPrime ← sampleP io
  let w := dot (skipWeights rPrime) zr
  return (⟨rounds, zr⟩, { w, rPrime, rIn := ts.reverse })

/-- Claim `c`'s ring-switch vector: `s[v] = Σ_y eq(x[1..], y)·z[128y + v]`. -/
def switchVec (tr : Trace) (c : Claim) : Array F128 :=
  let eY := eqTable (c.x.extract 1 c.x.size)
  (Array.range 128).map fun v => sumF eY.size fun y => if tr.z[128 * y + v]! then eY[y]! else 0

/-- **The ring-switch prover** (`Flock.Opening`): each claim's `s_k`, and what the verifier derives: `T` and the switched
claims. -/
def ringSwitchP (io : Asks M) (tr : Trace) (claims : Array Claim) :
    TMW M (Array (Array F128) × F128 × Array Switched) := do
  let n := claims.size
  liftTM (label "flock-pcs-open-batch-v0")
  let mut ss : Array (Array F128) := #[]
  let mut betas : Array F128 := #[]
  let mut perClaim : Array (Array F128 × Array F128) := #[]
  for k in [0:n] do
    let c := claims[k]!
    let s := switchVec tr c
    liftTM (label "flock-ring-switch-v0")
    liftTM (absorbSlice s)
    let rpp ← sampleVecP io 7
    let t := (Array.range 128).map fun u =>
      (Array.range 128).foldl (fun acc v => if (s[v]!).bit u then acc + F128.monomial v else acc) (0 : F128)
    betas := betas.push (dot (eqTable rpp) t)
    perClaim := perClaim.push (rpp, c.x.extract 1 c.x.size)
    ss := ss.push s
  let gamma ← sampleVecP io n
  let T := dot gamma betas
  let sws := (Array.range n).map fun k => { gamma := gamma[k]!, rpp := (perClaim[k]!).1, z := (perClaim[k]!).2 : Switched }
  return (ss, T, sws)

/-- **The batched basis on the Boolean cube** (`Flock.Opening`): `b(y) = Σ_k γ_k·Φ_k(eq(z_k, y))` for `y < 2^ℓ`, with
`Φ_k(x^u) = eq(r''_k, u)`. -/
def basisVals (sws : Array Switched) (l : Nat) : Array F128 :=
  let tabs := sws.map fun sw => (sw.gamma, eqTable sw.z, eqTable sw.rpp)
  (Array.range (2 ^ l)).map fun y =>
    tabs.foldl (fun acc (g, ez, er) =>
      let v := ez[y]!
      acc + g * sumF 128 fun u => if v.bit u then er[u]! else 0) 0

/-- **The packed table** (`Flock.Opening`): `t[y] = Σ_v z[128y + v]·x^v`, for `y < 2^(m − 7)`. -/
def packed (tr : Trace) (m : Nat) : Array F128 :=
  (Array.range (2 ^ (m - 7))).map fun y =>
    (List.range 128).foldl (fun acc v => if tr.z[128 * y + v]! then acc + F128.monomial v else acc) 0

end FlockComplete
