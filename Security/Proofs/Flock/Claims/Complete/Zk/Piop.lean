import Proofs.Flock.Claims.Complete.Prover.Piop
import Proofs.Flock.Claims.Complete.Zk.Tape

/-!
# The masked prover's zerocheck and lincheck

Live PROTOCOL.md §1 and §4 from the prover's side: `zerocheckP` and `lincheckP` with every sent value `v` sent as
`v + h_i`, `h_i` its pad at the layout's index (`Zk.PadLayout`), and the pads' root absorbed after the zerocheck's label
(`LABEL_PADS`), as `Zk.replayWith` reads them. The prover computes on the true values; what it absorbs and sends is
masked, so its coins are the verifier's. `final_c` is sent masked and absorbed only by the inner proof. Each returns
the masked proof part and the true `ZcOut`, `LcOut`, from which the ring switch's claims are drawn (their points are
the coins', their values unused).
-/

namespace FlockComplete

open Flock Transcript

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The masked zerocheck prover**: `zerocheckP` with the pads' root bound and each sent value padded. -/
def zerocheckPZk (io : Asks M) (m : Nat) (digest cap root : ByteArray) (lay : Zk.PadLayout) (h : Array F128)
    (tr : Trace) : TMW M (ZerocheckProof × ZcOut) := do
  liftTM (label "flock-r1cs-v0")
  liftTM (absorbBytes digest)
  liftTM (absorbBytes cap)
  liftTM (label "flock-zerocheck-v0")
  liftTM (label Zk.LABEL_PADS)
  liftTM (absorbBytes root)
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
  let r1ab' := (Array.range 64).map fun i => r1ab[i]! + h[lay.r1ab i]!
  let r1c' := (Array.range 64).map fun i => r1c[i]! + h[lay.r1c i]!
  liftTM (absorbSlice r1ab')
  liftTM (absorbSlice r1c')
  let z ← sampleP io
  let pc := dot (lagrange nodesL den6Inv z) r1c
  let wz := skipWeights z
  let mut a := (Array.range nx).map (skipHat tr.az wz)
  let mut b := (Array.range nx).map (skipHat tr.bz wz)
  let mut rounds : Array (F128 × F128) := #[]
  let mut rho : Array F128 := #[]
  for i in [0:m - 6] do
    let half := a.size / 2
    let eqT := eqTable (r.extract (i + 1) (m - 6))
    let g1 := sumF half fun k => eqT[k]! * (a[2 * k + 1]! * b[2 * k + 1]!)
    let ginf := sumF half fun k => eqT[k]! * ((a[2 * k]! + a[2 * k + 1]!) * (b[2 * k]! + b[2 * k + 1]!))
    let g1' := g1 + h[lay.ml i 0]!
    let ginf' := ginf + h[lay.ml i 1]!
    liftTM (absorbF g1')
    liftTM (absorbF ginf')
    let rr ← sampleP io
    rho := rho.push rr
    rounds := rounds.push (g1', ginf')
    a := foldLow rr a
    b := foldLow rr b
  let fa := a[0]!
  let fb := b[0]!
  let fa' := fa + h[lay.fa]!
  let fb' := fb + h[lay.fb]!
  liftTM (absorbF fa')
  liftTM (absorbF fb')
  return (⟨r1ab', r1c', rounds, fa', fb', pc + h[lay.fc]!⟩, { z, rho, rRest := r, vA := fa, vB := fb, vC := pc })

/-- **The masked lincheck prover**: `lincheckP` with each round's message and `z_partial` padded. -/
def lincheckPZk (io : Asks M) (c : CircuitFold) (m : Nat) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace)
    (zc : ZcOut) : TMW M (LincheckProof × LcOut) := do
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
  for j in [0:kLog - 6] do
    let half := comb.size / 2
    let q1 := sumF half fun i => comb[i + half]! * zr[i + half]!
    let qinf := sumF half fun i => (comb[i]! + comb[i + half]!) * (zr[i]! + zr[i + half]!)
    let q1' := q1 + h[lay.lc j 0]!
    let qinf' := qinf + h[lay.lc j 1]!
    liftTM (absorbF q1')
    liftTM (absorbF qinf')
    let t ← sampleP io
    ts := ts.push t
    rounds := rounds.push (q1', qinf')
    comb := halve t comb
    zr := halve t zr
  let zp' := (Array.range 64).map fun i => zr[i]! + h[lay.zp i]!
  liftTM (absorbSlice zp')
  let rPrime ← sampleP io
  let w := dot (skipWeights rPrime) zr
  return (⟨rounds, zp'⟩, { w, rPrime, rIn := ts.reverse })

end FlockComplete
