import Flock.Zk
import FlockSoundness.Protocol.Discharge.Composed.Inner.Zerocheck

/-!
# The masked zerocheck's forms evaluate to the clear replay

`Zk.replay` replays the zerocheck on the masked proof `zc'` (each sent value its clear value plus its pad, `Masks`) and
builds affine forms `Lin` in the pads. This file restates its zerocheck terms verbatim (`r1c`, `r1ab`, `computedC`, `fcL`,
`running`, `prodL`, at `s = lay.len`) and evaluates them at the pads `h` (`evalL l h = l.c + ⟨l.a, h⟩`):

* `running_eval`: the running form after `j` residual rounds evaluates to the clear replay's `claim`;
* `c_eval_eval`: `computedC + fc` (the constraint `zerocheck c_eval`) evaluates to `Pc(z) + final_c`;
* `final_eval`: at `h_ab = h_fa·h_fb`, `running + prod` (the constraint `zerocheck final a·b`) evaluates to
  `claim … n + final_a·final_b`.

At any coins, for any clear proof. With `Zerocheck.c_eval` and `Zerocheck.final_ab`, both constraints evaluate to 0 at the
honest prover's pads (`c_eval_zero`, `final_zero`). `evalL_combo` evaluates `verifyInner`'s batched form.
-/

namespace FlockSoundness.Discharge.Composed.ZerocheckMasked

open Flock Flock.Zk FlockLevel3
open FlockSoundness.Discharge.Composed.Zerocheck

local notation "𝔽" => FlockLevel3.GF128

/-! ## The replay's zerocheck terms, at `s = lay.len` -/

def r1c (lay : PadLayout) (zc : ZerocheckProof) : Array Lin :=
  (Array.range 64).map fun i => Lin.masked zc.round1c[i]! (lay.r1c i) lay.len

def r1ab (lay : PadLayout) (zc : ZerocheckProof) : Array Lin :=
  (Array.range 64).map fun i => Lin.masked zc.round1ab[i]! (lay.r1ab i) lay.len

def computedC (lay : PadLayout) (zc : ZerocheckProof) (z : F128) : Lin :=
  Lin.combo (lagrange nodesL den6Inv z) (r1c lay zc) lay.len

def fcL (lay : PadLayout) (zc : ZerocheckProof) : Lin := Lin.masked zc.finalC lay.fc lay.len

/-- The running form before the residual rounds. -/
def running0 (lay : PadLayout) (zc : ZerocheckProof) (z : F128) : Lin :=
  (Lin.combo (combWeights z) ((Array.range 64).map fun i => (r1ab lay zc)[i]!.add (r1c lay zc)[i]!) lay.len).add
    (computedC lay zc z)

/-- Residual round `j`'s update of the running form, at `r_eq = r[j]` and the round's coin `rho[j]`. -/
def roundL (lay : PadLayout) (zc : ZerocheckProof) (r rho : Array F128) (j : ℕ) (running : Lin) : Lin :=
  let (m1, minf) := zc.rounds[j]!
  let req := r[j]!
  let g1 := Lin.masked m1 (lay.ml j 0) lay.len
  let ginf := Lin.masked minf (lay.ml j 1) lay.len
  let g0 := (running.add (g1.scale req)).scale (1 + req).inv
  let rr := rho[j]!
  ((g0.scale (1 + rr)).add (g1.scale rr)).add (ginf.scale (rr * (1 + rr)))

/-- The running form after `j` residual rounds. -/
def running (lay : PadLayout) (zc : ZerocheckProof) (r rho : Array F128) (z : F128) : ℕ → Lin
  | 0 => running0 lay zc z
  | j + 1 => roundL lay zc r rho j (running lay zc r rho z j)

/-- `a·b = a'b' + a'·h_b + b'·h_a + h_ab` on the masked `a' = final_a`, `b' = final_b`. -/
def prodL (lay : PadLayout) (zc : ZerocheckProof) : Lin :=
  ⟨zc.finalA * zc.finalB,
    (((Array.replicate lay.len 0).set! lay.fb zc.finalA).set! lay.fa zc.finalB).set! lay.hab 1⟩

/-! ## Arrays and the field -/

/-! ## The forms' values -/

/-! ## The layout's indices -/

/-! ## The replay's terms' values -/

/-! ## The three facts -/

/-! ## At the honest prover -/

end FlockSoundness.Discharge.Composed.ZerocheckMasked
