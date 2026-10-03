import Flock.Zk
import FlockSoundness.Discharge.Composed.Inner.Zerocheck

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

/-- The value of the affine form `l` at the pads `h`: `c + Σ a_i h_i`. -/
def evalL (l : Lin) (h : Array F128) : F128 := l.c + dot l.a h

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

/-- `zc'` is the clear `zc` masked by the pads `h` at `lay`'s indices, on its first `n` residual rounds. -/
structure Masks (lay : PadLayout) (h : Array F128) (zc zc' : ZerocheckProof) (n : ℕ) : Prop where
  round1ab : ∀ i < 64, zc'.round1ab[i]! = zc.round1ab[i]! + h[lay.r1ab i]!
  round1c : ∀ i < 64, zc'.round1c[i]! = zc.round1c[i]! + h[lay.r1c i]!
  rounds : ∀ j < n, zc'.rounds[j]! = (zc.rounds[j]!.1 + h[lay.ml j 0]!, zc.rounds[j]!.2 + h[lay.ml j 1]!)
  finalA : zc'.finalA = zc.finalA + h[lay.fa]!
  finalB : zc'.finalB = zc.finalB + h[lay.fb]!
  finalC : zc'.finalC = zc.finalC + h[lay.fc]!

/-! ## The three facts -/

/-- **The running form evaluates to the clear claim**, after `j ≤ n ≤ nMlv` residual rounds. -/
theorem running_eval (lay : PadLayout) (h : Array F128) (hs : h.size = lay.len) (zc zc' : ZerocheckProof) (n : ℕ)
    (hn : n ≤ lay.nMlv) (hm : Masks lay h zc zc' n) (r rho : Array F128) (z : F128) (j : ℕ) (hj : j ≤ n) :
    evalL (running lay zc' r rho z j) h = claim zc r rho z j := by
  sorry

/-- **`zerocheck c_eval` evaluates to its two clear sides**: `Pc(z) + final_c`. -/
theorem c_eval_eval (lay : PadLayout) (h : Array F128) (hs : h.size = lay.len) (zc zc' : ZerocheckProof) (n : ℕ)
    (hm : Masks lay h zc zc' n) (z : F128) :
    evalL ((computedC lay zc' z).add (fcL lay zc')) h = dot (lagrange nodesL den6Inv z) zc.round1c + zc.finalC := by
  sorry

/-- **`zerocheck final a·b` evaluates to its two clear sides** at `h_ab = h_fa·h_fb`: `claim … n + final_a·final_b`. -/
theorem final_eval (lay : PadLayout) (h : Array F128) (hs : h.size = lay.len) (zc zc' : ZerocheckProof) (n : ℕ)
    (hn : n ≤ lay.nMlv) (hm : Masks lay h zc zc' n) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!)
    (r rho : Array F128) (z : F128) :
    evalL ((running lay zc' r rho z n).add (prodL lay zc')) h = claim zc r rho z n + zc.finalA * zc.finalB := by
  sorry

/-! ## At the honest prover -/

/-- `zerocheck c_eval` vanishes at the honest prover's pads, at every coin. -/
theorem c_eval_zero (lay : PadLayout) (h : Array F128) (hs : h.size = lay.len) (n : ℕ) (a b c : ℕ → 𝔽)
    (r rho : Array F128) (z : F128) (zc' : ZerocheckProof) (hm : Masks lay h (honest n a b c r rho z) zc' n) :
    evalL ((computedC lay zc' z).add (fcL lay zc')) h = 0 := by
  sorry

/-- `zerocheck final a·b` vanishes at the honest prover's pads with `h_ab = h_fa·h_fb`, for valid tables, at every coin with
`r_k ≠ 1` for `k < n`. -/
theorem final_zero (lay : PadLayout) (h : Array F128) (hs : h.size = lay.len) (n : ℕ) (hn : n ≤ lay.nMlv)
    (a b c : ℕ → 𝔽) (hvalid : ∀ i < 2 ^ (n + 6), a i * b i = c i) (r rho : Array F128) (hr : ∀ k < n, r[k]! ≠ 1)
    (z : F128) (zc' : ZerocheckProof) (hm : Masks lay h (honest n a b c r rho z) zc' n)
    (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!) :
    evalL ((running lay zc' r rho z n).add (prodL lay zc')) h = 0 := by
  sorry

/-- **`Lin.combo` evaluates to the weighted sum of its forms' values**, all on `s` pads. -/
theorem evalL_combo (ws : Array F128) (ls : Array Lin) (s : ℕ) (h : Array F128) (hs : h.size = s)
    (hl : ∀ l ∈ ls, l.a.size = s) :
    evalL (Lin.combo ws ls s) h = dot ws (ls.map fun l => evalL l h) := by
  sorry

/-- A form's value as a sum over its `s` pads in the field (`InnerHolds`'s shape). -/
theorem of_evalL (l : Lin) (h : Array F128) (s : ℕ) (hl : l.a.size = s) :
    GF128.of (evalL l h) = GF128.of l.c + ∑ k : Fin s, GF128.of l.a[k.val]! * GF128.of h[k.val]! := by
  rw [evalL, of_add, of_dot _ _ s hl]

end FlockSoundness.Discharge.Composed.ZerocheckMasked
