import FlockSoundness.Protocol.Discharge.ZkExec.VerdictInner

/-!
# The lincheck is complete on the masked replay (live PROTOCOL.md §4, §10)

`Zk.replay`'s lincheck reads the masked `final_a`, `final_b`, the `k_log − 6` round messages and `z_partial`, each a sent
value plus its pad, as affine forms in the pads (`Zk.Lin`); it gives the "lincheck final" constraint, one of the ten the
inner proof batches, and the `ab` claim's value `w` the ring switch reads. `lcFinal` and `lcAbValue` are that code with
its coins given (`α`, `β`, the round coins `ts`, `r′`, and the zerocheck's `ζ`, `ρ`), and `linVal` reads a form at the
pads `h`.

`lincheck_complete`: for a witness `w` of the statement, when the unmasked `final_a`, `final_b` are the zerocheck's true
`a`, `b` claims at `(ζ, ρ)` and the honest prover sends `lcMsg`'s round messages and the folded columns as `z_partial`,
each under its pad, the constraint vanishes at `h` and the `ab` claim holds for `w`, at every coin: the lincheck neither
divides nor refuses.
-/

namespace FlockSoundness.Discharge.Composed.Lincheck

open Flock FlockLevel3 Model Refine

local notation "𝔽" => FlockLevel3.GF128

/-! ## Pads and affine forms -/

/-! ## `Zk.replay`'s lincheck, its coins given -/

/-- Round `j` of `Zk.replay`'s lincheck loop at the coin `ts[j]`, on the masked messages `rounds[j]`. -/
def lcStep (lay : Zk.PadLayout) (rounds : Array (F128 × F128)) (ts : Array F128) (running : Zk.Lin) (j : ℕ) :
    Zk.Lin :=
  let s := lay.len
  let rr := ts[j]!
  let e1 := Zk.Lin.masked rounds[j]!.1 (lay.lc j 0) s
  let einf := Zk.Lin.masked rounds[j]!.2 (lay.lc j 1) s
  let e0 := running.add e1
  let c1 := (e0.add e1).add einf
  ((einf.scale (rr * rr)).add (c1.scale rr)).add e0

/-- `Zk.replay`'s lincheck claim after `n` rounds, from `α·a + b + β` on the masked `final_a`, `final_b`. -/
def lcRunning (lay : Zk.PadLayout) (fA fB alpha beta : F128) (rounds : Array (F128 × F128)) (ts : Array F128)
    (n : ℕ) : Zk.Lin :=
  let s := lay.len
  let a := Zk.Lin.masked fA lay.fa s
  let b := Zk.Lin.masked fB lay.fb s
  (List.range n).foldl (lcStep lay rounds ts) (((a.scale alpha).add b).addConst beta)

/-- `Zk.replay`'s masked `z_partial`. -/
def lcZp (lay : Zk.PadLayout) (zPartial : Array F128) : Array Zk.Lin :=
  (Array.range 64).map fun i => Zk.Lin.masked zPartial[i]! (lay.zp i) lay.len

/-- **`Zk.replay`'s "lincheck final" constraint** after `n` rounds, on the circuit's 64 folded entries `combv`. -/
def lcFinal (lay : Zk.PadLayout) (fA fB alpha beta : F128) (rounds : Array (F128 × F128)) (zPartial ts : Array F128)
    (n : ℕ) (combv : Array F128) : Zk.Lin :=
  (lcRunning lay fA fB alpha beta rounds ts n).add (Zk.Lin.combo combv (lcZp lay zPartial) lay.len)

/-- **The `ab` claim's value** `w` that `Zk.replay` hands the ring switch (`Replayed.abValue`). -/
def lcAbValue (lay : Zk.PadLayout) (zPartial : Array F128) (rPrime : F128) : Zk.Lin :=
  Zk.Lin.combo (skipWeights rPrime) (lcZp lay zPartial) lay.len

/-- The `ab` claim `Zk.replay` hands the opening (`abcClaims`' first, its value left at the masked replay's `0`):
skip weights `L_s(r′)`, point the round coins reversed, then `ρ[k_log − 6 …]`. -/
def lcAbClaim (m kLog : ℕ) (rho ts : Array F128) (rPrime : F128) : Flock.Claim :=
  { value := 0, skip := skipWeights rPrime, x := ts.reverse ++ rho.extract (kLog - 6) (m - 6) }

/-! ## The honest prover -/

/-- The circuit's column vector at the coins (§10, `Model.comb`): `Σ_i L_{i mod 64}(ζ)·eq(ρ_in, ⌊i/64⌋)·(α·A₀[i,c] +
B₀[i,c])`, plus `β` at the pin, with `ρ_in = ρ[.. k_log − 6]`. -/
noncomputable def lcU (S : Statement) (ζ α β : F128) (rho : Array F128) : ℕ → 𝔽 :=
  comb execArith S (fOf ζ) (fOf α) (fOf β) (fun j => fOf rho[j.val]!)

/-- The witness's block-folded columns at `ρ_out = ρ[k_log − 6 ..]` (`hcol`). -/
noncomputable def lcV (S : Statement) (rho : Array F128) (w : Witness S.m) : ℕ → 𝔽 :=
  hcol S (fun j => fOf rho[S.kLog - 6 + j.val]!) w

/-- One round's honest message `(e(1), e(∞))` for `Σ_{i<2H} u_i·v_i`, where
`e(X) = Σ_{i<H} (u_i + X·(u_{i+H} + u_i))·(v_i + X·(v_{i+H} + v_i))`. -/
def roundMsg {F : Type} [Field F] (H : ℕ) (u v : ℕ → F) : F × F :=
  (∑ i ∈ Finset.range H, u (i + H) * v (i + H),
    ∑ i ∈ Finset.range H, (u (i + H) + u i) * (v (i + H) + v i))

/-- **The honest prover's round-`j` message**: `roundMsg` on `u`, `v` of length `len` folded at the first `j` coins. -/
noncomputable def lcMsg (len : ℕ) (u v : ℕ → 𝔽) (ts : List 𝔽) (j : ℕ) : 𝔽 × 𝔽 :=
  roundMsg (len / 2 ^ (j + 1)) (foldAll (ts.take j) len u) (foldAll (ts.take j) len v)

/-! ## The forms' values at the pads -/

/-! ## The rounds -/

variable {F : Type} [Field F]

/-! ## Completeness -/

end FlockSoundness.Discharge.Composed.Lincheck
