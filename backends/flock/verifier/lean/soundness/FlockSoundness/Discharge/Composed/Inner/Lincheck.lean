import FlockSoundness.Discharge.ZkExec.VerdictInner

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

/-- Pad `i` of `h`, and `0` past the end, where `Zk.Lin.masked` puts no coefficient either. -/
noncomputable def pad {s : ℕ} (h : Fin s → 𝔽) (i : ℕ) : 𝔽 := if hi : i < s then h ⟨i, hi⟩ else 0

/-- **An executable constraint's value at the pads** `h`: `c + Σ_{i<s} a_i·h_i` in GF(2^128) (`ZkSession.Lin.eval` of
the row it is under `LinRel`). -/
noncomputable def linVal {s : ℕ} (h : Fin s → 𝔽) (l : Zk.Lin) : 𝔽 := fOf l.c + ∑ i : Fin s, fOf l.a[i.val]! * h i

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

/-- An executable claim's skip weights and point, read in GF(2^128), with the value `v`. -/
noncomputable def claimOf (m : ℕ) (c : Flock.Claim) (v : 𝔽) : Model.Claim 𝔽 m :=
  ⟨v, fun s => fOf c.skip[s.val]!, fun i => fOf c.x[i.val]!⟩

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

/-! ## Completeness -/

/-- **The lincheck is complete on the masked replay** (live PROTOCOL.md §4, §10), at every coin. For a witness `w` of
`S`, a circuit fold `c` realizing `S`, and pads `h`: if the unmasked `final_a`, `final_b` are the true `a`, `b` claims at
the zerocheck's `(ζ, ρ)`, the unmasked round messages are the honest ones (`lcMsg` on the circuit's columns `lcU` and the
witness's `lcV`) and the unmasked `z_partial` is `lcV` folded at the round coins, then on the circuit's 64 folded
entries `combv` the "lincheck final" constraint vanishes at `h`, and the `ab` claim, valued at `w`'s `linVal`, holds
for `w`. -/
theorem lincheck_complete (S : Statement) (c : CircuitFold) (hc : FoldRealizes c S) (w : Witness S.m)
    (hw : S.Satisfies w) (lay : Zk.PadLayout) (h : Fin lay.len → 𝔽) (ζ α β rPrime : F128) (rho ts : Array F128)
    (hrho : rho.size = S.m - 6) (hts : ts.size = S.kLog - 6) (fA fB : F128) (rounds : Array (F128 × F128))
    (zPartial combv : Array F128)
    (ha : (⟨fOf fA + pad h lay.fa, execArith.lagS (fOf ζ), vecOf (S.m - 6) rho⟩ : Model.Claim 𝔽 S.m).Holds (aVec S w))
    (hb : (⟨fOf fB + pad h lay.fb, execArith.lagS (fOf ζ), vecOf (S.m - 6) rho⟩ : Model.Claim 𝔽 S.m).Holds (bVec S w))
    (hround : ∀ j < S.kLog - 6,
      (fOf rounds[j]!.1 + pad h (lay.lc j 0), fOf rounds[j]!.2 + pad h (lay.lc j 1)) =
        lcMsg (2 ^ S.kLog) (lcU S ζ α β rho) (lcV S rho w) (ts.toList.map fOf) j)
    (hzp : ∀ i < 64,
      fOf zPartial[i]! + pad h (lay.zp i) = foldAll (ts.toList.map fOf) (2 ^ S.kLog) (lcV S rho w) i)
    (hcomb : c.folded α β (skipWeights ζ) (rho.extract 0 (S.kLog - 6)) ts = .ok combv) :
    linVal h (lcFinal lay fA fB α β rounds zPartial ts (S.kLog - 6) combv) = 0 ∧
      (claimOf S.m (lcAbClaim S.m S.kLog rho ts rPrime) (linVal h (lcAbValue lay zPartial rPrime))).Holds w := by
  sorry

/-- `lcFinal` is a form on the layout's pads (`LinRel`'s size). -/
theorem lcFinal_size (lay : Zk.PadLayout) (fA fB alpha beta : F128) (rounds : Array (F128 × F128))
    (zPartial ts : Array F128) (n : ℕ) (combv : Array F128) :
    (lcFinal lay fA fB alpha beta rounds zPartial ts n combv).a.size = lay.len := by
  sorry

/-- `Zk.replay`'s `ab` claim is `lcAbClaim` at its coins: the lincheck's `r_in` is the round coins reversed. -/
theorem abcClaims_ab (m kLog : ℕ) (zc : Flock.ZcOut) (rPrime : F128) (ts : Array F128) :
    (abcClaims m kLog zc { w := 0, rPrime, rIn := ts.reverse })[0]! = lcAbClaim m kLog zc.rho ts rPrime :=
  rfl

/-- `lcAbValue` is a form on the layout's pads. -/
theorem lcAbValue_size (lay : Zk.PadLayout) (zPartial : Array F128) (rPrime : F128) :
    (lcAbValue lay zPartial rPrime).a.size = lay.len :=
  ZkExec.combo_size _ _ _

end FlockSoundness.Discharge.Composed.Lincheck
