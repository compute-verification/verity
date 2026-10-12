import Proofs.Flock.C3.Reps.OpensRead

/-!
# n9: R's openings, read from the layers' read wires (`OpensReadAt 2`)

The last round's message is the openings step `os`, and at stratum 2 the source's openings are R's: a Ligerito proof's
rows and siblings (`openRow`), as `LevelTie`'s last conjunct has them. So the last round needs `OpensReadAt 2`, as S's
`opensReadAt_of_layers` gives strata 0 and 1.

* `ROpensOfLayers` (the hypothesis, in S's `OpensOfLayers` form): at every accepted outer proof, R's opened rows and
  siblings at every `(ℓ, q)` are `rd ℓ q` of the layers' bits on the read wires of `W 2`.
* `opensReadAt_two_of_layers`: `OpensReadAt 2` at any source whose R openings are `prR`'s rows, from it;
  `opensReadAt_two_norm` at H's `tiedSrc (normSrc stp)` (`normOs` keeps strata 0–2).
* The arrays' sizes are pinned, not the indices restricted: `openRow` reads past the arrays' ends as defaults, so the
  hypothesis holds at every `(ℓ, q)` where the rows past the schedule's ranges don't depend on the layers (`RPinned`).
  `rOpensOfLayers_of_pinned` gives it from the in-range read fact there.
-/

namespace FlockC3.RegLevels

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.Family (maskOf)

local notation "𝔽" => FlockLevel3.GF128

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Hypothesis: R's opened rows are read off the layers at `W`** (S's `OpensOfLayers` at R): at every accepted outer
proof, R's opened row and siblings at level `ℓ`, query `q` (`openRow` of `prR` at the layers, at R's schedule `sch`)
are `rd ℓ q` of the layers' bits on the read wires of `W 2` (`maskOf`), at every `(ℓ, q)`. -/
def ROpensOfLayers (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (rd : ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig)
    (prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof) (sch : Flock.Schedule) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ ℓ q,
    openRow (prR (layers zs dg N k Rw R ω t R₂ τ)) sch ℓ q = rd ℓ q (maskOf Vs W 2 (layers zs dg N k Rw R ω t R₂ τ))

/-- **`OpensReadAt 2` from R's rows read off the layers**, at a source whose openings at stratum 2 are `prR`'s rows. -/
theorem opensReadAt_two_of_layers
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof} {sch : Flock.Schedule}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hv : ROpensOfLayers zs dg N k Rw W rd prR sch)
    (hsrc : ∀ R ω t Lb ℓ q, (src R ω t Lb).os.1 2 ℓ q = openRow (prR Lb) sch ℓ q) :
    OpensReadAt zs dg N k Rw src 2 := by
  intro R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX
  funext ℓ q
  rw [hsrc, hsrc, hv R ω t R₂ τ o hA ℓ q, hv R' ω' t' R₂' τ' o' hA' ℓ q, FlockC3.Reps.maskOf_congr W 2 hX]

/-- **`OpensReadAt 2` at H's normalized tied source**, where `stp`'s openings at stratum 2 are `prR`'s rows. -/
theorem opensReadAt_two_norm
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof} {sch : Flock.Schedule}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hv : ROpensOfLayers zs dg N k Rw W rd prR sch)
    (hsrc : ∀ R ω t Lb ℓ q, (stp R ω t Lb).os.1 2 ℓ q = openRow (prR Lb) sch ℓ q) :
    OpensReadAt zs dg N k Rw (tiedSrc (normSrc stp)) 2 :=
  opensReadAt_two_of_layers hv fun R ω t Lb ℓ q =>
    (congrFun (congrFun (normOs_os_of_lt (u := stp R ω t Lb) (by decide : 2 < 3)) ℓ) q).trans (hsrc R ω t Lb ℓ q)

/-- **R's rows are pinned past the schedule's ranges**: at a level past the schedule's, or a query past the level's
`q`, `prR`'s opened row and siblings don't depend on the layers (the arrays' sizes are the schedule's). -/
def RPinned (prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof) (sch : Flock.Schedule) : Prop :=
  ∀ Lb Lb' ℓ q, ¬ (ℓ < sch.levels.size ∧ q < (sch.levels.getD ℓ default).q) →
    openRow (prR Lb) sch ℓ q = openRow (prR Lb') sch ℓ q

/-- **`ROpensOfLayers` from the in-range read fact**, where the rows are pinned past the ranges: the read function is
`rd` in range and the pinned row (at any `Lb₀`) past it. -/
theorem rOpensOfLayers_of_pinned
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof} {sch : Flock.Schedule}
    (Lb₀ : (j : Fin S) → (Vs j).Bits) (hpin : RPinned prR sch)
    (hin : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      ∀ ℓ < sch.levels.size, ∀ q < (sch.levels.getD ℓ default).q,
        openRow (prR (layers zs dg N k Rw R ω t R₂ τ)) sch ℓ q =
          rd ℓ q (maskOf Vs W 2 (layers zs dg N k Rw R ω t R₂ τ))) :
    ROpensOfLayers zs dg N k Rw W (fun ℓ q X => if ℓ < sch.levels.size ∧ q < (sch.levels.getD ℓ default).q then
      rd ℓ q X else openRow (prR Lb₀) sch ℓ q) prR sch := by
  intro R ω t R₂ τ o hA ℓ q
  show _ = if ℓ < sch.levels.size ∧ q < (sch.levels.getD ℓ default).q then rd ℓ q _ else openRow (prR Lb₀) sch ℓ q
  split_ifs with h
  · exact hin R ω t R₂ τ o hA ℓ h.1 q h.2
  · exact hpin _ _ ℓ q h

end

end FlockC3.RegLevels
