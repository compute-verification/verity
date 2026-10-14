import Proofs.Flock.C3.Reps.OpensRead

/-!
# n9: R's openings, read from the layers' read wires (`OpensReadAt 2`)

The last round's message is the openings step `os`, and at stratum 2 the source's openings are R's: a Ligerito proof's
rows and siblings (`openRow`), as `LevelTie`'s last conjunct has them. So the last round needs `OpensReadAt 2`, as S's
`opensReadAt_of_layers` gives strata 0 and 1.

* `ROpensOfLayers` (the hypothesis, in S's `OpensOfLayers` form, at in-range indices): at every accepted outer proof,
  R's opened rows and siblings at every `ℓ < levels.size` and `q < levels[ℓ].q` are `rd ℓ q` of the layers' bits on
  the read wires of `W 2`.
* `RRowsPinned` (the arrays' sizes, as a hypothesis): `prR`'s rows agreeing in range agree at every `(ℓ, q)`. A row
  past the ends reads `openRow`'s defaults, and a level past the schedule's reads the last level's (`openAt`), so the
  rows past the ranges are pinned by the in-range ones, not constant.
* `opensReadAt_two_of_layers`: `OpensReadAt 2` at any source whose R openings are `prR`'s rows, from both;
  `opensReadAt_two_norm` at H's `tiedSrc (normSrc stp)` (`normOs` keeps strata 0–2).
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

/-- **Hypothesis: R's opened rows are read off the layers at `W`** (S's `OpensOfLayers` at R, in range): at every
accepted outer proof, R's opened row and siblings at level `ℓ < levels.size`, query `q < levels[ℓ].q` (`openRow` of
`prR` at the layers, at R's schedule `sch`) are `rd ℓ q` of the layers' bits on the read wires of `W 2` (`maskOf`). -/
def ROpensOfLayers (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (rd : ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig)
    (prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof) (sch : Flock.Schedule) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    ∀ ℓ < sch.levels.size, ∀ q < (sch.levels.getD ℓ default).q,
      openRow (prR (layers zs dg N k Rw R ω t R₂ τ)) sch ℓ q = rd ℓ q (maskOf Vs W 2 (layers zs dg N k Rw R ω t R₂ τ))

/-- **Hypothesis: R's rows are pinned by their in-range ones** (the arrays' sizes are the schedule's): two layer
assignments whose `prR` rows agree at every `ℓ < levels.size`, `q < levels[ℓ].q` agree at every `(ℓ, q)`. -/
def RRowsPinned (prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof) (sch : Flock.Schedule) : Prop :=
  ∀ Lb Lb', (∀ ℓ < sch.levels.size, ∀ q < (sch.levels.getD ℓ default).q,
      openRow (prR Lb) sch ℓ q = openRow (prR Lb') sch ℓ q) →
    ∀ ℓ q, openRow (prR Lb) sch ℓ q = openRow (prR Lb') sch ℓ q

/-- **`OpensReadAt 2` from R's rows read off the layers**, at a source whose openings at stratum 2 are `prR`'s rows. -/
theorem opensReadAt_two_of_layers
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof} {sch : Flock.Schedule}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hv : ROpensOfLayers zs dg N k Rw W rd prR sch) (hpin : RRowsPinned prR sch)
    (hsrc : ∀ R ω t Lb ℓ q, (src R ω t Lb).os.1 2 ℓ q = openRow (prR Lb) sch ℓ q) :
    OpensReadAt zs dg N k Rw src 2 := by
  intro R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX
  funext ℓ q
  rw [hsrc, hsrc]
  refine hpin _ _ (fun l hl j hj => ?_) ℓ q
  rw [hv R ω t R₂ τ o hA l hl j hj, hv R' ω' t' R₂' τ' o' hA' l hl j hj, FlockC3.Reps.maskOf_congr W 2 hX]

/-- **`OpensReadAt 2` at H's normalized tied source**, where `stp`'s openings at stratum 2 are `prR`'s rows. -/
theorem opensReadAt_two_norm
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof} {sch : Flock.Schedule}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hv : ROpensOfLayers zs dg N k Rw W rd prR sch) (hpin : RRowsPinned prR sch)
    (hsrc : ∀ R ω t Lb ℓ q, (stp R ω t Lb).os.1 2 ℓ q = openRow (prR Lb) sch ℓ q) :
    OpensReadAt zs dg N k Rw (tiedSrc (normSrc stp)) 2 :=
  opensReadAt_two_of_layers hv hpin fun R ω t Lb ℓ q =>
    (congrFun (congrFun (normOs_os_of_lt (u := stp R ω t Lb) (by decide : 2 < 3)) ℓ) q).trans (hsrc R ω t Lb ℓ q)

end

end FlockC3.RegLevels
