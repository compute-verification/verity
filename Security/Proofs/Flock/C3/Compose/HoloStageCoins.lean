import Proofs.Flock.C3.Compose.SparseEnds
import Proofs.Flock.C3.Compose.SrcClauses
import Proofs.Flock.C3.Compose.Tied
import Proofs.Flock.C3.RegLevels.LigWhole
import Proofs.Flock.C3.Reps.RepCoins
import Proofs.Flock.Recursive.HoloCoins

/-!
# `StageCoins` at the guarantee's coin types

`stageCoins_holoCTs`: `StageCoins` at `holoCTs st hp` (`Recursive/HoloCoins.lean`), stage `i` taking as many of the
history's rounds as it has coins, at a run whose stages are tied. Each tied stage takes its whole shape: a rep
`holoRepShape` (`repKinds_of_tie`), a sparse phase `sparseKinds` (`sparseKinds_of_tie`), R's opening `opShape` at two
claims (`opening_exact`) and R's Ligerito `ligShape` (`ligeritoC_exact`). So the stages' coins add up to `holoKinds`',
which `holoCTs` counts after the link points.
`stageCoins_holo`: `srcOK_holo`'s `hcoins` at `holoCTs` and the tied source of `repsOfVals`. The reps' two conjuncts
come from custody's coins half (`repCoins_repsOfVals`), and the rest from `LaterCoins`.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (coinSteps)
open FlockC3.Reps (nCoins RepTie SrcRepTie repsOfVals UnitVals ProofOfVals SetupOfVals FramesOfVals RepRoundsHeld
  RepsDrawn CoinsEncoded repCoins_repsOfVals)
open Flock.Guarantees (holoCTs holoKinds)

theorem kindCoinTys_length : ∀ σ : List Exec.Kind, (Exec.kindCoinTys σ).length = kindCoins σ
  | [] => rfl
  | .msg _ :: σ => kindCoinTys_length σ
  | .coin _ :: σ => congrArg (· + 1) (kindCoinTys_length σ)

theorem kindCoins_append : ∀ a b : List Exec.Kind, kindCoins (a ++ b) = kindCoins a + kindCoins b
  | [], b => (Nat.zero_add (kindCoins b)).symm
  | .msg _ :: a, b => kindCoins_append a b
  | .coin _ :: a, b => by
      show kindCoins (a ++ b) + 1 = kindCoins a + 1 + kindCoins b
      rw [kindCoins_append a b]
      omega

/-- **The link points' and the later stages' coins are the history's** (named; `StageCoins` but its reps'
conjuncts): the link points are the history's first coin, and the sparse phases', R's opening's and R's Ligerito's
coins are the history's at their rounds, which follow the reps'. -/
def LaterCoins {r : ℕ} {CT : Fin r → Type} {e : ExecHolo} (cs : List ((k : Fin r) → CT k)) (u : C3Run e) : Prop :=
  (coinSteps CT 0 cs).head? = some ⟨Fin 2 → Fin e.cls.mPts → GF128, u.pts⟩ ∧
    CoinsAt cs (1 + nCoins (u.ts 0) + nCoins (u.ts 1)) (nCoins (u.sp 0)) (u.sp 0) ∧
    CoinsAt cs (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0)) (nCoins (u.sp 1)) (u.sp 1) ∧
    CoinsAt cs (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1)) (nCoins u.op) u.op ∧
    CoinsAt cs (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op)
      (nCoins u.lg) u.lg

/-- **`StageCoins` at `holoCTs`** at a run whose stages are tied: stage `i` takes its coins' number of rounds, and
those add up to `holoKinds`' coins, `holoCTs`' count after the link points. -/
theorem stageCoins_holoCTs {st : Flock.Stmt} {hp : HoloParams} {u : C3Run (holoAt st hp)} {cap₀ : Cap Dig}
    {sh : Flock.Shape} {w : Array F128} {ops : FlockVBridge.AccChain.Openings}
    {cs : List ((k : Fin (1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k)}
    (hr : ∀ i, RepTie (holoAt st hp) u i) (hs : ∀ i, FlockC3.Sparse.SparseTie (holoAt st hp) u i)
    (ho : FlockC3.Opening.OpenTie sh w ops u) (hor : OpeningReadAt (runSteps u) u)
    (hl : FlockC3.RegLevels.LevelTie cap₀ (holoAt st hp) u)
    (h0 : CoinsAt cs 1 (nCoins (u.ts 0)) (u.ts 0)) (h1 : CoinsAt cs (1 + nCoins (u.ts 0)) (nCoins (u.ts 1)) (u.ts 1))
    (hlater : LaterCoins cs u) : StageCoins (CT := holoCTs st hp) cs u := by
  obtain ⟨hpts, h2, h3, h4, h5⟩ := hlater
  have hkr := fun i => repKinds_of_tie (hr i)
  have hks := fun i => sparseKinds_of_tie (hs i)
  obtain ⟨_, ha⟩ := opening_plays_of ho hor
  have hko : u.op.map Exec.kindOf = Exec.opShape GF128 2 := Exec.Exact.kinds ha (Exec.opening_exact _ _ _)
  obtain ⟨_, hpl⟩ := lig_plays_of_levelTie hl
  have hkl : u.lg.map Exec.kindOf = Exec.ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch :=
    Exec.Exact.kinds hpl (FlockC3.RegLevels.ligeritoC_exact _ _ _ _)
  refine ⟨![nCoins (u.ts 0), nCoins (u.ts 1), nCoins (u.sp 0), nCoins (u.sp 1), nCoins u.op, nCoins u.lg], ?_,
    hpts, h0, h1, h2, h3, h4, h5⟩
  show 1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op + nCoins u.lg =
    1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length
  simp only [nCoins, coins_eq_kindCoins, hkr, hks, hko, hkl, kindCoinTys_length, holoKinds, kindCoins_append]
  omega

section src

variable {nT : ℕ} {T : Fin nT → Type} {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n}
  {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T (holoCTs st hp) st hp

/-- **`srcOK_holo`'s `hcoins` at `holoCTs`**, at the tied source of `repsOfVals` whose stages are tied: the reps'
coins from custody's coins half (`repCoins_repsOfVals`), the link points' and the later stages' from `LaterCoins`. -/
theorem stageCoins_holo {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
    {m : FlockSoundness.Assumptions.InnerRecordMap Cm (ℐ).Coin} {s : Fin 2 → String} {ix : Fin 2 → ℕ → ℕ}
    {cap₀ : Cap Dig} (hleaf : HLeafC3 zs dg N k Rw)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hsv : SetupOfVals zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse m)
    (hheld : RepRoundsHeld zs dg N k Rw sh rep wr held s ix)
    (henc : CoinsEncoded (CT := holoCTs st hp) m.selC m.coin)
    (hdr : RepsDrawn zs dg N k Rw sh rep vals m.selC s ix)
    (hrt : SrcRepTie zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hst : FlockC3.Sparse.SrcSparseTie zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hot : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => ∃ sh w ops,
      FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u)
    (hlt : FlockC3.RegLevels.SrcLevelTie zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) cap₀)
    (hlater : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      LaterCoins (CT := holoCTs st hp) (t.map Prod.snd) u) :
    AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      StageCoins (CT := holoCTs st hp) (t.map Prod.snd) u := by
  intro R ω t R₂ τ o hA
  obtain ⟨h0, h1⟩ := repCoins_repsOfVals hleaf halg hacc hpr hsv hf hcus hheld henc hdr R ω t R₂ τ o hA
  obtain ⟨_, _, _, ho, hor⟩ := hot R ω t R₂ τ o hA
  exact stageCoins_holoCTs (hrt R ω t R₂ τ o hA) (hst R ω t R₂ τ o hA) ho hor (hlt R ω t R₂ τ o hA) h0 h1
    (hlater R ω t R₂ τ o hA)

end src

end

end FlockC3.Compose

#print axioms FlockC3.Compose.kindCoinTys_length
#print axioms FlockC3.Compose.kindCoins_append
#print axioms FlockC3.Compose.repKinds_of_tie
#print axioms FlockC3.Compose.sparseKinds_of_tie
#print axioms FlockC3.Compose.stageCoins_holoCTs
#print axioms FlockC3.Compose.stageCoins_holo
#print axioms FlockSoundness.Discharge.Recursion.check_ok
#print axioms FlockSoundness.Discharge.Recursion.checkStmt_ok
