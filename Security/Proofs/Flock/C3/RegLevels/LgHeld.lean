import Proofs.Flock.C3.Sparse.HeldStages
import Proofs.Flock.C3.RegLevels.RRounds

/-!
# `StageHeld` part 3 at R's Ligerito stage: `laterCoins_of_held`'s `h5`

`holoStreams_four`: with a sparse shape and an R shape, `holoStreams`' streams before R's are both reps' and both
sparse phases', the list `stageStarts` sums.
`stageHeld_lg`: R's Ligerito stage is `StageHeld` at the map's selector (`holoSelOne`) from where R's opening ends. Its
coins are the verifier's hold of `stacked/rep0` from the ring switches' `k₂` on (`heldStage_of_lgCoins`); the stage
starts one past the four streams' rounds (`stageStarts`) and R's opening's `k₂`, which is R's history round `k₂`
since every round of R's draws coins (`stackedRounds_coins`), so `drawn_stacked` draws it in order; and it ends below
`r` where the run's stages take at most `r - 1` rounds in all (`htot`, `Compose.coins_total_holo` at `holoCTs`).
-/

namespace FlockC3.RegLevels

open Flock.RecVStar (Inner InnerCommits SchedRound StackedShape STACKED STACKED_CLAIMS stackedRounds)
open Flock.Guarantees (HoloStream heldBefore holoStreams holoRounds sparseRounds)
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockC3.Reps (nCoins StageHeld)

/-- **`holoStreams` with both shapes**: both reps' streams, both sparse phases' and then R's. -/
theorem holoStreams_four {ish : Inner} {rep rs : Array SchedRound} {sp : Flock.RecSparse.Shape} {sh : StackedShape}
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rep) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some sh) (hrs : stackedRounds sh = some rs) :
    holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ([((.rep 0 : HoloStream), rep), (.rep 1, rep),
      (.sparse 0, sparseRounds sp), (.sparse 1, sparseRounds sp)] ++ [(.stacked, rs)]) := by
  rw [hsp, hsh]
  simp [holoStreams, hrep, hrs]

/-- **R's Ligerito stage is `StageHeld` from where R's opening ends** (`laterCoins_of_held`'s `h5`, at the map's
selector `holoSelOne`): where the verifier holds `cs` (`hh`) and builds its `C3` of it (`hc3`), the inner shape has
the reps' rounds `rep`, a sparse shape and R's, at R's two claims, with as many rounds of coins held of `stacked/rep0`
as R's rounds (`hcsz`), the reps' and sparse phases' stages take their streams' rounds (`stageStarts`' `hts`, `hss`),
R's opening's coin steps are `stacked/rep0`'s rounds before `k₂` and R's Ligerito's those from `k₂` on (`MapsLeft`'s,
`lgSteps_of_lig`'s), and the run's stages take at most `r - 1` rounds (`htot`). -/
theorem stageHeld_lg {r : ℕ} {CT : Fin r → Type} {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String InnerCommits} {e : FlockSoundness.Discharge.Recursion.ExecHolo}
    {u : FlockC3.Compose.C3Run e} {ish : Inner} {tbl : String} {cs : InnerCommits} {c3 : Flock.RecCoins.C3}
    {unit : String} {rep rs : Array SchedRound} {sp : Flock.RecSparse.Shape} {sh : StackedShape} {k₂ : ℕ}
    (hh : held (wr t) = .ok cs) (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3)
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rep) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some sh) (hcl : sh.claims = STACKED_CLAIMS) (hrs : stackedRounds sh = some rs)
    (hcsz : (FlockC3.Opening.rep0 c3 unit).size = rs.size) (htbl : tbl ≠ s!"{STACKED}/")
    (hts : ∀ i : Fin 2, nCoins (u.ts i) = heldBefore (.rep i) rep rep.size)
    (hss : ∀ x : Fin 2, ∃ w cw, u.sp x = FlockC3.Sparse.stepsOf (FlockC3.Compose.sparseOf sp) w cw)
    (hop : u.op.filterMap Sum.getRight? = ((List.range' 0 k₂).map fun i => (FlockC3.Opening.rep0 c3 unit)[i]!).map
      (coinArr FlockSoundness.Refine.fOf))
    (hlg : u.lg.filterMap Sum.getRight? = ((List.range' k₂ ((FlockC3.Opening.rep0 c3 unit).size - k₂)).map fun i =>
      (FlockC3.Opening.rep0 c3 unit)[i]!).map (coinArr FlockSoundness.Refine.fOf))
    (htot : 1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op + nCoins u.lg ≤ r) :
    StageHeld t wr held (FlockC3.Compose.holoSelOne ish tbl)
      (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op) u.lg := by
  obtain ⟨L, hS', hL, hf⟩ := holoStreams_stacked hrep hsh hrs
  have hS := holoStreams_four hrep hsp hsh hrs
  obtain rfl := List.append_cancel_right (Option.some.inj (hS'.symm.trans hS))
  have hpos := stackedRounds_coins hcl hrs
  have hop' : nCoins u.op = k₂ := by
    show (u.op.filterMap Sum.getRight?).length = _
    rw [hop]
    simp
  have hlg' : nCoins u.lg = (FlockC3.Opening.rep0 c3 unit).size - k₂ := by
    show (u.lg.filterMap Sum.getRight?).length = _
    rw [hlg]
    simp
  have hj := (FlockC3.Sparse.stageStarts (u := u) hts hss).2.2.2
  obtain ⟨h1, h2⟩ := heldStage_of_lgCoins (unit := unit) hc3 hlg
  refine ⟨List.range' k₂ ((FlockC3.Opening.rep0 c3 unit).size - k₂), fun q => (FlockC3.Opening.rep0 c3 unit)[q]!,
    s!"{STACKED}/rep0", id, h1, fun q c h => h2 q c (by rw [hh] at h; exact h),
    by rw [List.length_range']; omega, ?_⟩
  intro p q hp
  have hlt : p < (FlockC3.Opening.rep0 c3 unit).size - k₂ := by
    simpa using (List.getElem?_eq_some_iff.1 hp).1
  have hk : heldBefore .stacked rs k₂ = k₂ := by
    have e := heldBefore_stacked (rs := rs) (k₂ := 0) k₂ fun p' hp' => hpos (0 + p') (by omega)
    rw [Nat.zero_add] at e
    rw [e]
    simp [heldBefore]
  have hd := drawn_stacked (n := (FlockC3.Opening.rep0 c3 unit).size - k₂) hS hL (hf tbl htbl)
    (fun p' hp' => hpos (k₂ + p') (by omega)) p q hp
  rw [hk] at hd
  rw [hop', hj]
  exact hd

end FlockC3.RegLevels
