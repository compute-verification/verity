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
`lgCoins_of_weave`: a weave's coins are a prefix of the coins it is given (`weave_coins_prefix`), so at `MapsLeft`'s
weave R's Ligerito's coins are `rep0`'s rounds from `k₂`, as many as it takes.
`stageHeld_lg_of_run`: `stageHeld_lg` at R's run (`run_of_sessionAt`'s) and `MapsLeft`, its `hop` and `hlg` from the
run's ring switches and Ligerito, and `hcsz` from the verifier's stream (`stackedStreams_of_covers`' `hx`, `hxs`).
-/

namespace FlockC3.RegLevels

open Flock.RecVStar (Inner InnerCommits InnerStream SchedRound StackedShape STACKED STACKED_CLAIMS stackedRounds)
open Flock.Guarantees (HoloStream heldBefore holoStreams holoRounds sparseRounds)
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockSoundness.Discharge.Recursion (VStmt HoloParams holoAt)
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
R's opening's coin steps are `stacked/rep0`'s rounds before `k₂` and R's Ligerito's `n` of those from `k₂` on
(`MapsLeft`'s, `lgCoins_of_weave`'s), and the run's stages take at most `r - 1` rounds (`htot`). -/
theorem stageHeld_lg {r : ℕ} {CT : Fin r → Type} {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String InnerCommits} {e : FlockSoundness.Discharge.Recursion.ExecHolo}
    {u : FlockC3.Compose.C3Run e} {ish : Inner} {tbl : String} {cs : InnerCommits} {c3 : Flock.RecCoins.C3}
    {unit : String} {rep rs : Array SchedRound} {sp : Flock.RecSparse.Shape} {sh : StackedShape} {k₂ n : ℕ}
    (hh : held (wr t) = .ok cs) (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3)
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rep) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some sh) (hcl : sh.claims = STACKED_CLAIMS) (hrs : stackedRounds sh = some rs)
    (hcsz : (FlockC3.Opening.rep0 c3 unit).size = rs.size) (htbl : tbl ≠ s!"{STACKED}/")
    (hts : ∀ i : Fin 2, nCoins (u.ts i) = heldBefore (.rep i) rep rep.size)
    (hss : ∀ x : Fin 2, ∃ w cw, u.sp x = FlockC3.Sparse.stepsOf (FlockC3.Compose.sparseOf sp) w cw)
    (hop : u.op.filterMap Sum.getRight? = ((List.range' 0 k₂).map fun i => (FlockC3.Opening.rep0 c3 unit)[i]!).map
      (coinArr FlockSoundness.Refine.fOf))
    (hlg : u.lg.filterMap Sum.getRight? = ((List.range' k₂ n).map fun i =>
      (FlockC3.Opening.rep0 c3 unit)[i]!).map (coinArr FlockSoundness.Refine.fOf))
    (hn : n ≤ (FlockC3.Opening.rep0 c3 unit).size - k₂)
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
  have hlg' : nCoins u.lg = n := by
    show (u.lg.filterMap Sum.getRight?).length = _
    rw [hlg]
    simp
  have hj := (FlockC3.Sparse.stageStarts (u := u) hts hss).2.2.2
  obtain ⟨h1, h2⟩ := heldStage_of_lgCoins (unit := unit) hc3 hlg
  refine ⟨List.range' k₂ n, fun q => (FlockC3.Opening.rep0 c3 unit)[q]!,
    s!"{STACKED}/rep0", id, h1, fun q c h => h2 q c (by rw [hh] at h; exact h),
    by rw [List.length_range']; omega, ?_⟩
  intro p q hp
  have hlt : p < n := by simpa using (List.getElem?_eq_some_iff.1 hp).1
  have hk : heldBefore .stacked rs k₂ = k₂ := by
    have e := heldBefore_stacked (rs := rs) (k₂ := 0) k₂ fun p' hp' => hpos (0 + p') (by omega)
    rw [Nat.zero_add] at e
    rw [e]
    simp [heldBefore]
  have hd := drawn_stacked (n := n) hS hL (hf tbl htbl) (fun p' hp' => hpos (k₂ + p') (by omega)) p q hp
  rw [hk] at hd
  rw [hop', hj]
  exact hd

/-- **A weave's coins are a prefix of the coins it is given**: `weave` stops at the first step it has none for. -/
theorem weave_coins_prefix : ∀ σ ms cs,
    (FlockSoundness.Discharge.Exec.weave σ ms cs).filterMap Sum.getRight? <+: cs
  | [], _, _ => List.nil_prefix
  | .msg _ :: _, [], _ => List.nil_prefix
  | .msg _ :: σ, _ :: ms, cs => by
      simpa [FlockSoundness.Discharge.Exec.weave, List.filterMap_cons] using weave_coins_prefix σ ms cs
  | .coin _ :: _, _, [] => List.nil_prefix
  | .coin _ :: σ, ms, _ :: cs => by
      simpa [FlockSoundness.Discharge.Exec.weave, List.cons_prefix_cons] using weave_coins_prefix σ ms cs

theorem eq_range'_of_prefix {α : Type _} {f : ℕ → α} {l : List α} {k N : ℕ} (h : l <+: (List.range' k N).map f) :
    l = (List.range' k l.length).map f ∧ l.length ≤ N := by
  have hle : l.length ≤ N := by simpa using h.length_le
  refine ⟨?_, hle⟩
  conv_lhs => rw [List.prefix_iff_eq_take.1 h]
  rw [← List.map_take, List.take_range'_of_length_ge hle]

/-- **R's Ligerito stage's coins at `MapsLeft`'s weave**: steps woven from `c`'s rounds from `k₂` on have `c`'s rounds
from `k₂` as their coins, as many as they take, at most as many as there are. -/
theorem lgCoins_of_weave {ts : List FlockSoundness.Game.Step} {σ ms} {c : Array (Array Flock.F128)} {k₂ : ℕ}
    (h : ts = FlockSoundness.Discharge.Exec.weave σ ms
      (((List.range' k₂ (c.size - k₂)).map fun i => c[i]!).map (coinArr FlockSoundness.Refine.fOf))) :
    ts.filterMap Sum.getRight? =
        ((List.range' k₂ (nCoins ts)).map fun i => c[i]!).map (coinArr FlockSoundness.Refine.fOf) ∧
      nCoins ts ≤ c.size - k₂ := by
  have hp := weave_coins_prefix σ ms
    (((List.range' k₂ (c.size - k₂)).map fun i => c[i]!).map (coinArr FlockSoundness.Refine.fOf))
  rw [← h, List.map_map] at hp
  rw [List.map_map]
  exact eq_range'_of_prefix hp

/-- **`h5` at R's run** (`stageHeld_lg` with its coin hypotheses supplied): where R's opening at the map's openings runs
with zero residuals on `stacked/rep0`'s rounds of coins (`run_of_sessionAt`'s `hrun`, `hz`), its ring switches give
`MapsLeft`'s opening coins (`hop`) and its Ligerito `MapsLeft`'s weave (`lgCoins_of_weave`, `hlg`), and the verifier's
`stacked/rep0` stream `x` has a round of coins for each of R's rounds (`stackedStreams_of_covers`' `hxs`, `hcsz`). -/
theorem stageHeld_lg_of_run {r : ℕ} {CT : Fin r → Type} {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String InnerCommits} {st : Flock.Stmt} {hp : HoloParams}
    {u : FlockC3.Compose.C3Run (holoAt st hp)} {ish : Inner} {tbl : String} {cs : InnerCommits}
    {c3 : Flock.RecCoins.C3} {unit : String} {rep rs : Array SchedRound} {sp : Flock.RecSparse.Shape}
    {sh : StackedShape} {x : InnerStream} {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt}
    {Reg' : Type} {M : FlockC3.Reps.StackedMaps stmt Vs} {zs : (s : Fin S) → (Vs s).Outer Reg'}
    {Lb : (s : Fin S) → (Vs s).Bits} {osh : Flock.Shape} {w : Array Flock.F128}
    {kap : Array (Array (Array Flock.F128))} {e0 sb e cap₀} {res : Array Flock.F128} {k₁ : ℕ}
    (hh : held (wr t) = .ok cs) (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3)
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rep) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some sh) (hcl : sh.claims = STACKED_CLAIMS) (hrs : stackedRounds sh = some rs)
    (hx : cs.stream? "stacked/rep0" = some x) (hxs : x.coins.size = rs.size) (htbl : tbl ≠ s!"{STACKED}/")
    (hts : ∀ i : Fin 2, nCoins (u.ts i) = heldBefore (.rep i) rep rep.size)
    (hss : ∀ x : Fin 2, ∃ w cw, u.sp x = FlockC3.Sparse.stepsOf (FlockC3.Compose.sparseOf sp) w cw)
    (hrun : Flock.RecOpening.run osh (msgsO osh w (opsAt (readMaps M osh.sched) osh Lb kap c3 unit w))
      (FlockC3.Opening.claimsO osh w c3 unit) 0 (Flock.RecAlgebra.Coins.ofRounds (FlockC3.Opening.rep0 c3 unit)) =
        .ok (res, k₁))
    (hz : ∀ y ∈ res, y = 0) (hleft : MapsLeft M zs Lb osh unit c3 w kap e0 sb e cap₀ u)
    (htot : 1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op + nCoins u.lg ≤ r) :
    StageHeld t wr held (FlockC3.Compose.holoSelOne ish tbl)
      (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op) u.lg := by
  have hname : s!"{STACKED}/rep0" = "stacked/rep0" := by decide +kernel
  have hcsz : (FlockC3.Opening.rep0 c3 unit).size = rs.size := by
    rw [FlockC3.Opening.rep0_eq (c := x.coins) (by
      unfold Flock.RecCoins.C3.coins
      rw [commits_c3Of hc3, hname, hx]
      rfl), hxs]
  obtain ⟨rsr, rsp, k₂, lgr, lg, h₁, h₂, rfl⟩ := opening_parts hrun
  have hlast : Flock.RecAlgebra.Val.add
      ((msgsO osh w (opsAt (readMaps M osh.sched) osh Lb kap c3 unit w)).tAtom.getD rsp.T) rsp.T = 0 :=
    hz _ (by simp)
  rw [Flock.RecAlgebra.f128_eq_of_add_eq_zero hlast] at h₂
  obtain ⟨-, -, hop, hR, -⟩ := hleft
  obtain ⟨hlg, hn⟩ := lgCoins_of_weave (hR rsr rsp k₂ lgr lg k₁ h₁ h₂).2
  exact stageHeld_lg hh hc3 hrep hsp hsh hcl hrs hcsz htbl hts hss (hop rsr rsp k₂ h₁) hlg hn htot

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.weave_coins_prefix
#print axioms FlockC3.RegLevels.eq_range'_of_prefix
#print axioms FlockC3.RegLevels.lgCoins_of_weave
#print axioms FlockC3.RegLevels.stageHeld_lg
#print axioms FlockC3.RegLevels.stageHeld_lg_of_run
