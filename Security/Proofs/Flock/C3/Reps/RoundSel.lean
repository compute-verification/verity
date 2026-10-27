import Proofs.Flock.C3.Reps.RoundReads
import Proofs.Flock.C3.Compose.HoloStreamSizes

/-!
# `holoSel_parts` at the verifier's streams

`holoSel_parts` takes the streams' list as `(.rep 0, rs) :: (.rep 1, rs) :: L` at `holoRounds`' `rs` (its `hS`, `hr`)
and its `c₀` as the reps' held rounds (`hc₀`). At any list `holoStreams` builds, `holoStreams_eq` gives that form, so
`holoSel_parts_streams` is `holoSel_parts` at `holoStreams`' own list, the one `RoundReads`' `hsel` leaves open being
`hm`.
-/

namespace FlockC3.Reps.RoundReads

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.PubRows
open FlockC3.Compose
open FlockVBridge.Family (partsOf)
open Flock.RecVStar Flock.Guarantees

/-- **`holoSel_parts` at the verifier's streams**: where `holoStreams` builds `L`, `L` is the reps' two streams at
`holoRounds`' `rs` ahead of the rest, and the map's `sel` reads each part's `m<m>` at `stageOf σ c₀` with `c₀` the reps'
held rounds, wherever `σ` puts message `m` at the reps' round `(msgRounds rs)[m]` (`hm`). -/
theorem holoSel_parts_streams {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
    {bm : Cm → String → ℕ → List UInt8} {L : List (HoloStream × Array SchedRound)}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L) :
    ∃ rs L', holoRounds ish.m ish.kLog ish.claims = some rs ∧ L = (.rep 0, rs) :: (.rep 1, rs) :: L' ∧
      ∀ {σ : List FlockSoundness.Discharge.Exec.Kind} {S : ℕ} {Vs : Fin S → VStmt} {sh : Flock.Shape}
        {sess : ℕ → Fin S},
        (∀ p < (partsOf sh).size, ∀ (m : ℕ) (q : Fin (Vs (sess p)).c.ports.size),
          (Vs (sess p)).c.ports[q].name = s!"m{m}" → ∃ hm : m < (msgRounds rs).length,
            FlockC3.Sparse.roundOf σ m = heldBefore (.rep 0) rs ((msgRounds rs)[m]'hm)) →
        ∀ p < (partsOf sh).size, ∀ (m : ℕ) (ρ : Fin 2) (q : Fin (Vs (sess p)).c.ports.size),
          (Vs (sess p)).c.ports[q].name = s!"m{m}" →
            (holoMap st hp Cm ish tbl bm).sel (stageOf σ (heldBefore (.rep 0) rs rs.size) ρ m) s!"rec-c{m}" ρ.val := by
  obtain ⟨rs, L', hr, rfl⟩ := holoStreams_eq hS
  exact ⟨rs, L', hr, rfl, fun hm => holoSel_parts hS hr rfl hm⟩

end FlockC3.Reps.RoundReads
