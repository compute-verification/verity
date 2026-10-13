import Proofs.Flock.C3.Reps.SentKeys
import Proofs.Flock.Recursive.HoloRepCoins

/-!
# A rep's consumed rounds at the schedule's coin rounds, and the held rounds at the schedule's count

`repIdxB_of_size`: a rep's consumed rounds (`repIdxB`) are the coin rounds of the reps' hidden-unit schedule
(`holoRepRounds`, `coinRounds_holoRep_split`) where its transcript consumed as many rounds as the schedule has
(`t₄.k = rs.size`). The verifier holds each rep stream's coins to the schedule's count by H's check in
`checkCommitRounds` (`cursor/flock-c3-refusals-95d4`); until that check is on this branch its result is the hypothesis
`RepRoundsCounted`, from which `heldRounds_size_of_counted` reads a rep stream's held rounds' count. The check's
round-for-round form gives it (`repRoundsCounted_of_rounds`).
-/

namespace FlockC3.Reps

open Flock Flock.RecVStar Flock.Guarantees FlockSoundness

/-- **Hypothesis: the verifier's count check of the reps' held rounds** (H's, in `checkCommitRounds`, on
`cursor/flock-c3-refusals-95d4`): every stream the verifier holds but the sparse phases and R's holds as many rounds of
coins as the reps' schedule `rep` has rounds. -/
def RepRoundsCounted (cs : InnerCommits) (rep : Array SchedRound) : Prop :=
  ∀ st ∈ cs.streams, (st.name.startsWith s!"{SPARSE}/" || st.name.startsWith s!"{STACKED}/") = false →
    st.coins.size = rep.size

/-- **A rep stream's held rounds are as many as the schedule's** under the count check. -/
theorem heldRounds_size_of_counted {cs : InnerCommits} {rep : Array SchedRound} {s : String} {st : InnerStream}
    (hc : RepRoundsCounted cs rep) (hs : cs.stream? s = some st)
    (hn : (s.startsWith s!"{SPARSE}/" || s.startsWith s!"{STACKED}/") = false) :
    (heldRounds (.ok cs) s).size = rep.size := by
  have hs' : cs.streams.find? (·.name == s) = some st := hs
  have hname : st.name = s := by
    first
    | exact beq_iff_eq.1 (Array.find?_some hs')
    | simpa using Array.find?_some hs'
  have hn' : (st.name.startsWith s!"{SPARSE}/" || st.name.startsWith s!"{STACKED}/") = false := by
    rw [hname]
    exact hn
  show (((cs.stream? s).map (·.coins)).getD #[]).size = rep.size
  rw [hs]
  exact hc st (Array.mem_of_find?_eq_some hs') hn'

/-- The count check's round-for-round form (each held round's coins the schedule's round's count) gives
`RepRoundsCounted`. -/
theorem repRoundsCounted_of_rounds {cs : InnerCommits} {rep : Array SchedRound}
    (h : ∀ st ∈ cs.streams, (st.name.startsWith s!"{SPARSE}/" || st.name.startsWith s!"{STACKED}/") = false →
      st.coins.map (·.size) = rep.map (·.1)) : RepRoundsCounted cs rep := fun st hst hn => by
  have e := congrArg Array.size (h st hst hn)
  rwa [Array.size_map, Array.size_map] at e

/-- **A rep's consumed rounds are the reps' schedule's coin rounds** where its transcript consumed as many rounds as
the hidden-unit schedule at the rep's `m` and `k_log` has, at a claim. -/
theorem repIdxB_of_size {S₀ : Model.Statement} {b : ValsSrc} {m kLog claims : ℕ} {rs : Array SchedRound}
    (hc : 0 < claims) (hrs : holoRepRounds m kLog claims = some rs) (hm : b.st.m = m) (hkl : S₀.kLog = kLog)
    (hk : b.t₄.k = rs.size) : repIdxB S₀ b = coinRounds rs := by
  have h22 := HoloRepCoins.holoRepRounds_m hrs
  have h6 := S₀.hk6
  have hz : 3 + (b.st.m - 6) + 2 + (S₀.kLog - 6) = zpRound m kLog := by
    unfold zpRound
    omega
  unfold repIdxB
  rw [hz, hk]
  exact (HoloRepCoins.coinRounds_holoRep_split hc hrs).symm

end FlockC3.Reps
