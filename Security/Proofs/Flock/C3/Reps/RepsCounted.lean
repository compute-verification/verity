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

`t₄k_of_counted` closes the count on the proof's side: where the record was consumed whole (the verifier's S15/R2,
`t₄.k` is the stream's round count) and the rep's rounds are its held stream's, `t₄.k` is the schedule's count, and
`repIdxB_of_counted` is RepsDrawn (ii)'s `repIdxB S₀ b = coinRounds rs` from it.
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

/-- **A rep's closing round is the schedule's count** where its record was consumed whole (`hk`, the verifier's S15/R2:
`t₄.k` is the stream's round count), its rounds are the ones the verifier holds of its stream (`hd`), and the count
check holds that stream to the schedule's count. -/
theorem t₄k_of_counted {cs : InnerCommits} {rep : Array SchedRound} {s : String} {st : InnerStream} {b : ValsSrc}
    {sh : Flock.Shape} {d : FlockVBridge.Compiled.RepData}
    (hc : RepRoundsCounted cs rep) (hs : cs.stream? s = some st)
    (hn : (s.startsWith s!"{SPARSE}/" || s.startsWith s!"{STACKED}/") = false)
    (hd : d.rounds = heldRounds (.ok cs) s) (hk : b.t₄.k = (b.vals sh d).t.rounds.size) : b.t₄.k = rep.size := by
  have e : (b.vals sh d).t.rounds.size = d.rounds.size := by
    show (FlockVBridge.Family.streamRounds d.rounds b.fr).size = d.rounds.size
    first
    | (unfold FlockVBridge.Family.streamRounds; rw [Array.size_mapIdx])
    | simp [FlockVBridge.Family.streamRounds]
  rw [hk, e, hd, heldRounds_size_of_counted hc hs hn]

/-- **RepsDrawn (ii)'s count**: a rep's consumed rounds are the hidden-unit schedule's coin rounds where its record
was consumed whole, its rounds are its held stream's, and the count check holds that stream to the schedule. -/
theorem repIdxB_of_counted {S₀ : Model.Statement} {b : ValsSrc} {sh : Flock.Shape}
    {d : FlockVBridge.Compiled.RepData} {m kLog claims : ℕ} {rs : Array SchedRound} {cs : InnerCommits} {s : String}
    {st : InnerStream} (hcl : 0 < claims) (hrs : holoRepRounds m kLog claims = some rs) (hm : b.st.m = m)
    (hkl : S₀.kLog = kLog) (hc : RepRoundsCounted cs rs) (hs : cs.stream? s = some st)
    (hn : (s.startsWith s!"{SPARSE}/" || s.startsWith s!"{STACKED}/") = false)
    (hd : d.rounds = heldRounds (.ok cs) s) (hk : b.t₄.k = (b.vals sh d).t.rounds.size) :
    repIdxB S₀ b = coinRounds rs :=
  repIdxB_of_size hcl hrs hm hkl (t₄k_of_counted hc hs hn hd hk)

/-- **`t₄k_of_counted` through `ProofOfVals`' coins**, at any rep unit's values `v`: the proof's record coins are the
rep's rounds (`UnitProof.coins`), so the transcript's rounds are as many as the held stream's. -/
theorem t₄k_of_coins {cs : InnerCommits} {rep : Array SchedRound} {s : String} {st : InnerStream}
    {d : FlockVBridge.Compiled.RepData} {v : UnitVals}
    (hc : RepRoundsCounted cs rep) (hs : cs.stream? s = some st)
    (hn : (s.startsWith s!"{SPARSE}/" || s.startsWith s!"{STACKED}/") = false)
    (hd : d.rounds = heldRounds (.ok cs) s) (hpc : d.rounds = v.t.rounds.map (·.coins))
    (hk : v.t₄.k = v.t.rounds.size) : v.t₄.k = rep.size := by
  rw [hk, ← heldRounds_size_of_counted hc hs hn, ← hd, hpc, Array.size_map]

/-- **RepsDrawn (ii)'s count at the verifier's round-for-round check** (H's, in `checkCommitRounds` on
`cursor/c3-n19-bridge-95d4`: every held stream but the sparse phases' and R's holds the reps' schedule's rounds, round
for round). -/
theorem repIdxB_of_rounds {S₀ : Model.Statement} {b : ValsSrc} {sh : Flock.Shape}
    {d : FlockVBridge.Compiled.RepData} {m kLog claims : ℕ} {rs : Array SchedRound} {cs : InnerCommits} {s : String}
    {st : InnerStream} (hcl : 0 < claims) (hrs : holoRepRounds m kLog claims = some rs) (hm : b.st.m = m)
    (hkl : S₀.kLog = kLog)
    (hr : ∀ st ∈ cs.streams, (st.name.startsWith s!"{SPARSE}/" || st.name.startsWith s!"{STACKED}/") = false →
      st.coins.map (·.size) = rs.map (·.1))
    (hs : cs.stream? s = some st) (hn : (s.startsWith s!"{SPARSE}/" || s.startsWith s!"{STACKED}/") = false)
    (hd : d.rounds = heldRounds (.ok cs) s) (hk : b.t₄.k = (b.vals sh d).t.rounds.size) :
    repIdxB S₀ b = coinRounds rs :=
  repIdxB_of_counted hcl hrs hm hkl (repRoundsCounted_of_rounds hr) hs hn hd hk

#print axioms t₄k_of_counted
#print axioms repIdxB_of_counted
#print axioms t₄k_of_coins
#print axioms repIdxB_of_rounds

end FlockC3.Reps
