module

public import Properties.Spot.RootHeld
public import Proofs.Properties.Challenger

/-! # Proof that the per-member form tests its record against the root its coins verify under (`Properties.Spot.MembersRootHeld`) -/

@[expose] public section

open Lean (Json)
open Spot
open Flock.HmNets (bind_ok)

namespace Properties.Spot.RootHeldProof

theorem atCoins_accept {Coins Pub Prof : Type} {x : Flock.V Coins} {k : Coins → Core.Outcome Pub Prof} {pub : Pub}
    {prof : Prof} (h : Core.atCoins x k = .accept pub prof) : ∃ ξ, x = .ok ξ ∧ k ξ = .accept pub prof := by
  unfold Core.atCoins at h
  cases hx : x with
  | ok ξ => rw [hx] at h; exact ⟨ξ, rfl, h⟩
  | error r => rw [hx] at h; cases h

/-- Where v2's coins are read under the challenger `d`, the parameters name its key, family and window, and its digest
is the registration's. -/
theorem source_some {p : OneStageZk.Params} {c : OneStage.Commitment} {d : Properties.Challenger.Draws}
    (h : Properties.Challenger.oneStageZkSource p c = some d) :
    p.base.challenger = some (d.key, d.label) ∧ p.base.verifier.windowRoot = some d.window ∧ d.after = c.digest := by
  unfold Properties.Challenger.oneStageZkSource Properties.Challenger.oneStageSource at h
  rcases hch : p.base.challenger with _ | ⟨key, label⟩ <;> rcases hw : p.base.verifier.windowRoot with _ | w <;>
    simp only [hch, hw] at h <;> try cases h
  exact ⟨rfl, rfl, rfl⟩

/-- **The coins v2 reads are the challenger session's on the verifier's window root.** -/
theorem coins_root {p : OneStageZk.Params} {c : OneStage.Commitment} {e : OneStageZk.Evidence} {ξ : ByteArray}
    (h : OneStageZk.coinsOf p c e = .ok ξ) :
    ∃ key label R, p.base.challenger = some (key, label) ∧ p.base.verifier.windowRoot = some R ∧
      Challenger.sampleCoins key R label c.digest e.base.coins = .ok ξ := by
  obtain ⟨d, hd⟩ := Properties.Challenger.oneStageZk_source h
  obtain ⟨raw, hr, rfl⟩ := Properties.Challenger.oneStageZk_reads p c d hd e ξ h
  obtain ⟨hch, hw, ha⟩ := source_some hd
  refine ⟨d.key, d.label, d.window, hch, hw, ?_⟩
  rw [← ha]
  exact hr

theorem check_of_checkAt {p : OneStage.Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : OneStage.Evidence}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : OneStage.checkAt p c (some ξ) e = .accept pub prof) :
    OneStage.check p c e = .accept pub prof := by
  unfold OneStage.checkAt at h
  split at h
  · cases h
  · exact h

/-- **What v1's check accepts passed the registration check.** -/
theorem docket_of_check {p : OneStage.Params} {c : OneStage.Commitment} {e : OneStage.Evidence}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : OneStage.check p c e = .accept pub prof) :
    Docket.check c.record p.verifier = #[] := by
  unfold OneStage.check at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hne
    simpa [Array.isEmpty_iff] using hne

theorem verifier_with {v : Docket.Verifier} {R : String} (h : v.windowRoot = some R) :
    { v with windowRoot := some R } = v := by
  rw [← h]

/-- **The trees v2 reads are the challenger session's on the verifier's window root.** -/
theorem trees_root {p : OneStage.Params} {e : OneStage.Evidence} {key : Challenger.ChallengerKey} {label R : String}
    {ts : Array Challenger.LoggedTree} (hch : p.challenger = some (key, label)) (hw : p.verifier.windowRoot = some R)
    (h : OneStageZk.treesOf p e = .ok ts) : Challenger.trustedTreesOf key R e.coins = .ok ts := by
  unfold OneStageZk.treesOf at h
  rw [hch, hw] at h
  try dsimp only at h
  cases ht : Challenger.trustedTreesOf key R e.coins with
  | error r => rw [ht] at h; cases h
  | ok t => rw [ht] at h; exact h

/-- **Where the per-member form's structure holds, the trees it read are the challenger session's on the verifier's
window root.** -/
theorem membersStructure_trees {p : OneStageZk.Params} {ms : Array OneStageZk.Member} {c : OneStage.Commitment}
    {e : OneStageZk.Evidence} {sts : Array OneStageZk.StatementFiles} {bs : Core.Blobs}
    {key : Challenger.ChallengerKey} {label R : String} (hch : p.base.challenger = some (key, label))
    (hw : p.base.verifier.windowRoot = some R) (h : OneStageZk.membersStructure p ms c e sts bs = .ok ()) :
    ∃ ts, Challenger.trustedTreesOf key R e.base.coins = .ok ts := by
  unfold OneStageZk.membersStructure at h
  obtain ⟨⟨⟩, -, h⟩ := bind_ok h
  obtain ⟨held, -, h⟩ := bind_ok h
  dsimp only at h
  split at h
  · obtain ⟨_, ht, -⟩ := bind_ok h
    cases ht
  obtain ⟨ts, hts, -⟩ := bind_ok h
  exact ⟨ts, trees_root hch hw hts⟩

/-- **Where the members' roots return, every member's run input trees were held** (`OneStageZk.runInputRoots`, at the
verifier's own window root). -/
theorem membersRootsOf_runInputRoots {p : OneStageZk.Params} {ms : Array OneStageZk.Member} {rec : Json}
    {roots : Std.HashMap String (Array Json)} (h : OneStageZk.membersRootsOf p ms rec = .ok roots) :
    ∃ P rs roots₀ roots', Program.parse p.own.program = .ok P ∧
      Partition.evaluate P (getD p.own.partition "query") = .ok rs ∧
      OneStageZk.runInputRoots p.base.verifier P.prog rs ms rec roots₀ = .ok roots' := by
  unfold OneStageZk.membersRootsOf at h
  obtain ⟨w, hw, -⟩ := bind_ok h
  unfold OneStageZk.membersRootsWired at hw
  try dsimp only at hw
  obtain ⟨_, -, h⟩ := bind_ok hw
  obtain ⟨P, hP, h⟩ := bind_ok h
  obtain ⟨rs, hrs, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨⟨⟩, -, h⟩ := bind_ok h
  obtain ⟨r2, h2, -⟩ := bind_ok h
  exact ⟨P, rs, _, r2, hP, hrs, h2⟩

end Properties.Spot.RootHeldProof

namespace Properties.Spot.SecurityProofs

open Properties.Spot.RootHeldProof

/-- **`MembersRootHeld`**: where the audit accepts, the window root its record's tests read is the root of the
challenger session whose signed coins it drew its sample from and whose coin trees it held the sessions to. -/
theorem membersRootHeld : MembersRootHeld := by
  intro p c e bs pub prof h
  unfold OneStageZk.checkBlobs at h
  obtain ⟨ξ, hξ, hacc⟩ := atCoins_accept h
  obtain ⟨key, label, R, hch, hw, hs⟩ := coins_root hξ
  refine ⟨key, label, R, ξ, hch, hw, hs, hacc, ?_⟩
  intro ms hm
  obtain ⟨sts, -, hm'⟩ := OneStageZk.checkBlobsAt_accept_members hm hacc
  obtain ⟨r, hr, -, hc⟩ := OneStageZk.checkMembersAt_accept hm'
  rw [verifier_with hw]
  unfold OneStageZk.membersRun at hr
  obtain ⟨⟨⟩, hst, hr⟩ := bind_ok hr
  obtain ⟨roots, hroots, -⟩ := bind_ok hr
  exact ⟨docket_of_check (check_of_checkAt hc), membersStructure_trees hch hw hst,
    membersRootsOf_runInputRoots hroots⟩

end Properties.Spot.SecurityProofs
