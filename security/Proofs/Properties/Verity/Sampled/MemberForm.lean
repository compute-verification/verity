module

public import Proofs.Properties.Verity.Sampled.MemberForm.Evidence
public import Proofs.Properties.Verity.Sampled.MemberForm.Order

/-!
# The per-member form of one-stage v2, as its proofs read it

What the per-member form's checks (`Spot.OneStageZk.checkMembersAt`, which accepts nothing until its guarantee is in
the lock) give of the bytes and the release log they accept, each stated at the code's own functions:

* `MemberForm.Evidence`: the bytes a session's closing round is opened after the SHA-512 of decode one way
  (`evidenceParts_evidenceBytes`, `evidenceBytes_inj`).
* `MemberForm.Order`: the members' sessions on v2's read of the release log (`members_read`): each session's tree
  less its closing round the session's, its closing round answering the bytes its evidence's SHA-512 names with no
  opening of its tree after, the sessions' trees opened in the sessions' order, and each session's closing round before
  the next session's tree.
-/
