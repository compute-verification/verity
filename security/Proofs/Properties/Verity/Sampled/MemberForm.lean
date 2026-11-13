module

public import Proofs.Properties.Verity.Sampled.MemberForm.Evidence
public import Proofs.Properties.Verity.Sampled.MemberForm.Files
public import Proofs.Properties.Verity.Sampled.MemberForm.Order
public import Proofs.Properties.Verity.Sampled.MemberForm.Consts
public import Proofs.Properties.Verity.Sampled.MemberForm.Opens
public import Proofs.Properties.Verity.Sampled.MemberForm.Run
public import Proofs.Properties.Verity.Sampled.MemberForm.SetsUp
public import Proofs.Properties.Verity.Sampled.MemberForm.States
public import Proofs.Properties.Verity.Sampled.MemberForm.Passes
public import Proofs.Properties.Verity.Sampled.MemberForm.Tree
public import Proofs.Properties.Verity.Sampled.MemberForm.Opened
public import Proofs.Properties.Verity.Sampled.MemberForm.Place
public import Proofs.Properties.Verity.Sampled.MemberForm.Sessions
public import Proofs.Properties.Verity.Sampled.MemberForm.Links
public import Proofs.Properties.Verity.Sampled.MemberForm.Read

/-!
# The per-member form of one-stage v2, as its proofs read it

What the per-member form's checks (`Spot.OneStageZk.checkMembersAt`, which accepts nothing until its guarantee is in
the lock) give of the bytes and the release log they accept, each stated at the code's own functions:

* `MemberForm.Evidence`: the bytes a session's closing round is opened after the SHA-512 of decode one way
  (`evidenceParts_evidenceBytes`, `evidenceBytes_inj`).
* `MemberForm.Files`: a session's proof files, decoded from the bytes its closing round is opened after the SHA-512
  of, are the ones its verdict read (`statementFiles_evidence`).
* `MemberForm.Order`: the members' sessions on v2's read of the release log (`members_read`): each session's tree
  less its closing round the session's, its closing round answering the bytes its evidence's SHA-512 names with no
  opening of its tree after, the sessions' trees opened in the sessions' order, and each session's closing round before
  the next session's tree.
* `MemberForm.Consts`: the vocabulary's openings hold at each member's constants at every window, a member at a
  time (`constsWith_read`), as the claim at a window reads its members.
* `MemberForm.Opens`: the vocabulary's openings hold wherever each member's deployed commitment opens, at members
  that open only at their own files' places (`opensWith_read`).
* `MemberForm.Accept`: what the proofs read of an accept of the form (`MembersAccept`), and that the check's accepts
  hold it (`FormAccepts`, of the check as it is by `formAccepts_refused`).
* `MemberForm.Run`: the form's run where every session accepted (`membersRun_ok`): the statements members' in order,
  one session each, at the statement read for its member (`statementAt_files`), the sessions' verdicts in order.
* `MemberForm.SetsUp`: what a window's instance sets up at the form's parameters (`MembersSetUp`: the window on
  the parameters' members and the files sent, `WindowOn`, laid out, each member's files passing, `FilesPass`), and
  the window's instance's setup at every accepted play from it (`windowAt_members`).
* `MemberForm.States`: the statement the form read for a member is its window's member's (`statesMember`), so a
  session's verdict at it is flock-verify `--zk`'s at the window's member.
* `MemberForm.Passes`: a member whose files the form's run held has its files pass at the window's member
  (`filesPass_of_held`, `WindowOn.passes`), and a member whose files pass has roots that fix their domains
  (`FilesPass.rootsFix`).
* `MemberForm.Tree`: what a session's record's check gives of its own tree of a log of several, less the tree's
  closing round (`tree_round`, `tree_keyAt`, `tree_rounds_le`, `helloSpec_of_text`).
* `MemberForm.Opened`: each opening the read shows is the opening its tree shows of that round, with its count, the
  bytes it answers and the coins the records give it (`read_opening`), and a record's tree opens only the record's
  rounds (`tree_recorded`).
* `MemberForm.Place`: the `j`-th tree the read opens is the `j`-th session's, and the read opens as many trees as
  sessions (`members_place`, `members_count`).
* `MemberForm.Sessions`: what the window's theorem asks of the `j`-th session on a device's log the read reads
  (`sessionRead_logged`: the session's record's rounds on its own tree and its closing round answering its
  evidence), at a read whose trees each open their `session` stream first (`LinkFirst`).
* `MemberForm.Links`: the read of lines `membersTreeReason` accepts opens each tree's `session` stream first
  (`linkFirst_members`), by the check's scan of the lines (`Spot.Challenger.linksFirst`).
* `MemberForm.Read`: v2's read at an accept of the form (`accept_read`): the releases it read the release log as,
  whose reads are the device's log's, the trees `membersTreeReason` accepts the sessions against, each statement's
  one session as the structure names it, and the device's log in order at the sample's family and digest.
-/
