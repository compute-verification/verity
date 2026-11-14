module

public import Proofs.Properties.Verity.Sampled.MemberForm.Evidence
public import Proofs.Properties.Verity.Sampled.MemberForm.Files
public import Proofs.Properties.Verity.Sampled.MemberForm.Order
public import Proofs.Properties.Verity.Sampled.MemberForm.Consts
public import Proofs.Properties.Verity.Sampled.MemberForm.Opens
public import Proofs.Properties.Verity.Sampled.MemberForm.Run
public import Proofs.Properties.Verity.Sampled.MemberForm.Named
public import Proofs.Properties.Verity.Sampled.MemberForm.SetsUp
public import Proofs.Properties.Verity.Sampled.MemberForm.States
public import Proofs.Properties.Verity.Sampled.MemberForm.Passes
public import Proofs.Properties.Verity.Sampled.MemberForm.Tree
public import Proofs.Properties.Verity.Sampled.MemberForm.Opened
public import Proofs.Properties.Verity.Sampled.MemberForm.Place
public import Proofs.Properties.Verity.Sampled.MemberForm.Sessions
public import Proofs.Properties.Verity.Sampled.MemberForm.Links
public import Proofs.Properties.Verity.Sampled.MemberForm.Read
public import Proofs.Properties.Verity.Sampled.MemberForm.Ranges
public import Proofs.Properties.Verity.Sampled.MemberForm.Sample
public import Proofs.Properties.Verity.Sampled.MemberForm.Units
public import Proofs.Properties.Verity.Sampled.MemberForm.Registered
public import Proofs.Properties.Verity.Sampled.MemberForm.Apart
public import Proofs.Properties.Verity.Sampled.MemberForm.Laid
public import Proofs.Properties.Verity.Sampled.MemberForm.WindowPasses
public import Proofs.Properties.Verity.Sampled.MemberForm.Sent
public import Proofs.Properties.Verity.Sampled.MemberForm.Reads
public import Proofs.Properties.Verity.Sampled.MemberForm.OpensOwn
public import Proofs.Properties.Verity.Sampled.MemberForm.ClaimReads
public import Proofs.Properties.Verity.Sampled.MemberForm.Sound

/-!
# The per-member form of one-stage v2, as its proofs read it

What the per-member form's checks (`Spot.OneStageZk.checkMembersAt`, which `checkBlobsAt` decides a case of the form
by) give of the bytes and the release log they accept, each stated at the code's own functions:

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
* `MemberForm.Named`: the form's structure, the sessions `membersTreesHeld` names and a member's record at its tree
  (`membersStructure_ok`, `membersTreesHeld_ok`, `tree_round`), below the binding, which reads an accept of either
  form.
* `MemberForm.Accept`: what the proofs read of an accept of the form (`MembersAccept`), and that the check's accepts
  hold it (`FormAccepts`, proved of the check by `formAccepts`).
* `MemberForm.Run`: the form's run where every session accepted (`membersRun_ok`): the statements members' in order,
  one session each, at the statement read for its member (`statementAt_files`), the sessions' verdicts in order.
* `MemberForm.SetsUp`: what a window's instance sets up at the form's parameters (`MembersSetUp`: the window the
  parameters' members on the registered files, `WindowOn`, laid out, each member's files passing, `FilesPass`),
  and how v2 reads the members' files (`membersRead`).
* `MemberForm.States`: the statement the form read for a member is its window's member's (`statesMember`), so a
  session's verdict at it is flock-verify `--zk`'s at the window's member.
* `MemberForm.Passes`: a member whose files the form's run held has its files pass at the window's member
  (`filesPass_of_held`, `WindowOn.passes`), and a member whose files pass has roots that fix their domains
  (`FilesPass.rootsFix`).
* `MemberForm.Named`: the structure's parts (`membersStructure_ok`), what it names each session
  (`membersTreesHeld_ok`) and a member's record at its tree less the closing round (`memberTreeReason_none`,
  `tree_round`), below the binding (`Binding.accepted_run`), which reads them at an accept of the form.
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
* `MemberForm.Ranges`: one range of units and lists in one order, over natural numbers alone: a range's first
  unit, count and units, lists strictly ascending under a key with the same members (`eq_of_sorted`), and the
  place of a member a filter keeps (`filter_at`).
* `MemberForm.Sample`: the sample a session's record holds at its member's statement, and the units it proved
  (`session_sample`): the subset print of slots ascending below the member's count, the units the member's first
  unit plus each slot. The window's population is the one the registration names (`WindowOn.population`).
* `MemberForm.Units`: the units the form's sessions proved, member by member (`run_over`), and the session of a
  member the sample holds (`member_session`): its statement's place among the statements, its verdict at the
  member's own statement, and its record's sample the sample file of the member's part of the window's sample.
* `MemberForm.Registered`: on blobs that hold each member's file as the registered files' file under its digest,
  v2 reads the members' files among the registered files (`readsFiles_of_sent`), at registered files that hold
  each file the parameters' members name (`RegisteredHolds`).
* `MemberForm.Apart`: no unit is in two members' ranges at parameters the form's parse accepts (`membersOf_apart`,
  `membersParamsOfJson_apart`), and what the proofs read of that at an accept (`FormApart`, proved of the check
  by `formApart`).
* `MemberForm.Laid`: the window the parameters' members are on is laid out (`WindowOn.laid`): its members' units'
  counts sum to its units (`WindowOn.total`), and each member's units lie one after another from where the members
  before it end.
* `MemberForm.Sent`: each session of the form in what the prover sent: the evidence sent for the `u`-th session
  (`sentEvidenceAt`), the `j`-th statement's one session as the evidence's `j`-th, named at place `j` (`named_at`),
  its record and evidence among what was sent (`sentRecordAt_session`, `sentEvidenceAt_session`), its one table and
  that table's coin tree (`rootBTables_one`, `Member.spec_coinTree`).
* `MemberForm.Reads`: at a play the form accepts, every release of the sample's family precedes every opening and
  the play has the sessions of the window's sample (`windowRead_members`).
* `MemberForm.WindowPasses`: what a window's instance takes of the form's acceptance. At each play the form
  accepts, the window the verifier sets up is laid out and each member's files pass (`windowPasses_members`), so
  the verifier sets the deployment's window up at every accepted play (`executionPhaseAt_form`), and each play has the
  window's read (`windowReads_members`).
* `MemberForm.OpensOwn`: a checked member whose files are ready for its commitment's openings opens only at its own
  files' places (`opensOwn_of_ready`), and files that pass the files check are ready for them (`FilesPass.ready`),
  so at a window the verifier sets up the vocabulary's openings hold wherever each member's deployed commitment
  opens (`MembersSetUp.opensWith`).
* `MemberForm.ClaimReads`: what the claim reads at the windows the verifier sets up (`claimReads`): each
  window laid out with its own share and units, and the committed files read at the claim by the window's members
  in order, at the seats' traces side by side (`readsWith_of`).
* `MemberForm.Sound`: SPOT's soundness and remote's over windows of several members, at the three facts above as
  proved of the per-member form (`spotSound_form`, `soundnessAt_form`), with what is left as their premises.
-/
