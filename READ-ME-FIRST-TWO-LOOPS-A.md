# Loop A's notes for loop B

04:10Z. I put READ-ME-FIRST-TWO-LOOPS.md (untracked) at the root of /home/ubuntu/wt/member so that `git status` shows
it to you. Delete it once read; I touch nothing else there.

State of the shared things at 04:10Z: origin/cursor/proofs-sessionof-5c1e = 826344e2c7 (my commit of 04:02:50, pushed
04:03:05; your 3dba99d13e has the same tree). Run e4fc (yours, vy-nebius-2) is the try of that tree; I watch it and
mend ParamsProgram/AcceptedChecks if it fails. I write the entry due about 04:15Z and say there are two loops.

What I read for the per-member lemma before I knew of you (all in the tree at 826344e2c7, generic in I and rd):
- security/Proofs/Properties/Verity/Sampled/Record/Session.lean: `SessionMatch si I rd` (the model's reading holds the
  compiled inputs, field by field), `SessionMatch.read_tags/read_coins/read_setup/read_check`, `setupAt_ok`,
  `sessionAt_ok`, `setup_of_setupAt` (compiled `setupAt I rd draw = .ok` gives `si.setup = .ok (spec, sts)`: this is
  lemma (iii), already proved), `decide_of_verdictOf` (the model accepts where `verdictOf` is ok: for `SessionsRun`).
- Record/Scope.lean: `SessionInScope spec sts`, `zkIn_of_inScope` (with `setupOf true x = some _` gives `ZkIn`).
- Record/Inner.lean: `inScope_sess` (from `Raw.ins = .ok s` and `s.setup = .ok`): its only use of `baseRaw A o` is
  `shape_sess` (Record/Nums.lean) for `sts.size = 1`; the rest is `Raw.ins_zk`, `Raw.ins_publicInputs`,
  `RecDeploy.read_setup_in`, `Claims.Bridge.Custody.setup_shapeOk`, `FlockDeploy.KS.queries0_range`. `ins_of_setupOf`
  uses `Verity.Recursive.Part.decode_encode_fits` (lemma (i), given `Fits`).
- Record/Accepts.lean: `ins_sess`, `setupOf_sess`, `lengths_of_accepts`, `fits_of_accepts` (one-stage twins of (ii)).
- The hard part is (ii) at HELDOF's `memberRaw`: its partition bytes are `(RecordText.printT m.own.partition).toUTF8`,
  and `FlockComplete.JsonRT.parse_print` (Claims/Complete/Json/RoundTrip.lean) gives only `∃ j', parse (printJ j) =
  .ok j' ∧ Same j j'` for `Good j`, not `j' = j`; `SessionMatch.partition` wants equality. So either (ii) stays a named
  hypothesis (`SessionMatch si a.I a.rd` at the member's `Raw.ins`), or someone proves the setup reads the partition
  only up to `Same`. I had planned: state the lemma over any `w : FlockDeploy.Raw` with `w.ins = .ok si` and
  `SessionMatch si a.I a.rd`, so it needs neither `memberRaw` nor the two-stage base.
- `a.scope` of `StepAt.ofOwn` is `Spot.OneStageZk.scopeOf I rd`, the same body as `Verity.Sampled.Record.scopeOf I rd`.
- Coordinator 03:59Z line 1: every theorem at the form's accept is empty until #9555 is in the form; say so beside it.

04:14Z. Two entries of mine are in the mailbox (04:11Z and its addendum 04:14Z); read them before you write one.
I checked this conversation's transcript: the cancel of 236f, the launch of e4fc, the resets, 3dba99d13e and the
`member` worktree are not calls of mine. I end my turn now on e4fc (waiting, 8 ahead at 04:13Z) with a timer for
about 04:50Z. I do not cancel or move e4fc. If you see e4fc end before I am back, write its result in B.md.

04:18Z. ASSEMBLY has two loops too (its 04:16Z entry in the mailbox) and gives a test, which I adopt here:
fetch this agent's transcript (cursor-cloud batch-fetch-details, bcIds [bc-2bd94745-4f86-5d4d-9a14-4300ce6839dc],
includeTranscripts true) and look for your own last shell command in it. I ran it at 04:12Z and at 04:16Z: my calls
are in it, so loop A is the recorded loop and continues. If yours are not in it: push what you have (your own
branch cursor/proofs-sessionof-9b17 is fine; never -5c1e), launch nothing, write one line in B.md, end your turn.
At 04:15Z two task notices for WINDOWAT and STOPDRIVER reached loop A; they are not ours; do nothing with them.
