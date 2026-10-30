import Proofs.Flock.Soundness.Discharge.Exec.Event
import Proofs.Flock.Soundness.Discharge.Exec.SessionL
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.Discharge.Exec.Tables
import Proofs.Flock.Soundness.Discharge.Exec.Headline
import Proofs.Flock.Soundness.Discharge.Exec.CompiledL
import Proofs.Flock.Soundness.Discharge.Exec.CompiledTableL
import Proofs.Flock.Soundness.Discharge.Exec.SessionSoundL
import Proofs.Flock.Soundness.Discharge.Exec.SessionAccL
import Proofs.Flock.Soundness.Discharge.Exec.SessionBatchL
import Proofs.Flock.Soundness.Discharge.Exec.SessionBatchAccL
import Proofs.Flock.Soundness.Discharge.Exec.FlockBatchedL
import Proofs.Flock.Soundness.Discharge.Exec.FlockBatchedAccL
import Proofs.Flock.Soundness.Discharge.Exec.StrictCRL
import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkL
import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkedL
import Proofs.Flock.Soundness.Discharge.Exec.FlockPublicL
import Proofs.Flock.Soundness.Discharge.Exec.E2EL
import Proofs.Flock.Soundness.Discharge.Exec.BindingE2EL
import Proofs.Flock.Soundness.Discharge.Exec.HeadlineL
import Proofs.Flock.Soundness.Discharge.Exec.Live
import Proofs.Flock.Soundness.Discharge.Exec.Merkle
import Proofs.Flock.Soundness.Discharge.Exec.Retained
import Proofs.Flock.Soundness.Discharge.Exec.VerifyK
import Proofs.Flock.Soundness.Discharge.Exec.VerifyKJ
import Proofs.Flock.Soundness.Discharge.Exec.Decode
import Proofs.Flock.Soundness.Discharge.Exec.Emit
import Proofs.Flock.Soundness.Discharge.Exec.FramesLigerito
import Proofs.Flock.Soundness.Discharge.Exec.FramesRep
import Proofs.Flock.Soundness.Discharge.Exec.Rounds
import Proofs.Flock.Soundness.Discharge.Exec.Verdict
import Proofs.Flock.Soundness.Discharge.Exec.Shape
import Proofs.Flock.Soundness.Discharge.Exec.Pts
import Proofs.Flock.Soundness.Discharge.Exec.LiveOn
import Proofs.Flock.Soundness.Discharge.Exec.LawDraw
import Proofs.Flock.Soundness.Discharge.Exec.SessionLDecisive

/-!
# R11: flock-verify's acceptance and the audit as Lean defines it

The acceptance lemmas (`accepts_*`) are about public-output statements only: each takes `hh : hiddenOutputs = false`,
since with hidden outputs flock-verify sets up its tables with `setupHidden`, about which nothing here is proved.

* `Event`: flock-verify's verdict on a session directory (`Accepts`), the live audit whose outcome carries it
  (`auditLive`), and R11 at one strategy pair (`ExecRefines`);
* `SessionL`: the batched session under a leaf scheme (`tableAfterL`, `sessionBL`, `batchedSessionL`), and over any
  table run (`sessionG`, `batchedSessionG`);
* `Setup`: under a tag without hidden outputs, flock-verify's acceptance sets up each table with `setupH`
  (`accepts_setupH`, `setupH_public`);
* `Tables`: under such a tag, each accepted table is one accepting run of `tableAfterL` at `leaf512`
  (`accepts_tables_hm96`);
* `Merkle`: each table `setupH` accepts commits with the tags' leaf scheme (`setupH_find`), so with `hm96Sha512` under
  every `hm96Rows` tag pinned today (`setupH_hm96`), and `accepts_tables_hm96` without `hms`
  (`accepts_tables_hm96_of_leaf`, without hidden outputs);
* `Retained`: every `hm96Rows` tag pinned today retains round bytes, so, without hidden outputs, every round of a
  record flock-verify accepts carries its framed bytes (`accepts_retained`), as `liveTable`'s rounds do, and every
  stream satisfies `Refine/Frames.lean`'s `AllRetained` (`accepts_allRetained`);
* `VerifyK`: what the acceptance walkers read of `verify`, at the verdict with its digest comparisons read through
  `keep` (`AcceptsK`, `verifyK_ok_tables`, `decodeWithK_points`, `decodeWithK_allRetained`), `verify`'s at `keep := id`;
* `CompiledL`: compiled soundness of one table under any leaf scheme (`table_sound_compiledL`), the salted form of
  `table_sound_compiled`;
* `CompiledTableL`: the same for any level-0 table (`table_sound_compiled_of_tableL`), the reps' terms independent of
  it (`advRTL_eq`);
* `SessionSoundL`, `SessionAccL`, `SessionBatchL`, `SessionBatchAccL`: `Session.lean`, `SessionAcc.lean`,
  `SessionBatch.lean` and `SessionBatchAcc.lean` over a leaf scheme: each table of the salted batched session is
  knowledge sound (`session_knowledge_soundL`, `session_knowledge_sound_accL`, `joint_le_BL`, `joint_le_accBL`);
* `FlockBatchedL`, `FlockBatchedAccL`: the batched audit's analysis over the salted session (`analysisBL`,
  `analysisBEL`), its knowledge term discharged (`flock_batched_knowledgeSoundEL`);
* `StrictCRL`: the salted finders under `SHA512CRStrict`: the compiled theorem (`table_sound_compiled_strictL`) and the
  knowledge term, at most `StrictCR.ksAvgStrict`, the unsalted closed form (`ksAvgBE_le_strictL`);
* `FlockLinkL`, `FlockLinkedL`, `FlockPublicL`: the link theorem over the salted session
  (`flock_batched_linkSoundEL`), and with it the count curve and drawn-unit bound at the public transcript
  (`flock_batched_count_publicL`, `flock_batched_drawn_publicL`); the binding (`ValueBinding`), the value layer's
  bound (`linkBoundE`) and the public wires' facts are the unsalted ones;
* `E2EL`, `BindingE2EL`: the end-to-end theorems over the salted session (`flock_e2e_countL`, `Prog.flock_e2e_countL`,
  `flock_e2e_count_hm96L`, `flock_e2e_drawn_hm96L`);
* `Headline`: the headline carried to the live audit through `ExecRefines` (`Prog.flock_headline_exec`,
  `flock_headline_exec_L`);
* `HeadlineL`: the headline over the salted session (`Prog.flock_headline_L`, the bound of `Prog.flock_headline`), and
  at the executable (`Prog.flock_headline_execL`), where only `hRef` stands for R11;
* `Live`: R11 for one-table sessions of the challenger, with `bad = ⊥`, given `Decodes` (`execRefines_one`), and
  the headline at flock-verify's verdict with no `bad` term (`Prog.flock_headline_live`);
* `Decode`: two pieces of `DecodesAll`: a run on two lists is a run on one tape with those messages and coins
  (`Plays.toT`), and the accepted proof's cap is a function of `root_B` (`capOfRoot`, `verify_cap_hm96`);
* `Emit`, `FramesLigerito`, `FramesRep`: the frames a run emits, round by round (`Emitted`, `chunks`), for each phase
  of a rep and for the whole rep (`zerocheck_emitted`, `lincheck_emitted`, `ringSwitch_emitted`, `ligerito_frames`,
  `rep_rounds`), on streams whose rounds all retained their bytes;
* `Rounds`: so, when flock-verify accepts under tags that retain rounds and have no hidden outputs, round `k` of each
  rep's stream retained the bytes of chunk `k` of the rep's frames (`RepRounds`, `accepts_rep_rounds`);
* `Verdict`: the openings and salts the proof files decode to (`opnOf`), and the verdict part of `Decodes`: at
  flock-verify's acceptance under a tag without hidden outputs, each table's salted verdict at `capOfRoot root_B[j]`
  and `opnOf` accepts on the reps' runs (`table_verdict_hm96`, `accepts_verdict_hm96`);
* `Shape`: where a rep's coins fall among its messages is the game's: every run of a rep follows a static spine read
  off `m`, `k_log`, the number of claims and the schedule (`repShape`, `repC_spine`, `repC_kinds`), so two runs with the
  same messages and coins are runs on the same tape (`repC_tape`);
* `Pts`: a `setupH` statement's link points, `mPtsOf` of the circuit's `g` and the units it proves (`setupH_facts`),
  for a drawn session the draw's proved units (`setupH_mPts_drawn`, `setupH_mPts_of_proved`), so equal at two draws
  that prove as many units (`setupH_mPts_eq`);
* `LiveOn`: `Live`'s R11 and headline with the decoder right at the law's draws only (`DecodesOn`, `execRefines_on`,
  `Prog.flock_headline_live_on`), which `DecodesAll` implies (`decodesOn_of_all`, `simAudit_eq_on`);
* `LawDraw`: every draw of the live stratified law has `∑ s, k s` units, or is every unit where the run reads past the
  source's bytes (`execOS_draw_card`, `execOS_draw_card_or`), so a draw file with no closure sets up at
  `mPtsOf c.g (∑ s, k s)` or `mPtsOf c.g n` (`setupH_mPts_execOS`, `setupH_mPts_execOS_or`);
* `SessionLDecisive`: some run of the salted batched session accepts (the all-zero run) and, on a plan with a table,
  some run rejects, and every message type is inhabited, so the session and the audit over it are decisive
  (`batchedSessionL_decisive`, `audit_tile_decisive`, `audit_tile_decisive_verdict`). These are existential over the
  coins; completeness of the reference prover stays open.
-/
