import FlockSoundness.Discharge.Exec.Event
import FlockSoundness.Discharge.Exec.SessionL
import FlockSoundness.Discharge.Exec.Setup
import FlockSoundness.Discharge.Exec.Tables
import FlockSoundness.Discharge.Exec.Headline
import FlockSoundness.Discharge.Exec.CompiledL
import FlockSoundness.Discharge.Exec.CompiledTableL
import FlockSoundness.Discharge.Exec.SessionSoundL
import FlockSoundness.Discharge.Exec.SessionAccL
import FlockSoundness.Discharge.Exec.SessionBatchL
import FlockSoundness.Discharge.Exec.SessionBatchAccL
import FlockSoundness.Discharge.Exec.FlockBatchedL
import FlockSoundness.Discharge.Exec.FlockBatchedAccL
import FlockSoundness.Discharge.Exec.StrictCRL
import FlockSoundness.Discharge.Exec.FlockLinkL
import FlockSoundness.Discharge.Exec.FlockLinkedL
import FlockSoundness.Discharge.Exec.FlockPublicL
import FlockSoundness.Discharge.Exec.E2EL
import FlockSoundness.Discharge.Exec.BindingE2EL
import FlockSoundness.Discharge.Exec.HeadlineL
import FlockSoundness.Discharge.Exec.Live
import FlockSoundness.Discharge.Exec.Merkle
import FlockSoundness.Discharge.Exec.Retained
import FlockSoundness.Discharge.Exec.Decode
import FlockSoundness.Discharge.Exec.Emit
import FlockSoundness.Discharge.Exec.FramesLigerito
import FlockSoundness.Discharge.Exec.FramesRep
import FlockSoundness.Discharge.Exec.Rounds
import FlockSoundness.Discharge.Exec.Verdict
import FlockSoundness.Discharge.Exec.Shape
import FlockSoundness.Discharge.Exec.Pts

/-!
# R11: flock-verify's acceptance and the audit as Lean defines it

* `Event`: flock-verify's verdict on a session directory (`Accepts`), the live audit whose outcome carries it
  (`auditLive`), and R11 at one strategy pair (`ExecRefines`);
* `SessionL`: the batched session under a leaf scheme (`tableAfterL`, `sessionBL`, `batchedSessionL`), and over any
  table run (`sessionG`, `batchedSessionG`);
* `Setup`: flock-verify's acceptance sets up each table with `setupH` (`accepts_setupH`, `setupH_public`);
* `Tables`: each accepted table is one accepting run of `tableAfterL` at `leaf512` (`accepts_tables_hm96`);
* `Merkle`: each table `setupH` accepts commits with the tags' leaf scheme (`setupH_find`), so with `hm96Sha512` under
  every `hm96Rows` tag pinned today (`setupH_hm96`), and `accepts_tables_hm96` without `hms`
  (`accepts_tables_hm96_of_leaf`);
* `Retained`: every `hm96Rows` tag pinned today retains round bytes, so every round of a record flock-verify accepts
  carries its framed bytes (`accepts_retained`), as `liveTable`'s rounds do, and every stream satisfies
  `Refine/Frames.lean`'s `AllRetained` (`accepts_allRetained`);
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
* `Live`: R11 for one-table sessions of the live coin server, with `bad = ⊥`, given `Decodes` (`execRefines_one`), and
  the headline at flock-verify's verdict with no `bad` term (`Prog.flock_headline_live`);
* `Decode`: two pieces of `DecodesAll`: a run on two lists is a run on one tape with those messages and coins
  (`Plays.toT`), and the accepted proof's cap is a function of `root_B` (`capOfRoot`, `verify_cap_hm96`);
* `Emit`, `FramesLigerito`, `FramesRep`: the frames a run emits, round by round (`Emitted`, `chunks`), for each phase
  of a rep and for the whole rep (`zerocheck_emitted`, `lincheck_emitted`, `ringSwitch_emitted`, `ligerito_frames`,
  `rep_rounds`), on streams whose rounds all retained their bytes;
* `Rounds`: so, when flock-verify accepts under tags that retain rounds, round `k` of each rep's stream retained the
  bytes of chunk `k` of the rep's frames (`RepRounds`, `accepts_rep_rounds`);
* `Verdict`: the openings and salts the proof files decode to (`opnOf`), and the verdict part of `Decodes`: at
  flock-verify's acceptance, each table's salted verdict at `capOfRoot root_B[j]` and `opnOf` accepts on the reps' runs
  (`table_verdict_hm96`, `accepts_verdict_hm96`);
* `Shape`: where a rep's coins fall among its messages is the game's: every run of a rep follows a static spine read
  off `m`, `k_log`, the number of claims and the schedule (`repShape`, `repC_spine`, `repC_kinds`), so two runs with the
  same messages and coins are runs on the same tape (`repC_tape`);
* `Pts`: a `setupH` statement's link points, `mPtsOf` of the circuit's `g` and the units it proves (`setupH_facts`),
  for a drawn session the draw's proved units (`setupH_mPts_drawn`, `setupH_mPts_of_proved`), so equal at two draws
  that prove as many units (`setupH_mPts_eq`).
-/
