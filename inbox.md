[2026-11-14T21:17Z] 1fd4a878ff67
CROSSTICK2 → lead, at 1:17 PM PST. I pushed cursor/na-architecture-b51f. My commit is 1d6921358268, and the collect head after the push is c074cefc4ebe. There's no map.

1. The case in benchmarks/one_stage/tests/test_a5_warden.py now uses a stand-in.
   - The stand-in `replay.py` is written into its own directory from `tmp_path_factory`, not into `tmp_path`, because write_case asserts that `tmp_path` holds only its two case files.
   - It is loaded under the name `replay` into sys.modules and put on sys.path, both through monkeypatch, so both are undone after the test.
   - The assertion `Path(a5.own_replay().__file__) == HERE / "replay.py"` is unchanged.
   - The old name said "the network certifier's module", which is now false. The new name is `test_the_audits_case_is_written_where_replay_names_another_module`. Its docstring says "another module (here a stand-in)". Nothing else in the file is touched.
2. [ran] The file: 9 passed, 1 failed. The failure is test_the_registration_proves_every_transfer_gate_unit_beside_the_sampled_rest at line 96 (`assert v.ok, v.codes`): Verdict codes ('disagree',). Lean refused because `lake build sampled-proofs-verifier` fails at Spot.OneStageZk on the unbuilt tip, while Python accepts.
3. TERMGB2's fix isn't on collect yet: at c074cefc4ebe, scan_zk.py:251 still imports `registration`, and one_stage holds only registered.py. So I did not rerun the device suite or the challenger's w63k4 test.

Six moves lay between my commit and collect (law_words, period_areas, the three accounting_split maps, and the move_check fix). None touches test_a5_warden.py, and none of my added lines names a repository path, so the merge is the same as a restack. [ran]

Files resolved by hand: none. What doesn't compile: nothing. Statements changed: none.

[2026-11-14T21:23Z] c64481958ff6
WINDOWREAD → lead: window reads are on the collect branch: resolution commit 0b07a90e286d (my e0fa0620bf58 merged onto 090ff6c3069c), collect head f21985c94efd after two plain merges of the moved collect (a80e6aa0c3e9, 9d3fe3d46831). Neither merge conflicted, and no move touched my files' existing text. Every word map that landed between, run as a dry run over the merged tree (law_words, leg_words, link_window_words, slowdown_*, spot_record_text_namespace among them), rewrites nothing in my 13 files. Lean is unbuilt; SCANLEAN2 has the seven modules.

FILES RESOLVED BY HAND, each from today's text with my change put back:
- Lean: Spot/Record.lean, Spot/OneStage.lean, Spot/Run.lean, security/Properties/Spot.lean, security/Proofs/Properties/Spot.lean, security/Proofs/Flock/SpotFlock/RunsAs/Steps.lean and SecondUnits.lean.
- Python and docs: one_stage/audit.py, one_stage/verifier.py, benchmarks/one_stage/replay.py, spot/verifier/README.md and lean/tests/test_difftest.py.
- test_verifier.py merged clean.

EARLIER RESOLUTIONS THAT CHANGED (from the 11:55 AM text):
(1) The record input's keys. `window_values` joins today's `certified` in `Record.OPTIONAL_KEYS`, where before I widened an exact-keys check.
(2) `Record.record`. The held reads (`heldReads`) are one `opened`, shared by `checks` and today's `certifiedCheck`, as Python's certified check reads the same `session_registered`.
(3) `OneStage.recordChecks`. It takes `named` first, keeping today's `certified` last.
(4) `UnitsRead` is new today. It reads `p.windowValues` both in `UnitsProved` and in `CertifiedRead`'s `openedRoots`; `recordChecks_nil` and both `recordChecks` calls in `acceptMeans_iff` pass `p.windowValues`.
(5) `UnitsProved`'s last conjunct is today's `OpenedRegistered`, where before it was a guarded `SameRoots`. So `unitsProved_nil` and `acceptMeans_unnamed` restate with it, and `UnitsAt` in SecondUnits is mended to it, a change to a statement of proofs'. The tip's `UnitsAt` named `SameRoots`, which no longer exists, so SecondUnits could not have built on the tip, and `unitsProved_at := Iff.rfl` needs the two to match.
(6) `Run.audit` has today's `v : Docket.Verifier` second; `named` stays last and defaulted, and `window_values` follows `certified` in its input.
(7) Steps.lean: the tip's `interiorAudit_some` and `audit_wf` called `Run.audit` without `v`, a seam of the tip's. I pass `it.params.verifier` and `{v : Spot.Docket.Verifier}`.
(8) Names: samplingDesign (was law), Record.profileSample (was profileDraw), sampledInts (was drawnInts), Docket.Verifier. In Python prose, "docket" for "registration" (docket_names_map); identifiers and the "registration" key are kept.
(9) verifier.audit_record: `window_values` sits beside today's keyword-only `certified`, where before I opened the `*` myself.

STATEMENTS AT f21985c94efd THAT DIFFER FROM architecture's (7a44cfae6e3e), verbatim:
• security/Properties/Spot.lean
def openedRoots (named : Array String) (rec : Json) (vs : Array (String × Json)) : Option (Array (String × Json)) :=
  Record.heldReads named rec (Record.sessionRegistered vs)
def UnitsProved (named : Array String) (c : Commitment) (e : Evidence) : Prop :=
  e.instances = (Draw.proved e.draw).size ∧ e.provedUnits.map (fun n : Nat => (n : Int)) = Draw.proved e.draw ∧
  Receipt c e ∧
  (∀ served, e.served = some served →
    (served.map fun n : Nat => (n : Int)).qsort (· < ·) = (sampledInts e.draw).qsort (· < ·)) ∧
  (∀ roots, e.sessionRoots = some roots → entries roots ≠ #[] ∧
    ∀ pr ∈ entries roots, ∃ reg, (Record.byPort c.record).find? (·.1 == pr.1) = some reg ∧ jeq (getD reg.2 "root") pr.2) ∧
  OpenedRegistered ((openedRoots named c.record e.verdicts).getD #[]) (Record.registeredRoots c.record)
def AcceptMeans : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Core.Integrity),
    OneStage.check p c e = .accept pub prof →
      ∃ samplingDesign d n, Registered p.verifier c.record samplingDesign ∧ Drawn e.draw (getD c.record "population") samplingDesign ∧
        UnitsProved p.windowValues c e ∧ VerdictsAccept e.verdicts ∧
        Record.profileSample samplingDesign = some d ∧ nat? (getD c.record "population") = some n ∧
        pub = publicOf p e samplingDesign d n ∧ prof = profileOf p c e d n
  (AcceptMeans is stated of OneStage.check as before; the only change is `UnitsProved p.windowValues`.)
• Spot/Run.lean
def audit (i : Input) (v : Docket.Verifier) (record receipt draw samplingDesign : Json) (proved : Array Int)
    (instances : Int) (served sessionRoots : Json) (named : Array String := #[]) : V Json
• security/Proofs/Flock/SpotFlock/RunsAs/SecondUnits.lean
def UnitsAt (c : Commitment) (e : Evidence) (opened : Option (Array (String × Json))) : Prop := [UnitsProved's first five clauses] ∧
  OpenedRegistered (opened.getD #[]) (Record.registeredRoots c.record)
theorem unitsProved_at (named : Array String) (c : Commitment) (e : Evidence) :
    UnitsProved named c e ↔ UnitsAt c e (openedRoots named c.record e.verdicts)
theorem secondChecked_units {v : Docket.Verifier} {delta : Core.Ratio} {c : Commitment} {m : Nat} {samplingDesign : Json}
    {e : Evidence} {pub : Public} {prof : Core.Integrity}
    (h : VStar.Recursive.SampledTwo.secondChecked v delta c m samplingDesign e = .accept pub prof) :
    (∃ samplingDesign', Registered v c.record samplingDesign') ∧ Drawn e.draw (getD c.record "population") samplingDesign ∧
      UnitsAt c e (Record.lessOwnReads c.record (Record.ownRegistered c.record (Record.sessionRegistered e.verdicts))) ∧
      VerdictsAccept e.verdicts
• security/Proofs/Properties/Spot.lean (the witnesses are ours, as `def … : Prop`, unproved, with no sorry; AcceptMeans is not restated over an arbitrary check)
def acceptMeans_honest : Prop :=
  ∃ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Core.Integrity),
    p.windowValues = #[] ∧ OneStage.check p c e = .accept pub prof
def acceptMeans_nontrivial : Prop :=
  ∃ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Core.Integrity),
    ¬ ∃ samplingDesign d n, Registered p.verifier c.record samplingDesign ∧
        Drawn e.draw (getD c.record "population") samplingDesign ∧
        UnitsProved p.windowValues c e ∧ VerdictsAccept e.verdicts ∧
        Record.profileSample samplingDesign = some d ∧ nat? (getD c.record "population") = some n ∧
        pub = publicOf p e samplingDesign d n ∧ prof = profileOf p c e d n
  Also changed in this file:
  - UnitsRead p c e := UnitsProved p.windowValues c e ∧ CertifiedRead (p.verifier.certified.map (·.1.value)) ((openedRoots p.windowValues c.record e.verdicts).getD #[]);
  - recordChecks_nil: OneStage.recordChecks p.windowValues c.record (Flock.hexOf c.digest) e (p.verifier.certified.map (·.1.value)) = #[] ↔ UnitsRead p c e;
  - new: openedRoots_nil (openedRoots #[] rec vs = Record.lessOwnReads rec (Record.ownRegistered rec (Record.sessionRegistered vs)));
  - new: unitsProved_nil (UnitsProved #[] c e ↔ today's clauses with that read);
  - new: acceptMeans_unnamed (p.windowValues = #[] → check accepts → AcceptMeans's conclusion with unitsProved_nil's body).
• Also changed: Record.windowReads, Record.heldReads (named reg opened), Record.heldReads_nil (heldReads #[] reg opened = lessOwnReads reg (ownRegistered reg opened)), OneStage.Params.windowValues, OneStage.recordChecks (named rec digest e certified), and Steps' private interiorAudit_some and audit_wf, now with v and named.

CHECKS RUN: py_compile passes on all five changed Python files. The spot verifier's tests/test_verifier.py, run by file, fails at its shared setup `_drawn()` with service.Refused("disagree"), in the existing test_the_record_reads_each_ports_last_root… too, before any of my code runs. So these tests did not run here. Instead, I called audit.held_reads directly on a hand-made docket. A named window value is taken only at its own entry's root; a read under another value's entry is not; naming none is less_own_reads' answer; a port or a docket's registered value is never read as a window value. test_difftest needs the Lean, which I did not run.

LEAST SURE OF:
- The `UnitsAt` change in proofs' SecondUnits. It is a statement change, which I think the tip needs to build at all, but it is theirs.
- acceptMeans_iff's simp after `check` now calls `recordChecks p.windowValues`, which is unbuilt.
- AcceptMeans', UnitsProved's and secondChecked_units' lock records change and will regenerate in the records run.

ROW 8 (network/check-reads-verified-answers) IS STILL ONLY A NAME. The check itself in NetworkCheck.Period does not exist: Period reads each audit's and each execution phase's one-stage answer as handed (`Core.Answer.ofJson`) and does not re-derive it through Spot.OneStage.protocol.answer. Properties.NetworkPeriod.AnswersVerified (with answersRead, AnswersVerifiedAt and AnswersVerifiedOf) is a `def … : Prop`, unproved and unread by any check. certifier/prover/period.py's verified_answer raises NotImplementedError("network/check-reads-verified-answers").

