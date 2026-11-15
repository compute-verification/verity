[2026-11-15T11:25Z] 7602fb19f7ab
ROWSPROTO2, TOP-424 env-reads mend PUSHED (3:25 AM PST). Branch cursor/na-guard-env-b51f, one commit 0e43de06ab4 on architecture a4dfd60dc5d0. Nothing pushed to architecture or collect, and no allowlist lines.
Option (a), cwd at the tree, with no environment read. The lint counts .get, [...] and `in`, and doesn't count passing os.environ whole. I chose cwd over setting PYTHONPATH to the tree alone because the suite runner's tracer rides on the child's PYTHONPATH: replacing it would leave the child's reads unchecked (tools/check/suites.py, "a PYTHONPATH without the tracer: what they read is unchecked"). With cwd, the child keeps the caller's environment whole, and `-m` puts the tree first on its path. Every Path argument is made absolute, so a caller's relative run directory still resolves. Record paths (`--public NAME`, which are str) pass unchanged.
There is one helper, served_session.guard_argv, which replaces guard_env. certifier_piece and test_served_session use it.
Changed lines:
served_session.py:165-168
 before: def guard_env(): return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(TREE), os.environ.get("PYTHONPATH")]))}
 after:  def guard_argv(*args): return [sys.executable, "-m", LEAK_GUARD, *(str(a.absolute()) if isinstance(a, Path) else a for a in args)]
served_session.py release
 before: subprocess.run([sys.executable, "-m", LEAK_GUARD, str(out), ...], capture_output=True, text=True, env=guard_env())
 after:  subprocess.run(guard_argv(out, quarantine, *gates, "--hold", hold, "--release-to", out, "--report", out / "leak-guard.json"), capture_output=True, text=True, cwd=TREE)
certifier_piece.py:956-961 guard
 before: captured([sys.executable, "-m", "benchmarks.network_traces.leak_guard", str(d), ...], env={**os.environ, "PYTHONPATH": ...os.environ.get("PYTHONPATH")...})
 after:  captured(SS.guard_argv(d, quarantine, ..., "--report", quarantine / "piece-guard.json"), cwd=TREE)
tests/test_served_session.py:286
 before: [sys.executable, "-m", SS.LEAK_GUARD, str(published), ...], env=SS.guard_env()
 after:  SS.guard_argv(published, q, ...), cwd=SS.TREE
Counts, with ci's Python path (no repository root on any subprocess's path) and fixtures fetched:
- tests/lint/ (all 8 files): 62 passed, 1 failed. test_repo_env_reads passes; on the tip it fails with the two reads. The one failure, test_repo_silent_excepts::test_no_new_silent_excepts, fails identically on the tip. It is not ours: four adapters/torch files (tests/frontend/test_derive_negative.py; verity_torch/frontend/rules/family.py, reference.py, vocab.py), 1 found, 0 allowed each, so the torch adapter's lane.
- tests/test_repository.py + tests/test_boundaries.py: 45 passed. test_boundaries alone: 28 passed, the same as the tip's 28.
- test_certifier_piece + test_served_session + test_wipe_gate + test_certifier_term: 133 passed (133 before).
- py_compile ok; ruff on the 3 files: 65 findings against the base's 68, none new.
- The child run from a directory that is not the repository root: from /home/ubuntu/gx with relative paths, SS.release(Path('out'), Path('hold'), Path('q'), None) gave rc 0, "clean ... 1 of 1 held released". The old form, `-m` without cwd, fails there with "No module named 'benchmarks'".

[2026-11-15T11:46Z] 8097709071e8
ONSITEMAIN2 to network-accounting lead, Sun 15 Nov 3:46 AM PST. The MonitoringPeriodLeakAccepted pin is mended: branch cursor/na-onsite-rests-on-b51f, commit 0bd0d494aeeaeef8ba2df87c2af5aa5a648657f9, cut from origin/architecture 54e1b5c0f933. Pushed only there; nothing to architecture or collect.

1. Only moved names; nothing in substance changed. [read]
- The pin is the lock's digest of the guarantee's statement Prop. onsite_record.statement takes the signature's type, NetTiming.Guarantees.MonitoringPeriodLeakAccepted, and reads its digest among the lock's reads.
- That digest: 41de25b3444487593edcbd858f2daee2767204c6475b0aa1ae05516f315d1772 at 009698aa9c0a (read module Definitions.NetworkCertifier.Guarantees.MonitoringPeriod); e7686773821a0c9646d03a906d16c534d2253ed2a566aedf686d01fccf4a1e36 at the tip (read module Definitions.Certifier.Guarantees.MonitoringPeriod).
- The guarantee's record, 009698aa9c0a against the tip:
  - signature, type_hash, assumptions ([]), owner, domain ([Claims.Sampling.CommitmentScheme.Binds]) and reads ([]): identical.
  - open: the same four but Verity.Assumptions.TrustedHost → Verity.Assumptions.TrustedDevice.
  - unfolded_hash: 16b44efd… → 893f1985….
- The Lean file security/Definitions/Certifier/Guarantees/MonitoringPeriod.lean is byte-identical at 009698aa9c0a and the tip. Its last changes are the moves b1a80682b1d (certifier_area_map) and aba8e048ae7 (wrong_units_words_map), both below 009698aa9c0a.
- The 009698aa9c0a lock was a record kept from before those moves: its reads still say TrustedHost while the Lean there already says TrustedDevice. ci's close (958aec66a02) regenerated the digest over the definition as it now reads, with the new names.
- Against main's Lean (security/Definitions/NetworkCertifier/Guarantees/MonitoringPeriod.lean), def MonitoringPeriodLeakAccepted's text is identical but for:
  - faultySets → incorrectSets (lines 8 and 29 of the def: hβ, and the count's #incorrectSets);
  - Verity.Assumptions.TrustedHost → Verity.Assumptions.TrustedDevice (line 12: htrust).
  No other token differs.

2. The takes still match. [read] RESTS_ON's takes for it are Claims.SamplingGeneric.Compiled.KS, Claims.Sampling.CommitmentScheme.Binds, Verity.Assumptions.TrustedDevice, hPw, hβ, hcap, hfit, hs, hdz, NetTiming.ScopeAcc and NetTiming.CertifierRegisters.
- They already say TrustedDevice, and they cover each of the tip record's open premises (KS, CertifierRegisters, ScopeAcc, TrustedDevice) and its domain (Binds).
- The test (line 941) compares only the lock's assumptions, [], against takes ∪ STANDING, so no takes key moves.

3. The changed line, the only change, in benchmarks/one_stage/onsite_record.py:576:
-        "statement": "41de25b3444487593edcbd858f2daee2767204c6475b0aa1ae05516f315d1772",
+        "statement": "e7686773821a0c9646d03a906d16c534d2253ed2a566aedf686d01fccf4a1e36",
I took the hash from the tip's lock, not from ci's line.

4. Counts: [ran] benchmarks/one_stage/tests/test_onsite_record.py whole gives 225 passed, 0 failed. py_compile passes on both files. The tests/lint/ and tests/test_repository.py results follow in my next line. TIMER: none.

[2026-11-15T11:47Z] 68873a015070
ONSITEMAIN2 to network-accounting lead, Sun 15 Nov 3:47 AM PST. The rest of item 4 at cursor/na-onsite-rests-on-b51f 0bd0d494aeeaeef8ba2df87c2af5aa5a648657f9:
- [ran] tests/lint/ (8 files) and tests/test_repository.py: 80 passed, 0 failed, rc 0.
- 8 bookkeeping warnings, which warn and don't fail: lane_text x2, retired_terms, ruff, and test_repository's every_path_has_owner, markdown_size_caps, no_report_genre_basenames and owned_paths_exist. None names onsite_record; they come from the tip, not the one-line pin.
- Earlier: test_onsite_record.py 225 passed, and py_compile passes. TIMER: none.

