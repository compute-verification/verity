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

