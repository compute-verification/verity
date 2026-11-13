[2026-11-13T18:11Z] 396ce0fadf49
PROBE, Fri 13 Nov 10:11 AM PST. **#9474 is carried clean across all 28 moves, a pure carry with nothing of mine on top, and the branch moved by fast-forward to `2d8e9bac34c759127a484f03dbea648e846e9329`. Every test passes at it except one suites group that's still running (its first launch never got a slot).**

- **New head** (RAN): `2d8e9bac34c759127a484f03dbea648e846e9329`. `git push -u origin cursor/nci-record-check-6111` printed `b23c853a35b..2d8e9bac34c`, a fast-forward, with no force. It contains `main` 43e3af9c16d.
- **restack.py** (RAN, cpu-9, a clone of `main` 43e3af9c16, run r20261113-163506-0ea2):
  - `tools/move/restack.py origin/cursor/nci-record-check-6111 --dry-run --kept --json` gave status `clean`, result `2d8e9bac34c7`, 28 crossed, skipped `[]`. The plain merge, for comparison, `conflicted` in 4 files. All 28 generated checks were already kept on cpu-9.
  - The job had to supply two things the node's clone lacks: GitHub as `origin` (the clone's own remote is the shipped tree), and this checkout's own git identity as `GIT_AUTHOR_*`/`GIT_COMMITTER_*` in the job's environment. Without them the tool stopped at `Author identity unknown` (r20261113-163301-d66f). No git config was changed.
  - **No `--push-to` run**: the job bundled the dry run's result commits into `$OUT`, and I fetched them here. A push of that same commit to a new branch, `cursor/nci-record-check-carry-6111`, was refused with HTTP 403 from this VM. So I fast-forwarded the PR branch to the tool's result itself, as the brief's step 2 ends. The pushed commit is the tool's commit, byte for byte (READ: `git cat-file`, its parents a79fd44a8d and 43e3af9c16).
- **Pure carry** (RAN: `git log origin/main..2d8e9bac34` is the tool's commits above b23c853a35; I committed nothing).
- **Test 1, retired names** (RAN locally, via the queue's own `research.queue.map_retires`, each map read at its move's base, with `Retired.named` on `own_text(origin/main, 2d8e9bac34)`). The counts per map are 0 for 24 maps. Four maps name something, all in the lines the branch adds:
  - `terms_map.toml` (833703ef0e1d): 2, `window` and `windows`.
  - `period_map.toml` (9fa40840bc5c): 1, `windows`.
  - `period_net_map.toml` (ea6f83438142): 1, `window`.
  - `protocol_groups_map.toml` (0932f6acd0a8): 0, where the old head had 3.

  The 4 left are word maps matching substrings in two names `main` itself still holds, so the scripts leave them on purpose and I left them (READ: `main`'s tree):
  - `from verity.commitments.windows.interface import circuit_bytes` in `served_record_case.py`. `verity/commitments/windows/` exists on `main`, and `main`'s own tests import `verity.commitments.windows.interface`.
  - The registration log's key `"window"`, in `log[-1]["window"]`: one line in `served_record_case.py` and one in `test_nci_record_check.py`. `main`'s `record_difftest_vectors.py` reads the same key (`tr["log"][at]["window"]["circuit"]`).

  For comparison, the old head named 7, matching the queue's line exactly.
- **Test 2, the tree** (RAN): `git diff --stat origin/main...2d8e9bac34` shows the 8 files and nothing else, at their moved paths: `verity/protocols/accounting/certifier/verifier/network/{pyproject.toml,tests/test_nci_record_check.py}`, and under `verity/protocols/policy/inference_only/verifier/nci/`: `README.md`, `lean/NciCheck/RecordCheckMain.lean`, `lean/lakefile.toml`, `lean/lean-audit.json`, `lean/scripts/nci_record_reference.py`, `lean/scripts/served_record_case.py`. Read as data, `lean-audit.json` differs from `main`'s only by `exempt/NciCheck.RecordCheckMain`; its 43-line text diff is the lock merge driver's key order.
- **Ratchet gate** (RAN locally): `adds 3 retired-term site(s)`, the same three as you ruled on, at the moved paths: `RecordCheckMain.lean:3` and `:15`, `lakefile.toml:57`.
- **Test 3, lean try**, r20261113-170845-7334, rc 0 (RAN): `lean exit 0 in 3 s`; last line `axiom probe: no theorems in it`.
- **Test 4, the package's Lean check** (`check.py --build verity/protocols/policy/inference_only/verifier/nci/lean`; no `security*` package), r20261113-170817-cfef, rc 0 (RAN): `AUDIT: PASS; reports in …/r20261113-170817-cfef/lean-audit`. The package line: `PASS 1730 declarations in 17 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees`. The record difftest: `670 cases compared, 0 differences`.
- **Test 5, suites** `--changed 43e3af9c16d --quick`, 60 suites in five groups (RAN):
  - r20261113-171005-4142: `6 suites: 6 passed (0 cached), 0 failed`.
  - r20261113-171039-9bbb: `12 suites: 12 passed (0 cached), 0 failed`. It includes the network suite, `94 passed` with none skipped, with `test_nci_record_check.py` at its new path run whole, the compiled command's test included.
  - r20261113-171101-ae84: `1 suites: 1 passed (0 cached), 0 failed` (verity-vllm).
  - r20261113-171158-3249: `20 suites: 20 passed (0 cached), 0 failed`.
  - **Group 4 (21 suites) is pending.** r20261113-171143-9b21 waited 37 min for a check slot on cpu-8 without starting; I cancelled it and relaunched as r20261113-181008-a458 (cpu-3). At the old head this group took about 4 min of test time. I'll send its last line the moment it ends.
- **Test 6, the PR body** (RAN: `research queue edit 9474`): paths are in today's layout, and "Run and read" has every run above with its id and last line, at the new head. Group 4 is marked pending.

In flight: r20261113-181008-a458.

