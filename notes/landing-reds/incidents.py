"""Every red or stall that held `main` since `checked` began (29 Oct 17:34Z), from ci's checkpoint log and drive's queue
thread: when it was named, what failed, its culprit, category, fix, when the fix reached `checked`, why the fix waited, the
main stall it fell in, and the mechanism that makes it a non-event. Times UTC; '~' marks a time read from an hourly log line."""
import csv, json

COLS = ["id", "named", "tip", "what", "culprit", "category", "fix", "fix_on_checked", "fix_wait_h", "why_wait", "stall",
        "mechanism"]
S1, S2, S3, S4, S5, S6 = ("10-30 18:08->10-31 11:35", "10-31 11:35->11-01 14:53", "11-02 02:36->11-03 15:50",
                          "11-04 13:36->11-05 05:47", "11-05 05:47->11-05 10:53", "11-05 10:53->open")
R = [
    ("R01", "10-30 19:11", "8a258310", "repository: test_retired_terms", "?", "lint-ratchet", "fix marked", "", "", "", S1,
     "repository invariants on next's tip before landing"),
    ("R02", "10-31 ~00:00", "5b7e9f32", "6 suites, no error line (OOM rc=137 shards)", "-", "machine", "rerun", "", "", "", S1,
     "slot memory sizing; a no-error-line red reruns once at once"),
    ("R03", "10-31 04:31", "8c8cbfdf", "verity-flock: test_vectors_coverage", "?", "semantic", "fix marked", "", "", "", S1,
     "changed suites on next's tip before landing"),
    ("R04", "10-31 17:09", "9c85ea64", "lean-audit: stale Security records", "#5950 (bisect named #5268 wrongly)",
     "lock-or-records", "#6771", "10-31 21:09", 4.0, "fix needed its own Lean check, run twice", S2,
     "lock records regenerated at landing (regenerate_locks); revert at once"),
    ("R05", "10-31 20:03", "25374685", "verity-pouw-benchmarks: read outside inputs; verity-vllm P8; pous pins; ratchet",
     "#5101 and others", "undeclared-input+lint+lock", "#6434 #6786 #6791 #6793", "11-01 00:03", 4.0,
     "behind #6392's Lean check on next", S2, "read guard and lints on next's tip before landing"),
    ("R06", "11-01 00:36", "-", "drive ticks crash (NFS ENOTEMPTY)", "-", "infra", "#6802 + usercustomize shim",
     "11-01 02:05", 1.5, "", S2, "landing's own code canaried before it runs"),
    ("R07", "11-01 04:25", "a754df09", "repository: #6802's lane-text dates", "#6802 (the crash fix)", "lint-ratchet",
     "#6808 #6809", "11-01 ~08:45", 4.3, "behind stacked Lean checks that failed on #5960 and #5667", S2,
     "repository invariants before landing; fix lane past next's Lean prefix"),
    ("R08", "11-01 12:40", "39adad68", "lean-audit: 3 stale Security/Properties records; verity-check", "?",
     "lock-or-records", "#6821 #6823 (baselined 41be424e instead)", "11-01 14:53", 2.2, "", S2,
     "records regenerated at landing"),
    ("R09", "11-02 05:12", "11e19607", "test_queue records test", "#6389 with #6639", "semantic", "#6893", "11-02 ~07:17",
     2.1, "", S3, "changed suites on next's tip before landing"),
    ("R10", "11-02 12:02", "e373a46d", "repository (#6749), verity-vllm (#5763), ratchet (#6719), research (disk full)",
     "#6749 #5763 #6719", "lint+semantic+machine", "#6911 #6921 #6866", "11-02 19:17", 7.3,
     "GitHub 500 on push 16:09Z; tick restacked #6670's move 1h42m; second loop on #6867", S3,
     "fix lane; drive rings on a refused push; move PRs land alone"),
    ("R11", "11-02 20:46", "2ed02f6b", "verity-check; test_settings; lean-suites undeclared read (compact_stage)",
     "? / #6840", "semantic+undeclared-input", "#6948 #6971 #7009", "11-03 05:55", 7.2,
     "#6971 touches tools/verity/check so left the low-risk lane; #6429 sank the prefix Lean checks; 87-min restack tick",
     S3, "fix lane; changed suites before landing"),
    ("R12", "11-03 05:00", "d156a449", "repository: test_domains; lane-text lint", "#5674 / ?", "semantic+lint",
     "#7062 #7047", "11-03 12:03", 7.0,
     "needed a Lean check; launches 07:15Z never registered (verdict gathering > 30-min wait); stale Lean read (#6928)",
     S3, "fix lane; launch wait above verdict gathering"),
    ("R13", "11-04 03:02", "43d09bf4", "verity-graph CPU budget (flake); repository", "load / ?", "flake+semantic",
     "#7198 #7238", "11-04 04:38", 1.6, "", "11-03 21:23->11-04 06:48", "delete flaky tests at first flake"),
    ("R14", "11-04 05:50", "701c8c00", "flock-rust-tests skipped: group stopped at off-gate circuit-check red", "-",
     "tooling", "hand rerun", "11-04 06:41", 0.8, "", "11-03 21:23->11-04 06:48",
     "gated jobs before off-gate ones in a group"),
    ("R15", "11-04 08:01", "-", "drive ticks crash in ci's pod priority wrapper", "-", "tooling", "shim fix",
     "11-04 08:31", 0.5, "", "11-04 06:48->11-04 08:28", "landing's own code canaried"),
    ("R16", "11-04 12:10", "f0c4b135", "verity-core-composition: 1 h cap", "-", "timeout-cap", "#7271", "11-04 16:34",
     4.4, "never marked until 12:52Z", "11-04 08:28->11-04 13:36", "a cap red reruns uncapped at once"),
    ("R17", "11-04 13:55", "455cf797", "tools/verity/graph: imports circuit_check.known (retired 1 Nov)", "#6903",
     "rename-or-path", "#7286", "11-04 ~15:35", 1.7, "", S4, "path and import check on next's tip before landing"),
    ("R18", "11-04 18:23", "8ae7a2a3", "pytest: #7086/#6890, #7049 lints, test_placement disk floor", "#7086 #6890 #7049 ci",
     "semantic+lint+test-of-tooling", "#7325 #7320 #7319", "11-05 03:17", 8.9,
     "16:36Z tick's sync 2.5 h of NFS merges; then the fixes waited for two Lean checks and the port", S4,
     "fix lane past next's Lean prefix"),
    ("R19", "11-05 07:33", "c6ba5690", "flock-rust-tests: upstream flaky take_reuses_given_buffer", "-", "flake",
     "#7471 (skip)", "11-05 ~09:40", 0.9, "", S5, "rerun a failed task once on another node"),
    ("R20", "11-05 09:16", "c6ba5690", "lean-suites crawled in a 64 GB lane slot", "-", "machine", "#7480; rerun in train slot",
     "11-05 09:58", 0.7, "", S5, "slot sizing in code (#7480)"),
    ("R21", "11-05 12:42", "4bc15be9", "lean-suites: sampled-proofs v2 case; test_check regex; _train_repo", "#6980 vs #7037; #7471",
     "semantic+test-of-tooling", "#7499", "11-05 ~14:00", 1.3, "", S6,
     "lean-suites of the Lean prefix before landing; revert at once"),
    ("R22", "11-05 ~12:42", "4bc15be9", "lean-suites: composition (5 difftests), verity-flock (7 tests)", "#6980, #6996",
     "semantic", "#7399 #7502", "open", None, "read 4.6 h late: one red step's other failing suites unread", S6,
     "machine-read red list per run; lean-suites before landing; revert at once"),
    ("R23", "11-05 14:45", "3b3f745e", "pytest: lane text of #6991", "#6991", "lint-ratchet", "#7508", "11-05 15:19", 0.6,
     "", S6, "repository invariants before low-risk landing (ci lever 3)"),
    ("R24", "11-05 17:00", "aa7843fd", "2 gated suites: #6414's rename (circuit-check pyproject lists, test_infra_paths)",
     "#6414", "rename-or-path", "#7515", "open", None, "", S6, "path check on next's tip before landing"),
]
rows = [dict(zip(COLS, r)) for r in R]
json.dump(rows, open("/tmp/reds/incidents.json", "w"), indent=1)
with open("/tmp/reds/incidents.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, COLS, delimiter="\t")
    w.writeheader()
    w.writerows(rows)
waits = sorted(r["fix_wait_h"] for r in rows if isinstance(r["fix_wait_h"], (int, float)))
print(len(rows), "incidents;", len(waits), "with a fix wait; median", waits[len(waits) // 2], "sum", round(sum(waits), 1))
