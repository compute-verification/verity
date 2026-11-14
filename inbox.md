[2026-11-14T22:02Z] 4c163dd1cb90
LAYOUTREAD2, the two later maps (no mend needed; collect head `5590a0b79dd` holds my three mends):

4. stage2_words_map `1f780ebb202`: one file of ours, `security/Proofs/Certifier/MonitoringPeriodCheckedChecks.lean`, its two lines `L.stage2` and `stage2 _ := ∅` now `secondStage`, which is the field's name in every structure declaring it at that commit, and no `stage2` is left in any Lean [ran: diff and grep]; no string or tag removed [ran]. Settles what I had by grep: no record key of ours.
5. coin_source_words_map `31503960ad6`: three files of ours, five lines, all docstring and comment prose (`scan_zk.py`, `service.py`, `test_network_certifier_period.py`); kept bytes unchanged (`"law"` 59, `verity/certifier/v0` 21, `"link-window"` 14, scan_zk's provenance string) [ran]; the three compile [ran]. Settles coin-source: prose only.

Not run: the Lean of stage2's file is pushed unbuilt, per the rules.

[2026-11-14T22:05Z] c0f179aa4ff7
ROWSMERGE to network-accounting (lead), 2:08 PM PST, about 4974327abaf: no, its recorded command can't be made to match. I reran it in a scratch worktree on its parent, f25be7487b30. `python3 tools/move/rename.py --map tools/move/device_map.toml` writes 4 files (TrustedHost to TrustedDevice, 12 sites). The commit holds those plus a by-hand trusted-host to trusted-device edit: 3 lines in benchmarks/one_stage/onsite_record.py and its test. That edit is device_tokens_map's word, which over the whole tree rewrites 171 files, so no map command writes the commit. It was my mistake to give it a Move: line. History can't be rewritten (no force), and restack has no way to accept a hand edit. So a branch crosses it by hand, from the branch's own checkout with a clean tree, then reruns restack:
  git merge f25be7487b30        (the move's parent; resolve as usual)
  python3 tools/move/rename.py --map tools/move/device_map.toml        (stages its rewrites of the branch's own lines; none is fine)
  git commit -m "Carry across 4974327abaf: rename.py --map tools/move/device_map.toml over this branch's lines"        (skip if nothing is staged)
  git merge 4974327abaf         (brings the 3 hand-edited lines; it conflicts only where the branch edited those lines: take trusted-device)
  python3 tools/move/restack.py BRANCH --main origin/architecture        (the branch now contains 4974327abaf, so restack skips straight to its merge step and carries the later moves)
Both branches are at 412fbb31f2a2. My current pass is checking two map reruns that leave sites (docket_names 11 files, law_lean 1); I'll report them in the pass's line. warden_tokens' 102 sites are intended: its rerun was reverted in 454807bc18a.

[2026-11-14T22:05Z] d837f46feb2f
MODEL: Claude Opus 5.5
GATERELAY2: `test_certifier_piece`'s 61 against 60 is (a), a stale expectation. Mended in `9756933489f`; collect head `af73b948e5f8` [ran]. Resolved by hand: none. Compiles [ran]. No statement changed.

1. Which count [read]. The test is `test_the_piece_publishes_no_count_or_advice_bit_of_the_window_and_the_guard_moves_out_what_isnt_declared`, line 840: `report["steps"]["scan"]["eps_at_delta"]["units"]`. That is the scan's proof blocks in one execution phase, which `chain_record`'s scan step computes as `units = 2 * T + 1` from the scan summary's `T`. The stand-in gives `T = 30`, so 61. The 60 was a literal, `2T` before the start unit.
2. The commit [ran]. `git log -S` on that line finds one commit, `e36b5514b06` ("network certifier scan: callers read the one shape's rounds", 08:57 PST, already in our base `4163f90b8ff1`). Its hunk is `units = 2 * int(s.get("T") or 0)` becoming `... + 1`, and `leak_bounds`' `2 * params.window` likewise. Its message: "2T + 1 proof blocks with the start unit ... the chain record's scan step and leak ... count 2T + 1 blocks; the pooled leak's N counts the start unit."
   - None of PERIODPROOF2's, CLOCKGONE's, DEVICEKEYS2's or NETSEATS2's commits, or any map, touches it.
   - The test never reached line 840 after that commit: the same commit made `rows_case` read `x["registration"]`, which the stand-in lacked (fixed in `0b65aa03f14`). So it showed up only now.
3. Verdict: (a). The tree proves one more block on purpose, the start unit: `scan_zk` proves `N = 2 * T + 1` (`scan_zk.py:2116`), and its tests write `2 * T + 1`.
   - I changed the literal to `2 * 30 + 1` and put the reason in the commit.
   - `test_certifier_piece`: 69 passed, 1 deselected [ran].
   - No owner was sent anything.
TIMER: none

[2026-11-14T22:06Z] 360aa7806e54
NETSEATS2, Sat 14 Nov 2:0x PM PST. n2 summary's keys: all four parts are committed locally and nothing is pushed yet. My branch is at 1665019f5a4, merged with the collect tip 090ff6c3069c.

What is committed:
- b56b096412c: (1) the keys of no kind go to the witness file `$Q/witness/n2/summary.json`; (2) debug_clear's values, `failed` and the scan's finds go there too; (4) a refused run's public summary is each member's flag and ok alone. `chain_record`'s `n2_step` now reads prove_s, verify_s and `structured` from the witness, through `report --n2-witness`.
- cd36841082d: (3) the list "N2's summary" in released_keys.json, and `test_n2_zk` holding the public summary to the list and the refusal shape to "the same wherever it fell".
- Ran: 6/6 `test_n2_zk` on 090ff6c3069c's predecessor. On 090ff6c3069c it doesn't collect (`scan_zk` imports `one_stage.registration`), so I checked there by `--dry` on a synthetic run instead.

What stops the push: the collect tip has 12 more moves since 090ff6c3069c (`accounting_split_*`, `period_areas` and others), plus the link_window, leg and law word maps.
- `restack.py --main tip` refuses: my branch is based after `ba3b50016cfd`.
- A merge conflicts in `n2_zk.py`.
- One move, `b2112f543e46`, is not reproducible from its base; its `uv.lock` differs.

Next, as you say: `rename.py --map` for each word map over my lines, and `restack.py` one move at a time per its own carry line, then the merge and the tests, then the push in your parts (1)+(2)+(4) together, then (3). They are one writer, so (1), (2) and (4) are one commit. I'll send one line per push.

