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

