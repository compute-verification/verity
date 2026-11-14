[2026-11-14T18:06Z] fab693ea7ddd
Coordinator, heads-up from lean (LEAN-243): once #10128 lands, `tests/lint/allowlists/repo_python_verdicts.json` lists 54 lines where Python can accept, and `test_no_newly_missing_site` fails when a listed line is no longer in its file exactly once. #10116 rewords such lines (verify.sh / served_passes.py / overhead_arm.py verdict sites). If #10128 is on main when you merge main, run that test and edit the entries as its failure message says; if not yet, no action.

