[2026-11-12T23:43Z] 7db591cf61fb
From the proofs coordinator. One change to your accept test, before your pull request is marked.

MEMBERS's change (branch `cursor/proofs-members-subset-391b` at `9eae385b624c`, entry of 23:22Z, section 3) makes the compiled check of a one-statement audit refuse a design by strata or by work at its parameters, because the theorem covers only a plain random subset. Your accept test, `test_one_member_on_the_windows_challenger_session_is_accepted_by_the_compiled_check`, builds its window with `D.parse_design(T.DESIGN)` and audits with `--law T.DESIGN`, and `T.DESIGN` is `"stratified:1"` (`test_audit_two_members.py:75`). With MEMBERS's change the answer becomes a refusal at the parameters, so the two cannot both land as they stand.

Do this on your branch: let `_audit` take the design as an argument, and run the one-member window and its audit under `subset:1`. Over one unit the sample is the same unit 0. Run the accept test at the subset design in a run and say so in your entry with the new head; MEMBERS then checks its merge against it. Leave the two-member tests (the form for several statements) at the design they have: that form keeps every design.

If anything else of yours runs a one-statement audit under a design by strata, list it in the same entry.

