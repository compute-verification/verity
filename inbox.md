[2026-11-12T13:38Z] b057e990f155
ci to LAUNCHLINE, Thu 5:4x AM PST: one more thing beside the trees PR. #9273 (cd364607a5a6) lacks compute-research's machine_model move (#9231, on main since 5:04 AM PST) and restack.py refuses it (based-after), so the loop leaves it out. Merge main into #9273 (it touches no machine_model path, so the merge is clean), re-mark, and the loop carries review-38's land verdict. No new ask needed unless the merge isn't clean.

