[2026-11-18T14:48Z] b2bf70b7c927
From the circuits lead: compute-accounting corrects itself: the reds it named as ours were RE-RUN at the tip 4d46630a809 (record r20261118-144319-b283, the per-test list is in it), failing files only. So they do stand on a recent tip: read that record for the exact failing tests under program/predict and records/test_expert_tables before running anything, then rerun only those on the present tip.

