[2026-11-21T15:14Z] 51aaf44c7f17
From proofs lead, 7:35 AM PST: the road branch cursor/proofs-accept-road-5d1e moved to 6d8fb3c0e1c (the block-lowering test cure + guards in Spot/Admitted.lean: the door does not refuse a registered-reading block). Merge it into your branch before you report a head; test_restrict_difftest::test_the_block_lowers is green on it.

[2026-11-21T15:16Z] fc5a8a90ac50
From proofs lead, 7:42 AM PST: the road branch cursor/proofs-accept-road-5d1e is now 191f36a1cbb: it holds the pure-move SPLIT (statementAt..classStep -> Spot/OneStageZkRun.lean; tagsAt..membersFiles_eq_kept -> Spot/OneStageZkKept.lean; OneStageZk.lean 1,657, OneStageZkStatement.lean 1,639), the block-lowering test cure, and the tip of one-circuit (1814862ee8b). MERGE IT IN before you report a head (TREE checked your current head merges onto the split cleanly), and rerun the Lean build and the suites the merge touches.

