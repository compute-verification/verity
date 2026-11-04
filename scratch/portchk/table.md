| PR | Goal | Now | Port | Conflicts | Try | The PR's own hand fix | Size | Inherited, fixed below the port |
|---|---|---|---|---|---|---|---|---|
| #7166 | finders-charged | open | above | 3: 1 records/pins; 2 gone on main | pass (E) [head moved] | none | - | - |
| #7164 | sampled-audits-sound | open | above | 7: ExecStrataOf.lean; 4 records/pins; 1 deleted; 1 gone on main | pass (E) | none | - | Soundness.ExecStrataOf (#7098, next) |
| #7160 | sampled-audits-sound | open | above | 5: 4 records/pins; 1 gone on main | fail (E) [head moved] | none | - | Json.RoundTrip (#5105, main), Properties.Composition (#6677, main), Nci.Protocol (#6677, main), VStarRecursive.Protocol (#6677, main) |
| #7159 | sampled-audits-sound | open | above | 5: 4 records/pins; 1 gone on main | fail (E) | none | - | Json.RoundTrip (#5105, main), Properties.Composition (#6677, main), Nci.Protocol (#6677, main), VStarRecursive.Protocol (#6677, main) |
| #7158 | finders-charged | open | above | 0 | fail (E) | Test.RegBudget: #7150's own parse error, not the port (fails on main's toolchain too); fixed at #7150's head 13076c1ddc, stacked here | - | - |
| #7153 | terminology-clean | open | above | 3: 1 records/pins; 2 gone on main | no Lean [head moved] | none | - | - |
| #7150 | recursion-knowledge-sound | open | above | 0 | fail (A) [head moved] | Test.RegBudget: #7150's own parse error, not the port (fails on main's toolchain too); fixed at #7150's head 13076c1ddc | - | - |
| #7144 | ~sampled-audits-sound (#7159, stacked on it) | open | above | 5: 4 records/pins; 1 gone on main | fail (A) | none | - | Json.RoundTrip (#5105, main), Properties.Composition (#6677, main), Nci.Protocol (#6677, main), VStarRecursive.Protocol (#6677, main) |
| #7134 | accounting/sampled-draw-from-log | open | above | 5: 4 records/pins; 1 gone on main | pass (A) | none | - | - |
| #7133 | zkp-system-done | open | above | 5: 4 records/pins; 1 gone on main | fail (A) | none | - | Json.RoundTrip (#5105, main), Properties.Composition (#6677, main), Nci.Protocol (#6677, main), VStarRecursive.Protocol (#6677, main) |
| #7127 | zkp-system-done | open | above | 0 | pending | none | - | - |
| #7123 | zero-knowledge | open | above | 4: 4 records/pins | fail (E) | none | - | Json.RoundTrip (#5105, main), Recursive.FirewallAny (#7064, main), Recursive.FirewallJson (#7026, main), Recursive.FirewallWire (#7026, main) |
| #7121 | zero-knowledge | open | above | 4: 4 records/pins | pending | none | - | - |
| #7119 | sampled-audits-sound | open | above | 7: ExecStrataOf.lean; 4 records/pins; 1 deleted; 1 gone on main | pass (E) | none | - | Soundness.ExecStrataOf (#7098, next) |
| #7118 | accounting/sampled-draw-from-log | next | below | 6: ExecStrataOf.lean; 4 records/pins; 1 gone on main | fail (A) | none | - | Json.RoundTrip (#5105, main), Soundness.ExecStrataOf (#7098, next), Properties.Composition (#6677, main), Nci.Protocol (#6677, main), VStarRecursive.Protocol (#6677, main) |
| #7116 | sampled-audits-sound | open | above | 5: 4 records/pins; 1 gone on main | fail (A) [head moved] | none | - | Json.RoundTrip (#5105, main) |
| #7107 | sampled-audits-sound | open | above | 6: ExecStrataOf.lean; 4 records/pins; 1 gone on main | fail (B) [head moved] | none | - | Json.RoundTrip (#5105, main), Soundness.ExecStrataOf (#7098, next), Nci.Protocol (#6677, main), SampledProofs.Json (#7092, next) |
| #7104 | zkp-system-done | next | below | 0 | pass (A) | none | - | - |
| #7099 | sampled-audits-sound | next | below | 0 | pending | none | - | - |
| #7098 | sampled-audits-sound | next | below | 6: ExecStrataOf.lean; 4 records/pins; 1 gone on main | fail (A) | Soundness.ExecStrataOf: keep the port's qsort_perm beside #7098's chain_covers (my first resolution took #7098's qsort_perm) | 3 | - |
| #7095 | zkp-system-done | open | above | 5: 4 records/pins; 1 gone on main | fail (B) | none | - | Json.RoundTrip (#5105, main), Properties.Composition (#6677, main), VStarRecursive.Protocol (#6677, main) |
| #7093 | ~sampled-audits-sound | open | above | 0 | no Lean [head moved] | none | - | - |
| #7088 | sampled-audits-sound | open | above | 6: ExecStrataOf.lean; 4 records/pins; 1 gone on main | fail (A) [head moved] | none | - | Soundness.ExecStrataOf (#7098, next) |
| #7084 | sampled-audits-sound | open | above | 6: ExecStrataOf.lean; 4 records/pins; 1 gone on main | fail (E) | none | - | Json.RoundTrip (#5105, main), Properties.Composition (#6677, main), Nci.Protocol (#6677, main), Sampled.Deployed (#7137, next), VStarRecursive.Protocol (#6677, main), Soundness.ExecStrataOf (#7098, next) |
| #7081 | sampled-audits-sound | open | above | 5: 4 records/pins; 1 gone on main | fail (A) [head moved] | none | - | Json.RoundTrip (#5105, main), Properties.Composition (#6677, main), Nci.Protocol (#6677, main), VStarRecursive.Protocol (#6677, main) |
| #7064 | zero-knowledge | main | below | 4: 4 records/pins | pending | none | - | - |
| #7051 | sampled-audits-sound | open | above | 6: ExecStrataOf.lean; 4 records/pins; 1 gone on main | fail (A) [head moved] | none | - | Soundness.ExecStrataOf (#7098, next) |
| #7045 | zero-knowledge | main | below | 4: 4 records/pins | pending | none | - | - |
| #7037 | sampled-audits-sound | main | below | 5: 4 records/pins; 1 gone on main | fail (B) | none | - | Properties.Composition (#6677, main) |
| #7030 | sampled-audits-sound | open | above | 5: 4 records/pins; 1 gone on main | pending | none | - | - |
| #7026 | zero-knowledge | main | below | 1: 1 records/pins | pending | none | - | - |
| #7010 | zero-knowledge | main | below | 4: 4 records/pins | pass (A) | none | - | - |
| #7003 | sampled-audits-sound | open | above | 0 | pass (C) | none | - | - |
| #6998 | ~sampled-audits-sound | next | below | 5: 4 records/pins; 1 gone on main | pass (B) | none | - | - |
| #6996 | sampled-audits-sound | open | above | 5: 4 records/pins; 1 gone on main | fail (E) [head moved] | none | - | Json.RoundTrip (#5105, main) |
| #6995 | zero-knowledge | next | below | 1: 1 records/pins | pending | none | - | - |
| #6980 | sampled-audits-sound | next | below | 4: 3 records/pins; 1 gone on main | pending | none | - | - |
| #6976 | zero-knowledge | main | below | 3: 3 records/pins | no Lean | none | - | - |
| #6961 | zk-leaks-closed | main | below | 1: 1 records/pins | pending | none | - | - |
| #6955 | zero-knowledge | main | below | 1: 1 records/pins | pending | none | - | - |
| #6935 | — (cleanup/no-lane-text) | open | above | 0 | pass (C) | none | - | - |
| #6934 | sampled-audits-sound | open | above | 0 | no Lean | none | - | - |
| #6929 | recursion-knowledge-sound | next | below | 0 | pass (A) | none | - | - |
| #6923 | zero-knowledge | main | below | 1: 1 records/pins | fail (C) | none | - | SessionLoop.FinPiopRet (#6842, main), SessionLoop.LiveInner (#6912, main) |
| #6915 | zk-leaks-closed | main | below | 1: 1 records/pins | pending | none | - | - |
| #6912 | zero-knowledge | main | below | 1: 1 records/pins | fail (C) | SessionLoop.LiveInner: `import all Init.Data.String.Defs` (toByteArray = toUTF8 by rfl) | 1 | SessionLoop.FinPiopRet (#6842, main) |
| #6882 | zk-leaks-closed | main | below | 0 | pending | none | - | - |
| #6871 | zero-knowledge | main | below | 0 | pending | none | - | - |
| #6855 | zero-knowledge | main | below | 1: 1 records/pins | pending | none | - | - |
| #6854 | ~zero-knowledge-any-worker | main | below | 0 | pending | none | - | - |
| #6852 | zk-leaks-closed | main | below | 1: 1 records/pins | pass (A) | none | - | - |
| #6844 | ~zero-knowledge | main | below | 1: 1 records/pins | fail (C) | none | - | SessionLoop.FinPiopRet (#6842, main) |
| #6842 | zero-knowledge | main | below | 0 | pending | none | - | - |
| #6841 | zero-knowledge | main | below | 4: Assumptions.lean*, Finish.lean*, Frames.lean*, Session.lean* | pending | resolve Assumptions.lean*, Finish.lean*, Frames.lean*, Session.lean* | - | - |
| #6837 | zero-knowledge | main | below | 0 | pending | none | - | - |
| #6836 | zero-knowledge | main | below | 0 | pending | none | - | - |
| #6822 | sampled-audits-sound | open | above | 5: 4 records/pins; 1 gone on main | pass (A) | none | - | - |
| #6803 | sampled-audits-sound | main | below | 4: 3 records/pins; 1 gone on main | no Lean | none | - | - |
| #6801 | sampled-audits-sound | open | above | 5: 4 records/pins; 1 gone on main | pass (E) [head moved] | none | - | - |
| #6789 | ~sampled-audits-sound | open | above | 0 | pending | none | - | - |
| #6770 | ~sampled-audits-sound | open | above | 0 | pending | none | - | - |
| #6768 | ~sampled-audits-sound | open | above | 0 | pending | none | - | - |
| #6735 | recursion-knowledge-sound | main | below | 0 | pending | none | - | - |
| #6727 | ~recursion-knowledge-sound | main | below | 0 | pending | none | - | - |
| #6716 | ~sampled-audits-sound | open | above | 0 | no Lean | none | - | - |
| #6714 | ~recursion-knowledge-sound (PRF-282: zkp-system-done) | main | below | 5: 4 records/pins; 1 gone on main | fail (B) | none | - | Json.RoundTrip (#5105, main), Properties.Composition (#6677, main), Nci.Protocol (#6677, main), VStarRecursive.Protocol (#6677, main) |
| #6681 | ~recursion-knowledge-sound | open | above | 0 | pass (C) | none | - | - |
| #6676 | zero-knowledge | open | above | 5: 4 records/pins; 1 gone on main | pass (E) [head moved] | none | - | - |
| #6667 | sampled-audits-sound | open | above | 1: 1 records/pins | pending | none | - | - |
| #6655 | sampled-audits-sound | main | below | 4: 3 records/pins; 1 gone on main | pending | none | - | - |
| #6632 | recursion-knowledge-sound | main | below | 1: 1 records/pins | pending | none | - | - |
| #6624 | recursion-knowledge-sound | main | below | 0 | no Lean | none | - | - |
| #6591 | ~zero-knowledge (PRF-281: zkp-system-done) | open | above | 0 | pass (D) | none | - | - |
| #6533 | ~sampled-audits-sound | open | above | 0 | pass (D) | none | - | - |
| #6532 | ~sampled-audits-sound | open | above | 4: 3 records/pins; 1 gone on main | pass (C) | none | - | - |
| #6529 | ~sampled-audits-sound | open | above | 0 | pass (D) | none | - | - |
| #6513 | — (vocabulary) | open | above | 0 | no Lean | none | - | - |
| #6511 | — (vocabulary) | open | above | 0 | pass (D) | none | - | - |
| #6509 | — (vocabulary) | open | above | 0 | pass (D) | none | - | - |
| #6503 | — (tree cut) | open | above | 4: 4 records/pins | pending | none | - | - |
| #6497 | — (vocabulary) | open | above | 0 | pass (D) | none | - | - |
| #6495 | — (vocabulary) | open | above | 0 | no Lean | none | - | - |
| #6493 | — (vocabulary) | open | above | 0 | no Lean | none | - | - |
| #6486 | — (vocabulary) | open | above | 0 | no Lean | none | - | - |
| #6484 | — (vocabulary) | open | above | 0 | no Lean | none | - | - |
| #6479 | terminology-clean | open | above | 5: 4 records/pins; 1 gone on main | no Lean [head moved] | none | - | - |
| #6478 | — (vocabulary) | open | above | 0 | no Lean | none | - | - |
| #6477 | — (vocabulary) | open | above | 0 | no Lean | none | - | - |
| #6476 | — (vocabulary) | open | above | 2: pyproject.toml*; 1 records/pins | no Lean | resolve pyproject.toml* | - | - |
| #6471 | — (vocabulary) | open | above | 0 | pending | none | - | - |
| #6460 | ~sampled-audits-sound | main | below | 0 | pending | none | - | - |
| #6426 | — (tree cut) | open | above | 80: 80 deleted | pass (D) | none | - | - |
| #6422 | sampled-audits-sound | open | above | 2: pyproject.toml*; 1 records/pins | no Lean [head moved] | resolve pyproject.toml* | - | - |
| #6418 | — (vocabulary) | open | above | 0 | pending | none | - | - |
| #6404 | sampled-audits-sound | open | above | 0 | pending | none | - | - |
| #6403 | — (vocabulary) | open | above | 0 | pending | none | - | - |
| #6401 | ~sampled-audits-sound | open | above | 0 | empty patch (in the port already) (D) | none | - | - |
| #6378 | sampled-audits-sound | open | above | 2: pyproject.toml*; 1 records/pins | no Lean | resolve pyproject.toml* | - | - |
| #6298 | — (vocabulary) | open | above | 2: pyproject.toml*; 1 records/pins | no Lean | resolve pyproject.toml* | - | - |
| #6280 | ~sampled-audits-sound | main | below | 0 | pending | none | - | - |
| #6140 | — (cleanup) | open | above | 0 | no Lean | none | - | - |
| #6075 | sampled-audits-sound | open | above | 5: 4 records/pins; 1 gone on main | no Lean [head moved] | none | - | - |
| #6072 | — (cleanup/no-lane-text) | open | above | 7: 4 records/pins; 3 gone on main | no Lean | none | - | - |
| #6068 | sampled-audits-sound | open | above | 0 | pending [head moved] | none | - | - |
| #5978 | ~sampled-audits-sound | main | below | 0 | pending | none | - | - |
| #5969 | ~sampled-audits-sound | main | below | 0 | pending | none | - | - |
| #5953 | ~sampled-audits-sound | main | below | 5: HmRow.lean*, Setup.lean*, SetupEq.lean*, SimSetup.lean*, PROTOCOL.md* | pending | resolve HmRow.lean*, Setup.lean*, SetupEq.lean*, SimSetup.lean*, PROTOCOL.md* | - | - |
| #5924 | sampled-audits-sound | open | above | 0 | no Lean [head moved] | none | - | - |
| #5864 | ~sampled-audits-sound | open | above | 0 | pending | none | - | - |
| #5825 | sampled-audits-sound | open | above | 0 | pending | none | - | - |
| #5796 | ~sampled-audits-sound | open | above | 0 | no Lean | none | - | - |
| #5774 | ~sampled-audits-sound | main | below | 0 | pending | none | - | - |
| #5766 | ~sampled-audits-sound | open | above | 1: 1 records/pins | pending | none | - | - |
| #5747 | ~sampled-audits-sound | open | above | 0 | no Lean | none | - | - |
| #5746 | accounting/pouw-lead-on-main | open | above | 0 | pending | none | - | - |
| #5674 | ~sampled-audits-sound | open | above | 0 | pending | none | - | - |
| #5664 | ~sampled-audits-sound | open | above | 0 | pending | none | - | - |
| #5617 | ~sampled-audits-sound | open | above | 0 | no Lean | none | - | - |
| #5534 | — (cleanup/no-lane-text) | open | above | 0 | pending | none | - | - |
| #5512 | — (cleanup/no-lane-text) | open | above | 0 | no Lean | none | - | - |
| #5489 | ~zero-knowledge | next | below | 0 | pending | none | - | - |
| #5483 | — (cleanup/no-lane-text) | open | above | 0 | pending | none | - | - |
| #5370 | ~sampled-audits-sound | open | above | 0 | pending | none | - | - |
| #5302 | recursion-knowledge-sound | open | above | 1: 1 records/pins | pending | none | - | - |
| #5280 | — (docs) | open | above | 0 | pending | none | - | - |
