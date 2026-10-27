import Definitions.Accounting.JointPeriod

/-!
# Retired: H1's checked-credit period bridge

`RealizesWorkOfH1CheckedCreditPeriod` is gone; the joint period's PoUW bridge is `JointBridge.CreditedWords` and
`JointBridge.H200` (#4690). This module declares nothing. It stays only because `Definitions.Accounting`'s import list
on `next` still names it: `merge.py --lean` merges import lists as a union, so a removal on one side doesn't survive a
merge whose other side changed the list. Delete it, and that import, once nothing else carries the old list.
-/
