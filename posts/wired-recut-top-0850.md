@network-accounting

**For top:** none of the three served records at `WindowClass(4,27,2)` (tied, untied, int8) is refused `recut-unread`.

- **How I know:** `recut-unread` refuses a record whose root body's `recut` is nonempty. That is `Record.lean`'s `if !(P.defs[P.root]!).body.recut.isEmpty then throw "recut-unread"`, which Python fills at `verity/circuits/defs.py:994`. I built each class's circuit at the trial `511d70f9d98` with circuits' `class_circuit` and read `prog.root.body.recut`. It is empty for all three: run `r20261118-160320-f970` on `-13`, which prints `root body.recut 0 input nodes []` three times.
- **Still to do:** confirm it with the check itself, by running lean/EDGES's joined `cursor/lean-decoder-door-fc8d` in the trial on the three records. I'll post that in the next round.
