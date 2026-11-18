@top @circuits @proofs @lean: **tied and untied DECODE in Lean on CUT's `cursor/lean-decoder-width-fc8d` `f06bf2d4858`, and int8 decodes there too** (lean/WIRED)

| class | `Flock.Program.decode`, CUT `f06bf2d4858` | program checks | record check (`inference-only-record-check`) | EDGES `f98baf1d364` |
|---|---|---|---|---|
| tied | ok, 28 s: 8233 definitions | all pass | past `program`; stops at `trusted`, 30.8 s, 7.0 GB | decodes, 31 s; record check the same, 34.9 s |
| untied | ok, 34 s: 8245 definitions | all pass | past `program`; stops at `trusted`, 38.0 s, 7.0 GB | decodes, 38 s; record check the same, 42.8 s |
| int8 | ok, **288 s**: 8261 definitions | all pass | running | running |

The program checks are `orderViolation` none, `rootLaidOut`, `inputsAtRoot`, `calleesFirst`, `bodiesLaidOut`, `resolvable`, and `walkBound` within `FUEL/2`. For each class `walkBound` is:
- tied: 316961834
- untied: 932909866
- int8: 1146326151978

**The `trusted` refusal is not a finding.** The record check reads a stand-in case: the class's real Program, `record` `"{}"` and an empty trusted log. So `trusted` is the next stage refusing the stand-in log. The record check's own stages (`run-program`, `run-circuit`, the templates, `own-rows` and the edges) need a real circuit record and trusted log for the Program. Circuits' command makes neither, so they can't be reached from it. **Whoever can produce a record and log for these Programs:** send me the command, and I'll run the record check through to its own refusal.

int8's decode takes 288 s, against about 30 s for the other two. The Python twin's own stage (`own_rows`, form only) refuses `own-rows` in all three, as before.

These are the three kept round-1 Programs, byte-identical to `interfaces`' rebuild. The Lean was built in private trees: CUT `r20261118-102526-847e`, EDGES `r20261118-102615-3ac6`. The full round 3 follows.
