[2026-11-12T18:35Z] 43f539aeedf4
TRN-145 spec synced: note:training-20261112T1835Z-spec-trn145-refusal (lanes/training/20261112T1835Z-spec-trn145-refusal.md).

Branch cursor/trn145-random-point-refusal-914d, head aba9b3338b, pushed by an ordinary push, unmarked, no PR.

Settled, one line each in the spec. There is no aliasing to handle, because words flow through the views. Every input word repeats, not only the outermost rows, because a loop's inputs have its iterations outermost. Lookups read their index modulo the table's rows. A claim past a scan the point doesn't run whole is refused, with the scan named. On lf that count is 0 (RAN on my VM). On the training_slowdown suite it is not counted yet: the next worker's first job.

The next worker keeps all of it and adds two things: the fixes in the messages and the cross-call test. Then come the node checks the spec lists.

Not run: node suites, gate, revert job, lf dump, s-t20.

[2026-11-12T18:35Z] d3a41aff658d
RENAME to training, TRN-133: handed over. The note is `lanes/training/20261112T1832Z-handoff-RENAME.md` in the notes, with the VM-only scripts in `20261112T1832Z-handoff-RENAME-scripts.md` beside it.

Heads, all pushed:
- A2, `cursor/trn133-slowdown-maps-2-51f9`: 4ff47565e6eb, three commits on d8b4a44698a6 (the 5 lines with the references map deleted; the harness's maps; grade's local). Not in it yet: the block for #9170's `work.py`. No pull request, nothing run on a node.
- Part A, #9275: d8b4a44698a6, unchanged since my 18:10Z message.
- The harness's dry run: 676dc8803b.

Runs:
- Part A's suites, r20261112-174205-9914: still running, 47 of 58 passed and none failed at 18:35Z. The new worker reads its last line and sends it to you.
- The harness branch at 676dc8803b (RAN on nodes): r20261112-175040-e3cd "5 suites: 5 passed (0 cached), 0 failed; 26.2 min of test time"; its gate, r20261112-181110-5453, "1 change(s), 0 failed".

The four lines of `move_check.py` for ci's tier-2 request are in the note, verbatim (`scoped_map`, lines 1416 and 1424: two out, two in).

#8676: the maps were built before it. Part A holds main 9ddae97ba2f7; 30d6fcc92ff7 is not in it. For the 144 allowlisted files under what the maps and moves touch, no count differs between the two mains (RAN). No map or move has run on a tree that holds 30d6fcc92ff7.

Not checked: A2 on a node; the stage.py trial on A2's own head (it ran on the same 5 lines as a trial branch).

