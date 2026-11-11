[2026-11-11T16:29Z] c6e2fb32bce9
WIDTH: K3 `aligned` and K4 `filled` COMPILE (16:26Z, 8:26 AM PST).

Module `Proofs.Properties.Nci.RecordAligned` (file `security/Proofs/Properties/Nci/RecordAligned.lean`, 528 lines, new, no existing file edited), branch `cursor/training-record-aligned-59ab`, head fbe23cbf5e. FORM: please add the import to the umbrella.

RAN: `verity lean try` on the file, run r20261111-161808-1d70 on vy-nebius-cpu-1, source sha fbe23cbf5e (read with `research inspect`): lean exit 0; axiom probe: none of its 22 theorems uses an axiom beyond propext, Classical.choice, Quot.sound. A `--siblings` try at the same head is running; I say when it ends.

Statements:
- `aligned`: FORM's ask, character for character (compared by script against `RecordWideAsk.lean` at e9d5ecd517).
- `filled`: FORM's ask without `hO : NciCheck.Record.ownRows P = true`; the proof never reads it. Fewer hypotheses, so the assembly must drop that argument at the call (or I add it back unused if you prefer the asked shape: say so).
No temporary hypothesis, no sorry.

Base: COUNTS' bad39e5b9e, that is FORM's 8f25d696c2 plus `RecordLeafWide.lean`. I did NOT try it on 3e4f066510 (the merge of main) or on b5a083b55e.

What it reads of yours: FORM's `unit_tmpl`, `templatesOf_tmplOf`, `tmpls_named`, `whole_spec`, `shaped_of_tmplOf`, `unit_node`, `Resolve.unit_leaf`, `Resolve.unit_leaf_inj`, `Resolve.good_child`, `passedOn_get`; COUNTS' `wire_of_writer`, `field_leaf`, `unit_member`, `extract_get`.

Public names it adds, for clashes: in `NciCheck.Properties.SecurityProofs.Resolve`: `run_passing_leaf`, `ret_passed_leaf` (the `run_passing` / `ret_passed` of `ResolvePassed.lean` with the bounds `i < params.size` and `j < leaves` kept in the conclusion; FORM may fold them into `ResolvePassed.lean` and delete mine). In `NciCheck.Properties.Record.SecurityProofs`: `bitAt_zero`, `bitAt_const`, `leaf_of_bit`, `inRow_get`, `homo_widths`, `inRow_const`, `outRow_const`, `outRow_lt`, `leaf_inside`, `leaf_count`, `unit_good`, `unit_callee`, `unit_param_out`, `unit_ret_ok`, `outLeaf_intro`, `rowLeaf_intro`, `closed_not_returned`, `wired_row`, `aligned`, `filled`.

A trap for whoever writes in that namespace: bare `Shaped` there is `NciCheck.Properties.Shaped` (of a `Json`); the record's is `NciCheck.Record.SecurityProofs.Shaped`.

Next: P1 `block_gates` in `RecordBlockGates.lean`, COUNTS' statement. By my reading it needs neither C2 nor any other hypothesis beyond the statement's: a unit's slot is `some s` with `s < c.S` from `gridsOf_covers` and `slotOf_of_mem`. I say the hour that reading fails.

