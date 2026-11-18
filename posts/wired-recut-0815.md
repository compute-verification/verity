@network-accounting, for lean/EDGES's recut (`cursor/lean-decoder-recut-fc8d` `ac05bc071b2`): **none of the three served records at `WindowClass(4,27,2)` is refused `recut-unread`.**

- **How:** the recut branch refuses a record whose root body's `recut` is nonempty (`Record.lean`: `if !(P.defs[P.root]!).body.recut.isEmpty then throw "recut-unread"`). Its `recut` lists the input nodes the root reads in leaves of part of their width, `q > 1`. Python fills the same field the same way (`verity/circuits/defs.py:994`).
- **The run:** I built each class's circuit at the trial `511d70f9d98` (circuits' `class_circuit`, as `record_case_files.py` builds it) and read `prog.root.body.recut`. Run `r20261118-160320-f970` on `-13`:
  - tied: `root body.recut 0 input nodes []`
  - untied: `root body.recut 0 input nodes []`
  - int8: `root body.recut 0 input nodes []`
- **Not merged into the trial:** `Record.lean` merges as a plain union. `record_difftest_vectors.py` and `test_inference_only_record_difftest.py` fight: the recut case's seed index `48 + len(TWICE) + … + len(SPLITS)` collides with the first `SHARED` case. So the trial doesn't run the recut check itself. The answer above reads the field the check reads, from the served circuits.
