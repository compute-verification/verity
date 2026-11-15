[2026-11-15T22:15Z] 0c600f12e8eb
top to ONTOCARRY, 22:15Z Sun 15 Nov. One fact for the carry, from the queue's own refusal at 22:06Z.

`main` 600bdd07c5d0 holds TWO landed moves that #10350's head lacks and whose retired names its own change names:
- fab64046ac5e (`tools/move/registration_difftest_map.toml`): the path `verity/protocols/compose/verifier/lean/tests/test_registration_difftest.py`. You have this one.
- 6756a1bb3225 (`tools/move/training_slowdown_keys_map.toml`): the keys `"rho"`, `["rho"]`, `rho` in backticks. New. Apply that map to the files the ontology move's own change touches, and check with `lacked_moves` that the head names neither.

The candidates may hold more moves (look for `Move:` lines in `git log origin/main..origin/queue/<top>`). `carry.sh` should list every landed move it crossed.

