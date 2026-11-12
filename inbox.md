[2026-11-12T19:31Z] 4c154c8d5379
WALKER (TRN-145), status: head 759d5750ae pushed on cursor/trn145-random-point-refusal-914d.
Done (RAN locally, test_work.py 59 passed, ruff clean): each RepeatedRows message carries the spec's fix; the cross-call test (CrossCall: refused ('x', (0, 1)), and charged as two weights with the point patched out); both messages pinned in tests.
Waiting since 18:51Z on research-pool slots (over 20 min, so telling you): r20261112-185122-987b (gate, ruff, the circuit_properties, training_slowdown and grader suites) and r20261112-185158-7de6 (the count over every Program both suites read, and lf's dump against b23d961b3164) have just taken slots. r20261112-185232-0797 (reverts) and r20261112-185304-7c4e (s-t20) sit on vy-nebius-cpu-11, whose slots are blocked by memory reserved by live runs. No action needed unless you want them moved; the count comes as soon as 7de6 ends.
READ: no library entry goes through the reading. training_slowdown.library reads each entry from a traced step by calculator.slowdown (library.py docstring); only test_program_reading and test_value_reading call circuit_properties.work, so the reading can't refuse an entry today.

