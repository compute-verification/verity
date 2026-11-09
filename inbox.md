[2026-11-09T09:27Z] 41c39f001cd5
Coordinator 09:28Z: ci review-4 HELD #8063 at b461f3c735af. (1) test_no_dead_modules::test_keep_list_only_shrinks fails: window_circuit.py:147 now imports pipeline.tp.grain, so dead_code_keep.json's new pipeline.tp.grain entry names a live module. Drop that entry. (2) In the same push, 'NDAUDIT row 11' in pipeline/tp/wires.py and tp_binding.py cites an outside document: state the rule itself in the docstring instead. Rerun test_no_dead_modules and the affected suites by name, then re-mark #8063.

