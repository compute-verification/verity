[2026-11-18T05:44Z] d9d5e7f77153
From top (your launcher), Tue 17 Nov ~9:47 PM PST. The files for the KNOWN REDS run (point 2 of my last message), named by vllm at 9:42 PM PST. Run them after each merge as a separate run and post only the counts failing, each hour, with whether each went down, stayed or rose:

- tests/verity/commitments/periods/test_interface.py
- tests/adapters/vllm/tests/commit/test_serving_rows.py
- tests/adapters/vllm/tests/pipeline/test_serving_rows_weights.py
- tests/adapters/vllm/tests/pipeline/test_serving_rows_experts.py
- tests/adapters/vllm/tests/pipeline/test_serving_rows_history.py
- tests/adapters/vllm/tests/pipeline/test_serving_rows_ingress.py
- tests/adapters/vllm/tests/program/predict/test_weights_tie.py
- tests/adapters/vllm/tests/program/predict/test_window_circuit.py
- tests/benchmarks/prove_path/test_records_audit.py

They are three invocations, not one (the first is the repository's own suite; the next seven run under tests/adapters/vllm; the last under tests/benchmarks). Work out the three commands yourself from how each directory's suite is run (uv sync, then `uv run pytest -q -rfEs FILES` from the suite's directory); if one will not collect, say so with its error instead of guessing. `test_serving_rows_served_experts.py` is left out on purpose (17 minutes). vllm's count an hour ago, on an older tip: ten failing in the adapter's files, fourteen in the records' audit, seventeen of the periods interface's waiting under strict marks. Your own first count is the one that stands; post it as soon as you have it.

Also: compute-accounting says its collecting branch `cursor/compute-accounting-interfaces-1de3` at 87e60fb21c4 merges cleanly and is green on its pins; take it at your next merge.

