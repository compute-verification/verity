[2026-11-16T21:38Z] 016c489460d1
top, Mon 16 Nov 1:38 PM PST. The leads answered; the pass's scope and the deciders' rule change as below. Put these in pass/decider-brief.md and in the shard list before you report.

SCOPE OF TONIGHT'S PASS: Python, Lean and shell only. OUT, and listed in pass/skipped.tsv with the reason:
1. Every C, C++, CUDA and Rust source and header (.c .cc .cpp .h .hpp .cu .cuh .rs), and Triton or template text held in a source file. A served kernel is held to a stored binary's digest, line numbers reach a binary through __LINE__ and asserts, and one record hashes the source bytes. They get a change of their own later, with binaries built and compared.
2. Any file, of any language, whose bytes or source text some code hashes into a pin: find them by rg for inspect.getsource, getsourcelines, source_sha256, _src_hash, and for a read of a source file's bytes next to a sha256/sha512 call. circuits names: adapters/vllm/verity_vllm/definitions/registry/ref_prims.py; adapters/torch/verity_torch/frontend/provenance.py and the modules it names; adapters/vllm/verity_vllm/acquire/ (fa2_tap_src/, fa3_tap_src/, inductor_templates.py, norm_tap_src); adapters/torch/verity_torch/pouw/pearl_c/. PROSEONLY is told to build this list too; take the union.
3. hardware/cards/*.toml and hardware/nodes/*.toml (already sent).

LEFT WHOLE inside files the pass does edit:
4. A Definition's body docstring: verity/circuits/defs.py reads it at run time (doc=fn.__doc__) and through the catalog. Every function registered as a Definition's body keeps its docstring whole.
5. A shell script's header block where it is the script's usage (options, what it writes, exit codes). History lines inside it are deleted; the usage stays.

DECIDERS' RULE, added:
6. In Lean, listed or not: a docstring that says what is NOT proved, or which premise nothing enforces ("no theorem proves it", "two premises no refusal of the check enforces", "Open:"), is the fact that passes. Keep it, as one to three lines. Never delete it.
7. A docstring that is the only statement of a contract a caller needs (what each line of a report means, a script's exit codes, the rule a lint test enforces, what a record's key holds) is "what a caller must know": keep it at the length that needs, cut the rest.
8. Every deletion made when unsure goes on the owner's list. In that list, deletions in these files come first, since their owners deleted a DESIGN.md on the strength of them: compute-accounting's (benchmarks/numerical/run.py, hawkeye_close.py, checker/params.py; benchmarks/pouw/filler/{arms,scoring,score}.py, harness/{bench,verify,arm}.py, ppl/{methodology,gates,run}.py, unit_lowering/run.py, pearl_c4/{build,run}.sh; machine_model's eps.py; PoUW's prover/sass_gate.py, prover/serving's package docstring, verifier/parties.py, verifier/window.py, FOLDS in receipt_folds.py) and vllm's (prod_repro.py, the lint tests' docstrings, records/__init__.py).

