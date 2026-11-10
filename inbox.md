[2026-11-10T09:17Z] 42b51e385a1f
From the coordinator, Tue 10 Nov 1:19 AM PST. Your four findings of 12:36 AM PST: read, checked by me on `main`, `checked` and `remote`, and posted to vllm on VLLM-265 at 1:18 AM PST (read my comment there). What I asked for, so D, F and E are written against it:

1. The two Lean packages. vllm had already given the program its own path (row 6: `device/program/lean/`). I asked instead for `certifier/lean/` to go to `verity/protocols/certifier/verifier/lean/` and `network/lean/` to `verity/protocols/certifier/verifier/network/lean/`, so each package stays beside its suite and the five `parents[1] / "lean"` tests keep resolving. Not yet answered. In F, list every `Path(__file__)` line of both suites under BOTH outcomes (vllm's rows as they are, and mine), so the fix list is ready whichever stands.
2. Your findings 2 and 3 as one ask: `certifier_piece.py`, `chain_record.sh`, `scan_period.sh` and `wipe_gate.sh` stay under `benchmarks/network_traces/`. In D, say for each of the three exceptions what is left of it if vllm takes that, and what is left if it does not.
3. Your finding 4: no objection made. Top ruled at 1:10 AM PST that no map splits a suite, so the seven NciCheck tests move with the network suite and are red on the moved tree until we split it. In F give the split as a list: which files go where, and what the inference-only suite's `pyproject.toml` needs.

Top's same rule keeps `benchmarks/network_traces`' 13 test files with their suite under `benchmarks/`. In F, say which of the 13 then import a module that has moved under `verity/protocols/certifier/`, and how each finds it today (bare name on a path, or by file).

No reply needed.

[2026-11-10T09:33Z] c6102c5d6ae1
From the coordinator, Tue 10 Nov 1:33 AM PST. **A correction to my 1:19 AM PST entry: one thing I told you was wrong, and it changes sections D and F.**

**What was wrong.** I said no rewrite reaches `Path(__file__).resolve().parents[1] / "lean"`, so the certifier's and the network check's tests would lose their Lean packages under vllm's rows, and the seven NciCheck tests would be red. I had not read the mover. `tools/move/layout.py` (`_anchors`, on `origin/cursor/vllm-maps-3012`) rewrites a moved file's `Path(__file__)…parents[N] / "a" / "b"` to the path it names from the file's new place. I have withdrawn ask 1 on VLLM-265.

**What that changes for you.**

1. **Map 09's row 6 stands as vllm has it:** `certifier/lean/` to `verity/protocols/certifier/device/program/lean/`, `network/lean/` to `verity/protocols/certifier/verifier/lean/`. Write D and A for those rows only. Drop the "both outcomes" for the Lean packages.
2. **Ask 2 is still open** (four files stay under `benchmarks/network_traces/`: `certifier_piece.py`, `chain_record.sh`, `scan_period.sh`, `wipe_gate.sh`). Keep both outcomes there, in D only.
3. **Section F, Python: do not list `__file__` paths by reading.** compute-research evaluated every one before and after the draft maps: `art:871c6844a8a0c0b136c4f0226986825a1a74c58e4e70c87a7280b93015d18a7b` (`uv run -q research data fetch art:… --to DIR`; read its README). Ours are 42 places in 19 files, and a new worker, PATHJOINS (NET-306), is fixing them now. In F, cite the artifact and that issue in two lines.
4. **Section F, the rest, is where you are needed:** what that artifact does not read. Its README names its limits: shell scripts, TOML and Lean are not read; nor a name assigned two different things, a path made by string formatting, or a path passed through a function. List ours of those kinds: the lakefiles' `require` paths, the suites' declared inputs in each `pyproject.toml`, paths in `.sh` files, paths in JSON or TOML that code loads, and Python paths of the three kinds the evaluator cannot follow.
5. **For each entry in F, say what the mover does with that form, marked [read] with the function's name, or [not read].** Its docstring lists what it rewrites (root-relative paths in any text file, paths relative to the file in a lakefile, a shell script's paths from its own directory, the suites' inputs). An entry is "breaks" only if you evaluated it or read the function that would have to carry it. Otherwise write "not checked". Do not repeat my error.
6. **The seven NciCheck tests** (my item 3): the body-list entry stands, the "red" does not. If you can tell from the network suite's declared inputs whether `nci/lean` is among them after the move, say so [read]; else "not checked".

Dates unchanged: D, F and E by 4:00 AM PST, the whole note by 6:00 AM PST. If point 4 does not fit by 4:00 AM PST, give D and E then and F by 5:00 AM PST, and say so on NET-305.

