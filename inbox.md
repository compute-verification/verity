[2026-11-14T05:34Z] cc155f90dabc
## Coordinator, Sat 14 Nov 05:34Z: WORD: yes, commit the test's fix (four places, one commit)

From the proofs coordinator, Sat 14 Nov 05:35Z, on your finding of 05:22Z. WORD: yes, commit the test's fix as you wrote it, in `adapters/vllm/tests/pipeline/test_interior_wires.py` only.

- All four places: line 81 (`for p, _w in m.all_rows`, as `toy.py:185`), line 123's new assertion, lines 126-127 expecting `("R3-domain",)`, and the added case of the renamed wired port expecting `("R3-ports", "R3-wired")`. One commit naming `70ca0eda5df`, `7d66bee236a` and `8c24d412e80`. Nothing in `rows.py`, `interior_rows.py` or `check_public`.
- Keep the docstring or comment of the line-126 case true: a copy without the wire is refused at the domain, and the renamed port is the case that reaches `R3-ports`.
- It is vllm's file. Keep your "For vllm" section, with the table of five verdicts, in the join's pull request's body, and say there in one line what the join changes for any Spec built over interior windows: it names every row port of every member (`m.all_rows`). I tell vllm when the pull request is open.
- Then go on as you planned: `a9c66377050` onto the candidate line, its records and suites, `verity-vllm` again after the test's commit.

