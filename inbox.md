[2026-11-15T06:43Z] fb945e78ba44
From top, Sat 14 Nov 10:4x PM PST. Daniel answered the scope question at 10:23 PM PST: "Yes all tests should just be in tests".

So the move is no longer `verity/` only. Size it for the whole tree: verity (1,074 on the branch), the adapters (1,532), benchmarks (979) and tools (444) all go under the root `tests/`, as one move, the first after the landing, combined with my ontology map so `verity/` moves once.

What I need in your answers (first at 07:30Z, final 14:00Z), in this order:
1. The destination layout under `tests/` (mirror of the source path, or by suite), with the one you recommend and what each costs.
2. What `tools/check/suites.py` discovery, `[tool.verity.tests] inputs`, the read guard, conftest files, fixtures beside tests and the Lean difftests need, each with the files that change. ci owns suites.py: write that part as a plan I can hand to ci.
3. What cannot move by a path map alone (relative imports, `Path(__file__).parents[n]`, data files beside a test, package-private helpers), with counts.
4. The order that keeps every suite green at each step, and whether it can be one commit.

Still no PR and nothing on the branch `architecture`. Say in your note that you took this message.

