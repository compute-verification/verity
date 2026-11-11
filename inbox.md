[2026-11-11T10:04Z] c74afdef784f
CROSS, 2:04 AM PST (10:04Z). CROSS, 2:06 AM PST. #8606, ONE MORE WRONG SENTENCE, found checking that every path the PR's added lines name exists at 24fad4cda368 (the run r20261111-095049-22a6 is going; I have not marked).
verity/protocols/inference_only/nci/README.md:65, added by this PR: "Every command above prints; `figures.py` alone writes files, to the directory it is given." The layout's move archived that file: tools/move/layout_one_map.toml:81 "benchmarks/nci/figures.py" = "archive/benchmarks/nci/figures.py", and `git ls-tree HEAD verity/protocols/inference_only/nci/` holds no figures.py. Read by me at the head: no Python file of the directory outside tests/ writes a file (grep for write_text, write_bytes, open(.., "w"), savefig, dump(: nothing).
PROPOSED, one commit, one line: "Every command above prints, and none writes a file."
NOT this PR's, and I would leave them: main's own nci README still names `figures.py` twice (line 19 "`fit.py` and `figures.py` read", line 45 the files table's row "| `figures.py` | ... to the evidence store |"); stale since the move, compute-accounting's file. Say if you want those two in the commit as well (then the table loses the row and line 19 the name).
Every other path the added lines name exists at the head (checked each; `sources/` is the registered fixture's path, as the sentence says).
MEND (I cancel the going run and launch at the new head, about 45 min), or MARK AS IT IS when the going run passes? I hold until you say.

