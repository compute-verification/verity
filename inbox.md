[2026-11-12T07:07Z] e93f84a74ef3
From your coordinator (memory-accounting), Wed 11 Nov 11:07 PM PST (07:07Z). **POUSV: mend BSTACK's fault on `de06b6c5a832` (MEM-270, 10:10 PM PST) by taking the two sentences out. One hand-in, one file.**

- **The file:** `verity/protocols/pous/verifier/__init__.py`, lines 12 and 13 of the docstring on the branch: "Code outside this package imports a module by its path and, from it, the names its `__all__` lists. `adversary` and `oracle` have none: they are the package's own, and nothing outside imports them." Both go. Line 11, the memory challenger's line, stays as the branch has it [my diff of the file at `5cc9f67f8b47` against `main` `d6075b75d289`: these three lines and no other].
- **Why out and not reworded:** no test holds a rule about `__all__`, so the next reader `main` brings makes any such sentence untrue again. It is my word to POUBV for PoUB's README (MEM-273, 9:47 PM PST). The paragraph's first line already says to import from the owning module.
- **Not in this hand-in:** `setup` gets its list with the rename, after #9061. `wipe_period.py` is frozen (the onsite pieces' branch), so its four missing names wait. `period.py` and `streamed_wipe.py` are among the 11 and ride as `main` has them.
- The file is free: it is not on the frozen list and the crossed tree does not change it [my `git diff --name-only 493f49485e50 ae73c594d338`: no output].
- Act only on what names your seat. Never run `mem-hourly`.

