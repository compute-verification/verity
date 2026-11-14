[2026-11-14T04:16Z] bf9e2384efc1
@training, Fri 13 Nov 8:16 PM PST. One addition to Sunday's 4 PM message, from network-accounting on TRN-25 (8:14 PM PST; it says yes to its lakefiles and Lean being renamed by the map). No reply needed now; do not stop what you are doing.

**The question: does any renamed name sit in a kept run's files, and is it compared at a later read?** On Wed 18 Nov network-accounting reads two kept serve directories again with `main`'s code (Mon 9 Nov's and Sat 14 Nov's). If a name the maps rename is written into a run's files (a record, a registration log, a verdict, a provenance line) and a reader on the renamed `main` compares it, the kept run stops reading. Their rule, and now ours: **such a site keeps the older name**, by `[keep]` or `[keep_in]`, and the table says so.

What I want in the table, as its own section:
- Every string a program WRITES into a stored file whose text a map changes: format and schema ids, a tool's or executable's name, a kind, a key, a Lean name in a string. Start with what the check's executables print or write (`nci-check`, `nci-check --count`, the difftests), `nci_reference`, `adapters/vllm/verity_vllm/pipeline/registration_log.py` (its keys `program`, `registered`, `previous`), and `benchmarks/one_stage/served_verdicts.sh` (`$OUT/record/inference-only.json`). [READ by me on main: `served_verdicts.sh` names the executable's path and `lake build nci-check` at lines 1082 and 1704; those are build paths, not stored text.]
- For each: the file that writes it, the reader that compares it, and whether the map renames it or keeps it. One you cannot tell is listed as unknown, not guessed.
- The ids map's rows `check_baseline`, `expand_baseline`, `baseline-difftest`, `BaselineRegistered` and the check's lock names: say for each whether any stored file holds it.

A renamed executable or path that only a script on `main` names is fine: the script is renamed with it.

