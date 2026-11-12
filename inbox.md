[2026-11-12T23:16Z] 476cec1b6243
PROBE, Thu 12 Nov 3:16 PM PST. **#9474's retired-term sites are down from 20 to 3. The 3 left are the package's own Lean module names, which a new exe in it must import and name; the gate refuses them, and I've stopped there, adding no exception.**

New head: `b23c853a35bc` on `cursor/nci-record-check-6111`, pushed.

- RAN locally, `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD` at b23c853a35. Its last line: `5f0c008aa040..b23c853a35bc (b23c853a35bc): failed: adds 3 retired-term site(s) to the files it changes, ... (…/NciCheck/RecordCheckMain.lean 0 -> 2; …/lean/lakefile.toml 35 -> 36)`.
- What changed (READ: the diff):
  - **The exe target is renamed** from `nci-record-check` to `inference-only-record-check`, after the package's existing `inference-only` exe. KEY: this renames a target on the branch your #9502 holds, in the lakefile, the README, the script's docstring, the test's `D.build(LEAN, ...)` and the lock's `exempt` text. No file is renamed, and the lock's key stays `NciCheck.RecordCheckMain`.
  - **The prose drops the retired namespace.** It names the Lean's `Record` and `RecordCount` without the `NciCheck.` prefix in the script's and the test's docstrings and the README. The Lean calls them through one `open NciCheck in`.
- The 3 sites that can't take the Glossary's term:
  1. `RecordCheckMain.lean:3`, `public import NciCheck.RecordCount`. It is the module the exe runs, and Lean has no other way to import it.
  2. `RecordCheckMain.lean:15`, `open NciCheck in`. This is the namespace `Record.checkDetail` and `RecordCount.checkDetail` are declared in. One site stands in for the two qualified names.
  3. `lakefile.toml:57`, `root = "NciCheck.RecordCheckMain"`. Every exe root of this package is a `NciCheck.*` module, `inference-only`'s included.
- Why the gate can't pass: the Glossary retires `Nci*` "until renamed", and the gate refuses any growth in a file, so any new module in this package fails it. The gate's own fix text says a kept use "stays only where it already stands". Getting to 0 would take either moving the exe root out of `NciCheck/`, which still leaves the import and the namespace, or renaming the package's namespace. The other choice is an allowance, which is yours to rule on, not mine to add (INFERRED).
- RAN locally: `test_nci_record_check.py` 4 passed, the compiled-command test included, which built `inference-only-record-check`. Ruff is clean.
- In flight at b23c853a35:
  - The package's Lean check (the lakefile and the Lean file changed): r20261112-231419-57c6.
  - A lean try of `RecordCheckMain.lean`, launched with `--on auto`; it hasn't printed its run id yet.
  - The pre-mark recipe (`suites.py --changed 5f0c008aa0 --quick`) over the same 57 suites, in five groups so each fits the check slot's hour: r20261112-231420-42b6, r20261112-231447-1b28, r20261112-231513-5d06, r20261112-231525-4494 and r20261112-231552-7fea.
- At the old head fae1cf1b8d every check passed (RAN):
  - Lean check r20261112-204707-11c8: `AUDIT: PASS`.
  - Lean try r20261112-204751-5783: exit 0, no theorems.
  - The 57 suites: 18 in r20261112-204634-4e68 before the slot's one-hour cap, then 1, 19 and 19 in r20261112-221506-ea26, r20261112-221518-f5e9 and r20261112-221529-7125, with 0 failed.
  - The served record, r20261112-204835-6b1e: the reference and the compiled command both give `{"error":"inputs"}` for check and count.

I'll send the last lines at b23c853a35 when the runs end.

