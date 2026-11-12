[2026-11-12T12:08Z] f03357ca637d
RENAME to training, TRN-133, Thu 12 Nov 4:10 AM PST: the rehearsal's result on main bcaf58a45d53

THE RENAME RUNS ON MAIN bcaf58a45d53 WITH #9003 IN IT: EVERY MOVE CERTIFIES AND EVERY SUITE THE CHANGE TOUCHES PASSES. Part A is open as a draft, #9275 (https://github.com/compute-verification/verity/pull/9275), nothing marked. The whole chain is `cursor/trn133-slowdown-stack-4-51f9` at e034ac9f52. The full report is the note `lanes/training/20261112T1207Z-result-trn133-rehearsal-on-bcaf58a45d53.md`.

#9003's FILES (RAN). `layout.py` rewrites `CALCULATOR`'s path and the import in `calculator.py`, `config.py`'s refusal text and the path strings in the test's toys. It does not follow `CALCULATOR / "rho" / "min_cut.py"` or `MODULES = {"rho", "periods"}`, and the harness failed after the move ("FileNotFoundError"). One commit in part A, b4af0fac3a, in those two files of yours: both take the package's name from `calculator.CALCULATOR.name`. `NAMED_ONLY` is unchanged.

SITES (RAN, on bcaf58a45d53 and on e034ac9f52; they replace every earlier count)
- Before: 1,179 occurrences in 139 files.
- Renamed, 665: path 221, Python name 149, module 145, prose 103, key 21, file name 16, printed text 10.
- Left, 514: key 177, prose 107, Python name 87, Lean prose 49, path 45, printed text 36, Lean binder 6, data string 5, module 2. They are: the harness 369, Lean 55, the move checker's toy tree 36, records of moves 26, stored keys and strings 19, the Glossary's row 6, `archive/` 2, a card 1.
- Not touched: the dampener's ρ 79, ACCEL's rate 348, the three unknown sites, and 32,303 of other quantities, the same in each file before and after.

MOVE CHECK (RAN on a node, r20261112-075958-1a34)
- B: "certified b4af0fac3a..769ef9c092 is a pure move (1 renames, 15 path-only edits)".
- C, two commits as one range: "certified 769ef9c092..b5d782dfa9 is a pure move (8 renames, 91 path-only edits)".
- D: "certified b5d782dfa9..c90df3ce87 is a pure move (0 renames, 6 edits by tools/move/slowdown_prose_map.toml's words)".
- Each map run again on its result changes nothing.

SUITES (RAN on nodes, `suites.py --changed PARENT`, each run's last line)
- The head against main: r20261112-083403-9622, "63 suites: 63 passed (0 cached), 0 failed; 401.9 min of test time".
- A: r20261112-080031-11d2, "55 suites: 55 passed (0 cached), 0 failed; 318.0 min of test time".
- B: r20261112-083555-c07a, "49 suites: 49 passed (0 cached), 0 failed; 136.7 min of test time".
- C: r20261112-083525-9a56, "54 suites: 54 passed (0 cached), 0 failed; 124.9 min of test time".
- D: r20261112-083628-f4be, "44 suites: 44 passed (0 cached), 0 failed; 97.4 min of test time".
- `uv lock --check` 0 and `pins.py --stale` 0 on the head: r20261112-083452-aa47.
- The harness's suite passed in all five (532 tests), each on a node of its own. Last night's refusal on vy-nebius-cpu-14 did not come back.

MAIN MOVED AGAIN, to 19fb19a21e40 (READ at 09:00Z). On the VM, nothing pushed (RAN): part A merges clean; the three moves certify (B 14 path-only edits, C and D as before); one hand-written commit of part E, the formulas, conflicts in nci's `slowdown.py`, whose docstring main rewrote, and is redone by hand in three hunks. No suite was run there.

NOT CHECKED: any suite on 19fb19a21e40; B, C and D on the eight heavy suites at their own heads (the head's run and A's hold them); stored copies of the records whose keys stay.

NEXT: nothing runs now. The harness's map (S `slowdown`, ρ `bound`) is next, as a map of its own on the chain's head: its dry run and site list by Fri 13 Nov noon PST. Parts B to E are opened one at a time, each staged again on that day's main.

WHAT I NEED FROM YOU
1. Yes or no to five pull requests in this order, C as two commits certified as one range. Under tonight's rule (two deep at most) I open each part when the one before it has landed, unless you want the chain handed in as one head.
2. Yes or no to part A changing two files of the harness and one line of nci's `statement.py`, whose owner I do not know.
3. Whether the move checker's toy tree (36 sites) is renamed or stays as a record.
4. Daniel's answer on which quantity keeps the word: it is one word in the harness's map.
5. For others: the ratchet gate's fix (`cursor/ratchet-gate-move-check-51f9`, part C fails the gate as main has it) for vllm; one flaky test in `tools/verity/check/tests/test_lean_slot.py`; five superseded branches of mine I may not delete.

