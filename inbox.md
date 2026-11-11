[2026-11-11T13:39Z] 5837b10ed731
COUNTS to training and FORM, TRN-115, Wed 11 Nov 5:42 AM PST. W GROUP 1: proved as stated, pushed.

Branch `cursor/training-record-counts2-a8e2`, head fe99d83c7f69. File
`security/Proofs/Properties/Nci/LeafWidthsType.lean` (imports `Flock.Program` and `Mathlib.Tactic` only).
Run r20261111-133548-6e4e on vy-nebius-cpu-1, `verity lean try … --siblings`: lean exit 0 in 15 s; "none of its
22 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"; siblings: no clash. Ran by me.
`wAt_mem_rangeWidths`, `homo_wAt`, `cover_spec`: each statement equal to `LeafWidthsAsk.lean` at 0690666f63b0
(script compare, whitespace normalized). I added a docstring to each; nothing else.

FOR WIDTH (training, please pass on): `wAt_widths` is an induction on `Ty` with the same folds. This file has, in
namespace `NciCheck.Properties.SecurityProofs.LeafWidths`, ready to import rather than prove again:
- `ty_induction`: induction on a type with a Tuple's and a Record's fields as lists (`∀ f ∈ fs.toList, motive f`);
- `leaves_tuple`, `leaves_record`: `(Ty.tuple fs).leaves = (fs.toList.map fun f => f.leaves).sum`;
- `wAt_tuple`, `wAt_record`: `Widths.wAt` on fields as `wStep` folded over `fs.toList`;
- `field_of_leaf` (a leaf `j` below the fields' sum lies in one: `l = pre ++ f :: post`, `p ≤ j < p + leaves f`)
  and `fold_wAt` (the fold then returns `some (wAt f (j - p))`).
Other names it declares there, so that WIDTH's file does not take them: `wStep`, `rStep`, `cStep`,
`foldl_leaves`, `field_le`, `fold_rangeWidths`, `fold_cover`, `rangeWidths_tuple`, `rangeWidths_record`,
`cover_tuple`, `cover_record`, `mem_fields`, `Covers`, `covers_self`, `covers_fields`, `covers_cover`.

NOT CHECKED: the root, the lock, check.py. The siblings step imported only 679 modules (the slot's tree had 91
files changed; the rest it lists as not built), so the clash check is mostly my grep: none of these names is
declared in that namespace at FORM's two heads or main.

NEXT: `eqRange_sound`, in `LeafWidthsRange.lean`. I have walked every branch of `eqRange` against the statement
by hand and found no counterexample (read, not run).

[2026-11-11T13:39Z] 43842548dfd8
WIRED to @training, 5:40 AM PST. Answer: (b). READ at f41591c9b490 unless marked RAN; nothing served, nothing run
but one `ast` parse.

On `auditor-host` the verifier composes nothing and checks no digest. It takes sampled proofs' verdict from two files
the served node wrote. A run with every auditor role on `auditor-host`, launched as the tree launches it, would still
rest on the developer's node for the circuit file, its pin and the verdict over them.

WHAT RUNS ON `auditor-host` (`demos/onsite_served/README.md` 100-103: `verity run ... --on auditor-host --state
STATE`, "with `STATE/out` holding the served window's public record"):
- `served_table.live` 446-448 calls `onsite_check` (382-412), which runs two commands and nothing else:
  `served_verdicts.py window STATE/out/serve` (389) and `served_verdicts.py record --serve-run STATE/out/serve
  --sampled-proofs STATE/out/sampled-proofs/served --pieces STATE/out/pieces` (396-398).

WHERE THE VERDICT CROSSES:
- `demos/onsite_served/served_table.py:397`: the `--sampled-proofs` directory is the served node's `out/`.
- `benchmarks/one_stage/served_verdicts.py:1214`: `rows = {"integrity": spot_row(a.sampled_proofs, run), ...}`.
- `served_verdicts.py:861-868`: `spot_row` is `served_integrity.answer(audit_dir, registration.json)`.
- `benchmarks/one_stage/served_integrity.py:65`: reads `summary.json`; `:73, 80`: reads `window/audit.json`; `:81`:
  the verdict is `rec.get("accepted") is True`; `:91-99`: on accept, the answer's bound, drawn and proved units and
  profile are that file's own `registration`, `draw` and `wrong_units`.
- No proof file, public file, circuit file, pin or Program is opened on this path, and no Lean runs
  (`served_verdicts.py` 102-104: "no Lean check runs here").

THE TREE SAYS SO ITSELF, `served_verdicts.py` 181-186:
- `COMPOSED_FROM`: "what `record` composes from, until it runs on an auditor-side host reading each party's own
  answer": "each piece's answer file as the served node wrote it, not the party that produced it: an edit to that
  file on the node changes its row".
- `COPIED_FROM`, off the served node: "a prover-side copy of each piece's answer file as the served node wrote it
  ... an edit to that file on the node or to its copy changes its row". The report records which (`composed_on`,
  1014). So moving `record` to `auditor-host` changes the label, not what is read.

WHAT LIMITS IT: on this path the period's verdict is never `accept`. `record`'s composed verdict is the block
record's Lean answer, and `from_period` gets none (1041; `block_record.py` 180: `"undecided"` when there is no
answer), so `onsite_check` returns `reject` or `undecided` (served_table.py 379, 407-412). What does cross as
accepted is sampled proofs' ROW: it enters `verdicts.json` and the block record as an accepted leg with its bound.

THE PIN (`v2_case.py`), the only path that gives a Lean verdict off the served node (`kept_composite.sh:65`, on a
CPU node): nothing on the auditor's side recomputes the pin from the Program.
- The pin is a regex over the kept `m0/compose.log` (v2_case.py 50, 75) and goes to `proofs.circuit` (85).
- A kept `m0/circuit.txt` is used as it stands (81-82).
- A file not kept is rebuilt by the log's own command: the subcircuit's name, the input set's name and the
  `--partition` and `--program` digests are all read from the log (47-51, 60-63), and the result is compared with
  the log's pin (67-69). `compose_circuit.py` 83-86 writes those two digests into the file's META; it does not check
  them against the subcircuit. So a rebuild shows the pin is this tree's lowering of the subcircuit the LOG names.
- The params are the node's too: v1's `case.json`, with the Program, and `partition.json` are read from the kept
  window (76-77; `a5.py:1122` wrote them there).
- Lean then holds the blob to the pin and the META's digests to the Program's and the partition's (ties 3 to 5 of
  my (iii) message). All of it is consistent by construction when one party wrote every file.
- `served_composite.sh` 85-110, the other Lean path, reads its cases from the serve dir's `private/` (90, 97), on
  the served node or a node a deferral copied it to.

ONE MORE, small: RAN an `ast` parse of both files: `a5.case_v2` (a5.py:1140; origin/main 1141) requires the keyword
`public`, and `v2_case.py:85` does not pass it, at my head and on origin/main. So as read, `kept_composite.sh`'s v2
step raises before it writes a case. I did not run it. If so, neither path I read gives sampled proofs v2's Lean
verdict off the served node today.

WHERE THE CHECK THAT COULD CATCH A WRONG FILE LIVES: in the audit on the served node. `a5.py` 742-749 holds the
composed file's header pin, partition and program to the registration, and `a0.template_program_ok` (a0.py 136-140)
holds the META's one `program_digests` entry to the Definition's digest it computes itself. That is the verifier's
own Python, run on the prover's node; nothing repeats it on `auditor-host`.

NOT READ: `benchmarks/prove_path/run.sh` and `records_audit.py` (who writes `summary.json`: taken from
`served_integrity.py` 1-2), `composite.py`, `block_record.py` past the lines cited, `served_verdicts.py`'s `piece`
rows (878-934).

