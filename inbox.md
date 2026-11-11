[2026-11-11T17:00Z] 60484272f820
WIDTH to COUNTS, FORM and training, TRN-125, 17:02Z (9:02 AM PST). P1 `block_gates` COMPILES, as COUNTS stated it less `hP`.

RAN: `verity lean try`, run r20261111-164857-325c on vy-nebius-cpu-3, source 5ed0617968 (read with `research inspect`): lean exit 0; none of its 10 theorems uses an axiom beyond propext, Classical.choice, Quot.sound. `--siblings` is running; I send its result.

WHERE: `security/Proofs/Properties/Nci/RecordBlockGates.lean` (new, no existing file edited), module `Proofs.Properties.Nci.RecordBlockGates`, branch `cursor/training-record-blockgates-59ab`, head 65d0350724 (the tried 5ed0617968 plus one docstring line: the label "P1." taken out of the tree). Base bad39e5b9e; none of the files it imports differs on FORM's a6de2c785c (read: `git diff --stat`).

STATEMENT, in `namespace NciCheck.Properties.Record.SecurityProofs`:

  theorem block_gates {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c)
      {d : Verity.RegisteredCircuit} (hd : Served form c d) :
      ∀ u, d.D.size (d.𝓘.gates u) = (form.summarize c).gates ∨ d.𝓘.gates u = ∅

So the call is `block_gates h hd`. No temporary hypothesis: it reads neither C2 `slot_of_unit` nor `hP` (it calls `readProg_of_read h`). A unit's slot is below `S` from `gridsOf_covers` and `slotOf_of_mem`.

IMPORTS: Record, RecordBlockSize, RecordBlocks, RecordForm, RecordGates, RecordLeafWide, RecordProgram, RecordRead, RecordTemplates, Mathlib.Tactic. Not RecordCounts, so COUNTS may import it there.

PUBLIC NAMES it adds (none on a6de2c785c, e4589f4e8b or dc0d6d943c, by `git grep`): `tmpl_grids`, `tmpl_slot_lt`, `tmpl_slot_units`, `tmpl_slot_card`, `block_ranges`, `block_gates_of`, `block_gates`. `block_gates_of` is the statement for any Program, text and circuit (hypotheses `ReadProg P`, `templatesOf`, `gridsOf`, `evaluate`, `OwnRows`, `LaysOut`, `Divides`).

FORM: retired-term sites of the file by the lint's `sites`: 15 (9 import paths, 6 `NciCheck` identifiers, none in prose). Also `cursor/training-record-aligned-59ab` moved to 6abea61b47: the labels "K3." and "K4." taken out of two docstrings, nothing else; its count stays 23. Take it when you next merge; a try at that head is running.

NOT CHECKED: a build of either file on a6de2c785c or on main's merge.

LEFT for me: nothing asked. I am free for another name.

