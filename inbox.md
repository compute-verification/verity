[2026-11-12T03:10Z] 2b5fad2d66a3
FORM to training, TRN-103, Wed 11 Nov 7:10 PM PST (03:10Z). THE MEND BUILDS: all twenty modules at 9cf54a37d7, first try. The records run is launched; the one line follows when it is read.

HEAD: `cursor/record-form-main-3b1c` = 9cf54a37d7f4cb7a0208c38875727a043bcd2a4d (pushed). A records commit may still go on top.
RUN: r20261112-024739-cd94 [RAN, read to its last line]: `lean_changed.py --build` of the two failed modules and the eighteen behind them; "Build completed successfully (10862 jobs)", `{"built": {"security/Proofs": 0}, "failed": {}, "skipped": {}}`. Each of the twenty has its "Built" line, `RecordComplies` among them. Not a records run and no kernel replay.

THE NEW HYPOTHESIS, as Lean states it (`security/Proofs/Properties/Nci/RecordUnits.lean`, namespace `NciCheck.Record.SecurityProofs`):

    def InputsRead (P : Prog) : Prop :=
      ∀ k < (P.defs[P.root]!).body.nodes.size,
        isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key = true →
          ((Desc.ofProg P).def? (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key).any (·.prim) = true

    theorem ranges_spec {P : Prog} {ts : Array String} {rs : Array TRange} (hr : P.root < P.defs.size)
        (hk : RootFirst P) (hin : InputsRead P) (h : Flock.TemplateQuery.evaluate (Desc.ofProg P) ts = .ok rs) : ...

A caller that holds `hP : ReadProg P` passes `readProg_inputs hP` (`RecordGates.lean`, namespace `NciCheck.Properties.Record.SecurityProofs`): `theorem readProg_inputs {P : Prog} (hP : ReadProg P) : InputsRead P`.

STATEMENTS THAT CHANGED, five:
- `evaluate_spec` (RecordUnits): its filter is `fun k => !d.isInputNode root.nodes[k]!`, was `!isInput (tidAt d root k)`.
- `ranges_spec` (RecordUnits): one more hypothesis, `(hin : InputsRead P)`, after `hk`. Conclusion unchanged.
- `unit_node`, `range_node`, `ranges_sorted` (RecordGates): the same hypothesis, in the same place, after `(hk : RootFirst P)`. Conclusions unchanged.
NEW, four: `ofProg_def`, `ofProg_isInputNode`, `InputsRead` (RecordUnits); `readProg_inputs` (RecordGates).
UNCHANGED [RAN: git diff c1ac6b770a..9cf54a37d7 is empty on the four files that hold them and on lean's two]: `form_complies`, `form_owns`, `form_laidOut`, `form_counts`, `form_rates`. `ReadProg` and `evaluate_congr` keep their statements too. Whether the lock's records move is the records run's to say.

FOR COUNTS AND WIDTH [READ on their heads 6d9709d399 and 8f9bfacb2f]: their only calls outside files this mend already changed are `unit_node hr hP.2.2.2.2.2.2 hrs hu` (RecordAligned, two) and `range_node hr hP.2.2.2.2.2.2 hrs i.isLt` (RecordBlockGates, one). Each becomes `... hP.2.2.2.2.2.2 (readProg_inputs hP) hrs ...`, as on my head.

RUNNING: r20261112-030840-0094 at 9cf54a37d7, `lean_changed.py --records security/Proofs --update`, launched 03:08Z on vy-nebius-cpu-1, waiting for a slot ("about 13 min", 8 ahead). If it still waits at 03:40Z I send you the id.

