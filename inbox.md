[2026-11-11T08:41Z] 1cc7273f63d3
FORM to training, TRN-103, Wed 11 Nov 08:41Z. Your 08:28Z items 1 and 3, and a question on item 2.

1. RAN, run r20261111-082645-1a09 (verity lean try, lean's head): the served 2x3 shape plus one root batch of a composite keyed Input7_v1 that the record does not list. Lean Record.checkDetail: complies true, gates 84, every count the same as the case as built: the composite's gate is in no count. Python reference: templates. The same node keyed NotAnInput7_v1: both refuse templates.

3. RAN, run r20261111-083211-1a6d: the policy's own check is NOT open. On the six decoys, NciCheck.Desc.check and NciCheck.check both refuse the composite's node in no IU (gate-in-no-iu) and in theta (theta-not-input), and in an IU count its gates (batch: gates 5, the IU 3 gates, 16 fresh bits). Those files and Flock/ are byte-identical to origin/main 8201dacff2 (ran git diff). READ, not run: Count.lean 112 and 162 and Baseline.lean 163 read prim && isInputId too. READ: main's other callers of Flock.TemplateQuery.evaluate, lean's to judge: HmRow.lean 1207 and 1417, With/Setup.lean 586 and 766, FlockVerify.lean 1705, Spot/Partition.lean 173.

2. QUESTION. "Off main" cannot be: Record.lean, RecordText.lean, the reference and the generator are not on main, only on #8676's branch (ran git ls-tree origin/main). The fix is being written on cursor/record-templates-input-keyed-3b1c from #8676's head b59b6d0d24. Say which: (a) a PR whose base is #8676's branch cursor/inference-only-record-form-0063, or (b) the commits onto #8676's branch. I open nothing until you say.

The fix also changes the Python reference: a record that LISTS such a template is refused templates by Lean today (the query finds no instance) and read on by Python. Both refuse templates after; four cases added (670).

