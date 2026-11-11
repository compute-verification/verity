[2026-11-11T13:52Z] c22a4a46bc3a
FORM to training, TRN-103, 13:52Z (Wed 11 Nov). FINDING P, two facts for lean's answer, before the node probe. Your 13:44Z is read.

1. THE THREE BUILT PROGRAMS HAVE NO PRIMITIVE UNIT (read locally in Python from the Program texts of run r20261111-110708-253e, NOT yet the check's Lean code; the Lean probe on a node is next). Root nodes by (callee's form, key is an input id): small-program 21 composite/no + 56 primitive/yes; count-instance 1 + 171; class-program 63 + 171. Root nodes whose callee's kind and key disagree: 0, 0, 0. So way (1) should move no verdict on them (inferred until the probe says so).

2. A SURPRISE, in the difftest, not in a served Program: THE CHECK'S OWN DIFFTEST MAKES PRIMITIVE UNITS ON PURPOSE. scripts/record_difftest_vectors.py's random_circuit makes 8% of its templates a primitive (fn = prim(...)), and the reference carries the form (template_gates, passed_on and hands_on each have an isinstance(fn, PrimitiveDefinition) branch). Counted at eef1ce17ff (ran locally, the generator's case(seed) on all 600 seeds): 117 of the 600 random cases have a template that is a primitive; 72 of them are accepted today, 4 are already refused at templates. So #8676's check was written to take a unit of a primitive, and under way (1):
   - 72 accepted cases of the pinned difftest become refusals (templates) unless the generator stops making such templates; I change it on the refusal's branch to wrap that primitive in a composite of one gate (same draw, the other 483 seeds' cases unchanged), and pin the data again;
   - the new case I added (a primitive unit, alone in its record): the reference at eef1ce17ff reads it past its templates (it refuses at wires, as it does the same template wrapped in a composite); with the change both refuse at templates. Ran locally, Python side only.
   This is no verdict of a served Program, so I do not read it as your STOP; it is what lean should know when it settles (1) against PROPERTY's default-body lemma: the form is exercised today, and way (1) retires it from the difftest.

3. WHAT I DO NOW, as your 13:44Z orders: the refusal on its own branch cursor/record-units-composite-3b1c (from eef1ce17ff; nothing onto #8676 until you pass lean's answer); the Lean probe on a node on the three Programs (old test against new); then the difftest suite and the package's Lean check on that branch.
   A NAME: I renamed keyedAsInput to miskeyed (a root node's callee is a primitive exactly where its key is an input id, else refused), since the old name no longer says what it tests. Two proofs name it (RecordRead.lean's two lemmas, my RecordGates.lean's ReadProg); they follow in the same pass. Say if you want the old name kept.

4. Sent 13:45Z to WIDTH and you: ResolveWithin.lean compiled as written; commit d8d914e4c5 (one file), head 0a3af3096b with COUNTS' fe99d83c7f69 merged; run r20261111-133905-03cb.

