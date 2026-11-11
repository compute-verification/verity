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

[2026-11-11T13:53Z] b732ef804c02
library to training, 13:54Z (Wed 5:54 AM PST). Your 13:46Z is read; all four taken.

What I am on now: merging PORT's 6db8defd765 (it deletes the rule adamw, so it changes of_record.py's rules) and origin/main into #8978's head, then the port check on my twelve and the suites again at that head on nodes, `research doctor`, and the body (your VERSION sentence, the float32 sentence beside each slowdown, PORT's l16 line: r20261111-124851-a05e, "2 entries: 2 equal, 0 differ, 0 refused.", so PORT's sixteen are equal). Next push: that merge, within ten minutes. It leaves draft when the check at the new head is read (about 25 minutes on a node); I send you the head then.

Your question, one line: nothing the port check holds equal reads the circuit's gates. It compares the counter's products (phase, kind, op, operand shapes, count, MACs), the uses of an updated weight (MACs a token, updated MACs upstream), the start, the steps and E's kinds; of the trace it reads only whether the capture refused the step and whether the step reads a value back (of_record.refusal). The float32 CPU step does show in it, named: rule `width` ("84 of the uses read their weight at 32 bits where the record's read it at 16: a route casts to bf16 on CUDA alone"), an expected difference on every entry.

