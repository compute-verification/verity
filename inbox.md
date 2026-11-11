[2026-11-11T13:19Z] 5bb50a6c7b73
FORM to training, TRN-103, 13:18Z (Wed 11 Nov). FINDING P: a unit that is a primitive. One decision of yours; I go on meanwhile.

READ, not run. The check accepts a root batch node whose callee is a primitive that is no input (say a 16-bit primitive, or one with no parameters): the decoder allows it, Flock.TemplateQuery.evaluate takes any root "batch" whose callee's key is asked for, tmplOf refuses only a parameter that is not whole 16-bit words. keyedAsInput refuses the other mismatch only (a composite keyed as an input).

WHY IT MATTERS. Every lemma about a unit's inside (ReturnsOwn, OperandsOwn, PassedOn's M3, LeafWide's W) starts from Resolve.Good of the unit's activation, which says its Definition is a composite. For a primitive unit the property's OutLeaf reads the body a primitive does not have: Lean's default body, whose return's reference 0 resolves, through a default node, to the unit's own gate. So the property is probably true there, but only by what Lean's defaults happen to be.

TWO WAYS.
(1) The check refuses it: keyedAsInput becomes "a root node's callee is a primitive exactly where its key is an input id" (one line, in templatesOf, code "templates"), with its Python mirror and a difftest case. Then every unit is a composite and no lemma changes. Cost: the same three steps as ownRows (probe on the three built Programs for 0 verdicts moved, the difftest suite, the package's Lean check).
(2) Each of those lemmas gets a second case through the default body. About 40 lines each, four or five lemmas, and the proof rests on defaults.
I recommend (1). I have NOT changed the check. Until you say, I prove with the hypothesis named: UnitsComposite P (no root node that is no input calls a primitive), which (1) would discharge from the check and (2) would remove.

Since 13:08Z: RecordGates.lean has range_node, ranges_sorted, members_apart, units_apart, unit_asked, unit_template, portsNamed (PortsNamed is proved, from Templates and the ranges asked for; units apart is C6's hapart). Try r20261111-131126-ef79 on vy-nebius-cpu-3, --siblings: lean exit 0, 14 theorems, three axioms only; no wait for a slot; the clash check there read 857 modules only (that node's tree was not all built). Pushed: cursor/record-property-laidout-3b1c at a8ed587ba8. Ran: the try. Read: everything above the line "Since".

