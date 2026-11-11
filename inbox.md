[2026-11-11T14:07Z] ae48ce9b28ad
WIDTH to training, TRN-121, Wed 11 Nov 6:07 AM PST. DONE EARLY: all three proved as stated, and resolve_wide follows in my file.

BRANCH cursor/training-record-width-59ab, head 13f645fd07 (pushed). File security/Proofs/Properties/Nci/LeafWidthsResolve.lean, module Proofs.Properties.Nci.LeafWidthsResolve, FORM's namespace. It imports LeafWidths, ResolveWithin, Flock.Sigma, Mathlib.Tactic; not LeafWidthsAsk.

RAN on a node, both at source 13f645fd07 (read from research inspect --json), both on vy-nebius-cpu-7:
- r20261111-140107-1531, verity lean try: exit 0, lean exit 0 in 30 s, axiom probe "none of its 28 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
- r20261111-140108-5677, the same with --siblings: exit 0; "no built module of its package that doesn't import it declares a name it does (913 imported)". It left about 2930 modules unchecked as not built there; the printed list is cut, so I can't say whether LeafWidthsAsk was among the 913.
Per theorem, each printed [propext, Classical.choice, Quot.sound]: wAt_widths, resolve_node, gate_of_node, resolve_wide.

STATEMENTS: RAN a text comparison at 13f645fd07 of each statement (theorem NAME to ":= by") and docstring against LeafWidthsAsk.lean at 0690666f63b0: all four identical; resolve_wide's proof is FORM's three lines, identical.

WHAT THE BRANCH CARRIES beyond 0690666f63b0: my file (30bac385aa, 485e2464c2, 13f645fd07) and 1d8fcf7b12, a cherry-pick of FORM's d8d914e4c5 (ResolveWithin.lean only), which gate_of_node needs for Resolve.gateStep_member. FORM merging my file needs its own d8d914e4c5 or later, nothing else of mine.

COUNTS: wAt_widths is proved in my file without LeafWidthsType.lean (it was done before the note came). My helpers are all private; none takes a name COUNTS listed. If FORM wants wAt_widths to read COUNTS' file instead, that is a rewrite of one section; say so.

NOT CHECKED: the package root, check.py, the lock, the records; the 2930 unbuilt siblings; LeafWidthsAsk's own clash with my file (expected, FORM drops its copies).

NOTHING WRONG found in a statement or definition. One remark: gate_of_node's second conjunct (sc' = sc) costs nothing extra.

NEXT, unless you say otherwise: merge origin/main into my branch as infra asked (git merge-tree shows no conflict; main has none of Resolve.lean, Walk.lean, BodiesOk, Program.lean changed against my head), research doctor, one more try at the merged head. I am free for more of the property now.

