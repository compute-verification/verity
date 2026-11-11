[2026-11-11T13:04Z] b9ab30ec6e70
grade to training, 2026-11-11T13:04:38Z (from `date`). Two things.

1. TRN-102, 106f: POSITIONS NULL AGAIN; ITS THREE ROWS STAND. Posted on TRN-102 as training/grade.
- Ran (r20261111-105852-439e, cpu-7, 112 GB declared, peak 79.7 GB, rc 0; ended 12:57Z): with the fixed recorder
  (12e82e884b25) under both rules, positions, context_fwd_bits and context_bwd_bits are null in 20 of 20 rows, no
  data-dependent read. The two rules' records differ in version strings and stamps only; against the stored records
  (te_core 4.12.11, 4.12.13) each differs only in te_core, the stamps and the newer capture's keys (capture_code, he).
- Ran: its three published rows (search @324 at gains 1.1, 2, 11: rho_default 52315.26, rho_served 792.10, rho_own
  98094.81) equal the published row in every cell under both rules; no overlay row; no other run's row differs from
  the control, and the control equals the published table for these rows.
- Read in the trace's log: three recordings; the program plans 325 steps at the base size and 682 at the halved.
- Inferred, not run: the program's own schedule, as f0ed's. So all seven rows of the two runs stand; nothing entered
  the table. TRN-102 has nothing left of mine.

2. TRN-117, your 13:03Z message (the tokens fault on 168 search rows, 37 on the page). WHICH I DO: BOTH COLUMNS.
- Now: the as-if from the stored records, the published base (running: r20261111-130100-9098, cpu-7, 72 GB
  declared; a first launch, r20261111-125213-9eba, stopped at the price check in 2 s: the inputs tar held main's
  gate table from before the layout move, same prices, other module names). In what I send, each of the 168 rows is
  marked and set apart: its cells are printed as "on the published base, which has the tokens fault", never beside
  the others as if sound, and the least, median and largest factor are given without them and for them alone.
- Then: the same as-if started from REPLOT's restated records, when their ids exist (by Thu 12 Nov 6 PM PST, your
  message), which gives those rows their second column: restated base, before and after. I ask REPLOT by inbox now
  for the ids when its redo ends and for its list of the 168 rows.
- Nothing is published and no stored record changes: the job reads K and writes only its own scratch and out.
- Part 2 (every other op at s-t20): being built now, on a CPU node; onecalc's seven points are in hand.

[2026-11-11T13:04Z] 4ce5d309eb0f
FORM to training, Wed 11 Nov 13:04Z (UTC), TRN-103. W IS SENT TO COUNTS at 13:04Z (its inbox; the text is
/tmp/trn103/inbox-counts-w.md here, and the same message is below in short).

1. WHAT COUNTS HAS. Branch cursor/record-leafwide-statements-3b1c at 0690666f63b0 (scratch, statements with sorry,
   never a pull request): LeafWidths.lean (definitions) and LeafWidthsAsk.lean (13 statements and resolve_wide proved
   from two of them). RAN: r20261111-130112-1b13, vy-nebius-cpu-1, no wait, lean exit 0 in 106 s: every statement
   elaborates. Order given: (1) wAt_mem_rangeWidths, homo_wAt, cover_spec; (2) eqRange_sound, the long one; (3)
   residues_mem, partOk_sound; (4) seqBad_sound, bodyBad_sound; (5) wAt_widths; (6) gate_of_node, resolve_node.
   5 and 6 read nothing of 1 to 4, so a second worker can take them. NOT COUNTS': decoded_checked and decode_keys
   (the two frozen files, lean's word).

2. TWO STATEMENTS MOVED BEFORE THEY WENT, both found by reading the code against them (read, not run):
   a. partOk_sound WAS FALSE as I first wrote it. A strided part whose pattern has length 0 (inner = 0) passes
      Widths.partOk with nothing checked: both residue lists are of count 0, so the closed form's "for all" is over
      nothing and returns true (Program.lean 646-649). The parser refuses inner = 0 (line 276), so no decoded Program
      has one, but the decoder's proved invariant (PartNonneg, PartValid) does not say so. Fix: new definitions
      PartInner and BodyInner in LeafWidths.lean; partOk_sound, seqBad_sound, bodyBad_sound take them; and
      decoded_checked (W1a) now concludes BodyInner too. So W1a is THREE facts of the decoder, one pass: the
      width check ran, retType = ret, and every strided part has inner > 0. The third is a copy of the parser's
      step lemma in BodiesOk/Nonneg.lean (195 lines, not frozen) plus one conjunct threaded through NodesOk and
      BodyFacts (the frozen two). Not a soundness finding about the decoder: the parser is right, the lemma was
      short.
   b. gate_of_node lacked hD (a primitive is one gate, CalleeOk). Added; resolve_wide still proved from the two.
   W2's hole is marked in the message, with what I did not walk: resolve_node's three scan cases.

3. MY BRANCH cursor/record-property-laidout-3b1c is at 7c56b62dde1a (pushed). Since 78a418d0b3: COUNTS'
   d0333c11f8 merged (RecordLanding, C1); RecordGates.lean (unitsComputed, gate_located, gatesOwn: UnitsComputed and
   GatesOwn of a Program the check reads, under ReadProg, which carries RootFirst as its last clause);
   evaluate_asked in RecordUnits; LeafWidths.lean; all in the one root RecordCounts. RAN:
   - r20261111-124428-060e (joint, no wait): the W statements' first elaboration and RecordGates' first build, rc 0.
   - r20261111-125332-fcfd, RecordGates.lean --siblings, vy-nebius-cpu-5, no wait: lean exit 0, 7 theorems, the three
     axioms only, no name clash (2,496 modules).
   - r20261111-125719-19da, RecordCounts.lean --siblings at 5363a4c1829e, vy-nebius-cpu-5, no wait: lean exit 0 in
     153 s, so the 14 modules build together on my merged tree, RecordLanding among them (COUNTS' "not tried on
     FORM's head" is now tried), no name clash (2,511 modules). No theorem of its own to probe.
   Not run: check.py on the package, the lock. COUNTS' RecordLeafHolds (5ab325f38aa5) is not merged: it changes to
   one proof first, as you told COUNTS.

4. LeafHolds, the short answer now (read, not run; the paragraph on Thursday): it does not shorten W1b. Its clauses
   1 and 3 (a unit's parameter leaf) need the width check only on the root's arguments, whose sources are input
   nodes of one width: three of the eight statements, in a restricted form. Clause 2 (a leaf of a unit's output row)
   resolves down through the unit's own body, any nesting and any types, so it needs all eight. "≤" for "="
   saves nothing: the decoder's check is the only fact tying a gate's width to its leaf's, and it is an equality.
   Nothing but the landing bound reads LeafWide (grep at my head), and LeafWide/LeafHolds are hypotheses of the
   proofs tree's lemmas, not clauses of lean's property, so nothing in the property changes either way.

5. Suites at 83102b1b53 (r20261111-114434-b28f): still running at 13:03Z, two of six in and passed (research,
   pouw/verifier). Your item 2 (the three Programs and a shared Input gate) I answer with a run or a reading in my
   next message.

