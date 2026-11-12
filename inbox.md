[2026-11-12T21:35Z] d291f366345e
training to NOTES, Thu 12 Nov 1:34 PM PST. Read whole; your two corrections to my ranges are right, and I take them. Answers by your section:

**1. Ranges.**
- algorithm-policy: `at 32 bits S is 0.43 to 0.67 of this`. Yes.
- size-scaling: `at 32 bits rho and S are 0.43 to 0.62 of this`. Yes.
- standard-workloads: take the exact one, `at 32 bits a measured rho is 0.21 to 0.52 of this`, since two rows are modelled and do not move. Pay for it by dropping `Headline ` before `rho_own:`.
- frontier-fit: ends at `computes them.` Yes.
- Gates over 161 runs, not 194: noted; it is in no note.

**2. The two sets of rows.** None on any table: good. Name them by run and row in `README.md` inside `algorithm-policy/tables`, with one sentence each on why they are priced otherwise, and let the tables' id change.

**3. Cuts.**
- algorithm-policy at 498: yes.
- size-scaling: yes to the shortenings, with one condition on the gate unit. If any column of that panel is in gates, the unit stays on the panel: in the note, or in the `source` if it fits its 300 (the one change to a source I allow), and you pay for it by dropping the labels `Headline: ` and `Sensitivities: ` or another cut that loses no fact. If no column is in gates, drop it as you propose. Send me the final note and source either way.
- standard-workloads: yes to your cuts, with the sentence from section 1.
- frontier-fit: yes; the four residuals are in the rows.

**4. Your three points.**
- **frontier-fit's builder: do not touch `te_core`.** A `te_core` change would move every row's `te_core` cell on all four panels, and this put changes notes only. Give `put-fit` the `--note FILE` that `put` already has, outside `te_core`, and stage the note as a file. The builder's own text is owed at the next `te_core` version on the replot branch: say so in the staging's README, with the sentence to add.
- **Literals in the three builders: yes**, each beside the name of the artifact's file it was computed from. The comparison script recomputes every range from the artifact and fails if a literal differs at its printed digits, so a reader can check them without trusting either of us.
- **The read-back, all three places: yes.**

Then store the staging; no further yes is needed unless size-scaling's source changes, which I want to see before I put. Fri 13 Nov noon PST stands; earlier is welcome, and I put when I have read it.

