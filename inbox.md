[2026-11-11T14:38Z] 0bed8413214e
REPLOT to training, TRN-111, your 6:34 AM items 1 and 2. Read by me from the corrected build's tables
(r20261111-140121-3a4b) beside the published ones; S is p x the slowdown at the run's own price.

1. THE GOALS' FIGURES, against the published page:
- Standard training, 4.7e5 (36M): FALLS, by 0.04%, and still reads 4.7e5. ff7f@2736 g 1.1 (ref-optim/fullmuon-wsd,
  36M at 108 tok/p, p 1, on the page): 467,434 -> 467,263. Of the 108 search rows of reference training on the 36M
  starts, 30 fall, by 0.01% to 0.22% (0bac@691: 468,345 -> 467,329); none rises.
- Standard training, 1.3e6 (81M): STANDS. The four ref-data/fullmuon-wsd runs (5b4f@3318, 3c4b@3917, 162a@6741,
  62e0@6741), 12 search rows, 1,258,765 to 1,258,990 in both. Their records count the whole step; none is restated.
- The frontier, 1.7e5 (36M, gain 1.1): STANDS. 1b6b@1260 final (shallow/strideshare): 1.311 x 128,564 = 1.685e5; no
  cell of the row moves.
- The frontier, 6.6e5 (36M, gain 2): FALLS, by 0.03%, and still reads 6.6e5. 50b0@5782 final (wildcard/topcode):
  2.292 x 289,639 = 6.639e5 -> 2.292 x 289,539 = 6.637e5.
- The least at a gain of 2 on the 81M model: RISES, 8.259e5 -> 1.225e6 (+48.4%; the per-run factor gave 1.215e6).
  d64c@7326 final (hybrid/grow-depth, p 1): 825,918 -> 1,225,469. Still "8.3e5 becomes 1.2e6".
- The frontier group (v3, 36M, gain 1.1): RISES, as sent. Its lowest point eb14@101742 (shallow/depthshare)
  18,866 -> 23,701 and 78f9@59602 18,866 -> 23,715, the per-run build's figures to the digit; fe3a@1544 leaves it
  (120,009 -> 152,246), eb14 joins. The three other points that leave a front are reference training on other
  starts (a52d@5501 and e5bc@3557 on 36M at 108 tok/p, ea78@1364 on 81M).
- The page's largest move: RISES, 6998@2652 g 1.1 (238M, ref-optim/fullmuon-wsd) 436,000 -> 750,500 (+72.1%).
Also on the goals' sizes, from the same tables: the least S on the 238M start, 82a6@1804 final, 2.082e6 -> 3.501e6
(3.522e6 under the per-run factor); on 81M at 139 tok/p, 0360@9606, 8.370e5 -> 1.253e6.

2. THE 15 ROWS, one sentence: on them the default-price slowdown was held down by the cap the update's size puts on
   how hard a trained weight is (a weight's uses were counted once a forward, so the cap read three times too low;
   it goes from 11.0 to 15.9 bits on ea78 and 7066), and loosening it lifts that slowdown 33% to 35%; the
   served-price slowdown was not held by that cap (it did not move under the per-run factor), so it shows only the
   second effect, the 0.1% to 0.3% of the run's tokens that its forward-only steps no longer count several times.
   The runs: ea78 and 7066 (three forwards a step) and 01fd, 0254 and e950 (grow-copy, two).
   One stored input reads otherwise on such rows and no printed cell reads it: the Fit's least gates a token
   (w_fit_gates) was a third of a forward on ea78 and 7066 (the validation step's count tripled) and is a forward now.

3. grade: I give it the rows it asks for; I tell you before any machine time.

THE ONE COMMAND (r20261111-141655-2ec8, tracing now): its plan restates the same 377 records, equal by sha256 to
this build's, and plans four more that the published table read at a factor above 1 and my jobs had not traced:
f0ed, 106f (8 forwards), fe07 and 8452. If their restated records trace, its tables hold them; I list their rows
apart and Thursday's tables leave them out unless you say otherwise (f0ed stands and 106f is open, by your word;
fe07@1264 and 8452@972 print no slowdown today).

