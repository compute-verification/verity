[2026-11-13T06:46Z] 34d63733bea1
training, Thu 12 Nov 10:46 PM PST (TRN-109, your arguments message)

**One config for the two depth-18 runs, not two; your names stand; four things for the part 2 design.**

1. **18b2 and 714f are one job, so they are one entry:** `m-t20-2x-hybrid-grow-depth-18`, held to 18b2 in `of_record.toml`. Write no `-b` copy: two configs with one text are one definition written twice. For 714f, run the same comparison against its record once by whatever the command already offers for a record you name (a call of its own functions in a job is fine; no change to `of_record.py`), and give me the verdict and the lines that differ, RAN. Its point is listed in the body as the same job as that entry. Whether an entry may name a second run in `of_record.toml` is mine to decide once the two changes that edit `of_record.py` have landed.
2. Names and targets for the `ref-data` entries: as you propose.
3. For the part 2 design:
   - **Reuse, don't add.** The decay share and the schedule are `subset`'s dials with `subset`'s names and meanings; the average of the weights is route `ema`'s machinery, imported, not a second copy. Say what `grow` would import from each.
   - **The route does not choose between the weights and their average.** The program scored both on held-back windows and handed out the better: an entry choosing by a score is refused. As the fullmuon entries do, the two forms are two entries. Write an entry for each form a checkpoint of the run handed out, and none for a form no checkpoint used (so 6111 and 2137 get the average alone, 7ace and f13e the weights alone, b248 and ca11 both).
   - **Warmup:** say what rate the program takes at each of the first steps, and whether an integer `warmup` gives those rates. A new dial only if no integer does.
   - **Muon's norm:** the dial's default is the held entry's form, and the option's key enters the optimizer's state only when it differs, as `stack` does now.
4. Part 3: your reading is enough to expect that the widened run is listed as waiting. Send the design anyway, short: what the harness's mapped start and the evaluator would each have to take, and whether the noised copies can be buffers drawn at `build` from torch's global generator.

Dates unchanged.

