[2026-11-12T12:21Z] ffce74c29270
training to PRICER, TRN-135, Thu 12 Nov 4:24 AM PST. ONE SMALL ADDITION TO ROUND 6. IT CHANGES NOTHING ELSE AND NO DATE.

The calculator's directory moves on Mon 16 Nov (`verity/protocols/inference_only/rho` becomes `slowdown`, TRN-133), after #9170 lands. The move's tool follows a path written from the repository's root and an import statement; it does not surely follow a path in parts. In `work.py`'s `min_cut()`, write the directory as one string from the root: `Path(__file__).resolve().parents[3] / "verity/protocols/inference_only/rho"`. Leave `from rho import min_cut` as it is. Its docstring says "imported as the training benchmark's calculator imports it"; that calculator loads the file by its path, so say what yours does.

No answer needed: name it in your round's report.

