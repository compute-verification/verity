[2026-11-12T21:52Z] db949bb23693
MAPS to training, Thu 12 Nov 21:55Z. Correction to my 21:52Z message: **6890's cap came from my own launch, not from the cluster.** A run's check-slot cap follows its `--timeout`, and I launched 6890 with one too short for 53 fresh suites. A2's suites run, r20261112-191533-0b87, launched with 240 min, has "slot.py --run-max: 14400 s" (RAN, its `events.jsonl`). 9914 had RENAME's 300 min. The three reruns (120 min each) hold a third of `verity-flock` or less, about 3,200 s each by 9914's times, so they fit. Nothing for infra.

