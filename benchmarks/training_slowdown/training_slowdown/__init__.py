"""training_slowdown: the code that measures a training run's slowdown (README.md). A change to what it measures bumps
VERSION, which every record carries."""

VERSION = "4.17.0"
#: a job's exit status when it was told to stop (SIGTERM) and left its state: sysexits' "temporary failure, try again"
STOPPED = 75
#: a job past its budget's minutes by this factor is stopped at its next step
BUDGET_SLACK = 1.2
#: a job's `--continues` when whoever starts it found no state for it to continue from
NOTHING = "nothing"
