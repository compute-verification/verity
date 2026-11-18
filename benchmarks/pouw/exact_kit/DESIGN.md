# The exact kit: design

## A script as a recorded run

Without `pipefail` the run takes `tee`'s exit code, and a failed verdict records a passing run.
