[2026-11-09T00:31Z] ab9634f0fd42
Coordinator 00:37Z, #8057: go with (A). Port SampledLeakChecked like SampledLeakAccepted; check.py requires a changed statement to carry both witnesses, so (B) is out, and (C) waits too long. The PR body must say plainly: the honest witness is a toy where the check accepts nothing (miss 0), and the deployed-check honest instance stays open under CIR-190 (due 30 Nov). I will tell circuits on CIR-190. Merge checked (never next), try, ONE records run, quick suites, then mark.

