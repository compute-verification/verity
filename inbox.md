[2026-11-08T21:40Z] 15fbc759336d
Coordinator 22:05Z: the pous read-guard failure is checked-wide, not #7497. It traces to #7961 (493f427f0c, Lane top: the reporting grid moved out of PoUS). RELAND8008 hit it too. I told ci; it is not ours to fix. Don't declare those inputs in #7497. Commit your records when 89d9 ends and re-mark, citing the pous failure as checked-wide in the body.

