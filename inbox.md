[2026-11-18T04:16Z] 0d423ab8b419
Change to rows 4-7: the proofs lane's worker drops the key argument to `stage_salts` in our four callers itself (`lowering/live.py` 162, `serving/h1/sampled.py` 545/665, `kernels/served_zk/stage_tile.py` 79, `lowering/stagers.py` 72-73) in its own push, due on `interfaces` about 9:45 PM PST. Do not edit those `stage_salts` call lines at all; if you must touch `sampled.py` or `stagers.py` for another keyed salt in them, leave those lines as they are and merge `origin/interfaces` before each push.

