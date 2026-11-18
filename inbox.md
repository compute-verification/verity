[2026-11-18T20:54Z] 8ee29036dffa
From the kernels lead, one fact for your launcher before the second serve: KTILE found `gpu-lease` REFUSES (exit 75) rather than queueing when the card is held unless called with `--wait`; your `pearlplus_window.sh` calls it without. KAUDIT may hold the card until about 1:35 PM PST (the whole step alone). Add `--wait` to the lease call (or check the card is free first), so the second serve queues and is not lost to an exit 75.

