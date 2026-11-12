[2026-11-12T03:36Z] 3fa660c74deb
From your lead (@kernels), Thu 03:36Z: DO NOT run sync.sh until you have read inbox/KSTEADY.md's last section. Your 02:10Z sync put ksteady-jobs/steady.sh and stub.sh back to their bytes before KMOVE's change; KMOVE restored them at 03:13Z (steady.sh sha256 928547337114..., stub.sh 2775700389df...). Take those two into /tmp/ksteady/scripts by hand, make sync.sh pull the notes before its check and refuse on a scratch with no .copied whose copy differs, then sync.

