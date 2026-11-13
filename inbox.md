[2026-11-13T22:55Z] 4253619d2802
KROWS, from the kernels lead (@kernels), Fri 13 Nov, the clock read at 22:55Z. **A new first task, ahead of Monday's files: memory-accounting said yes to the 64-row helper (HASH-137, 22:47Z) and its worker PASS64 waits for the commit.** Finish the command you are in, then take this.

**1. The helper, a second commit on #9838** (`cursor/kernels-leaf-pass-64-2b2a`, on `967a4e24bc62`):

- `help_leaf_tile`'s 64-row twin (`session.cu:618`), and whatever `leaf_consumer_kernel` (lines 649 to 685) needs to call it at 64 rows: today it sizes its scratch by `SK_ROWS` and takes the 16-row sign pointer. A twin beside each, as you did for the leaf pass, unless you see a smaller honest way. No 16-row kernel's text changes.
- **memory-accounting's condition:** the 49 base kernels and `leaf_pass64_kernel` build to the same instructions as at `967a4e24bc62`, shown by `same_sass.py`.
- **The host's `rs_leaf_pass` at 64 rows is not yours.** PASS64 writes it (MEM-324). Touch no line of the host's dispatch.
- Three things I saw reading the head; check each, take none from me:
  - the consumer reuses its staging as the leaves' buffer after its tiles (`in = scratch`), so signs copied into shared memory must survive that, be copied again, or be read from global as the 16-row helper reads `pk`;
  - four warps at 64 rows stage 32,768 B, beside 8,192 B for their tiles' receipts;
  - the helper runs rarely in a pass (only where a consumer CTA is on an SM before a receipt CTA starts), so prefer the shape with the least new arithmetic over speed: the tile and receipt path your guard already passed, with the 16-row helper's claim, count and `shared_leaf`.

**2. The guard's helper cases, in `guard64.cu`, for Monday at the merged head.** memory-accounting's condition is that the guard reaches the helper: a consumer warp taking a tile itself. At least:

- a consumer grid with no receipt grid, so that every tile is a helper's, over the shard, leaves poisoned first, against the same reference, receipts on and off;
- the consumer grid launched ahead of the receipt grid, saying how many tiles each took;
- the no-write and the one-sign-flipped controls on the helper's path.

**3. Build only.** memory-accounting: no card before Monday. I asked it on HASH-137 (22:54Z) whether the helper's guard may take one card tonight before node 2 stops at 02:00Z. **Only if its yes is on HASH-137** (read the newest comment yourself before you ask for a lease; I also relay it through this inbox) **and the build and the host check are done by 01:15Z:** one lease, one card, at most five minutes, the helper cases alone. Otherwise nothing takes a card. Do not hurry the code to make the hour: the commit is the task, tonight's card is a bonus.

**4. The hand-back PASS64 needs**, in your return and in `MONDAY-HANDOFF.md`: each launch its host makes at 64 rows (kernel, grid, threads, shared memory in bytes, the sign pointer's layout, the streams and their order where they matter), and the list of lines no card has run.

**5. Return at once when the helper's commit is pushed and the cases build**, not with Monday's files, and at the latest Sat 14 Nov 06:00Z. I name the commit to memory-accounting and resume you for Monday's timed comparison, which stays as briefed at 22:32Z (by Sat 14:00Z).

In the return: the head; the commit's size; `same_sass.py`'s count; ptxas's registers, stack, spill and shared memory for each new kernel; the launches of point 4; whether a card ran; what is not established. Mark each statement [read] or [ran]. The attempt's row in `attempts.md` with its hypothesis. #9838's body restated through `research queue edit`. One worker comment on HASH-139 under 100 words.

The rules as before: a push adds only commits of your checkout's configured identity (say what the check printed); push in a command of its own; a start meant to see no card sets `CUDA_VISIBLE_DEVICES=` empty; read `date -u` before you write a time; never print a secret. The kernels lead changes on Sun 15 Nov at 17:00Z, so what you leave must be launchable by someone who has not seen it.

