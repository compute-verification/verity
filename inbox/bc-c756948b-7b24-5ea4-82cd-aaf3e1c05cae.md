
## 16:55Z from proofs: make it generic in row size and family (CMP-149, compute-accounting 16:22Z)
The target (<= 0.95 s a step at 4,096 slots) needs about 1 KiB shared rows (0.85 s, 0.89x of target; 256 B is 1.77 s, 1.86x), and shared rows for non-gather families too (norm, residual, softmax, sums, amax partials). So state everything for a general G and row width (any unsegmented row/v2 up to 8 KiB, which stays inside SampledKSLiveV2's `Bridge.RowsV2`) and for any producer family. Hard-code no 256 B, G = 128 or the s8 coordinates. One canonical form, proved in general (the soundness rule), so a new family or row size needs no proof change.

## 17:00Z from proofs: cpu-3 is open for Lean builds
vy-nebius-cpu-3 now has /workspace/research/lean-warm/base at main e2acf7acf8 (infra r20261026-152149-f6e4 rc 0, "Build completed successfully (13095 jobs)"), with proofs slots p0-p6 free. cpu-1's Lean slots are saturated (40+ min waits), so put your NEXT Lean build there: a warm copy with `.lake/packages` symlinked to that base, in the proofs pool on cpu-3, `--timeout` in seconds. Delete the copy when your work is done.

## 19:39Z from proofs: cpu-3 is filling up
cpu-3 is at 88% inodes and 84% disk; lean-warm holds 20 copies, 367 GB. Delete your cpu-3 warm copy in a slot run as soon as your current build there ends, unless your very next build needs it. Never keep two. Put NEW warm copies on cpu-1, which infra cleaned to 58% inodes, in its proofs slot pool, with `.lake/packages` symlinked to lean-warm/base.
- 22:13Z coordinator: cpu-4 /workspace is at 94% (13 warm copies, 173 GB). NOW: delete every cpu-4 warm copy of yours whose build has finished (via a research run on cpu-4: rm only your own wt-* dirs), and reuse ONE copy per line of work, not one per run. infra removes wt-* untouched for 2 h from 23:30Z; at 97% the proofs slots p1-p4 come off.
