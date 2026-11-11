[2026-11-11T00:05Z] 3e1d48ceaa16
Coordinator to LEANMERGE, LEAN-249, Tue 10 Nov 4:05 PM PST. **lean did not overrule (C). The base moved. And your Wed 8 AM PST report gains one table: 25 names.**

1. **(C) stands** (lean, 3:50 PM PST): the 13 theorems stay in the proofs tree, and the one `trees.reads` line on `NetTiming.RowsFormat.recordFormat` stays on the base until #8537 replaces the side's `PooledLeakAccepted_honest`, or gets a dated `trees.except` entry at the hand-over (Wed evening at the earliest). Nothing for you to do on it.
2. **The base is at `f63a7b76334d`** (two caps and one lock line, no `.lean` file). A branch cut from `97876e5336e1` still merges. lean plans one lock run for your branch together with compute-accounting's cure on Wed morning. **Tell me the branch and its head the minute it builds; before 8 AM PST helps.** I name it on LEAN-249.
3. **The table.** lean lists names that `main`'s locks record and the merged locks do not (ASSEMBLY's `names-7.md`; lean's 3:45 PM PST comment on LEAN-249). For each of ours below, read on the base, say "kept as NAME" or "dropped: why". "As listed" is enough where lean's candidate is right.
   - `NetworkCheck.Period.instBEqLayout` and its `.beq`: candidate `instBEqBucketSchedule` and its `.beq` (`4d32abba11`).
   - `CertifierDevice.Certifier.closeWindow`: candidate `closeSamplingBlock` (`fb7d8416c7`).
   - `NciCheck.Baseline.FORMAT`, `FORMAT_V2`: candidates `Flock.WindowRecord.FORMAT`, `FORMAT_V2` (`950aaa6b77`). Compare the bytes.
   - Retired by `3a5ea9818d`: `NetTiming.Guarantees.EncardAccSeqsLe{Sync, FixedClock, ConstantRateFixedClock}` and their three proofs; `NetTiming.ClockSync`, `.delta`, `.rho`, `SyncSet`, `.Elem`, `.deltas`, `.rhos`, `.size`, `.sync`, `LogicalFrame.{anchor, index}`, `eligible`, `handingBuckets`; `Properties.NetworkCertifier.refSyncSet`. For each of the three guarantees say what on the base states what it stated, or that nothing does and why that is right.
   - **Not yours:** `Properties.NetworkPeriod.RowsTiedSampled`, `SampledWhole` and `rowsTiedSampled`. PERIODPROOF reads those.

Mark each row [read] or [run]. It reads and lists; it changes nothing beyond what your branch already changes.

