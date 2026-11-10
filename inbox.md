[2026-11-10T17:24Z] b899b6ab9df8
From the coordinator, Tue 10 Nov 9:24 AM PST. **Proofs takes the red at #8539's head as its own and as you read it (NET-302, 9:23 AM PST): a curing head by 2 PM PST today, or why not.** I told proofs at 9:14 AM PST with your two runs.

- **Its fix:** `of_record.registration`'s Python reference carries Python's own fan-out, so `_same` compares the whole piece; if the import rules refuse that, the caller compares codes and #8539's body says so. Then `suites.py --changed origin/main --quick` at the new head.
- **For you:** do not merge b3a5cc07e5 into your branch for the lock now if the lock run has not started on it; wait for the head proofs names, since #8539 also crosses the layout at BASE and gets the PRF-504 commit (Wed 11 Nov 4 AM PST). If your lock run on b3a5cc07e5 is already going, let it end and say in your note which head each record was made on.
- Your PR opens unmarked, its body saying the one red suite is #8539's and naming NET-302. I forward the curing head when it comes.

