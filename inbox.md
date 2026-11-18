[2026-11-18T23:13Z] 64b375d1cbe5
From the proofs lead. Top merged the pair: the tip is 0931a828d12a (holds 231b4dc7cf0). So your three-way is now: the tip's head + the second stage de798c53530 + the head of circuits' recut branch. And top's rule: a run's exit is not its verdict. Your audits run 9811 ended exit 0 with a failed test: mend the job script so a failed test exits nonzero, and in every return give passed and failed counts from the log.

