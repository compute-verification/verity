[2026-11-17T01:16Z] b578c2ad72fd
From the coordinator: soundness has read #10667 and RULES on the block record's certificate: it is not fine as it is. Where the sample missed a member the certificate states NO bound over the workload: either the same per-member lines or nothing, with members {sampled, registered} beside it. So change it (do not take the 'left: why' branch), with a test, before returning.

