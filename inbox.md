[2026-11-18T04:25Z] 5c7806125787
From the proofs lead: kernels' worker has taken `commit_slots/` and `sha512_rows/` (row 17) and probably `commit_gpu` (row 16): do NOT open or size them; your part is the stager's stage_salts and our callers only. Also: soundness confirmed a wire's bits are not fresh; lean withdrew its own SampledGuarantee: Spot.Sampling.Sound is the sampling layer's one statement, and soundness states tonight the link from Sound to a scheme's GetsBack/AuditBound at Fin n.

