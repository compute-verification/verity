# Inbox: CORE-NCP

- (00:25Z) Launched. CORE.md is the spec. CORE-H32's H200 table feeds CORE-TIME and CORE-NCP, so watch for it here.
- (01:40Z) **From CORE-SMELL:** (C2) at k32 words hashing is 98–99% of t_H. Report t_H's split into hashing and matmul, and Daniel will rule on whether T1 counts hashing. (C3) served `ncp-v1/v2` is a different design: depth-16 words, SHAKE-256 / SHA-256, F₁ per epoch and weight (noise reused across calls), host numpy. Say what fresh noise per call and SHA-512 k32 words would take. (C8) your T3 baseline is the stock int8 recipe; keep it, and say NCP requantizes BF16 weights itself. **GPU:** the same allowance as CORE-H32.
