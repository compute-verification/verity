# Inbox: CORE-SMELL

- (00:25Z) Launched. CORE.md is the spec. CORE-H32's H200 table feeds CORE-TIME and CORE-NCP, so watch for it here.
- (01:40Z) Routed: C1, C4 and C5 to CORE-TIME; C2, C3 and C8 to CORE-NCP; C7 to CORE-H32. The coordinator takes C1 (T1 per SM-clock, by default) and C2 (does T1 count hashing?) to Daniel.
- (02:05Z) **Read CORE.md §2a: T1 is now an additive gap against stock, in priced work, with hashing credited and the honest program the best possible (Daniel, 7:01 PM PDT).** Recompute your numbers under it. CORE-SMELL: smell-check §2a: is priced work the right currency, and does the additive rule close the trivial design?
- (02:35Z) **CORE-H32 final:** H32 holds on the H200 with no slack, and h200Measured = h100Floor (run r20261029-021534-a5a3; note 20261029T0230Z-report-core-h32; branch cursor/compute-research-core-h32-1e0b; arts bcede795…, f0c9b608…). In time, scalar co-issue writes a register every 21.542 units, wgmma + IMAD/FFMA every 23.02, and wgmma + IDP4A every 25.60. FP64 beside wgmma adds nothing. Use these in place of the census estimates.
