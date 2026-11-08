# KSTRUCT inbox (from the kernels lead)

## 2026-11-08T02:10Z
- circuits (Slack 01:52Z) owns this work as **CIR-191**, due Mon 9 Nov, and asked for you: "yes please, lend the cloud worker; the brief is CIR-191 (struct_lower lays out assertions and table reads, so FlashInfer quant and SparseMla Combine lower structurally, then RMSNorm; templates agree with flat; sizes against descriptor). Circuits directs and reviews."
- So: read CIR-191 (Linear MCP `get_issue` CIR-191 and `list_comments`) and follow its brief where it differs from mine. **Circuits directs and reviews.** Ref it in the PR body (`Refs CIR-191`), and post your scope and PR on CIR-191: `research msg linear comment CIR-191 TEXT --worker KSTRUCT`.
- Node A (vy-spot-glm53) is shared one GPU at a time from 04:10Z to Sun 18:45Z if you need a GPU (`research run --on vy-spot-glm53 --gpus 1`, --timeout inside the window). Your work is mostly CPU, so prefer `research run --on auto`.
