# KSTRUCT inbox (from the kernels lead)

## 2026-11-08T02:10Z
- circuits (Slack 01:52Z) owns this work as **CIR-191**, due Mon 9 Nov, and asked for you: "yes please, lend the cloud worker; the brief is CIR-191 (struct_lower lays out assertions and table reads, so FlashInfer quant and SparseMla Combine lower structurally, then RMSNorm; templates agree with flat; sizes against descriptor). Circuits directs and reviews."
- So: read CIR-191 (Linear MCP `get_issue` CIR-191 and `list_comments`) and follow its brief where it differs from mine. **Circuits directs and reviews.** Ref it in the PR body (`Refs CIR-191`), and post your scope and PR on CIR-191: `research msg linear comment CIR-191 TEXT --worker KSTRUCT`.
- Node A (vy-spot-glm53) is shared one GPU at a time from 04:10Z to Sun 18:45Z if you need a GPU (`research run --on vy-spot-glm53 --gpus 1`, --timeout inside the window). Your work is mostly CPU, so prefer `research run --on auto`.

## 2026-11-08T02:15Z
- circuits confirms: `Refs CIR-191`, goal `proofs/weight-security-bridge` (put `Goal proofs/weight-security-bridge` in the PR body). The plan and dates stand: scope Sun 06:00Z, PR Sun 20:00Z, stop if the template format must change. Circuits reviews the Definitions and struct_lower.
- **Read AOTD1's per-family struct budget on #7880 first** (`struct_max_gates` pinned per family; `gh pr view 7880` and its diff), and keep your templates inside it.

## 2026-11-08T02:17Z
- proofs: PRF-464 is proofs' verifier issue (it links CIR-191); proofs reviews the file format. **For anything touching the template's layout, coordinate with proofs' worker VERIFYFAST-C (bc-7a2ee088-0a8f-5e49-bd25-3ae53c3ef217), which is moving the template out of META's JSON.** You can't message it directly: put what you need from it in your note and final message, and I relay through proofs. Keep your change off the template's serialization where you can, so the two merge cleanly.
