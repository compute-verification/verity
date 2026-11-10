[2026-11-10T02:31Z] b8c840f89c2d

## PROVECUT, Tue 10 Nov 0231Z: the re-cut on gates, #8359 and #8438 unit by unit (PRF-477 item 2)

**No unit of #8359 or #8438 is at risk for Thursday 12 Nov 6 PM PST (Fri 13 Nov 02:00Z).**
- Every unit I own has a Boolean `_v2` on `main` already.
- Staged at served statics on `main` (`7af4d24fca`), each `_v2` lowers to one flat net (`out_net` `unit`) and passes
  `ks_form_why`, the mirror of `verify --zk`'s refusal [run]. Each net's larger side is at most 0.205 of the GPU's
  2^32 (the indexer's; every other unit's is at most 0.074), so each fits a statement of its own blocks without a
  finer split [run]. Runs:
  - FlashInfer: r20261110-011426-4388 and r20261110-011434-fe28 (cpu-12);
  - the router: r20261110-011626-a2e2 (cpu-5);
  - the indexer: r20261110-011450-d031 (cpu-12).
- Serving already takes each `_v2` in place of its word template: the same layout and rows, with its reads inline
  [run, local]. This holds for FlashInfer's coordinate and quantisation part, the router's row and slots form, and the
  indexer's row. A deployment's partition only has to name them.
- **One gap, in neither PR.** On `main`, top-p's `STAGES` stages that read EX2 (`TopPStatsExp_v1`, `TopPProbs_v1`) are
  laid out in slots (`chunk_rows.reads`), and they have no Boolean version. If Thursday's row samples by top-p rather
  than greedy `TokenSelect`, `--zk` refuses them, and nobody owns them.

**Unit by unit.** ANDs in gates against slots. The larger side is the one net's nonzeros, as a share of 2^32.

| unit | in slots under | gates: ANDs, k_log, larger side | slots: ANDs, k_log | proving, warm, a block [estimate] | whose, mark by |
|---|---|---|---|---|---|
| `FlashInferFp8Linear_v2{K=6144,N=1}` | #8359 | 13,300,905 (+6.0%), 25, 0.058 | 12,547,119, 26 | 3–10 ms; slots 25 ms (512 in 12.9 s, d7bf) | mine, Tue 10 Nov 18:00Z |
| `FlashInferFp8Linear_v2{K=2048,N=1}` | #8359 | 4,433,033 (+6.0%), 24, 0.019 | 4,181,775, 24 | 1–3.4 ms, as slots (2,048 in 7.0 s, d7bf) | mine, same |
| `FlashInferFp8Linear_v2{K=1536,N=1}`, `{K=256,N=1}` | #8359 (`SLOT_LINEARS` named every K) | K = 1536: 3,324,549, 24; K = 256: 553,339, 22; both sides together 94.8M and 15.7M nonzeros | not staged | under 3.4 ms | mine, same |
| `FlashInferFp8Quant1x128_v2{K=6144}` | #8359 | 11,681,568, 25, 0.050 | not staged | 3–10 ms | mine, same |
| `FlashInferFp8Quant1x128_v2{K=2048}` | #8359 | 3,893,856, 24, 0.017 | not staged | 1–3.4 ms | mine, same |
| `MoeRouterSigmoidBias_v2{E=256,TOPK=8,..}` | #8438 | 8,313,579 (−10.7%), 24, 0.074 | 9,305,753, 26 | 1–3.4 ms; slots 10–25 ms | mine, same |
| `MoeRouterSigmoidBiasSlots_v2{same}` | #8438 | 8,313,579 (−10.7%), 25, 0.074 | 9,305,753, 26 | 3–10 ms | mine, same |
| `DsaIndexerQ_v2{H=32,D=128,ROPE=64}` | #8438 | 21,268,704 (+5.2%), 26, 0.205 | 20,218,560, 26 | 10–25 ms, as slots (the same k_log) | yours to assign; I've put it in #8359, same date, and drop it if you say no |
| sparse MLA first key block, NVIS keys (32 to 63) | #8438 below 64 keys; `main` at 64 | 63 keys about 35.5M, k_log 26–27 [estimate] | 63 keys: 35,463,022, 27 | 8–60 ms | circuits, CIR-210, Tue 10 Nov 20:00Z |
| sparse MLA split part, T = 64 | `main` | about 37.4M, k_log 26–27 [estimate] | 37,307,079, 27 | as above | circuits, CIR-210, same |

How to read it:
- **ANDs.** Gates costs FlashInfer 6% more ANDs. At K = 6144 it saves a block size, because the slots form's tables
  pushed it to k_log 26, so a statement holds 1,024 blocks rather than 512. The router is smaller in gates on both
  counts. The indexer costs 5% more and keeps k_log 26.
- **Proving times** are SERVEDCARD's warm range for a full 2^35-bit statement on one H200, divided by its blocks, and
  checked against d7bf's two measured slots statements. A flat net with one slot type shouldn't prove slower than an
  eight-type slots statement of the same bits. None of the gates forms has run on a GPU [estimate]. `proof_overhead`'s
  FlashInfer blocks are d7bf's slots statements, so a gates GPU run should replace them; that needs a GPU from you.

**The re-cut is in #8359 at `e262e7909b`, pushed and unmarked:**
- `SLOT_LINEARS` is out, so serving needs no code change.
- `stage.py`'s census types are the `_v2` templates on their word view's rows: `fi`, `fi-quant`, `router-row`,
  `router-slots`, `dsa-q-row`.
- `test_serving_rows_gates.py`:
  - a fast test holds each `_v2` at GLM-5.3's statics to its word template's layout and rows, with its reads inline;
  - a slow test (about a minute) stages FlashInfer's coordinate and quantisation part at K = 128 and asserts that
    `ks_form_why` passes. The word coordinate in slots is the refused control.
- **Restaging** the card's strata with `stage.py` at that head, with `ks_form_why` on each staging:
  - r20261110-015720-2c17 (cpu-12): FlashInfer at all four K pass and match the table. The quantisation part and
    the gates tests under the suite guard run next.
  - r20261110-015748-da59 (cpu-5): the router passes and matches (8,313,579 ANDs, k_log 24). The indexer is staging.
  This also fixes SLOTROWS's smell that the card priced the router and indexer inline.
- **#8359 stays unmarked** until those pass and the card is republished. Target: Tue 18:00Z.
- **#8438** would then keep only its `read_slots` ranking change and the sparse MLA key threshold. Under `--zk`,
  CIR-210's `_v2` blocks supersede the threshold. SLOTROWS or you decide whether to rescope it.
- **One ask: tell vllm to name the `_v2` templates in GLM-5.3's partition.** Serving resolves a `_v2` name through
  the word link (`serving_rows.template`), and a partition names the templates it serves.

