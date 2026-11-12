[2026-11-12T18:28Z] 6c0720e02d4d
BACKWARD to training, Thu 12 Nov 10:28 AM PST (18:28Z). **TRN-143, the report you asked for by 6 PM PST: seven Definitions, matched on captured words in a test; branch `cursor/fa2-train-backward-d232` now at `cb20f4171f` (its shape changed since 10:09 AM, tell SEAT). My date for done-conditions 1 to 3: today 6 PM PST if the CPU pool's line moves, Fri 13 Nov noon PST at the latest.**

**The Definitions** (`verity/definitions/registry/fa2_train_bwd.py`, words, `_v1`; `boolean_fa2_train_bwd.py` lifts each to bits, `_v2`):

- `Fa2TrainBwdPre_v1{D}`: a query row before the loop. (dO_i, O_i, LSE_i) to its dP sum, dO . O in f32 in `dot_do_o`'s thread order, and its LSE in log2 units.
- `Fa2TrainBwdRow_v1{D,BN,NVIS}`: a query row against one key block of BN keys, NVIS of them visible. To P and dS in bf16 (BN each) and the row's dQ part (f32, D). Holds the products S (visible keys only), dP (all BN keys) and dQ.
- `Fa2TrainBwdKeyFirst_v1{D,BM}` and `Fa2TrainBwdKey_v1{D,BM}`: a key through one query block of BM rows, its dV and dK accumulators (f32, D each), from +0 and from the block before. Hold the products dV and dK.
- `Fa2TrainBwdDkDv_v1{D}`: a key after the loop: dK = bf16(acc D^-0.5), dV = bf16(acc).
- `Fa2TrainBwdDq_v1{D,NP,NS}`: a row's dQ in bf16 from its NP key blocks' parts in NS buffers (deterministic mode's order), scaled after the sum.
- `Fa2TrainBwdHead_v1{T,D,BM,BN,NS}`: one head, (dO, q, k, v, O, LSE) to (dQ, dK, dV): key block n's query blocks from the last down to its own first, rows then keys in each. `train_sdpa_backward(B, H, T, D)` binds it at the four captured shapes (BM 64, BN 128, NS = ceil(188 / (B H)) = 12, 8, 6, 4) and refuses the rest.

**Where 1 to 3 stand.**

- **RAN on the VM** (the named test file, 24 s): dQ of five rows of an H 24 head (rows seeing 1, 6 and 128 keys of their diagonal block, a row over two key blocks, a row over five in four buffers); dK and dV of the last key block of an H 16 head, every word; a small head against the kernel's loop written out; each Definition on bits against its words. One word of the five-block row differs under the sum in the key blocks' order, so the test tells the kernel's buffers from it.
- **The fixture** is 227,163 bytes, cut by a generator beside it, its id only in `fixtures/artifacts.json`. The capture's two ids are in `verity/definitions/evidence.json`, where the forward's are and where the Definitions' conformance text reads them. Say if you meant otherwise.
- **Queued on CPU nodes at `cb20f4171f`, 7 to 13 checks ahead at 10:26 AM:** circuit-check on both sidecars (`r20261112-182341-0064`), the suites the change touches (`r20261112-182341-8493`), and every captured word of the 64 heads through the head Definition (`r20261112-182341-2949`, about 12 minutes a head on the reference evaluator). The draft PR opens when circuit-check and the suites are in.

**What is hard.**

- **A tile can't be a Definition.** On bits it is too large to build at any size, and circuit-check's by-calls root can't call another. My first circuit-check run failed on exactly that (3 findings, `by-calls/unbound`; RAN, `r20261112-180442-35f3`). The head now calls rows and keys itself, as the forward's does.
- **The every-word check is a node job, not a test.** A head is 21.3 million word gates. The test holds a cut near the 256 KiB cap, because a row that tells the buffer order needs five key blocks of k and v.
- **Only head dim 64 is measured.** At D 32 the dP sum's four-thread order is READ, not captured, and `train_sdpa_backward` refuses it.
- **Default mode has no Definition.** The line you asked for: without deterministic algorithms the key blocks' dQ parts are added by atomics in the order the card schedules the thread blocks, so dQ is not a function of the inputs (55 of 4,194,304 captured words differed from the ascending sum); a job whose circuit must give its words runs under `torch.use_deterministic_algorithms(True)`, or its dQ order is advice.
- **Not checked:** whether any captured word tells flush-to-zero from none.

**Job 3's table** (one head, T 1,024, D 64; RAN on the VM from the Definitions; 23,682 gates a k16 step on bits, 1,480.125 a multiply-add):

| product | (row, key) pairs | multiply-adds | gates on bits |
| --- | --- | --- | --- |
| S, in the circuit: visible pairs only | 524,800 | 33,587,200 | 49,713,254,400 |
| dP, dV, dQ, dK, each: the 72 tiles the kernel visits | 589,824 | 37,748,736 | 55,872,847,872 |
| any of them over the full rectangle | 1,048,576 | 67,108,864 | 99,329,507,328 |

- **A head on bits:** 293,929,461,888 gates at H 8 and 12, 293,873,993,856 at H 16, 293,744,568,448 at H 24 (only the dQ buffers differ). The five products are 273,204,645,888 of it, 93.0%.
- **One thing for the count to settle.** The card multiplies whole tiles for S too (37,748,736 multiply-adds) and then writes -inf over the masked scores. The circuit holds no gate for a masked score, as the forward's Definitions don't: 4,161,536 fewer multiply-adds a head.

Next from me: the checks' results and the draft PR's number as they land; then Friday noon's TRN-140 report (the efficient kernel's CPU counts ran, `r20261112-174820-99ef`, not yet written up).

