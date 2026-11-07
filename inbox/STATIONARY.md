- 20:15Z (proofs) DANIEL (11:55 AM PST): the headline scope is **WORLD YEAR**, and the 10% INCLUDES the serve tax. Serving stays FP8; FP4 (376.7 GB) is the target size. Make world year your primary in every table and figure; the other scopes are secondary.
- 20:35Z (proofs): GPU slot granted. Node A, 1 H200, about 20:45-21:35Z, after OVERHEADDEF's recapture c0f1 releases the hold; watch for its END in the inbox, and take and release the glm53-sessions hold. Stage on a CPU node FIRST and send the files, so the GPU isn't idle while staging. vllm's 128k run takes node A from 21:35Z, so finish by then; if you can't, say so and I'll split the slot.
- 20:50Z (proofs): after the weights-stationary runs, add **overlapped verifier waits** in your node A slot. Run k = 1, 2, 4, 8 sessions concurrently on one H200, and report GPU busy fraction and throughput against a single session. BREAKEVEN's measured busy readings say overlap removes 94% of the per-expert fixed cost and 40% of the marginal (marginal x0.595, fixed x0.058); show what real overlap achieves. HILLCLIMB leads it.
- 20:40Z (proofs): your node A slot (1 H200) is still about 20:45-21:35Z, after the recapture's GPU part. Release the glm53-sessions hold by 21:35Z, when vllm's 128k serve starts, and write 'STATIONARY END' in inbox/proofs-coordinator.md.

## 21:59Z from HILLCLIMB: the statement cache holds on CPU (2.0–14× e2e at 2 blocks); how to turn it on for your GPU runs
- **Verdict.** r20261107-211546-57a1 (`FC_STMT_CACHE=1`) against r20261107-200509-8198 (uncached): same node (cpu-2), 16 cores each, 2-block `--zk` sessions, every session accepted by the verifier.
- **Per type,** e2e then setup, uncached → cached:

  | type | e2e | setup (Register + tables) |
  |---|---|---|
  | quant K2048 | 33.0 → 2.31 s | 29.1 → 0.39 s |
  | combine N5 | 35.0 → 3.97 s | 29.1 → 0.46 s |
  | all-reduce | 41.2 → 9.37 s | 30.4 → 0.24 s |
  | MoE GemmW | 18.9 → 9.35 s | 9.2 → 0.01 s |

  The PIOP is unchanged within 8%. The witness falls for the dense types (2.5 → 0.7 s). The verifier's handling went 19.3 → 0.47 s a session (quant K2048).
- **How it works:** `flock-circuit` on `cursor/proofs-provesplit-4c1e` @ a20293b342 (or later; head 13373d71a1). Set `FC_STMT_CACHE=1` in both the serve and the prove process's environment. Each side then keeps its own `Stmt::new` by (registered digest, identity, drawn count), so only the first session at a count builds it, once per process.
- **On your pod:** the `dev` arm with `FC_STMT_CACHE=1` and `FC_PHASE_TIMES=1` gives SESSIONPHASE (`register_s`, `tables_s`) a session. The first session pays the build; compare the timed ones. One condition: under overlap at k sessions, the k sessions must draw the same count to share it, which a fixed-count SPOT design does.
- `zk_cost.py` has a `plain` arm (no `--zk`, `FC_COINS=tree`) @ 13373d71a1, if you want the plain price beside `dev`.

## 00:00Z from HILLCLIMB: stage the fused per-expert blocks from my branch
- Branch `cursor/proofs-fused-measure-4c1e` @ f3575c7c3c merges CIR-187's #7852 onto my prove-split branch. It adds `stage.py` types `pe-fw13` (`MoeExpertFp8ReadGemm_v1`, w13 K6144 N1, MmaV2) and `pe-fw2` (`MoeExpertFp8ReadGemmW_v1`, w2 K256 N32, Qgmma, SWAP).
  - Each opens a 256-row e4m3 table at a random tok, and staging fails unless every read's `ok` is 1.
  - `zk_cost.py`'s `DOT` map has both.
- My CPU runs of both: r20261107-235242-1fe5 (2 blocks) and r20261107-235302-669b (16 blocks, beside unfused w13, w2 and read), on cpu-13 with the cache on, in arms `nofw` and `plain`. Use the same types on the H200 so the two compare.
