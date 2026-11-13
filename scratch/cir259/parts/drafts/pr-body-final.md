Lane circuits
Refs CIR-259

`benchmarks/served_units/stage.py` no longer stages its 18 `PARTS` types, the `Q_word` classes of word Definitions whose statements claim proof blocks the Lean verifier cannot derive. In their place each caller stages a whole Definition, bound by default to the Program of its instances and that Program's partition object, with `partition.json` and `program.json` written beside the statement. `stage.py` refuses a retired name and says what to stage instead, or that nothing does. `--bind` is gone, and `--unbound` stages the old way for the two callers that hand Lean neither file. This PR holds the `--bind` commits from `cursor/circuits-cir245-census-bind-6f04` (head `31b1b56aec2a`) and merges `origin/main` at `5f0c008aa040`.

## What `parse()` says for each retired name

| retired type | what `parse()` says to stage instead |
|---|---|
| `moe-gemmw` | stage census:MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v1"}} in its place (its own Definition) |
| `moe-chunk` | stage census:MoeExpertFp8CoordinateChunk_v1{E=256,K=512,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v1"},FIRST=false,LAST=false} in its place (its own Definition) |
| `moe-w13` | stage census:MoeExpertFp8BlockGemm_v3{K=6144,N=1,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v1"}} or census:MoeExpertFp8BlockGemm_v3{K=6144,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v1"}} in its place (the census's whole Definition, its swapped variant or the decode one) |
| `dsa-krow` | stage census:DsaIndexerKRow_v1{D=128,ROPE=64,EPS={"f64":"0x1.0c6f7a0b5ed8dp-20"}} in its place (the census's Definition) |
| `dsa-logits` | stage census:DsaMqaLogits_v3{N=1,H=32,D=128} in its place (the census's whole Definition) |
| `moe-sum` | stage census:MoeSum_v1{TOPK=8,H=96} in its place (the census's Definition) |
| `silu-mul` | stage census:SiluMul_v2{I=256} in its place (the census's Definition) |
| `rope` | stage census:RoPEHeadGptJ_v2{D=64} in its place (the census's Definition) |
| `dsa-topk` | stage census:DsaTopKShortRow_v1{TOPK=2048} in its place (the census's Definition) |
| `s8-linear` | stage s8-coord in its place (its block, the W8A8 coordinate) |
| `router` | stage router-row or router-slots in its place (the router's Boolean version, its row form or its slots form) |
| `dsa-q` | stage dsa-q-row in its place (the indexer query's Boolean version) |
| `rmsnorm-cuda` | stage rmsnorm-v8 in its place (the fused RMSNorm the census serves) |
| `sparse-mla-head` | stage census:SparseMlaHeadBlock_v2{DR=64,DV=512,NVIS=64,FIRST=false,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v2"},SCALE={"f64":"0x1.0000000000000p-4"}} and census:SparseMlaHeadEnd_v1{DV=512,INV={"fn":"Fa3InvSum_v1"}} in its place (the census's cut, its key block (the Boolean version, whose reads are in gates: the census's own reads in lookup slots, which `verify --zk` refuses) and its end block) |
| `token-select` | stage census:GreedyChunk_v1{N=16384,FIRST=false} and census:GreedyCombine_v1{V=154880,N=16384} in its place (the census's cut, a chunk and the combine) |
| `moe-w2` | nothing stages in its place (the census does the down projection as moe-gemmw's Definition) |
| `rmsnorm-triton` | nothing stages in its place (the census has no Definition of it) |
| `silu-quant` | nothing stages in its place (the census has no Definition of it) |

The replacements are the ones CIR-259's first part staged and verified bound (CENSUSBIND's table on CIR-259). For the three with nothing in their place, that table and proofs found no Definition of the census to stage. `sparse-mla-head`'s key block is the Boolean version `SparseMlaHeadBlock_v2`: the census's own v1 block at 64 keys reads its exponentials in lookup slots, a form `verify --zk` refuses, and `chunk_rows.zk_refusal` names the v2 block as the one to stage. A test holds each replacement to `parse()` and each census replacement to `zk_refusal` being empty.

## What each caller hands the Lean verifier now

- **`benchmarks/served_units/job.sh`**, through `benchmarks/pouw/hidden_zk/job.sh`: stages bound, then hands the Lean verifier the statement, its partition object and its Program (`--program`, new here). `summarize.py` now expects the honest session to be accepted, the selftest build's statement to be refused by `servedForm` ("a selftest build's statement"), and the three forgeries to be rejected. Read at `fp8-coord:K=256`, N=2: `honest-own` ACCEPT, `st-honest` REFUSED, `st-output_claim_false`, `st-output_row_committed_false` and `st-unit_internal_bit_flipped` REJECT, `as_expected` true (run `r20261112-201828-47ac`, source `4dff5ca9743`).
- **`benchmarks/one_stage/zk_cost.py`** (and `zk_cost.sh`): stages bound, `verify` hands `--partition` and `--program`, and a session counts only on Lean's accept. Before this change, it counted the no-Program refusal as checked. Read for `census:SiluMul_v2{I=256}` at N=2: 362,496 ANDs a unit, `k_log` 22, and "Lean accepted 1 of 1 sessions and refused 0" (`r20261112-201828-47ac`).
- **`benchmarks/one_stage/phase_profile.sh`**: stages the census's decode gate-up block through `zk_cost.py stage`, bound, at full (2,048 units, 2,307,615 ANDs a unit, `k_log` 24). It hands Lean nothing; it pins the prover's transcripts (`r20261112-233111-2abf`).
- **`benchmarks/harness/verity_bench/steady_node.py`** (benchmarks'; one line): its stage command passes `--unbound`. `server_rate.py` is not edited and hands Lean what it handed on `main`, so a steady row's circuit, pin and verdict stay as on `main`. The steady head block (`SparseMlaHeadBlock_v2{…NVIS=64,FIRST=false…}`) staged `--unbound` at N=1 has pin `4b31e7caba011398…`, which is the pin `benchmarks/harness/specs/glm53-flock-steady-head-block-h200.toml:36` holds on `main` (`r20261112-201828-47ac`).
- **`benchmarks/verifier_cost/flock_sessions.py`** (proofs'; one line): passes `--unbound` and counts what it counted on `main`. Moving these two callers to hand the Program and read an accept is PRF-567's second step.
- **No Lean, so only their digests move:** `benchmarks/unit_walk/run.sh`, `benchmarks/served_units/prove_time.sh` (its `+bind` arm is gone: every arm stages bound) and `window.py` (its `KEEP` keeps `partition.json` and not `program.json`; unchanged here).
- `verity/protocols/flock/prover/tests/test_witness_host.py:208` names `moe-w13`'s shape in a comment of proofs'. It stays.

## What moved

- **`stage.py`'s default is bound.** A bound statement's `class_sha512` is the unbound one's, and its pin is its own. Examples: `fp8-coord:K=256` at N=2 goes from `a35ba0c14adafea4` unbound to `295e25ae636a8478` bound, and `census:SiluMul_v2{I=256}` at N=2 from `f85f500d3b170116` to `4b8b99d4c0f4d686`. `--unbound` gives the pin, `circuit.txt`, `inst.bin` and `pub.bin` that `main`'s `stage.py` with no flag gives, byte for byte (SHA-512 of each, both types; `r20261112-201828-47ac`). The same holds for `fp8-coord:K=256 --public-weights` (pin `d88246eae7750a06`; `r20261112-202406-85e0`).
- **`phase_profile.sh`:** `UNITS` defaults to `census:MoeExpertFp8BlockGemm_v3{K=6144,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v1"}}` instead of `moe-w13:N=1,SWAP=False`, and `PEAK_UNITS` to it and `gq`. `MAIN_ZK` goes from `493e51d7…` to `03825618…`, and `MAIN_M0` from `0f0dfa3c…` to `8f1d5806…`. The build takes `SM` (default 90, as before).
- **Examples:** `zk_cost.sh`'s example is `router-row`, and `served_units/README.md`'s is `SU_UNITS="fp8-coord:K=6144 s8-coord:K=2048"`.
- No spec's pin moves.

## Checks

- **Suites** (`suites.py --changed 5f0c008aa040`, without `--quick`): at `4dff5ca9743`, 9 of 9 passed, none cached (`r20261112-201917-61b8`). At the head: 9 of 9 passed, none cached (`r20261112-234331-2cc5`, source `9b280baba610`).
- **`--unbound` against `main`'s staging**, with four stagings each (head `--unbound`, `main`'s `stage.py`, `31b1b56aec2a`'s `stage.py`, head bound): `fp8-coord:K=256` and `census:SiluMul_v2{I=256}` at N=2 (`r20261112-201828-47ac`, source `4dff5ca9743`); `--public-weights` (`r20261112-202406-85e0`, source `4aafb2af879`).
- **Bound `SiluMul_v2{I=256}` at N=2**, the Lean verifier with the partition object: `statement` with the Program held derives the units; with another Definition's Program, `statement` and `verify --zk` give "setup: partition: the verifier's program is not the one the partition object names"; `verify --zk` with the Program held accepts one CPU session; `verify --zk` with no Program gives "refused: a session without the verifier's program (--program): …". Staged `--unbound`, `verify --zk` gives the same no-Program refusal (`r20261112-201828-47ac`).
- **GPU** (vy-nebius-2, one RTX PRO 6000, `SM=120`, `phase_profile.sh AS=base` with setup, build and stage outside the lease, then `ident` under `gpu-lease`): under Flock ZK and under Flock's circuit prover, proofs and transcripts are byte-identical across sessions, and `gpu_paths_agree` passes under both (`r20261112-233111-2abf`, source `b7d44e418ab`).
- **Ratchet gate** at the head: 0 failed.
- **Local**, one file at a time: `benchmarks/served_units/tests/test_stage_census.py` and `test_names.py`, `benchmarks/harness/tests/test_steady_point.py`, `benchmarks/one_stage/tests/test_zk_cost.py`, `benchmarks/pouw/tests/test_served_units_summarize.py`, and the tree's retired-term, doc-path and module-string tests.
- **`git merge-tree`** at the head: clean against `origin/main` (`7e70df68bd18`) and against `origin/cursor/lean-remote-merge-onmain-fc8d` (`67b2c6791982`). `main` since the merge base adds no caller of a retired name or of `--bind`.

## Not done, or not established

- **No session at a full statement's count** for any replacement on CPU or with Lean. Every Lean accept here and in CENSUSBIND's table is at 1 or 2 units. The one full statement, W13 at 2,048 units, ran on the GPU without Lean.
- **`--packed --public-weights` at `K=256`** stages on none of the three: "public inputs: leaf 132 reads a row word inside word 16, which holds a public word". `main`'s `stage.py` refuses it the same way, so this change doesn't cause it (`r20261112-202406-85e0`).
- **The steady head block at `FIRST=true`** was not staged here. Only the `FIRST=false` row's pin was compared with `main`'s.
- **The new `MAIN_ZK` and `MAIN_M0` were taken on sm_120**, where `main`'s were taken with `SM=90`. The prover's transcripts are device-independent, but no sm_90 run of the new default was made [inferred from `gpu_proofs_match_cpu`].
- **`SparseMlaHeadEnd_v2`** was not staged bound here. `sparse-mla-head`'s end block stays `SparseMlaHeadEnd_v1`.
- **`DsaIndexerQ_v1{H=32,D=128,ROPE=64}`, the census's own `dsa-q`, now verifies bound with all four verdicts at N = 1, and so does `sparse-mla-head`'s block at 64 keys (`r20261112-183158-1477`, posted on CIR-259). This PR still stages `dsa-q-row` in `dsa-q`'s place, until circuits' lead says otherwise.
