Lane circuits
Refs CIR-259

`benchmarks/served_units/stage.py` no longer stages its 18 `PARTS` types, the `Q_word` classes of word Definitions whose statements claim proof blocks the Lean verifier can't derive. In their place each caller stages a whole Definition. It is bound by default to the Program of its instances and that Program's partition object, with `partition.json` and `program.json` written beside the statement. `stage.py` refuses a retired name and says what to stage instead, or that nothing does. `--bind` is gone, and `--unbound` stages the old way for the two callers that hand Lean neither file.

Each statement below is marked: [run] means run by this PR's worker and its output read; [read] means read in a record, file or run of someone else's; [inferred] means neither.

## The head

- **Head `c9dcea5b385ca3642280fcbd705a771b4ea8ed98`.** It is `bbcfd2356a59` plus one commit, `c9dcea5b385`, which changes one line of `stage.py`: `token-select`'s replacement (below). [run]
- **`bbcfd2356a59` is the branch carried across the layout moves and merged with `main`.** The move tool's restack gave `9205f60f5f82` from `64bb330833c2`, and `bbcfd2356a59` merges `origin/main` `d119240998be` into it, cleanly. No move landed on `main` after the carry (`git log 4ae17f7eafd7..origin/main -- tools/move/` is empty). [run]
- **The carry check (`research.carry 64bb330833c2 bbcfd2356a59`) exited 0**, with `"kind": "merge-only"`, `"carries": true` and 68 commits; it reran 28 moves (`r20261113-171632-9535`). It was not rerun at `c9dcea5b385`: with a hand commit on top, the head is no longer merge-only, and the commit's diff is the one line below. [run]
- **Pins:** at `bbcfd2356a59`, `pins.py --stale` gave rc 0, `--regen` rewrote no file, and `--stale` again gave rc 0 (`r20261113-170442-6f08`, a clone). [run]
- **Paths:** every path and module this PR's added lines name exists at the head, and no old path needed a fix by hand. [run]
- **`git merge-tree`** of the head with `origin/main` `f479876a8758` is clean. [run]

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
| `dsa-q` | stage census:DsaIndexerQ_v1{H=32,D=128,ROPE=64} or dsa-q-row in its place (the census's whole Definition of the indexer query, or its Boolean version) |
| `rmsnorm-cuda` | stage rmsnorm-v8 in its place (the fused RMSNorm the census serves) |
| `sparse-mla-head` | stage census:SparseMlaHeadBlock_v2{DR=64,DV=512,NVIS=64,FIRST=false,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v2"},SCALE={"f64":"0x1.0000000000000p-4"}} and census:SparseMlaHeadEnd_v1{DV=512,INV={"fn":"Fa3InvSum_v1"}} in its place (the census's cut, its key block (the Boolean version, whose reads are in gates: the census's own reads in lookup slots, which `verify --zk` refuses) and its end block) |
| `token-select` | stage census:GreedyChunk_v2{N=7744,OFF=7744} and census:GreedyCombine_v2{V=154880,N=7744} in its place (the census's cut, a chunk and the combine) |
| `moe-w2` | nothing stages in its place (the census does the down projection as moe-gemmw's Definition) |
| `rmsnorm-triton` | nothing stages in its place (the census has no Definition of it) |
| `silu-quant` | nothing stages in its place (the census has no Definition of it) |

The table is `RETIRED` at the head, written the way `parse()` words it. [read]

- **The replacements but `token-select`'s** are the ones CIR-259's first part staged and verified bound (CENSUSBIND's table on CIR-259). For the three with nothing in their place, that table and proofs found no Definition of the census to stage. [read]
- **`token-select`'s replacement is the census's v2 cut, the circuits lead's decision on CIR-259 (Fri 13 Nov).** `main`'s `d3567e576ca` retired `GreedyChunk_v1`, which the replacement had named, so `test_retired` failed at `bbcfd2356a59` (below). [run] `main`'s census (`benchmarks/sampling_plan/glm53_blocks.json`) cuts the token select into 19 `GreedyChunk_v2{N=7744,OFF=k*7744}`, for k = 0 to 18, and one `GreedyCombine_v2{V=154880,N=7744}`. [read] `OFF=7744` names a chunk that isn't the first, as the old entry did; any of the census's chunks would do. [read: the lead's decision]
- **Both were accepted before, on proofs' tree.** The cover record v6 (`art:546ff3480ca9dd2abe6e019e1cd89e3a9bb2b908570a0d7a1f527915556b8fbb`) lists both as accepted in proofs' run `r20261112-191841-86b4`, on proofs' tree `7c2636e740e0` and not on `main`, with that run's limits: a stand-in serve, and the audit's own verdict there a reject on other grounds. `main`'s pins file `verity/definitions/registry/greedy_stages.circuit-check.json` is byte-identical to the one at proofs' `f461ac0edeb2`. [read: the circuits lead, on CIR-259]
- **Both are staged and accepted at this head** (`r20261113-192328-bcec`, source `c9dcea5b385`, one unit each, bound, 16 GB asked). [run]
  - `GreedyChunk_v2{N=7744,OFF=7744}`: 951,986 ANDs, `k_log` 24, staged in 36 s at 1.8 GB.
  - `GreedyCombine_v2{V=154880,N=7744}`: 954,687 ANDs, `k_log` 25, staged in 33 s at 1.8 GB.
  - For each, `ks-form` passes, and `statement` with the Program held derives the unit.
  - `verify --zk` on one CPU session gives exit 0 (accepted) with the Program, exit 1 with no Program ("refused: a session without the verifier's program"), and exit 1 with another Definition's Program (`SiluMul_v4{I=256}`: "setup: partition: the verifier's program is not the one the partition object names").
  - The largest peak in the run is 1.8 GB.
- **`dsa-q`'s first choice**, the census's own `DsaIndexerQ_v1{H=32,D=128,ROPE=64}`, has its verdicts from run `r20261112-183158-1477` (source `31b1b56aec2a`, one block, bound): [run]
  - `ks-form` passes;
  - `statement` with the Program held derives the units;
  - `verify --zk` accepts one CPU session;
  - `verify --zk` is refused with another Definition's Program and with none.

  `dsa-q-row`, its Boolean version, is named second.
- **`sparse-mla-head`'s key block is the Boolean version `SparseMlaHeadBlock_v2`.** The census's own v1 block at 64 keys reads its exponentials in lookup slots, a form `verify --zk` refuses, and `chunk_rows.zk_refusal` names the v2 block as the one to stage. [read]
- **`test_retired` holds each replacement** to `parse()`, and each census replacement to `zk_refusal` being empty. [read]

## What each caller hands the Lean verifier now

- **`benchmarks/served_units/job.sh`**, through `benchmarks/pouw/hidden_zk/job.sh`: stages bound, then hands the Lean verifier the statement, its partition object and its Program (`--program`, new here). [read: the diff] `summarize.py` now expects:
  - the honest session to be accepted;
  - the selftest build's statement to be refused by `servedForm` ("a selftest build's statement");
  - the three forgeries to be rejected.

  At `fp8-coord:K=256`, N=2, it gave `honest-own` ACCEPT, `st-honest` REFUSED, `st-output_claim_false`, `st-output_row_committed_false` and `st-unit_internal_bit_flipped` REJECT, and `as_expected` true (run `r20261112-201828-47ac`, source `4dff5ca9743`). [run]
- **`benchmarks/one_stage/zk_cost.py`** (and `zk_cost.sh`): stages bound, `verify` hands `--partition` and `--program`, and a session counts only on Lean's accept. Before this change, it counted the no-Program refusal as checked. [read: the diff] For `census:SiluMul_v2{I=256}` at N=2 it gave 362,496 ANDs a unit, `k_log` 22, and "Lean accepted 1 of 1 sessions and refused 0" (`r20261112-201828-47ac`). [run]
- **`benchmarks/one_stage/phase_profile.sh`**: stages the census's decode gate-up block through `zk_cost.py stage`, bound, at full (2,048 units, 2,307,615 ANDs a unit, `k_log` 24). It hands Lean nothing; it pins the prover's transcripts (`r20261112-233111-2abf`). [run]
- **`benchmarks/harness/verity_bench/steady_node.py`** (benchmarks'; one line): its stage command passes `--unbound`. `server_rate.py` is not edited and hands Lean what it handed on `main`, so a steady row's circuit, pin and verdict stay as on `main`. [read: the diff] The steady head block (`SparseMlaHeadBlock_v2{…NVIS=64,FIRST=false…}`), staged `--unbound` at N=1, has pin `4b31e7caba011398…`, the pin `benchmarks/harness/specs/glm53-flock-steady-head-block-h200.toml:36` holds on `main` (`r20261112-201828-47ac`). [run]
- **`benchmarks/verifier_cost/flock_sessions.py`** (proofs'; one line): passes `--unbound` and counts what it counted on `main`. Moving these two callers to hand the Program and read an accept is PRF-567's second step. [read]
- **No Lean, so only their digests move:** [read: the diff]
  - `benchmarks/unit_walk/run.sh`;
  - `benchmarks/served_units/prove_time.sh` (its `+bind` arm is gone: every arm stages bound);
  - `window.py` (its `KEEP` keeps `partition.json` and not `program.json`; unchanged here).
- **`verity/protocols/proofs/flock/prover/tests/test_witness_host.py:208`** names `moe-w13`'s shape in a comment of proofs'. It stays. [read]

## What moved

- **`stage.py`'s default is bound.** A bound statement's `class_sha512` is the unbound one's, and its pin is its own. `--unbound` gives, byte for byte (SHA-512 of each, both types), the pin, `circuit.txt`, `inst.bin` and `pub.bin` that `main`'s `stage.py` with no flag gives (`r20261112-201828-47ac`). The same holds for `fp8-coord:K=256 --public-weights` (pin `d88246eae7750a06`; `r20261112-202406-85e0`). [run] Examples of how the pins move, unbound to bound:
  - `fp8-coord:K=256` at N=2, from `a35ba0c14adafea4` to `295e25ae636a8478`;
  - `census:SiluMul_v2{I=256}` at N=2, from `f85f500d3b170116` to `4b8b99d4c0f4d686`.
- **`phase_profile.sh`:** [read: the diff]
  - `UNITS` defaults to `census:MoeExpertFp8BlockGemm_v3{K=6144,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v1"}}` instead of `moe-w13:N=1,SWAP=False`, and `PEAK_UNITS` to it and `gq`;
  - `MAIN_ZK` goes from `493e51d7…` to `03825618…`, and `MAIN_M0` from `0f0dfa3c…` to `8f1d5806…`;
  - the build takes `SM` (default 90, as before).
- **Examples:** `zk_cost.sh`'s example is `router-row`, and `served_units/README.md`'s is `SU_UNITS="fp8-coord:K=6144 s8-coord:K=2048"`. [read: the diff]
- **No spec's pin moves.** [run]

## Checks

- **At the head `c9dcea5b385`:**
  - `suites.py served_units repository`, without `--quick`, gave "2 suites: 2 passed (0 cached), 0 failed": `benchmarks/served_units` 48 passed, `test_retired` among them, and `repository` 361 passed (`r20261113-191707-881e`). [run]
  - The ratchet gate (`ratchet_gate.py origin/main HEAD --json`) exits 0 with `"failures": []`. [run]
  - The retired-terms lint's `sites()` on `stage.py` finds one site, `caps` in `getattr(st.low, "caps", None)`, which `main`'s `stage.py` holds too. [run]
- **At `bbcfd2356a59`:**
  - The nine suites this PR's paths select gave "9 suites: 8 passed (0 cached), 1 failed" (`r20261113-180845-6cde`). The failure was `benchmarks/served_units`' `test_retired`: "LookupError: template GreedyChunk_v1{N=16384,FIRST=false} names no registered Definition". `c9dcea5b385` fixes it. [run]
  - The full `suites.py` (`r20261113-171018-712b`, 130 suites selected) ended at its 3-hour timeout: "cancelled rc=137 class=STAGE_TIMEOUT", with no summary line. [run]
    - 119 suites passed, and `benchmarks/served_units` failed on `test_retired`, as above.
    - 10 didn't finish. `verity-vllm` was killed by SIGTERM after 10,508 s at 87% of its tests, and `verity-flock` after 6,555 s at 74%; the progress lines `stdout.log` keeps show no failure in either. The other eight never started: `research`, `verity-circuit-check`, `verity/commitments/windows`, `verity/circuits`, `verity-numerical`, `verity/commitments`, `verity-ops` and `verity-core-evaluate`.
- **The head differs from `bbcfd2356a59` in the files of `c9dcea5b385` only, so the other suites of those runs stand for the head: the 119 that passed in `r20261113-171018-712b`.** [inferred] The 10 unfinished suites aren't covered by any run at either head.
- **Before the carry** [run]:
  - Suites (`suites.py --changed 5f0c008aa040`, without `--quick`): at `4dff5ca9743`, 9 of 9 passed, none cached (`r20261112-201917-61b8`). At `9b280baba610`, 9 of 9 passed, none cached (`r20261112-234331-2cc5`). At `fe800897e74d`, 9 of 9 passed, none cached, 0 failed (`r20261113-011713-d278`).
  - `--unbound` against `main`'s staging, with four stagings each (head `--unbound`, `main`'s `stage.py`, `31b1b56aec2a`'s `stage.py`, head bound): `fp8-coord:K=256` and `census:SiluMul_v2{I=256}` at N=2 (`r20261112-201828-47ac`, source `4dff5ca9743`); `--public-weights` (`r20261112-202406-85e0`, source `4aafb2af879`).
  - Bound `SiluMul_v2{I=256}` at N=2, the Lean verifier with the partition object (`r20261112-201828-47ac`):
    - `statement` with the Program held derives the units;
    - with another Definition's Program, `statement` and `verify --zk` give "setup: partition: the verifier's program is not the one the partition object names";
    - `verify --zk` with the Program held accepts one CPU session;
    - `verify --zk` with no Program gives "refused: a session without the verifier's program (--program): …";
    - staged `--unbound`, `verify --zk` gives the same no-Program refusal.
  - GPU (`r20261112-233111-2abf`, source `b7d44e418ab`): vy-nebius-2, one RTX PRO 6000, `SM=120`. `phase_profile.sh AS=base` ran with setup, build and stage outside the lease, then `ident` under `gpu-lease`. Under Flock ZK and under Flock's circuit prover, proofs and transcripts are byte-identical across sessions, and `gpu_paths_agree` passes under both.

## Not run, or not established

- **No replacement has a Lean accept at a full statement's count.** Every Lean accept here and in CENSUSBIND's table is at 1 or 2 units, `token-select`'s v2 pair included. The one full statement, W13 at 2,048 units, ran on the GPU without Lean. [run]
- **Only one chunk position of the v2 cut was staged, `OFF=7744`.** The other 18 chunks of the census's cut weren't staged here. [run]
- **The carry check wasn't rerun at the head**, for the reason above. [run]
- **Ten suites have no verdict at either head:** `verity-vllm`, `verity-flock` and the eight that never started, all cut off by `r20261113-171018-712b`'s timeout. They weren't rerun. [run]
- **`--packed --public-weights` at `K=256`** stages on none of the three stagers: "public inputs: leaf 132 reads a row word inside word 16, which holds a public word". `main`'s `stage.py` refuses it the same way, so this change doesn't cause it (`r20261112-202406-85e0`). [run]
- **The steady head block at `FIRST=true`** wasn't staged here; only the `FIRST=false` row's pin was compared with `main`'s. [run]
- **The new `MAIN_ZK` and `MAIN_M0` were taken on sm_120**, where `main`'s were taken with `SM=90`. The prover's transcripts are device-independent, but no sm_90 run of the new default was made. [inferred from `gpu_proofs_match_cpu`]
- **`SparseMlaHeadEnd_v2`** wasn't staged bound here; `sparse-mla-head`'s end block stays `SparseMlaHeadEnd_v1`. [run]
- **The Lean verifier took 147,217,100 kB (about 147 GB) resident** on `DsaIndexerQ_v1{H=32,D=128,ROPE=64}` at one block, whose `circuit.txt` is 35,924,397,430 bytes (`r20261112-183158-1477`). Lean's memory at a full statement of it isn't known. [run]
