Lane circuits
Refs CIR-260
Goal vllm/production-vllm-reproducible

## What changed and why

Under `--kv-cache-dtype fp8_ds_mla`, vLLM's fused norm-and-RoPE kernel (`_fused_norm_rope_kernel`, program 1, at vLLM `d9105ea80`) writes three things per token:
- `kv_c_out`: the RMS norm's f32 output rounded to bf16;
- `k_pe_out`: the interleaved RoPE rounded to bf16;
- one 656-byte cache row: 512 e4m3 bytes, 4 f32 scales and the 64 RoPE bf16 words.

No registered Definition stated the cache row. This PR adds it as a new module with its own sidecars, and changes no registered Definition.

- **`verity/definitions/registry/dsa_ds_mla.py`** has two Definitions.
  - `DsMlaFp8Tile_v1{T}` (x: T f32) → (q: T e4m3, scale: f32) is one scale group.
  - `DsaLatentRowDsMla_v1{KVL, R, TILE, EPS}` (kv: KVL + R bf16, w: KVL bf16, cs: R bf16) → (latent: KVL + R bf16, nope: KVL e4m3, scale: KVL / TILE f32) is one token's whole write.
  - The row states the norm's f32 output once and both outputs read it, so a served Program states each served word once and never makes the unserved f32 a word. Two Definitions instead would make that f32 a word, or state the norm twice.
  - `latent` is built from the registered `DsaRMSNormSumSq_v1` and `RoPEHeadGptJ_v2`, and equals `DsaLatentRow_v1`'s words.
  - The cache row is `nope`, then `scale` little-endian, then `latent[KVL:]`.
  - One group: scale = `F32Fmaxf(F32Mul(max |x|, 0x3B124925), 0x00800000)`, and q = `F32ToE4m3Sat(F32Mul(DivFullScaleA(x, scale), DivFullRcp(scale)))`.
  - The docstring cites the kernel's source lines and the PTX instructions of both FP8 specialisations line by line, and states what the corners give: no scale is subnormal; no finite value saturates (the largest is 448); a NaN in a group gives `0x7F` and the max drops it; an infinity makes the scale +inf, so its byte is `0x7F` and every finite value becomes ±0; -0 gives `0x80`.
- **`boolean_dsa_ds_mla.py`** holds the Boolean versions, `DsMlaFp8Tile_v2` and `DsaLatentRowDsMla_v2`. circuit-check lowers no word composite this large whole, so without them it compares the word Definitions on no vectors. `boolean_version` maps each `_v1` to its `_v2`.
- **The sidecars** bind GLM-5.3's sizes and a small binding: the tile at T 128 and 64 (word) and at 32 and 128 (Boolean); the row at (512, 64, 128, 1e-5) and (256, 8, 64, 1e-5).
- **`verity/definitions/tests/test_dsa_ds_mla.py`** checks hand-worked vectors at the corners:
  - a group of zeros, and a group of NaNs, at the FLT_MIN floor;
  - amax · f32(1/448) one f32 step below FLT_MIN, at it, and one step above it;
  - ±448 at scale 1, and round-to-nearest-even ties;
  - a NaN in a finite group; ±inf in a group; -0;
  - the row's latent equal to `DsaLatentRow_v1`'s.
- **`verity/definitions/tests/catalog_ids.json`** gains the four new ids.
- **`adapters/vllm/tests/properties/glm53_ds_mla_write_capture_gpu.py`** is the capture. Its two phases:
  - `--phase gpu` runs vLLM's own kernel at the pin with the FP8 arm, at GLM-5.3's sizes, in both specialisations (with and without the indexer). The operands are 512 served-shaped tokens with the corners planted. The cache is poisoned to `0xA5` first, each case runs twice, and it keeps the compiled PTX.
  - `--phase compare` holds all 656 bytes of every row, and `kv_c_out` and `k_pe_out`, to `DsaLatentRowDsMla_v1`. In the same run it holds the norm and RoPE words to `DsaRMSNorm_v1{512}` and `RoPEHeadGptJ_v2{64}` as registered.
  - It writes rows compared and rows that differ, and exits nonzero on any difference.
- **`test_glm53_ds_mla_write_capture.py`** runs the compare on CPU over the reference's own output. It checks that every corner is reached with 0 rows differing, that a flipped byte, word or scale is caught, and that the PTX check refuses a different PTX.

The capture's record carries `ok` and the document's digest, unsealed. Registering its schema in `verity_vllm/properties/record.py`, and its row in the capture map, are vllm's.

## circuit-check

All 8 targets: 0 failures, 0 dead gates.

| target | gates | conformance | partition | dead | redundant | vectors | block | failures | warnings | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `DsMlaFp8Tile_v1{T=128}` | 644 | model-untested | checked, units/call 66, max_out_bits 32, isolated 0 | 0 | 0 | none | 606489 ANDs, k 20 | 0 | 1 | 0.57 |
| `DsMlaFp8Tile_v1{T=64}` | 324 | model-untested | checked, units/call 34, max_out_bits 32, isolated 0 | 0 | 0 | none | 306457 ANDs, k 19 | 0 | 1 | 0.57 |
| `DsaLatentRowDsMla_v1{KVL=256,R=8,TILE=64,EPS=1e-05}` | 3230 | model-untested | checked, units/call 657, max_out_bits 32, isolated 0 | 0 | 0 | none | 3523810 ANDs, k 22 | 0 | 1 | 0.83 |
| `DsaLatentRowDsMla_v1{KVL=512,R=64,TILE=128,EPS=1e-05}` | 6358 | model-untested | checked, units/call 1353, max_out_bits 32, isolated 0 | 0 | 0 | none | 7311282 ANDs, k 23 | 0 | 1 | 0.83 |
| `DsMlaFp8Tile_v2{T=32}` | 323234 | exact-model-tested | checked, units/call 5621, max_out_bits 16, isolated 0 | 0 | 0 | word: 1024 compared, 0 mismatches | 109940 ANDs, k 17 | 0 | 0 | 2.89 |
| `DsMlaFp8Tile_v2{T=128}` | 1260562 | exact-model-tested | checked, units/call 13941, max_out_bits 16, isolated 0 | 0 | 0 | word: 1024 compared, 0 mismatches | 432116 ANDs, k 19 | 0 | 0 | 2.45 |
| `DsaLatentRowDsMla_v2{KVL=256,R=8,TILE=64,EPS=1e-05}` | 7915960 | exact-model-tested | not checked: over the partition budget 5000000 | 0 | not checked | word: 1024 compared, 0 mismatches | 2762719 ANDs, k 22 | 0 | 0 | 3.1 |
| `DsaLatentRowDsMla_v2{KVL=512,R=64,TILE=128,EPS=1e-05}` | 16630260 | exact-model-tested | not checked: over the partition budget 5000000 | 0 | not checked | word: 1006 compared, 0 mismatches | 5840815 ANDs, k 23 | 0 | 0 | 5.77 |

How to read the table:
- **The one warning on each word target** is `vectors/none`: circuit-check lowers no word composite over 256 IR gates whole. Each word target is compared through its Boolean version, which the `_v2` rows show.
- **Pins.** `--update-pins` ran over both sidecars and wrote no pins. It pins only whole lowerings under 65,536 ANDs, and each of these targets is over that.
- **Which bindings `--all` checks.** The GLM-5.3 Boolean bindings are over `targets.MAX_GATES` and unpinned, so `--all` checks each Boolean family's smallest binding. The GLM-5.3 ones were checked by id, in run `r20261112-140059-2533`.

## Runs

| run | slot wait | slot held | what I read in it |
|---|---|---|---|
| `r20261112-134619-b909` | 73 s (13:46:44Z–13:47:57Z) | 21 s (to 13:48:18Z) | circuit-check of the word sidecar: 4 targets, 0 failures, 4 `vectors/none` warnings |
| `r20261112-135620-3b7d` | 5 s (13:56:50Z–13:56:55Z) | 3 min 25 s (to 14:00:20Z) | both sidecars with `--update-pins --no-cache`: word 4 targets, 0 failures; Boolean 2 targets, 0 failures, 2 compared on random vectors; peak 6.7 GB RSS; no pins written (git status empty) |
| `r20261112-140059-2533` | 10 s (14:01:20Z–14:01:30Z) | 2 min 39 s (to 14:04:09Z) | the GLM-5.3 Boolean tile and row and the word row by id: 3 targets, 0 failures, 1 `vectors/none` warning; peak 6.1 GB RSS |
| `r20261112-140537-334e` | 110 s (14:05:56Z–14:07:46Z) | 180 min, cancelled at its timeout (rc 143) | the suites the change from 3498d4ad6546 touches, 57: 49 passed (`repository` whole, `verity/definitions`, `verity-vllm` with 14972 passed, 481 skipped, 9 xfailed), 0 failed, 8 unfinished |
| `r20261112-173053-7c83` | 6 s (17:31:19Z–17:31:25Z) | 2 h 45 min (to 20:16:27Z), rc 0 | the 8 unfinished, all passed: `verity-flock` 2079 passed, 69 skipped; `research` 4626 passed, 3 skipped; `verity/definitions/ops` 985 passed, 2 skipped; `circuit-check` 428 passed, 5 skipped; `verity/protocols/pouw/verifier` 1975 passed; `verity-numerical` 641 passed, 7 skipped; `verity/commitments` 592 passed; `verity-core-evaluate` 67 passed, 1 skipped; their verdicts in `out/suites.json` |
| `r20261112-203126-e313` | 10 min 47 s (20:31:49Z–20:42:36Z) | about 8 min (to 20:50:21Z), rc 0 | `repository` alone, whole: 352 passed, 0 failed, its verdict in `out/suites.json` (the cancelled first run wrote no `suites.json`) |
| `r20261112-172953-93dc` | 35 s to start (17:29:53Z–17:30:28Z); the lease granted at once | card 2's lease 12 s; the run 63 s (to 17:31:31Z) | the capture on node A card 2 (NVIDIA H200, 132 SMs, cores 24-33), vLLM `0.28.1rc1.dev472+gd9105ea80`: GO, below |

The retired-terms lint (`tests/lint/test_repo_retired_terms.py`) passes on this VM (13 passed). Its scan finds 0 sites in each changed file it reads; it skips `catalog_ids.json`.

## At the new head

`ffaacd91c9dd` removes the capture script's three `# noqa: BLE001` catches. The vLLM lint ratchet refused them, and each is now fixed rather than suppressed:
- **The GPU phase** has no catch. A failure exits non-zero with its traceback before `gpu.json` is written, so the compare doesn't run and no record exists.
- **The PTX copy** catches `OSError` by name. That block only does filesystem work, and the record keeps `ptx.ok` false and the error, so the verdict is NO-GO.
- **The compare** has no catch. A worker's exception re-raises in the parent before the record is written, so the run exits non-zero with no record. I checked this off-tree with a missing words file: exit 1, the traceback printed, no record written.

The change is one file, diff stat ` adapters/vllm/tests/properties/glm53_ds_mla_write_capture_gpu.py | 40 +++++++--------------- (13 insertions, 27 deletions)`. No suite other than `repository` and `verity-vllm` needs a rerun for it.

What I ran at this head:
- **The ratchet gate,** `python3 adapters/vllm/tools/ratchet_gate.py origin/main ffaacd91c9dd --json` against origin/main `5f0c008aa040`: `failures: []`.
- **The capture's test file,** `test_glm53_ds_mla_write_capture.py`, on the VM: 3 passed.
- **Suites,** through `research run --on auto --cwd source --wait`:
  - `repository` in `r20261112-210538-041f`: 352 passed, 0 failed, in its `out/suites.json`. It waited 21 min 34 s for a slot (21:06:01Z–21:27:35Z) and ran about 6 min (to 21:33:42Z).
  - `verity-vllm` in `r20261112-213631-ff0e`: 14972 passed, 0 failed, 485 skipped, in its `out/suites.json`. It waited 19 min 19 s for a slot (21:37:02Z–21:56:21Z) and ran 1 h 16 min (to 23:12:22Z).

## The capture

The capture ran at `f912fe226f80`. The new head `ffaacd91c9dd` differs from it only in the capture script's three error catches (`glm53_ds_mla_write_capture_gpu.py`, 13 insertions and 27 deletions), and in nothing the capture computes: the Definitions, the operands, the comparison and the record's schema are byte for byte the same.

`r20261112-172953-93dc` ran vLLM's own `fused_norm_rope` with `mla_kv_cache_dtype="fp8_ds_mla"` in both specialisations, each twice.
- **Cases.** Two of 3,136 tokens each: "served", shaped like served operands with the fused-norm capture's special rows; and "corner", with planted groups.
- **Slots.** 3,128 tokens per case have a slot. Every byte outside the written rows kept its `0xA5` poison.
- **Repeats.** Both runs of each case gave the same bytes, and the two specialisations gave the same rows.
- **PTX.** The kept PTX of both specialisations has the code sha256 the docstring cites.

What the compare found:
- **Cache rows:** 3,128 per case and specialisation, 12,512 in all, compared byte for byte with `DsaLatentRowDsMla_v1{512,64,128,1e-5}`; 0 differ.
- **`kv_c_out` and `k_pe_out`:** 3,136 tokens per case and specialisation, against the Definition's latent; 0 differ.
- **The registered words:** the norm words against `DsaRMSNorm_v1{N=512}` and the RoPE words against `RoPEHeadGptJ_v2{D=64}`, as registered; 0 differ.
- **Corners reached in the corner case:**
  - 154 groups of zeros at the floor;
  - 37 groups whose amax · f32(1/448) is below FLT_MIN;
  - 223 groups with an infinite scale;
  - 2,905 NaN values in finite groups;
  - 3,351 `0x7F` bytes, 199,294 `0x80` bytes and 12,098 bytes at ±448.
- **An independent recomputation** of 11 of those rows from the saved words, outside the capture's compare, gives 0 differences. It covers rows with an infinite scale, rows at the floor and rows with NaN bytes.

## Not run

- Served vLLM with `fp8_ds_mla` end to end. The capture calls the kernel directly at served sizes, with operands shaped like served ones.
- FlashMLA's read of the row, and the dense prefill read-back. Both are other issues.
- `circuit-check --all`.

