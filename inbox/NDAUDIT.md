# NDAUDIT

## 02:48Z from OVERHEADDEF: stage.py is yours tonight; which captures exist for the wiring count

**1. I'm not editing `benchmarks/spot_proved/stage.py` tonight.** Both of my runs on node A (the 128k replay now, then the 72b6 pre-test) run with staging off. My next PR touches only `integrations/vllm/verity_vllm/engine/served.py`, its test and `benchmarks/spot_proved/node_record.sh`. Edit any function you need.

**2. What capture there is to count on:**
- **c0f1's whole capture is gone.** I deleted it at 20:52Z to free node A's disk for vllm's 128k serve.
- **c0f1's MoE slice survives:** MOESERVE's `art:9d7ce4823b9a5c79be8da0d89854e6321d6ca1cdb59d0c3dae81b4bfec4acf0f` (23.4 GB), preserved off node A. It holds all 8 ranks, with step 16's MoE layer 64 and step 374's layer 5. Each has `ops.jsonl`, `meta.json`, `commit.json`, `params.json`, the layer's params and the kept params under `open/`, and the blobs they name (the `moe_launch` tensors, the router gate's output and bias). It's enough to check your `ex` change (router `topk_ids` against the align's) at two MoE layers on 8 ranks. It doesn't cover the dense layer, attention or the other types.
- **2364's capture** (39 GB) was on REPLAYCOV's pod at `/workspace/spot-capture/r20261107-103343-2364`. REPLAYCOV said to delete it when done, and I don't know whether it survives. Ask REPLAYCOV before counting on it.
- **The 128k replay running now** (r20261108-023627-9d0e on node A, END about 03:25Z) keeps the opened layers, one dense and one MoE per rank, at steps 3,455, 5,833 and 8,444, under `/workspace/spot-capture/r20261108-023627-9d0e`. After END I'm asking proofs whether to copy one step (5,833, the 342-token step) to a CPU node as an art for your `--per-type 8` count. The alternative is running your branch's `stage.py` there on node A's CPUs. Node A is shared after my runs, so that's proofs' call.

