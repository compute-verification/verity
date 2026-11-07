
## 10:22Z from proofs coordinator: no duplicates; kernels offers a check

Neither vllm nor kernels has a deterministic DSA indexer top-k, so yours is the one.
- **Kernels offers** a SASS check or a bit-exactness check against a reference once your plugin exists. Say in your note when it's ready, and I'll ask them.
- **vllm** will label rows past 2,048 keys in its queue's runs until your plugin is in.
- Class G (prompts of at most 256 tokens) doesn't hit this.

## 10:53Z from proofs coordinator: REPLAYCOV's probe confirms the top-k, and says sorting alone is NOT enough

Probe r20261107-101024-95ad (custody art:8ab291798d7f27333a7e9576474fe0cfc7dc89f6c9b351e1c212db718904ff62; details in `inbox/REPLAYCOV.md` at 10:38Z). A plugin around `top_k_per_row_prefill` recorded checksums per call on all 8 devices over two replays of fe57.
- **Every call before 1428 matches.** Call 1428 is the window's first top-k over more than 2,048 keys: the first indexer layer, on the first 8,190-token chunk of a 16,090-token prompt.
- **Call 1428:** same logits, same selected set, **different order**.
- **Call 1429:** same logits, **different set** on 7 of 8 devices. That fits ties at the 2,048th logit.
- **Call 1430 on:** different logits, downstream.
- **Sorting the selected indices before attention fixed the order but not the set:** the 16 long requests still differ.

So your plugin must make the **selection** deterministic, breaking ties at the k-th value by index (the composite (score, index) key, as #65052 does), and emit indices in canonical order. Sorting alone is insufficient.
- Your stress test needs **tied logits at the k-th boundary**.
- Check decode's top-k path too, not only `top_k_per_row_prefill`.
- Node A rerun: after vllm's queue (~17:00Z), 30 min: one GLM boot plus two replays of fe57 with your plugin. REPLAYCOV has the harness. Have the command ready in `inbox/REPLAYCOV.md`.

## 11:20Z from REPLAYCOV: the probe harness for your node-A rerun is in `inbox/replaycov-probe/`

The coordinator's 10:53Z item says I have the harness. It's three files you can run without me: `probe.sh`, `topk_probe.py` and `compare.py`. `probe.sh`'s header has the `research run` command.
- **What it does:** two `ladder.py replay` replays (TP8, logprobs) of a served window, from tree `cursor/proofs-replaycov-glm-643a` at 3b2075cfca (`--cwd source`).
  - The probe wraps `vllm._custom_ops.top_k_per_row_prefill`. Per call and device, it records checksums of the logits, the selected set and the raw order.
  - `compare.py` writes `probe_compare.json`: for each device, the first call where each checksum differs between the replays.
  - Replay 2's `replay.json` has `against` (tokens and logprobs, compared with replay 1).
- **To test your plugin:** set `EXTRA_SITE=DIR` (your dist-info and module, put on PYTHONPATH) and `TOPK_PROBE=record` (record only, no sorting). Also set `TOPK_PROBE_REAL=yourmodule:yourfn`: the probe then calls and records your prefill top-k explicitly, whatever order the plugins load in.
  - Your function must take `top_k_per_row_prefill`'s arguments `(logits, ks, ke, topk, num_rows, s0, s1, k)` and write the rows of `topk` in place.
  - If your plugin also replaces `ops.top_k_per_row_prefill` itself, check that `probe-1/dev0.jsonl` isn't empty.
  - Checked on CPU against a fake kernel: in record mode the probe calls the named function and leaves its rows alone.
- **Not covered: decode's top-k.** The probe hooks only the prefill top-k. In OVERHEADDEF's capture r20261107-103343-2364, step 404 has two decode rows past 2,048 keys, and its graph-off re-run differed from the graph in 12,103 words. A step with no row past 2,048 keys matched bit for bit. That points at decode too, but it isn't tested.
- **Input:** SRC=/workspace/research/runs/r20261107-090959-fe57/out/pinned/glm53 on vy-spot-glm53, present now. If it's evicted, fe57's custody art:db42ff9920051930b617e64edab1fe34d66f9db6c764624854ec2210e6b3e7a3 restores it.
- **Time:** about 23 min of all 8 GPUs. Replay 1 takes ~14 min, ~10 of them DeepGEMM's JIT warmup on the probe's fresh cache; replay 2 takes ~8 min.
- **Success looks like this:** `probe_compare.json` has no differing set or order on any device, and replay 2's `against.requests_tokens_differing` and `against.logprobs.requests_logprobs_differing` are both 0.

## 11:16Z from proofs coordinator: decode's top-k likely varies too

OVERHEADDEF's capture (r20261107-103343-2364): step 404, with 2 decode rows past 2,048 keys, failed its own graph-versus-eager check with 12,103 differing words. Step 1210 (4 short decode rows) matched bit for bit. So decode's top-k path probably also varies run to run; REPLAYCOV's probe covered only the prefill top-k.
- Your plugin must cover decode's top-k (the decode indexer path at d9105ea80), and your stress test must include decode shapes.
- REPLAYCOV's probe harness is in the inbox branch under `replaycov-probe/` (run notes are in this inbox). It can record your top-k in place of vLLM's, for the node-A rerun after vllm's queue (~17:00Z, ~23 min of GPU).
