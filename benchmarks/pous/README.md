# POUS benchmark

One harness and one dataset for every POUS implementation. The metric is the frozen effective weight bandwidth (EWB):
the plaintext weight bytes that one pass decodes from `C`, per second. Every weight is decoded on every pass, at the
scheme's certified parameters.

- `weight_sets.json` (`pous-weight-sets/v1`) is the dataset. It holds the weight sets as the serving engine holds them,
  with their checkpoint pins:
  - Qwen2.5-0.5B: every linear weight and the tied head, 15,075 blocks of 64 KiB;
  - Qwen2.5-7B: layer 0 and the head, 23,744 blocks;
  - Llama-3.1-8B: layer 0 and the untied head, 22,688 blocks (band-gpu and the reference; not in `vllm-gpu`'s frozen
    harness).
- `bench.py` is the harness: one command per (scheme, implementation, weight set, hardware). It writes `bench-result.json`
  (`verity-pous/bench-result/v1`), and `result.json` under `research run`.
  - Every implementation passes the same gates:
    - the scheme's pinned conformance vectors;
    - the implementation's own gates: known answers, and decoded blocks equal to the reference's;
    - the timed audit, plus a negative control;
    - `certificate`: the scheme's certificate covers the audit's k and deadline.
  - Each run reports the storage profile the certificate gives the accepted audit, gaps included.
- `pod.sh` runs it on a pod as a research Attempt.
- `explorer.py` builds the average-memory explorer's data file (`pous-explorer/v1`): the certified average-ε bound's
  reference math, the honest-latency presets as survival curves, the schemes' serving overhead, ε_proof by in-degree, and
  test vectors the console page reproduces. `--histograms` pools the vLLM e2e series' `verify-honest.json` histograms
  into the `measured` preset. Publish with `research data put --kind pous-explorer/v1 --file F --preserve`.

Implementations:

- `reference`: `verity.core.protocols.pous` on CPU processes.
- `vllm-gpu`: the GPU kernels of `verity_vllm.engine.pous` (#172, #188), from their own tree, through its frozen
  harness `verity-vllm pous-dense-bench`, with the flags of the band's run of record.

- `band-gpu`: the same band and dense kernels, moved into `band_gpu/` with no vLLM dependency, run by the POUS-only
  harness `band_gpu/harness.py` (steps and metrics in its docstring). Its whole-chain decode runs on lane pairs
  (`harness.KERNELS`, `dense.CHAIN_PAIRS`); `pod.sh KERNELS='{"chain_pairs": false}'` runs the run of record's kernels:
  - kernel gates first: the pinned whole-segment vector on the device, sampled segments of `C` equal to `verity.core.protocols.pous`'s
    encoding, and the round trip. A failed gate stops the run: nothing is timed and no measurement is published;
  - encode GB/s, with `C` then resident;
  - decode overhead on a synthetic loop of the weight set's GEMMs at T = 1, 8, 32 and 2048 tokens, against the same GEMMs
    on plaintext, both forwards from a CUDA graph where the path is capturable, the plain arm's bandwidth against a measured
    device copy, and every T where it is launch-bound flagged; every rep is recorded;
  - the audit limit fixed before any load from a loopback calibration (labelled loopback: a remote memory challenger's round trip
    adds to it), then timed audits against the native responder under POUS forwards, and three controls against that one
    limit: a value control (one block in ten zeroed), a late control (one answer in ten held past the limit, which must
    reject on time alone; the only answers held), and a recompute control (a fraction of `C` dropped, each challenged dropped
    block recomputed on the device from its kept parents, `W`'s block XORed in on the device, as an adversary would: the limit
    minus each recomputed answer's recorded device time is the measured margin, labelled `recompute_margin_kind`: an upper
    bound on attacker latency, from the honest kernel with per-step launches and a host sync, so it overstates the margin; the
    gate fails when no challenge hit a dropped block; it bounds only an attacker that recomputes from kept parents, not one
    that resumes kept round states, as under the step floor below);
  - the `Π₂` step floor, the fastest honest kernel for a whole call, with the SM clock sampled while it runs, the floor
    scaled to the maximum clock and a +15% overclock, and one Keccak-f latency beside it, against the certificate's
    `call_ns`: 241.5 µs, the 4,830 sequential Keccak-f of a 64 KiB call in a chain at 50 ns each (the rounds' absorbs
    stream behind the squeezes; `verity/core/protocols/pous/PROTOCOL.md`), so in the ideal-permutation model `D = 5` covers
    `Δ + RTT < 1.449` ms and the band's cliff at `d = 12` is about 3.14 ms. It is not an upper bound on an attacker's
    floor, and neither are the margins derived from it: at the reference Feistel-SHAKE256 `Π₂` an attacker completes no
    call and rebuilds a label from 1.53% of `C` in one round's squeeze, 481 Keccak-f, about a tenth of a call (red team,
    4 Oct 2026, art:7617c8017e80). A floor that bounds the attacker must time that resume, not a whole call;
  - optionally (`--nvme-dir`), one block's read latency from NVMe;
  - what re-derives every number: every rep, each audit's challenge source and each answer's time, the clocks, torch,
    CUDA and nvcc versions (`pod.sh` pins torch and records the CPU topology, the NVMe mount and the installed packages).

- `p2-v1-gpu`: P2 v4, the specification (`verity/core/protocols/pous/p2/PROTOCOL.md`), in `p2_v1/`, with
  the same steps, controls and metrics as `band-gpu` so the two compare side by side (`side_by_side.py`):
  - `reference.py`: the scheme on the CPU on `verity.ml.randomness` (the counter-mode SHA-512 keys, the maps, the RFC
    6962 tree over v4 leaves, the mode-0 audit derivation), pinned by `vectors.json` (the file hash in the spec's §7);
    `wire.py` is `p2-audit-wire/v4` with its reference verifier, and `vectors.py` regenerates every vector file in one command;
  - `p2v1.cu`: decode with per-block keys on `p2_gpu/p2dec.cu`'s tuned squaring (keys expanded on the device just before
    use), and encode by two roots (an addition chain for `(p + 1)/4` on CGBN's generic squaring, reduced by `2^W ≡ c0`);
  - the kernel gates: the pinned vector through the device (keys, encode, decode), the round trip, sampled blocks against the
    reference's decode and keys, no kernel flag; a failed gate stops the run;
  - audits at k = 105 (`p2_v1/memory_challenger.py`: v4 leaves, `verity/pous/p2-audit-challenge/v4`), against a limit fixed before load (spec
    v4 caps the allowance at 354 µs), with the value, late and recompute controls (one block in ten of all of `C` dropped, each
    re-encoded on demand from its decoded row `Dec(C_i)`, which is `W_i` in the encoded head);
  - encode GB/s over the first `--p2-encode-blocks` blocks (the rest of `C` is random canonical blocks), and one block's two
    roots, the fastest of `--reps` on a pseudo-random block, against twice the limit (SeqRoot; the root is untuned, so an upper
    bound on the fastest GPU root).
  The transport is the harness's responder wire, not `p2-audit-wire/v4`'s frames.

- `p2-gpu`: the efficient-cryptography campaign's tuned P2 kernel (`p2_gpu/`, CGBN) for the **experimental**
  `p2-16448/v2`, whose ChaCha8 keys the spec does not allow (its F4 names counter-mode SHA-512). The registered P2 with the
  spec's keys is `p2-16448/v4`; `p2-v1-gpu` measures those keys on the device.
  - The measured build expands fresh per-block keys in the timed kernel, by ChaCha8 from one 32-byte seed per
    segment. That is M1's schedule, the one the pinned P2 theorems are about, and inside the 1.05× space bound.
  - A tweak candidate at or above p (probability c0/2^w per block) stops the kernel with an error instead of drawing
    the next candidate.
  - The campaign's shared-key build (M2) runs on the same C for comparison.

A new one is a class in `bench.py`'s `IMPLS`.

~~~bash
PYTHONPATH=. python benchmarks/pous/bench.py --impl reference \
  --weight-set qwen2.5-0.5b/linears+head --hardware local --out /tmp/pous-bench --max-segments 1   # about 50 s

git -C <#188 checkout> archive -o impl.tar 4e49163193c3750dad580f3a28f5428fbb722a8a
research run --on <L40S pod> --project verity --source . --cwd source --send impl.tar --campaign pous \
  -- bash benchmarks/pous/pod.sh IMPL=vllm-gpu IMPL_COMMIT=4e49163193c3750dad580f3a28f5428fbb722a8a
research run --on <L40S pod> --project verity --source . --cwd source --campaign pous \
  -- bash benchmarks/pous/pod.sh IMPL=p2-gpu [WS=qwen2.5-7b/layer0+head]                     # experimental P2
research run --on <RTX 4090 pod> --project verity --source . --cwd source --campaign pous \
  -- bash benchmarks/pous/pod.sh IMPL=band-gpu [WS=qwen2.5-7b/layer0+head] [NVME_DIR=/workspace/nvme]
research run --on <RTX 4090 pod> --project verity --source . --cwd source --campaign pous \
  -- bash benchmarks/pous/pod.sh IMPL=side-by-side [WS=...] [NVME_DIR=/workspace/nvme]      # band and P2 v1, then the table
research run --on <RTX PRO 6000 pod> --project verity --source . --cwd source --campaign pous \
  -- bash benchmarks/pous/pod.sh IMPL=side-by-side GATES_ONLY=1 MAX_SEGMENTS=2              # build check: both arms' kernel gates
research run --on <RTX PRO 6000 pod> --project verity --source . --cwd source --campaign pous \
  -- bash benchmarks/pous/pod.sh IMPL=band-gpu BAND_D=6                                      # the band at in-degree 6
~~~

`BAND_D=N` (`bench.py --band-d N`, band-gpu and side-by-side's band arm) runs the band at in-degree `d = N`, as
`band-chain/dN/v1`, with no byte-format name registered for it. It is refused for `d < 5`, which has no certificate. For
`d >= 5`, the certificate gate checks `Pous.SecurityProofs.BandMultiMeetsFamily`, for any segment count, with the cliff at `d`. The pinned
vector is still checked at the registered `d = 12`, and the reference gate checks the run's own `d`. `bench-result.json` records
the run's `d` in `scheme` and `scheme_params`.

`pod.sh` installs nothing on the system, so it runs as-is on a shared node. It builds with the toolkit already installed
(`CUDA_HOME`, `/usr/local/cuda`, or `/usr/local/cuda-13.0`, `-12.9` or `-12.8`). With CUDA 13 it uses torch 2.14.0 cu130 wheels,
and with CUDA 12.8+ it uses 2.7.1 cu128. It builds `pous_native` for the device's capability (`native.arch_list`) and `libp2v1.so`
for `ARCH=sm_120` on an RTX PRO 6000. Without a system `gmp.h`, CGBN's host pass takes gmpy2's wheel's from the venv.
`WORK`, `VCPU`, `RCPU` and `WORKERS` place it on a shared node's directories and cores. Every `nvidia-smi` query names the run's
own GPU (`--id`): `GPU_LEASE_UUID`, else `CUDA_VISIBLE_DEVICES`' first, else the node's only GPU (the harness: the lease's,
else the CUDA device's UUID). `pod.sh` refuses a GPU run on a node with several GPUs and neither. `bench-result.json`'s
`device` records the id queried (`nvidia_smi_id`), the `index`, `uuid` and `pci.bus_id` that answered, and the CUDA device's
own index and UUID. Each `clocks-*.txt` starts with the same id and answer.

`AUDITS=0` (`bench.py --audits 0`, band-gpu and p2-v1-gpu only) is for an untimed, preemptible placement. The gates, encode,
the synthetic loop (decode overhead and bandwidth) and the step floor (P2 v1's SeqRoot root) run as in a timed run. No limit,
audit or control runs, so there is no certificate gate and no storage profile. `bench-result.json` and `result.json`'s
fingerprint say `timed: false` and `audits: skipped (untimed placement)`, so no audit or latency number is cited from it.
