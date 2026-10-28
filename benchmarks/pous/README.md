# PoUS benchmark

One harness and one dataset for every implementation of proof of useful space (PoUS). The metric is the frozen effective
weight bandwidth (EWB): the plaintext weight bytes that one pass decodes from `C`, per second. Every weight is decoded
on every pass, at the scheme's certified parameters.

- `weight_sets.json` (`pous-weight-sets/v1`) is the dataset. It holds the weight sets as the serving engine holds them,
  with their checkpoint pins:
  - Qwen2.5-0.5B: every linear weight and the tied head, 15,075 blocks of 64 KiB;
  - Qwen2.5-7B: layer 0 and the head, 23,744 blocks;
  - Llama-3.1-8B: layer 0 and the untied head, 22,688 blocks (band-gpu and the reference; not in `vllm-gpu`'s frozen
    harness).
- `bench.py` is the harness: one command per (scheme, implementation, weight set, hardware). It writes `bench-result.json`
  (`verity-pous/bench-result/v1`), and `result.json` in `$RESEARCH_RUN_DIR` when that is set.
  - Every implementation passes the same gates:
    - the scheme's pinned conformance vectors;
    - the implementation's own gates: known answers, and decoded blocks equal to the reference's;
    - the timed audit, plus a negative control;
    - `certificate`: the scheme's certificate covers the audit's k and deadline.
  - Each run reports the storage profile the certificate gives the accepted audit, gaps included.
- `pod.sh` runs it on a pod as a research Attempt.
- `lateness/` measures how late honest answers to timed challenges arrive, on-node and from another machine, quiet and
  under load, and the wire's floor, with the tree's only UDP challenger and responder (its README).
- `explorer.py` builds the average-memory explorer's data file (`pous-explorer/v1`): the certified average-ε bound's
  reference math, the honest-latency presets as survival curves, the schemes' serving overhead, ε_proof by in-degree, and
  test vectors the console page reproduces. `--histograms` pools the vLLM e2e series' `verify-honest.json` histograms
  into the `measured` preset. Publish with `research data put --kind pous-explorer/v1 --file F --preserve`.

Implementations:

- `reference`: `verity.core.protocols.pous` on CPU processes.
- `vllm-gpu`: the GPU kernels of `verity_vllm.engine.pous`, from their own tree, through its frozen
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
    adds to it), then timed audits against the native responder under PoUS forwards, and three controls against that one
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
    call and rebuilds a label from 1.53% of `C` in one round's squeeze, 481 Keccak-f, about a tenth of a call (red team).
    A floor that bounds the attacker must time that resume, not a whole call;
  - optionally (`--nvme-dir`), one block's read latency from NVMe;
  - what re-derives every number: every rep, each audit's challenge source and each answer's time, the clocks, torch,
    CUDA and nvcc versions (`pod.sh` pins torch and records the CPU topology, the NVMe mount and the installed packages).

- `p2-v1-gpu`: P2 v4 (`verity/core/protocols/pous/README.md`, "P2"), in `p2_v1/`, with
  the same steps, controls and metrics as `band-gpu` so the two compare side by side (`side_by_side.py`):
  - `reference.py`: the scheme on the CPU on `verity.ml.randomness` (the counter-mode SHA-512 keys, the maps, the RFC
    6962 tree over v4 leaves, the mode-0 audit derivation), pinned by `vectors.json` (`VECTORS_SHA512`);
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
  - `serve.py`: `store.py`'s toy store (below) served under proof of useful bandwidth (PoUB, `remote_bandwidth.p2`), the CPU
    reference of the serving loop: each engine step decodes the store layer by layer, runs the layer's matmuls on what
    it decoded, sketches each block's bytes as it decodes them, and commits one root, which the next step's seed chains
    from; each window's openings are answered before the next step. `serve.py fixture` writes the decode kernel's
    sketch fixture: two chained passes' seeds, sign rows, every block's partial and every unit's sketch and receipt
    (the interface is `remote_bandwidth`'s `PROTOCOL.md` §11). It isn't tracked: `FIXTURE_SHA512` pins its bytes, and
    `tests/test_pous_p2_serve.py` regenerates it.
  - `store.py`: the served model's encoded weights store. It lays the MQA16K INT8 weights out in W with every tensor
    starting on a block (`store.py layout` writes the layout JSON), and its `decode_layer` is the reference a
    decode-to-staging kernel matches byte for byte. `store-toy/` is a 2-layer store at the registered width: its salt,
    C*, negative blocks, and spot vectors at the real store's indices, with W and the staging buffers pinned by digest
    (`store.py toy --plain` writes them). `tests/test_pous_p2_store.py` regenerates both. `store.py commit` is
    `setup.commit` of the real W, regenerated in this layout from the seeded generator (`e2e_h200.served_weights`),
    with each s8 tensor held to the generator's manifest by SHA-256 as the pass reads it. `store.py encode` draws the
    salt from the auditor's coin and encodes C* on every GPU of a node (`pod.sh IMPL=p2-store ... -- encode ...`), as
    canonical segment files a rerun resumes; `--segments LO:HI --order down` encodes a part and `finish` joins the
    parts. `verify --decode` rereads a store and decodes it into the commitment, `spot` decodes random blocks against W
    regenerated from the seed, and `tie` binds the store to the weights' registration digest. `p2_v1/parts/` moves
    parts between machines without an agent VM: `part-out.sh` publishes a node's segments through its run's custody,
    and `fetch-records.sh` fetches those records into another machine's store, checking every file against its
    segment's digests (`join.py`).

A new one is a class in `bench.py`'s `IMPLS`.

~~~bash
PYTHONPATH=. python benchmarks/pous/bench.py --impl reference \
  --weight-set qwen2.5-0.5b/linears+head --hardware local --out /tmp/pous-bench --max-segments 1   # about 50 s

# on the GPU machine, from the source tree's root; pod.sh writes to $RESEARCH_RUN_DIR and reads impl.tar from its inputs/
git -C <the implementation's checkout> archive -o "$RESEARCH_RUN_DIR/inputs/impl.tar" 4e49163193c3750dad580f3a28f5428fbb722a8a
bash benchmarks/pous/pod.sh IMPL=vllm-gpu IMPL_COMMIT=4e49163193c3750dad580f3a28f5428fbb722a8a   # an L40S
bash benchmarks/pous/pod.sh IMPL=band-gpu [WS=qwen2.5-7b/layer0+head] [NVME_DIR=/workspace/nvme]  # an RTX 4090
bash benchmarks/pous/pod.sh IMPL=side-by-side [WS=...] [NVME_DIR=/workspace/nvme]     # an RTX 4090: band and P2 v1, then the table
bash benchmarks/pous/pod.sh IMPL=side-by-side GATES_ONLY=1 MAX_SEGMENTS=2             # an RTX PRO 6000: both arms' kernel gates
bash benchmarks/pous/pod.sh IMPL=band-gpu BAND_D=6                                     # an RTX PRO 6000: the band at in-degree 6
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

`streamed_wipe/` times the streamed-label wipe (`verity/core/protocols/pous/PROTOCOL.md`, "The streamed-label wipe")
on one node: the labels one GPU derives (`labels.cu`, gated against `streamed_wipe.label`), what reaches pinned host
memory, the NIC's line rate and RDMA loopback, and 74,000-challenge round trips at batch sizes from 1 to 74,000, plus one
whole reference session, judged by the Lean verdict (`--verdict lean|both|python`, `wipe_verdict.py`'s `VERDICT` by
default). `bench.py`'s docstring lists the steps; `run.sh` builds for sm_120, takes one GPU through
`gpu-lease` and writes `timing.json`:

~~~bash
bash benchmarks/pous/streamed_wipe/run.sh   # on an RTX PRO 6000 node, from the source tree's root
~~~

`streamed_wipe/served_wipe.sh` is the wipe's piece of a served window: `demo.sh`'s wipe of the H200 a
window is served on, before its engine loads, with the memory challenger on vy-auditor-2. It writes the core answer
(verity/wipe/v0) to `DIR/answer.json`, with demo.sh's outputs and `served.json` beside it.
`benchmarks/one_stage/served_verdicts.sh` runs it as
`WIPE_CMD='bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"'`. It refuses a card that isn't idle or a gate
key that isn't on the node, and kills nothing. Its header lists the challenger's settings. `tests/test_pous_served_wipe.py`
runs it on a CPU, with one segment in host memory.

`wipe_resume/` runs the red team's resume attack (art:7617c8017e80) on both wipes, on CPU. `pi2_state_attack.py` rebuilds
band and dense blocks from the five odd rounds' SHAKE256 states (1,000 B a 64 KiB block) and times an answer against the
`D = 5` read deadline. `resume.py` adapts the attack to the streamed wipe, which the prover can't evaluate without `K`, and runs
it next to honest, naive-cheat and broken-premise provers (`K` on the prover's node, an undeclared tier), all judged by the
Lean verdict and `streamed_wipe.verdict`. It writes `resume.json` and fails if any prover ends other than expected:

~~~bash
bash benchmarks/pous/wipe_resume/run.sh   # on a CPU node, from a clone (the Lean verdict builds into it)
~~~
