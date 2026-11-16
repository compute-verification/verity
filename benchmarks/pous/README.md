# The PoUS benchmark

PoUS's codecs have several implementations: the CPU reference, and GPU kernels in more than one form. This benchmark
times them all with one harness on one dataset, so their numbers compare. The number is the frozen
effective weight bandwidth (EWB): the plaintext weight bytes that one pass decodes from `C`, per second. Every weight
is decoded on every pass, at the scheme's certified parameters.

Every implementation goes through the same gates: the scheme's pinned conformance vectors,
its own known answers and blocks equal to the reference's, the timed audit with a negative control, and the
scheme's certificate covering the audit's k and deadline. Each run reports the storage profile the certificate gives the
accepted audit, gaps included.

## Try it

The reference implementation on one segment of Qwen2.5-0.5B's weights, on the CPU, from the source tree's root:

~~~bash
PYTHONPATH=. python benchmarks/pous/bench.py --impl reference \
  --weight-set qwen2.5-0.5b/linears+head --hardware local --out /tmp/pous-bench --max-segments 1   # about 50 s
~~~

It prints one line, the scheme, the effective weight bandwidth and each gate's verdict, and writes `bench-result.json`
to `--out`. It exits 1: no Lean judges the band's timed audit (`timed_audit.lean_judges`: P2 v4 alone), so that gate
fails and the run's `passed` is false.

| Implementation | What it is |
| --- | --- |
| `reference` | PoUS's verifier, the Python reference, on CPU processes. |
| `vllm-gpu` | The vLLM adapter's PoUS kernels, through their frozen harness. |
| `band-gpu` | The same band and dense kernels with no vLLM dependency. |
| `p2-v3-gpu` | P2 v4 on the GPU, with the same steps, controls and metrics as `band-gpu`, so the two compare side by side. |

A new implementation is a class in `bench.py`'s `IMPLS`. `streamed_wipe/` times the streamed-label wipe on one node,
and `wipe_resume/` runs the red team's resume attack on both wipes.

## On a GPU

`pod.sh` runs the harness on a GPU pod as one Attempt: it builds the implementation's kernels and runs `bench.py`.

~~~bash
# on the GPU machine, from the source tree's root; pod.sh writes to $RESEARCH_RUN_DIR and reads impl.tar from its inputs/
git -C <the implementation's checkout> archive -o "$RESEARCH_RUN_DIR/inputs/impl.tar" 4e49163193c3750dad580f3a28f5428fbb722a8a
bash benchmarks/pous/pod.sh IMPL=vllm-gpu IMPL_COMMIT=4e49163193c3750dad580f3a28f5428fbb722a8a   # an L40S
bash benchmarks/pous/pod.sh IMPL=band-gpu [WS=qwen2.5-7b/layer0+head] [NVME_DIR=/workspace/nvme]  # an RTX 4090
bash benchmarks/pous/pod.sh IMPL=side-by-side [WS=...] [NVME_DIR=/workspace/nvme]     # an RTX 4090: band and P2, then the table
bash benchmarks/pous/pod.sh IMPL=side-by-side GATES_ONLY=1 MAX_SEGMENTS=2             # an RTX PRO 6000: both arms' kernel gates
bash benchmarks/pous/pod.sh IMPL=band-gpu BAND_D=6                                     # an RTX PRO 6000: the band at in-degree 6
~~~

`pod.sh` installs nothing on the system, so it runs as-is on a shared node. It builds with the toolkit already installed
(`CUDA_HOME`, `/usr/local/cuda`, or `/usr/local/cuda-13.0`, `-12.9` or `-12.8`). With CUDA 13 it uses torch 2.14.0 cu130 wheels,
and with CUDA 12.8+ it uses 2.7.1 cu128. It builds `pous_native` for the device's capability (`native.arch_list`) and `libp2v1.so`
for `ARCH=sm_120` on an RTX PRO 6000. Without a system `gmp.h`, CGBN's host pass takes gmpy2's wheel's from the venv.
`WORK`, `VCPU`, `RCPU` and `WORKERS` place it on a shared node's directories and cores. Every `nvidia-smi` query names the run's
own GPU (`--id`): `GPU_LEASE_UUID`, else `CUDA_VISIBLE_DEVICES`' first, else the node's only GPU (the harness: the lease's,
else the CUDA device's UUID). `pod.sh` refuses a GPU run on a node with several GPUs and neither. `bench-result.json`'s
`device` records the id queried (`nvidia_smi_id`), the `index`, `uuid` and `pci.bus_id` that answered, and the CUDA device's
own index and UUID. Each `clocks-*.txt` starts with the same id and answer.

[`DESIGN.md`](DESIGN.md) says what each implementation's run does step by step, and holds the untimed runs, the wipe's
and the served run's pieces, and what code outside this directory uses.
