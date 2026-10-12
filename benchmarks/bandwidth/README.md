# The bandwidth audit's GPU tagger

The honest prover's step of the bandwidth audit v0 (`verity/core/protocols/bandwidth/PROTOCOL.md`) on an RTX PRO 6000:
every block's tag over the store `E` in device memory, the tags' leaves and their RFC 6962 root, for the three tag
constructions: `ro-blake3/v1` and `ro-sha512/v1`, the certified ones, and `gf31-poly/v1`, the uncertified fast check.

- `tagger.cu` holds the kernels and builds two ways: `bwtag`, which checks the device's roots against the reference and
  then times, and `libbwtag.so`, a C ABI for a harness that holds `E` in its own device buffer. Its header comment
  explains how each kernel is laid out.
- `gpu.py` is the harness API over `libbwtag.so`. `Tagger.tag_pass(tag, store_ptr, blocks, seed, workspace)` returns a
  commitment whose `root` the prover sends; `commitment.open(j, store_tree)` gives the `audit.Opening` the core
  verifier (`audit.Window.verdict`) checks. `Prover.commit(window)` is the honest prover's window: it always prepares
  (an untimed pass under a throwaway seed, then a 512 MiB read of another buffer, so L2 starts clean) before it asks
  for the seed. `gpu.py windows` runs every window through it: the timed windows through `audit.Verifier`, the skip
  controls, and one window per tag at a deadline no certificate allows, which checks the openings of a tag whose timed
  windows are late. `Tagger.device()` is the GPU's id (`audit.device_id`'s 16 bytes, from the driver), for which the
  store's salt is drawn.
- `reference.py` computes the expected roots with the core reference on CPU processes, and the store's PoUS leaves
  under a salt (`store_leaves`), which `gpu.py windows` calls under the lease, once the GPU is known.
- `run.sh` is one Attempt on vy-nebius-2. It builds and runs the reference outside the lease, then under
  `gpu-lease 1 --wait` runs `bwtag`'s gates, its timing and `gpu.py windows`. A failed gate times nothing. Its header
  lists the outputs.
- `summarize.py` writes `summary.json`: the honest pass against the raw device read, the skip curve, the PCIe rates, the
  windows, and `audit.bound` at the measured numbers for the certified tags.

The store is a generated test store (`reference.block`, the bandwidth vectors' blocks) under P2's geometry at its
registered point: 2^19 blocks of 2,056 bytes. It is not a P2 encoding. The audit reads only the blocks and their tree
under `vk`, so a generated store exercises the same work. Its salt is the device's, `audit.draw_salt(secret,
reference.COMMITMENT, tagger.device())` from a secret drawn fresh in each run and never written out, as the verifier
requires (`audit.Verifier` refuses any other salt).

`tests/` holds CPU twins of every kernel's arithmetic and message layout, with the SHA-512 and BLAKE3 constants read out
of `tagger.cu`. It checks them against the core reference and checks that the bottom-up tree and its paths are RFC 6962's.
`tests/test_prover.py` checks, with a fake tagger, that every window the harness runs is prepared before its seed. Run it
with `uv run tools/verity/check/suites.py benchmarks/bandwidth`.
