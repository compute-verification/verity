# The bandwidth audit's GPU tagger

The honest prover's step of the bandwidth audit v0 (`verity/core/protocols/bandwidth/PROTOCOL.md`) on an RTX PRO 6000:
every block's tag over the store `E` in device memory, the tags' leaves and their RFC 6962 root, for the three tag
constructions: `ro-blake3/v1` and `ro-sha512/v1`, the certified ones, and `gf31-poly/v1`, the uncertified fast check.

- `tagger.cu` holds the kernels and builds two ways: `bwtag`, which checks the device's roots against the reference and
  then times, and `libbwtag.so`, a C ABI for a harness that holds `E` in its own device buffer. Its header comment
  explains how each kernel is laid out.
- `gpu.py` is the harness API over `libbwtag.so`. `Tagger.tag_pass(tag, store_ptr, blocks, seed, workspace)` returns a
  commitment whose `root` the prover sends; `commitment.open(j, store_tree)` gives the `audit.Opening` the core
  verifier (`audit.Window.verdict`) checks. `Prover(tagger, store, blocks, scrub)` is the honest prover of one store
  for every tag, and `Prover.commit(window)` its window under the window's tag: it opens the window
  (`window.open()`, whose record the verifier fsyncs), then always prepares (an untimed pass under a throwaway seed,
  then a 512 MiB read of another buffer, so L2 starts clean) before it asks for the seed (`window.start()`).
  `challenger_device(tagger, gpu)` is the id the challenger reads itself (`audit.read_device`: nvidia-smi's `GPU-…`
  UUID for `gpu`, a MIG slice refused), for which the store's salt is drawn; it refuses a tagger whose CUDA device
  (`Tagger.device()`, 16 bytes) has another UUID. `gpu.py windows --gpu` names the GPU (`run.sh`: the lease's).
- A store has one challenger state and one prover, whatever the tag: `audit.ChallengerState.create(state_dir, vk)`
  once, in the challenger's state directory under the name the store gives it (`audit.state_name`), then one
  `audit.ChallengerState(state_dir, vk)` that every tag's
  `audit.Verifier(vk, params, secret, state, device=, commitment=, salt_source=)` takes, and `verifier.window()`
  takes no number. `gpu.py windows` runs every window through them: the timed windows, the skip controls, and one
  window per tag at a deadline no certificate allows, which checks the openings of a tag whose timed windows are late.
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
reference.COMMITMENT, device)` for the device the challenger read, from a secret drawn fresh in each run and never
written out, as the verifier requires (`audit.Verifier` refuses any other salt).

`tests/` holds CPU twins of every kernel's arithmetic and message layout, with the SHA-512 and BLAKE3 constants read out
of `tagger.cu`. It checks them against the core reference and checks that the bottom-up tree and its paths are RFC 6962's.
`tests/test_prover.py` checks, with a fake tagger, that every window the harness runs is prepared before its seed and is
in the store's one challenger state under a nonce of its own. Run it with
`uv run tools/verity/check/suites.py benchmarks/bandwidth`.
