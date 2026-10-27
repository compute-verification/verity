# The streamed wipe

One GPU's memory wiped and proved wiped, end to end (MEM-67 3a.2): PoUS's streamed-label wipe
(`verity/core/protocols/pous/`, "The streamed-label wipe" in its `PROTOCOL.md`). The memory challenger streams labels
derived from its own secret into every byte of the device the prover declares, the prover keeps them in device memory
(`wipe_store.Cuda`: the CUDA driver through ctypes, no toolkit), and the challenger times its spot reads of them and signs
the session. The Lean (`wipe-protocol`, `verity/core/devices/memory_challenger/lean/`) and the Python
(`streamed_wipe.verdict`) judge the signed session and must agree. `wipe_period` is the leg, and its summary publishes
`foreign_bits`: what of the device could still hold prior content.

- **Protocol:** `pous` (the streamed-label wipe). Its bound is `Pous.Guarantees.StreamedWipe`, and `--reads 5231` is
  the H200's instance, `Pous.Guarantees.StreamedWipeMeets141GB5`.
- **Workload:** none. The fill is the challenger's labels, over the whole device less `--reserve-gib`.
- **Placement:** the prover on the GPU node, and the memory challenger on a host the prover doesn't control
  (`trusted-nebius-host`): `--challenger HOST --challenger-address H:P --challenger-command 'ARGV'`, where `ARGV` starts
  `wipe_party` there. On the auditor hosts the verifier's gate serves `wipe_party` as `auditor` from infra's pinned tree
  (`benchmarks/live_verifier/host.py`, INF-258). Without `--challenger` the challenger runs on the prover's node and the
  premise is unmet; `--simulated` names that stand-in. The verdict is computed on the GPU node from the signed session,
  and `wipe.case.json` replays it with `wipe-protocol` on any CPU node.
- **Owner:** memory-accounting.

## Commands

From the tree root, each in a clone (the Lean builds into it):

~~~bash
# the wipe, the challenger on a host of its own (demo.sh's header lists every option)
research run --on NODE --project verity --source . --cwd clone --timeout 90m -- \
  bash examples/wipe/demo.sh --challenger HOST --challenger-address H:P --challenger-command 'ARGV'

# the same with every step but the size and the placement: 64 segments (2 GiB), the challenger on this node
research run --on NODE --project verity --source . --cwd clone --timeout 90m -- bash examples/wipe/demo.sh --dry-run

# a cheat that keeps G GB of prior content, which must reject
research run --on NODE --project verity --source . --cwd clone --timeout 90m -- bash examples/wipe/demo.sh --foreign-gb G ...

# a rehearsal with the challenger on a cluster peer over ssh (the lane's machine, not the auditor's: it certifies nothing)
research run --on NODE --project verity --source . --cwd clone --timeout 60m -- \
  bash examples/wipe/peer.sh PEER [--tunnel] -- DEMO_OPTIONS
~~~

Outputs go under `$OUT`: `summary.json` (the verdict, `foreign_bits` and what the bound doesn't cover), `wipe.json`,
`wipe.case.json`, `period.json`, and the GPU's and the node's state. The GPU timing measurement of the wipe (the labels'
rate, the pinned-memory and NIC bandwidth, the round trips at each batch size) is `benchmarks/pous/streamed_wipe/`.
