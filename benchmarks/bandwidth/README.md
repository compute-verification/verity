# Device ceilings for PoUB

The measurements PoUB's timing cites (`verity/protocols/poub/verifier/timing.py`): how fast a node's writers can
fill a tier with labels, and how much a card holds on chip. Each probe is one Attempt on a GPU node.

**Run:** `verity bench bandwidth chip|write_gpu|write_host [ARGS]` (`run.sh`) runs one probe's script, named below, as
the command of a `research run`. `uv run tools/verity/check/suites.py benchmarks/bandwidth` tests the runner, on any
machine.

**Writes:** a probe's records are its result, under its run's `$OUT`: `chip.json` and `ur.jsonl` (`chip`),
`write.jsonl` (`write_gpu`) and `write_host.jsonl` (`write_host`), beside the build's and the device's logs. Each
script's header lists its files. No code reduces a probe's records to one number, so no probe writes a `result.json`.

## Write ceilings

The write ceilings of PoUB, `timing.GPU_WRITE_BYTES_PER_S` and
`timing.HOST_WRITE_BYTES_PER_S`, bound how fast a fill can write labels into a tier. These files measure what the node's
writers reach, each arm gated before it is timed (the destination poisoned with 0xA5, sampled slots checked, and a
control that writes nothing rejected).

- `write.cu` (`bwwrite`) on one card: labels derived on the device by 16-byte stores (plain and evict-first) and by
  bulk (TMA) stores from shared memory, the driver's memset, labels streamed in over PCIe by the copy engines (H2D)
  and by the SMs (zero copy), and the card's D2H into pinned host memory. `run_write_gpu.sh` is one Attempt under the
  queue's GPU.
- `write_host.c` (`bwwrite_host`) on the host's cores: labels by non-temporal and by cached stores, and `rep stosb`,
  one thread a core over 1, 2, 4, ... cores and each NUMA node's. `run_write_host.sh` is one Attempt on the CPUs the job
  was given.

## The chip's stores

`timing.GPU_CHIP_BYTES` bounds what an RTX PRO 6000 holds on chip.

- `chip.cu` (`bwchip`) measures the on-chip stores that bound has to cover: the instruction caches (code-footprint
  sweeps over PTX that it writes and ptxas assembles), whether code and constants share a cache, and where the evicted
  constants come back from; the constant cache, the TLBs, L1 and L2 retention; and how many uniform registers a warp
  has (`chip_ur.py` rewrites a cubin's uniform-register fields past what ptxas allocates). `run_chip.sh` is one
  Attempt, and it refuses any cubin whose SASS doesn't hold the code that its sweep says it does.

## A node's own report

A node file (`verity/hardware/nodes/`) states what a node's registration does and no card states. `node.py` reads
three of those on the node, in one Attempt that writes nothing to a device:

- the interfaces that carry traffic off the node, each with its line rate: every interface's own report, `ethtool`
  and the routes, and which interfaces the routes send packets out of (`net.json`, `net.txt`);
- the interface each disk is behind: its driver and bus, and the PCI Express link of a drive that is the node's own,
  which bounds its read; a disk the hypervisor gives the guest reports none (`disks.json`, `disks.txt`);
- what each NVIDIA card's firmware says it gives a program: the usable-memory register and the MMU's local memory
  range, read over BAR0 at the addresses of NVIDIA's open kernel modules, beside `nvidia-smi -q` (`regs.json`).

Its docstring has the command line, how it decides that an interface leaves the node, and each register's header.
