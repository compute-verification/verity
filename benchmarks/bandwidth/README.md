# Device ceilings for PoUB

The measurements PoUB's timing cites (`verity/protocols/poub/verifier/timing.py`): how fast a node's writers can
fill a tier with labels, and how much a card holds on chip. Each probe is one Attempt on a GPU node.

## Write ceilings

The write ceilings of PoUB, `timing.GPU_WRITE_BYTES_PER_S` and
`timing.HOST_WRITE_BYTES_PER_S`, bound how fast a fill can write labels into a tier. These files measure what the node's
writers reach, each arm gated before it is timed (the destination poisoned with 0xA5, sampled slots checked, and a
control that writes nothing rejected).

- `write.cu` (`bwwrite`) on one card: labels derived on the device by 16-byte stores (plain and evict-first) and by
  bulk (TMA) stores from shared memory, the driver's memset, labels streamed in over PCIe by the copy engines (H2D)
  and by the SMs (zero copy), and the card's D2H into pinned host memory. `run_write_gpu.sh` is one Attempt under the
  queue's GPU; its header lists the outputs.
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
