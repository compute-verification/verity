# Proof of useful bandwidth's device probes

What the device model of proof of useful bandwidth (PoUB), `verity/core/protocols/remote_bandwidth/timing.py`, takes as
measured on the card and the host: the on-chip stores S_chip has to cover, and the write ceilings that bound a fill.
Each file below is a probe, gated before it is timed; none is on the served path.

## The chip's stores

- `chip.cu` (`bwchip`) measures the on-chip stores that S_chip has to cover: the instruction caches (code-footprint
  sweeps over PTX that it writes and ptxas assembles), whether code and constants share a cache, and where the evicted
  constants come back from; the constant cache, the TLBs, L1 and L2 retention; and how many uniform registers a warp
  has (`chip_ur.py` rewrites a cubin's uniform-register fields past what ptxas allocates). `run_chip.sh` is one
  Attempt, and it refuses any cubin whose SASS doesn't hold the code that its sweep says it does.

## Write ceilings

PoUB's write ceilings, `timing.GPU_WRITE_BYTES_PER_S` and `timing.HOST_WRITE_BYTES_PER_S`, bound how fast a fill can
write labels into a tier. These files measure what the node's writers reach, each arm gated before it is timed (the
destination poisoned with 0xA5, sampled slots checked, and a control that writes nothing rejected).

- `write.cu` (`bwwrite`) on one card: labels derived on the device by 16-byte stores (plain and evict-first) and by
  bulk (TMA) stores from shared memory, the driver's memset, labels streamed in over PCIe by the copy engines (H2D)
  and by the SMs (zero copy), and the card's D2H into pinned host memory. `run_write_gpu.sh` is one Attempt under the
  queue's GPU; its header lists the outputs.
- `write_host.c` (`bwwrite_host`) on the host's cores: labels by non-temporal and by cached stores, and `rep stosb`,
  one thread a core over 1, 2, 4, ... cores and each NUMA node's. `run_write_host.sh` is one Attempt on the CPUs the job
  was given.

## Shared kernels

`tagger.cu` holds device code that `word_codec/wc.cu`, `word_codec/wide.cu` and `kv_filler/topup.cu` include. Its
header comment explains how each kernel is laid out.
