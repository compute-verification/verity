# For KSPLIT (HASH-101), from the kernels lead

## Tue 10 Nov 06:41Z: benchmarks' answer on where the table goes

benchmarks, the kit's owner, said yes (Slack thread 1794292569.687439, 06:40Z): `benchmarks/harness/verity_bench/nsys.py`
is the place for the per-kernel table.

- Its other kit calls land Thursday in `kit.py` (lease, host stamp, paired intervals, writer), which imports `nsys.py`
  and does not edit it, so your PR should not conflict.
- **Keep the one-sided SM-time cross check as it is.** Add your block; change nothing that exists in the file.
