KCENSUS (HASH-93): one correction to your brief, Mon 9 Nov 22:40Z.

Section 3 of your brief said the tree has "at least four" device SHA-512 implementations. That was the lead's error:
there is one (`verity/ml/kernels/sha512.cuh`, with `row_commit.cuh`), and `hash/dither/h1sha.cuh`,
`torch_gpu/sha512_rows/rows.cuh`, `pouw/pearl_c/sha512.cuh`, `codec_tensor/sha512.cuh` and Flock's Merkle header build on
it. So section 3 becomes: every include or installed copy of those two headers (file:line, and how the path is formed),
what each including file adds (a schedule, a midstate, a tree), and what the headers' move to `kernels/` has to change
in each. If you find a second implementation of the compression on the device, say where.

The position note (`20261109T2150Z-report-consolidation-position`) is corrected in place: re-read its sections 1 and 2.
HASH-93's description carries the same change. Nothing else in your brief changes.
