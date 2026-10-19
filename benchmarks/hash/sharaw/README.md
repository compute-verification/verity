# sharaw: hm96-sha512 at 4 KiB leaves near the raw SHA-512 rate

`sharaw.cu` commits a step of 4 KiB `sha512/row/v2` rows as hm96-sha512 leaves in a frame-v3-sha512 tree, byte for byte
the format of `verity.ml.commitments`, in kernel forms that keep the GPU throughput-bound at served step sizes.

- `gpu.sh` runs everything on one leased GPU: the build, `check.py` (every candidate form against compute-accounting's
  `benchmarks/pouw/commit_gpu/reference.py`, ragged leaf counts and a fresh key included), then each step size with the raw
  rate, `commit_gpu`, circuits' `native_rows.cu` (fetched at a pinned commit) and the sharaw forms interleaved rep by rep
  with each rep's SM clock. `PROF=1` adds nsys and ncu with `--clock-control none`.
- Measured (median of 7, 18 Oct; `note:hash/20261018T0650Z-draft-sharaw`): at a 13.49 GB prefill step 92.5% of raw on
  H200 (`r20261018-080737-be25`), 94.6% on B200 (`r20261018-080746-d042`), 95.0% on RTX PRO 6000
  (`r20261018-081929-15b3`), against `commit_gpu`'s 82-86%. A 142 MB decode step stays latency-bound at 42-57% of raw.
- On sm_100 and sm_120, a kernel that calls the non-inlined compression and folds tree levels in the same kernel produced
  wrong leaves (suspected ptxas); no form here uses that combination, and every form is checked before it is timed.
