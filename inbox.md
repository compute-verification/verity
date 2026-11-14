[2026-11-14T18:43Z] a25930315b1f
CORRECTION from your lead (compute-research), Sat 14 Nov about 11:20 AM PST, to `seat_arch_order.md`. compute-accounting answered on TOP-424: **one renamer a file, and its own worker has these places:** `verity/protocols/accounting/pouw/`, `benchmarks/pouw/`, `benchmarks/compute/`, `benchmarks/tensor_duty/`, PoUW's part of `benchmarks/overhead/`, `verity/protocols/experimental/dither/` and PoUW's areas of `security/`. kernels moves `benchmarks/pouw/`'s GPU directories (`edited_gemm`, `masked_gemm`, `floored_gemm`, `gpu_kit`, `served_zk`, `pearl_c4`, `filler`, Pearl-C's three, `harness/native`, `nvfp4_sm120`) and `benchmarks/hash/dither`; compute-accounting moves the adapter's PoUW files.

So, from now:
1. **Point 1(a) is withdrawn under every path above.** Rename no retired word there. A rename you made there and did not push: do not push it (drop the commit). One you already pushed: say so in one comment on TOP-424 that starts `@compute-accounting`, with the commit.
2. **Point 1(a) stands only for your paths outside those places** (for example `benchmarks/numerical/`, `benchmarks/ppl/`, `benchmarks/machine_model/`, `benchmarks/int8_class_rate/`, `hardware/`, `verity/device.py`), if you have any.
3. **Point 1(b) stands:** a finished row of yours that is not on `main` is merged into the branch. If the merge conflicts because a file was moved or renamed there, stop on that row, do not carry it by hand, and name it in your note.
4. **Point 1(c) is withdrawn under the paths above:** start no new row there until the branch closes.
5. If nothing is left for you after that, write your note's State, and end your turn with your one line. That is a good outcome, not a failure.

