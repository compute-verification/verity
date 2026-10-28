# u2_packed

The filler's exact GEMM, O = A Wᵀ with s32 out, on 2-bit operands (values in {0, 1, 2, 3}) streamed from HBM packed four
values a byte and unpacked on chip before the tensor cores read them. Streaming packed operands moves a quarter of the bytes
that int8-stored ones do, at the price of integer work on the CUDA cores that SHA-512 beside the GEMM also needs. This
directory measures both sides: the GEMM's MACs per SM clock against 4,096, and the hash rate beside it against the bar.

## The packed format

A row of A (M × K) or W (N × K) is K / 4 bytes. Its value k is bits 2i and 2i + 1 of byte 4 ⌊k / 16⌋ + k mod 4, with
i = ⌊k / 4⌋ mod 4. Each little-endian 32-bit word x of a row then holds 16 consecutive values, and
(x >> 2i) & 0x03030303 is the four bytes of values 16w + 4i to 16w + 4i + 3, in order. `bench.py`'s `pack` and `unpack`
are the format's reference, and `tests/` holds the kernel's arithmetic to them.

The masked form (`p2m`) packs W with each byte's pairs reversed (value k at bits 2 (3 − i) and 2 (3 − i) + 1, `pack(w,
reverse=True)`). Then x & (0x03030303 << 2i) is A's four values times 4ⁱ and y & (0x03030303 << (6 − 2i)) is W's times
4³⁻ⁱ, so every product is 64 times the values' product. Unpacking is one AND per four values with no shifts, the MMA runs on
u8, and the epilogue shifts O right by 6 (exact while K < 2³¹ / 576).

## The kernel (`packed.cu`)

A persistent CTA per SM with two consumer warpgroups and no producer warpgroup, each warpgroup 64 × 256 of a 128 × 256
tile, at most 168 registers each so that a 256-thread `hash_k` CTA fits beside it. Thread 0 issues every TMA. All 256
threads unpack each k-block (128 deep) from a packed ring into the SWIZZLE_128B layout `wgmma m64n256k32` reads, and one
named barrier per k-block orders the two. The arms:

| arm | operands | what it measures |
|---|---|---|
| `i8` | int8 by TMA straight into the MMA's slots | the same schedule on four times the bytes |
| `p2` | packed by TMA, unpacked by shifts and masks | the candidate |
| `p2u` | packed, unpacked as u8 × 64 (left shifts, which ptxas may put on the FMA pipe), shifted back in the epilogue | a cheaper unpack on the integer pipe |
| `p2m` | packed with W's pairs reversed, unpacked by masks alone (u8 × 64) | the fewest integer instructions an unpacked value |
| `res` | one block unpacked once and reused | the schedule's ceiling with nothing streamed |
| `p2x`, `p2ux`, `p2mx` | `p2`'s, `p2u`'s and `p2m`'s loads and unpacking, no MMA | the unpacking's own time |

## Running it

`run.sh` builds the cubin with nvcc, holds it to `sass_gate.py` (no spills, the right MMA atom, the register caps), and runs
`bench.py` under one gpu-lease. Each arm is first gated bit for bit against `torch._int_mm` on the same operands, with O
poisoned beforehand, and then timed in items of about 4 s (8 s with the hash beside), with clocks from `%clock64`, watts
from NVML's energy counter and the hash's compressions against 4.355 G/s.

`ARM+wN` runs one `hash_k` CTA of N warps on each SM beside the GEMM. The hash's width is how the split between the two is
set: SM warp schedulers give a free-running hash most of the issue slots, so a CTA of eight warps takes nearly the whole
bar and leaves the GEMM a fraction of its rate. The hash starts once the GEMM's first call is resident, logs its progress
every couple of milliseconds, and stops on a flag after the GEMM's last call. Both rates are taken over the same window:
the GEMM's segments inside the hash's span.

~~~bash
research run --on NODE --project verity --source . --cwd source -- bash benchmarks/torch_gpu/u2_packed/run.sh
~~~

`--arms`, `--shapes`, `--secs` and `--rounds` narrow it (`bench.py --help`); `BUILD_ONLY=1` stops after the gate.
