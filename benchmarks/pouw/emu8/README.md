# emu8: Dither's int8 words on FP8 E4M3 tensor cores

Dither commits every 32-step prefix word (exact s32) of C = ÃB̃, with Ã = A + R and B̃ = B + S, both int8. This
directory asks whether sm_120's E4M3 MMA (`QMMA.16832.F32.E4M3.E4M3`) can produce those words bit for bit, how fast,
and how many of the honest limb products a cheater can skip.

Each int8 is split into balanced base-16 limbs: x = 16h + l, with h = (x + 8) >> 4 in [−8, 8], l in [−8, 7], and
m = h + l in [−15, 14]. Every plane below is an E4M3 integer.

| Arm | Planes (A side; B side) | MMAs per k32 step | Recombination per step, in registers |
|---|---|---|---|
| `imma` | int8 | 1 IMMA | the s32 accumulator, which is the word |
| `k3` (Karatsuba) | [4h, m, −l]; [4h', m', l'] | 3 QMMA: acc1 = (4h)(4h') then (−l)(l'), from C1; acc2 = m m', from C2 | word += 15·bits(acc1) + 2·bits(acc2), two IMADs |
| `k4` (schoolbook) | [16h, l]; [16h', l'] | 4 QMMA chained into one accumulator, from C0 | word += bits(acc) − C0 |

The identity behind k3 is xy = 240hh' + 16mm' − 15ll'. Each step's MMAs start from a constant C in a binade whose ulp
divides the step's units, so the FP32 result is exactly C + T. With T1 = 16ΣHH' − ΣLL' and T2 = ΣMM':

- C1 = 0x4B7BAB6C has ulp 1 and |T1| ≤ 32768;
- C2 = 0x49E07A56 has ulp 1/8 and T2 is in [−6720, 7200];
- 15·C1 + 2·C2 ≡ 0 mod 2^32, so the step's increment 15·T1 + 16·T2 needs no constant;
- k4's C0 = 0x4B400000 (1.5·2^23) has |T| ≤ 2^19.

Nothing is recombined through memory. The k32 word is updated in registers after every step, and the GEMM stores only
the final word (or, in `DUMP` builds, every prefix word for the gate).

| File | What |
|---|---|
| `emu8.cu` | The counter-hash operand generator (`DISTS`: uniform, max-magnitude limbs, every product large and positive, 1024-step runs of one sign, max magnitude with independent signs, Dither-shaped, A = 0 with B̃ = S), the plane split, `gemm<ARM, FOLD, DUMP>` (a 128×128 CTA, cp.async, ldmatrix), and cuBLASLt int8 and FP8 baselines. Modes: `gate` (every prefix word of every arm at K up to 16384, under `gpu_kit/bench.cuh`'s 0xA5 poisoning with its no-write control, against the device's own int64 words), `time` (each shape's final words gated against cuBLASLt int8, then interleaved timed items) and `hostdraw` |
| `emu8.py` | The host's half: the limb identities over every int8 pair (`identities`), the numpy twin of the generator, `recheck` (operands against the twin, every arm's words against numpy int64 prefix sums, a replay of sampled steps through `verity.ml.circuits.silicon`'s E4M3 model), and `summarize` (ms per arm and ratios to `imma` and cuBLASLt int8) |
| `accounting.py` | The security accounting: for k3 and k4 and each range of U, how many limb products a cheater with A = 0 (or a one-limb A) and B = 0 still needs; the NVFP4 route's per-block coverage of U by two E2M1 terms; ε |
| `run.sh` | The job: build and SASS-gate each arm's mainloop (IMMA 32 per stage; k3 48 and k4 64 QMMA per stage) outside the lease, then under one `gpu-lease` the gate and the Llama-3.1-8B shapes at M = 4096, then the recheck and the summary |

~~~text
research run --on vy-nebius-2 --project verity --cwd source --env GPU_LEASE_WHO=LANE -- \
  bash benchmarks/pouw/emu8/run.sh [REPS]
python benchmarks/pouw/emu8/accounting.py [--json OUT]   # CPU
~~~

The tests are `../tests/test_emu8.py`. Its slow test builds `emu8.cu`, checks `hostdraw` against the twin, and runs
the SASS gate.
