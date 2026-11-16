# pearl_c4: Pearl-C on FP4 operands, experimental

PoUW's Pearl schemes credit a GPU's matrix products by having it compute them with noise in the operands, so that a
verifier can recompute a sampled tile bit for bit. Pearl-C does that on FP8 products. Pearl-C4 (`pearl-c-nvfp4-v0`, with
`pearl-c-mxfp4-v0` as a contrast line) is the same scheme on the FP4 product of the RTX PRO 6000's sm_120, one scale per
scale group (`mma.sync kind::mxf4nvf4`): the noise enters inside one FP4 atom per element, one noisy cast forms each
operand, and the BF16 clean-up is checked as in Pearl-C.

It is a protocol being tried: no demo lists it, and its guarantee rests on a premise that is stated and not proved.
PoUW loads nothing of `experimental/`: its registry names the scheme and refuses it with `NotImplementedError`.

## Try it

~~~sh
uv run tools/check/suites.py verity/protocols/experimental/pearl_c4 --quick
~~~

This runs its tests without the slow ones, on the CPU: the numpy twin of one kernel tile, the device check's fixture,
the offline SASS gate and the fast F1′ evaluation against the scalar reference, the scheme's pinned vectors, and its
circuits. It takes about a minute and a half.

Its kernels need an sm_120 card. Its Lean is in PoUW's areas of `security/`, and its device measurements are under
`benchmarks/pouw/pearl_c4/`. [`DESIGN.md`](DESIGN.md) says what each part of the package is and where the rest lives.
