# tc_probe: tensor-core instructions against their models

Verity's circuits for GPU arithmetic use each tensor-core instruction's exact semantics
([`verity/circuits/silicon`](../../verity/circuits/silicon/README.md)), so a served model's circuit computes the words
the GPU computes, bit for bit. Honest serving passes only where those models match the silicon. These probes check that
on the hardware: each runs an instruction on its GPU and compares every output word with the instruction's pinned
model. From that evidence, `trust.py` computes how far each instruction's model is trusted; nobody sets it by hand.

The instructions run from sm_80 to sm_120a, in BF16, E4M3 and E5M2. [`tc_probe_fp4`](../tc_probe_fp4/README.md) is
the Blackwell FP4 and FP8 counterpart.

## Try it

~~~text
uv run python benchmarks/tc_probe/trust.py --all [--root STORE] [--format md|json]
~~~

This needs no GPU. It prints a dossier for each subject: its trust level (UNPROBED, HYPOTHESIS, PINNED, TRUSTED or
TRUSTED_STABLE), each criterion with its evidence, and a note where the instruction registry's status is ahead of what
the evidence shows. It reads the evidence store; where the store holds no probe evidence, every subject is UNPROBED.

The probes themselves run from the repository root after `uv sync --all-packages`, on a GPU of the instruction's
architecture with nvcc (and `cuobjdump` for `--sass-check`):

~~~text
uv run python benchmarks/tc_probe/tc_probe.py --instruction sm89.mma.m16n8k32.e4m3 --out DIR --sweep --sass-check
uv run python benchmarks/tc_probe/tc_probe.py --instruction sm89.mma.m16n8k32.e4m3 --out DIR --canary
uv run python benchmarks/tc_probe/instances_hw.py --instances fp8-ada-synthetic/vu-k1536-fp8 --out DIR
~~~

The canary is the small fixed tile set to rerun after a driver or toolkit change, or on a second device of the SKU.
Each script's docstring lists what it takes and what it writes.

`tool.py` holds the store declarations, which say which parameters key a result; the evidence store's tool registry
loads it by name. A test of `verity/circuits/ops` loads the probe's tile families and holds them byte-identical for a
fixed seed, since recorded sweeps and canaries are keyed on them.
