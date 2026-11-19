# PoUW's prover

This is the developer's half of [PoUW](../verifier/README.md): what runs on the developer's machines so that each
registered matrix product can later be credited. It noises and commits each product as the model is served, builds the
circuits a sampled tile is proved against, and holds the GPU kernels that do the work. PoUW's verifier never imports
it, so nothing here is trusted.

## Try it

The kernel schemes are named once, in `../verifier/kernel_schemes.py`, and built here; a caller asks for one by name,
or for the one a resource scheme runs on a card:

~~~sh
uv sync --all-packages --extra torch-cpu
uv run python -c "
from verity.protocols.resource_accounting.compute.pouw import scheme
from verity.protocols.resource_accounting.compute.pouw.prover import schemes
from verity.protocols.resource_accounting.compute.pouw.verifier import kernel_schemes
print(', '.join(kernel_schemes.BY_NAME))
print(schemes.get('pearl-c-sm120-v1-h3').name)
print(schemes.of(scheme.PEARL_PLUS, 'h100', pinned=True, arch='sm_90a').name)
"
~~~

It prints every kernel scheme's name, then `pearl-c-sm120-v1-h3`, the scheme it built, then `pearl-c-h100-v1-h3`, the
one Pearl+ runs on the H100. It needs no GPU. `pearl-c-nvfp4-v0` and `dither` are named but raise when asked for until
`verity.protocols.experimental.pearl_c4` or `verity.protocols.experimental.dither` is imported, which registers its
builder.

## What is here

| Part | What it is |
| --- | --- |
| `schemes/` | The kernel schemes' builders: NCP, Pearl's FP8 scheme and Pearl-C, each a `PoUWScheme`, and Pearl-C's device record. |
| `circuit/` | Each scheme's Definitions, an execution phase's Program and sampling design, and what a call's rows are committed under. |
| `serving/` | What runs beside a served model, one directory a scheme. An adapter reaches it only through the names `serving` exports. |
| `lowering/` | H1's lowering: the statements Flock proves for a sampled H1 tile, and the commitments they are opened against. |
| `kernels/` | The device code, one directory each, each with its README. |
| `sass_gate.py` | The SASS gate: no shipped kernel computes a value the verifier's scalar model can't replay. |

A caller imports a module by its own name and takes what its `__all__` lists. Not yet true: some modules read by other
files still lack a full `__all__`.
