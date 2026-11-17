# PoUW's serving side

This package is what runs beside a served model so that [PoUW](../../verifier/README.md) can later credit its matrix
multiplies. A workload's adapter calls it once for each matrix multiply, and it commits what the scheme will need. It
holds no device code and loads no tensor library: the workload hands in its tensors, its int8 arm and H1's device, and
until it does the host paths run.

A workload reaches it only through the names the package exports. There are two calls:

- `run_matmul(run, site, x, weight, bias=None)` runs a matrix multiply under PoUW, served and committed by `run`.
- `observe_matmul(run, site, x, weight, y, ...)` records one the workload served itself, and computes nothing.

`site` is the linear's ordinal in the forward pass. The package's docstring says which run each scheme takes, and each
module's docstring says what it holds.

| Scheme | Where |
| --- | --- |
| H1 | [`h1/`](h1/README.md) |
| The lead | `lead/` |
| Pearl-C | `pearl_c/` |
| NCP (`ncp-v1`, `ncp-v2`) | `ncp.py` |

There is nothing here to run on its own: the vLLM adapter drives it while serving.
