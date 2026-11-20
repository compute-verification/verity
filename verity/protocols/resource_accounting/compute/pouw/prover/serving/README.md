# PoUW's serving side

This package is what runs beside a served model so that [PoUW](../../verifier/README.md) can later credit its matrix
multiplies. A workload's adapter calls it once for each matrix multiply, and it commits what the scheme will need. It
holds no device code and loads no tensor library: the workload hands in its tensors and its device halves, and until it
does the host paths run.

A workload reaches it only through the names the package exports. There is one call:
`run_matmul(run, site, x, weight, bias=None)` runs a matrix multiply under PoUW, served and committed by `run`.

`site` is the linear's ordinal in the forward pass. The package's docstring says which run each scheme takes, and each
module's docstring says what it holds.

| Scheme | Where |
| --- | --- |
| Pearl-C | `pearl_c/` |

H1 is deleted; a registered table commits through `verity.commitments.periods.rows`. The live coins a window draws (its
challenger and sample session) are in `coins/`.

There is nothing here to run on its own: the vLLM adapter drives it while serving.
