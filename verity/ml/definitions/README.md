# verity.ml.definitions

The Definitions of the kernels served models run: each kernel vLLM launches (a GEMM, an RMSNorm, attention, MoE routing,
sampling, a training step's optimizer) written as a composite Definition over the primitives of `verity.ml.circuits.ops`,
gate for gate what the kernel computes on the GPU, together with its Boolean version, the fast evaluators that reproduce
it, and the partition of a served Program into proof blocks. The integrations (`integrations/vllm/`) build served
Programs from these Definitions.

No verifier reads this package, so it is off the soundness path: a verifier holds a Program and the semantics in
`verity.ml.circuits`, never this code.

## Example

Load every registered Definition, bind one, and find its Boolean version:

```python
from verity.ml.circuits import Program, bind
from verity.ml.circuits.boolean import is_boolean
from verity.ml.definitions.boolean import boolean_version
from verity.ml.definitions.registry import catalog

REGISTRY, errors = catalog.load()       # errors == []: every registry module imported
rms = bind(REGISTRY.defs["RMSNormTriton_v1"], N=64, EPS=1e-05)
rms.id, is_boolean(rms), Program(rms).gates
# ('RMSNormTriton_v1{N=64,EPS=1e-05}', False, 1286): a word Definition, 1286 gates on BF16 and FP32 words
bits = boolean_version(rms)
bits.id
# 'RMSNormTriton_v2{N=64,EPS=1e-05,SQRT=MufuSqrtFtz_v2,SCALEA=DivFullScaleA_v3,RCP=DivFullRcp_v2}'
is_boolean(bits), Program(bits).gates
# (True, 785666): the same function as And/Xor/Not gates on bits
```

Building the Boolean version takes a few seconds and about 1.6 GB.

## The registry

`registry/` holds one module per kernel family, each registering its Definitions in `verity.ml.circuits.defs.REGISTRY`
under `Name_vN`: word Definitions in modules named for the kernel (`fp8`, `deepgemm`, `fa2_split`, `sampling`, ...) and
their Boolean versions in the matching `boolean_*` modules. `value_inputs` holds the versions that take what a program
brings (a norm's epsilon, the softmax scale, a softcap) as an input value rather than a static. A module's docstring lists
its Definitions with their signatures.

- `registry.catalog.load()` imports every registry module and returns `(REGISTRY, errors)`. A module that fails to import
  is named in `errors`, never read as "not registered". Consumers that resolve a Definition by id go through it, so a new
  module's Definitions resolve everywhere, not only where someone imported the module.
- Each Definition module has two sidecars beside it: `<module>.circuit-check.py`, the small bindings `circuit-check`
  checks it at, and `<module>.circuit-check.json`, its lowering pins (`tools/verity/circuit_check/`).
- `tests/test_catalog.py` checks what the catalog loads.

## Boolean versions

`boolean.boolean_version(spec)` is the registered Boolean Definition whose word Definition is `spec`, found among every module
of `verity.ml.circuits.ops.boolean` and of the registry, with each function-valued static bound to its own Boolean
version; it raises `NoBooleanVersion` naming the Definitions it finds none for. The same module measures how far a served
Program is from pure bits.

## Proof blocks

- `word`: `Q_word_v1{X,W,R=no-recompute}`, the cut of each Call into proof blocks whose committed outputs are at most `X`
  bits or one value of at most `W` bits, each gate computed in exactly one proof block. `word.unit_rule` gives a
  Definition's cut, and `word.check_query` checks a served Program's.
- `cross_call`: the same invariant across a Program's Calls, a value computed in two Calls.
- `call_scope`: what the two share (the Definition a spec id names, its parts, `QueryRuleViolation`).
- `encoding`: a specialization's Definitions in the codec's form (`encode_specialization`), a Definition's digest
  (`definition_digest`), and the values of a varying static a Program graph is partitioned at (`word_values`).

## Fast evaluators: `numerics/`

The Definitions are evaluated gate by gate by the reference in `verity.ml.circuits.evaluation`. `numerics/` holds the
evaluators that reproduce them fast, each registered as an evaluation kernel against the Definition id it computes
(`numerics.kernel_registry.register`) and held to the reference by `self_check`; where one cannot be exact on an input it
declines that input, and the reference computes it.

- Row kernels (`rows.py` and the `*_rows.py` modules, registered as kernel `"rows"`): numpy evaluators of whole Program
  rows, one per Definition family, which the sampled replay re-evaluates.
- Twins (`*_twin.py`, `*_twins.py`): vectorized re-implementations of single composites.
- CPU models of the GPU kernels, independent of the Definitions (`cpp/`, loaded through ctypes by `cpu_model.py` and its
  siblings), and the relations built on them (`*_relation.py`), which predict a kernel's committed bytes from its opened
  inputs alone.
- `cuda/`: probe kernels that capture, on a GPU, the approximate instructions and tensor-core tiles the CPU models and
  primitives reproduce (MUFU's `ex2`, `rcp`, `sqrt` and `tanh`, Ampere's `mma.sync` m16n8k16).

## Other modules

- `replay`: replays a recorded flat body over new arguments, the body builder of a derived Program's function. It is
  torch-free.
- `evidence`: the ids that conformance texts, provenance records and refusals cite, read from `evidence.json` and keyed by
  the capture, probe or rule they identify.

## Tests

`tests/` is the package's suite (`uv run tools/verity/check/suites.py verity/ml/definitions`).
