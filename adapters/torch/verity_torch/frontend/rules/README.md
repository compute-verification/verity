# verity_torch.frontend.rules

The lowering rules of the Build phase that translate plain PyTorch. A rule takes one node of a model's exported ATen
graph and emits one Call of a registered Definition, or refuses the node with a named reason. `TORCH_RULES` is the list
of the rules that target ATen ops only; a serving adapter adds the rules for its own ops and makes its default rule set.
`vocab` holds the kinds a rule may emit under each profile (`PROFILES`, which an adapter fills), `family` matches a
program's layers to its registered families, and `bindings` holds the plain bindings. Each module's docstring says what
it holds.

## What other code imports

A module's `__all__` lists the names it gives to library code outside the package, and across one line inside it: the
reference, pattern and training rules (`reference`, `ref_kinds`, `patterns` and the `training` modules) read the
package's other modules through these lists, and are read through their own. On either side of that line a module
reads its neighbours freely. The package itself gives `TORCH_RULES` and `vocabulary_for`, which load at first use;
`base` gives the types a rule is written in (`RuleSet`, `RuleInfo` and the rule classes). Private names read from
outside the package all the same, and not declared, are read by the vLLM adapter's rules: `family._field_intervals`
and `family._runs`, `padding._PROFILE`, `_bit_value`, `_materialize_bits`, `_scalar_of` and `_static_bit`,
`profile._bias_shape` and `_without_batch_invariance`, `views._RESHAPE`, and `bindings.operands._strides` and
`_untransposed`. Two are read by `patterns`: `views._STATIC_ARITH` and `views._STATIC_MAKERS`.
