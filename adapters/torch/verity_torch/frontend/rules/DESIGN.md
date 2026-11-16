# verity_torch.frontend.rules: design

What someone working on the rules needs beyond the [README](README.md) and each module's docstring.

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
