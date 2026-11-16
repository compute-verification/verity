# verity_torch.frontend.rules

A rule is one step of the [translator](../README.md): it takes one node of a model's exported ATen graph and emits one
Call of a registered Definition, or refuses the node with a named reason. A rule decides by the node's structure (its
operands' dtypes and shapes, sharing, mutation), never by the op's name alone, and each application is recorded in
the Program's derivation with the rule's id and version.

`TORCH_RULES` is the list of the rules that target ATen ops only. A serving adapter adds the rules for its own ops and
makes its default rule set, as the vLLM adapter does in
[`verity_vllm.program.frontend.rules`](../../../../vllm/verity_vllm/program/frontend/rules/README.md).

~~~python
from verity_torch.frontend.rules import TORCH_RULES

TORCH_RULES[:3]    # [<rule StaticIndex_v2>, <rule IdentityView_v1>, <rule Reshape_v1>]
~~~

`base`'s docstring says what a rule must do, and each other module's says what it holds. [`DESIGN.md`](DESIGN.md)
says which names other code imports from here.
