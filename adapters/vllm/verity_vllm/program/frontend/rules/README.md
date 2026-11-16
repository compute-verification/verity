# verity_vllm.program.frontend.rules

These are the translator's rules for vLLM's own ops. Like the PyTorch adapter's
([`verity_torch.frontend.rules`](../../../../../torch/verity_torch/frontend/rules/README.md)), a rule takes one node
of a model's exported ATen graph and emits one Call of a registered Definition, or refuses the node with a named
reason; these are the rules for the nodes plain PyTorch doesn't have, such as vLLM's RMSNorm, rotary embedding and
attention ops.

`DEFAULT_RULES` is the versioned default rule set: the members of the PyTorch adapter's `TORCH_RULES`, whose rules
target ATen ops only, and of `VLLM_RULES`, whose rules each target an op of vLLM's. `vllm_bindings.vllm_ruleset()` is
the default with the bindings of vLLM's custom ops, and `vocab` fills the PyTorch adapter's table of the kinds a rule
may emit with vLLM's profiles.

~~~python
from verity_vllm.program.frontend.rules import VLLM_RULES

VLLM_RULES[:2]    # [<rule RmsNorm[vllm]_v1>, <rule FusedAddRmsNorm[vllm]_v1>]
~~~

Each module's docstring says what it holds.
