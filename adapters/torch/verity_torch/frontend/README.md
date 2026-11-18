# verity_torch.frontend

This is the translator from PyTorch to circuits. It takes an ordinary torch model, with no Verity annotations,
exports its ATen graph, and translates every node by exactly one rule into a Call of a registered Definition. The
result is a Program, or a refusal with a named reason, never a partial Program. Nothing reads a tensor's values: the
export runs on shapes and dtypes. It also derives one training step as a Program, and consecutive steps with torch's
own optimizers.

## Try it

With the workspace synced with CPU torch (`uv sync --all-packages --extra torch-cpu`):

~~~python
import torch
from verity_torch.frontend.derive import derive

model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU())
program = derive(model, [torch.zeros(2, 8)], mode="reference")
program.annotations["derive"]["applications"][0]["emitted"]    # "LinearRef_v1{K=8,N=4,DT='f32',BIAS=True} x2"

derive(lambda x: x if x.sum() > 0 else -x, [torch.zeros(2, 8)], mode="reference").reason    # 'dynamic_control_flow'
~~~

The first is the Program's derivation record: which rule translated each node, and the Call it emitted. The second
is refused, because its control flow depends on a value.

The rules on ATen ops are in [`rules/`](rules/README.md), and the vLLM adapter adds the rules for vLLM's own ops
([`verity_vllm.program.frontend`](../../../vllm/verity_vllm/program/frontend/README.md)). `derive`'s docstring says how
a derivation works, and each other module's says what it holds.
