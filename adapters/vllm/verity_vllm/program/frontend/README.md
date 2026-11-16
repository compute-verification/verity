# verity_vllm.program.frontend

This is the vLLM half of the translator from PyTorch to circuits. The walk over a model's graph is the PyTorch
adapter's ([`verity_torch.frontend`](../../../../torch/verity_torch/frontend/README.md)); this package adds what vLLM
serving needs on top of it: the rules for vLLM's own ops ([`rules/`](rules/README.md)), the binding of a rule to the
kernels of the served hardware, the target a Program is derived for, and the two served rules applied after the walk,
the seed rule and the length rule. `vllm_derive` is the walk under all of them, and the adapter's public way to derive
under vLLM's rules.

## Try it

With the workspace synced with CPU torch (`uv sync --all-packages --extra torch-cpu`), derive a two-layer serving
loop, built on meta tensors:

~~~python
import torch
from verity_vllm.definitions.registry import b1
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.vllm_derive import derive_report

C = {**b1.B0_CONFIG, "NL": 2, "V": 64, "MAX_POS": 64}
with torch.device("meta"):
    model = ex.GreedyServe(C, 2)
rep = derive_report(model, (torch.empty(3, dtype=torch.int32, device="meta"),), mode="profile", profile="b1-eager",
                    config=C, name="Serve")
rep.family["family"]                                        # 'Serve_v1{C={f7fd46dceb48},LP=3,STEPS=2}'
rep.program.annotations["derive"]["rules_applied"][:3]      # ['ArgmaxTokenSelect_v1', 'Embedding[profile]_v1', 'FusedAddRmsNorm[vllm]_v1']
~~~

`GreedyServe` is an ordinary serving loop: a prompt's prefill, then greedy decode steps. Its derived Program is
recognised as the registered family `Serve_v1`, and the rules applied include vLLM's own, such as the fused add and
RMSNorm. Each module's docstring says what it holds, and [`DESIGN.md`](DESIGN.md) lists the names other packages
import from here.
