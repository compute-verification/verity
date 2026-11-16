# verity_vllm.correspondence

A derived Program says what is computed. The correspondence says where each of its Calls happens when vLLM runs the
model: the module that owns it, its graph node, and its row. The query reads it to decide which values a Commit must
commit, and the acquisition plan reads it too.

The record itself, `runtime-correspondence/v1`, is the PyTorch adapter's (`verity_torch.correspondence.runtime`),
which the translator writes into the Program it derives. This package holds its two readers, `reader_for_query` and
`reader_for_acquire`, and each one's docstring says what it reads. [`DESIGN.md`](DESIGN.md) lists the names other
packages import from here.
