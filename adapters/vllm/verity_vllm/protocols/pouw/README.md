# verity_vllm.protocols.pouw

PoUW on vLLM. Under PoUW, every linear layer's matrix multiply is computed as a PoUW scheme's honest work, so that a
verifier can check it. This package is the option that makes vLLM do that: at each linear layer it hands the call to
PoUW's serving package in `verity`, which runs the multiply, commits it and manages its execution phase.

It holds no PoUW logic and no device code of its own. The schemes are PoUW's, and the GPU side is the PyTorch
adapter's. The package's docstring states the option, and [`DESIGN.md`](DESIGN.md) says which schemes it hands on,
what each module holds, and what it loads and is loaded by.
