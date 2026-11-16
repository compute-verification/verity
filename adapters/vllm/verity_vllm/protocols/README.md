# verity_vllm.protocols

What each protocol needs from a running vLLM engine. The protocols themselves live in `verity`; this package only
attaches them to vLLM. A run turns protocols on as options, off by default, each on its own or several in one run, and
each option is one module written against one interface.

| Option | What it does to the engine |
| --- | --- |
| SPOT | Nothing of its own: the Commit's committer and taps commit the values the model produced. |
| PoUW | Replaces the GEMM at each linear layer with the scheme's honest work. |
| PoUS | Holds each linear layer's weight only encoded, and decodes it at every use. |

PoUB is not an option: serving turns it on by a flag. Its part here attaches the node's registered store to the
engine, with the hooks that keep the store current while vLLM serves.

As a library, `verity_vllm.LLM` takes the options as `protocols=`. The package's docstring says the order the options
install in, their phases and what a run is refused for. [`DESIGN.md`](DESIGN.md) lists which names other code imports
from each module.
