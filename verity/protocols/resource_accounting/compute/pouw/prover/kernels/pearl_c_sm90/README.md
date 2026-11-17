# Pearl-C on the H100 and H200

These are Pearl-C's device kernels for sm_90a (`pearl-c-h100-v1-h3`, G = 4, hashing format -h3), and what they cost
at the linears of GLM-5.3 as one tensor-parallel rank serves them. `run.py` checks every buffer the kernels write
against the CPU reference, byte for byte, before anything is timed. The vLLM adapter loads `run.py` and `grouped.py` by
path as the served kernel on these cards and records both files' sha256 in every pass, so a change to either is a new
variant, which serves only once it has passed its own gate. Nothing here is a verifier: a replay's ACCEPT is a
development check, never the protocol's verdict.

## Try it

Without a GPU, see which of GLM-5.3's linears the kernels take:

~~~sh
uv run python benchmarks/pouw/pearl_c_sm90/glm53.py
~~~

It prints one line for each linear one rank serves: its shape, the kernel vLLM serves it with, and either the
advantage ε at the H100 record, at both readings of the FP32 add's price, or why the kernels refuse it. The script's
docstring says how each is computed.

On the card, the kernels are built off it and checked and timed on one H100 or H200:

~~~text
bash benchmarks/pouw/pearl_c_sm90/ship.sh [OUT]    # off the card, nvcc 12.8 or later; default /tmp/pearl-c-sm90-ship
bash benchmarks/pouw/pearl_c_sm90/h200.sh          # under research run, with OUT sent as ship.tar
python3 run.py [check.json]                        # in the built tree, on the card
~~~
