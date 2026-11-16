# The stacked share of live KV

How much of a node's live KV a prover could hold as encoded carriers inside the honest compute budget the audit allows. That's
the stacking sampling design, read in bytes:

    stock(L) = 128 R_cap L / E + R_in L        f = stock(L) / M_dyn

- R_cap is the honest Program's nonlinear units a second: property/v1's budgeted units plus data products (a product of two run-time values: the attention row's Q′K′ᵀ and P V′; a MAC by a weight is free).
- E is the nonlinear units an encode of one 1024-bit word costs.
- L is a carrier's lifetime.
- R_in is the node's run-input bytes a second.
- M_dyn is vLLM's KV pool.

At f ≥ 1 no KV is credited.

The served rate comes from stock FP8 serving (vLLM's FP8 linears, BF16 KV and LM head): the int8 Program's units are counted at
a throughput int8 serving hasn't been measured at.

## Try it

`beta.py` bounds the stacked share by a min cut instead of the sampling design's rate. On a modeled request, on the CPU:

~~~sh
uv run python benchmarks/stacked_share/beta.py --decoder benchmarks/stacked_share/decoder-llama31-8b-int8.json --out /tmp/beta.json
~~~

It prints its progress, then the graph and β at each E and L, and writes the whole bound to `--out`.

| Script | Needs | Writes |
| --- | --- | --- |
| `units.py` | a Build's Program, where `verity_vllm` imports | the Program's units (`--out`) |
| `share.py` | `units.py`'s units and a served trace's `mix.json` | f for each E, L and R_in, as built and under each lever (`--out`) |
| `beta.py` | a Build's Program, or a modeled request (`--decoder`) | the min-cut bound (`--out`) |
| `atoms.py`, then `frontier.py` | where `verity_vllm` imports, without torch | the same sampling design for frontier models no Program has been built for, and its tables as `frontier.md` |
| `job.sh`, `beta_job.sh`, `frontier_job.sh` | one GPU (`job.sh`), a CPU node (the other two) | the scripts above as jobs |

[`DESIGN.md`](DESIGN.md) says what each script computes, and which quantities are measured and which modeled.
