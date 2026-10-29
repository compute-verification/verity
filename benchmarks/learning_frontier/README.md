# learning_frontier: loss against cost under restrictions, at small scale

Small decoder-only language models trained on one fixed corpus, to measure what a restriction on training costs at a fixed
loss: the compute penalty of a restricted family against dense AdamW, and how much of the restricted quantity (novel state
per window, bits communicated) it saves.

The corpus is `prep.py`'s: fixed shards of `HuggingFaceFW/fineweb-edu` (`sample-10BT`) at a pinned revision, GPT-2
tokens (tiktoken `gpt2`), and a fixed held-out split from a shard no training run reads.

~~~text
python benchmarks/learning_frontier/prep.py --dest DATA
~~~
