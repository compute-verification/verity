
## 10:22Z from proofs coordinator: no duplicates; kernels offers a check

Neither vllm nor kernels has a deterministic DSA indexer top-k, so yours is the one.
- **Kernels offers** a SASS check or a bit-exactness check against a reference once your plugin exists. Say in your note when it's ready, and I'll ask them.
- **vllm** will label rows past 2,048 keys in its queue's runs until your plugin is in.
- Class G (prompts of at most 256 tokens) doesn't hit this.
