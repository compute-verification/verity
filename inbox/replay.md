
## 02:28Z Thu 12 Nov, from the coordinator: Daniel, 6:22 PM PST: no restriction on context length

- **His words:** "I think it would be pretty reasonable to shoot for saying that we have no restrictions on context length, max context length is fine. Does that seem within reach or no?" I answered: yes as a property of the design, not as a result shown this week; the code holds 131,072 in four places, and one cost grows with context.
- **What changes for you:** the context column of your table is the model's own limit, 1,048,576 positions (GLM-5.3's config of record, `max_position_embeddings`, plain RoPE with no scaling; circuits on CIR-186), beside 131,072. For each route say what grows with context: at serving (bytes committed and time a token), in the replay of one sampled span (tokens recomputed), in advice, and in what the guarantee charges.
- **The point I gave Daniel. Confirm or correct it with file and line:** a replay that recomputes everything before the span it replays grows with context, as it does with reuse. One that reads the earlier attention values from the request's table committed while serving does not, and that is what a 128k request's replay blocks already do (`proofs/spot-128k-attention-proved`: "A replay block is a span of positions inside the request, and it reads the earlier attention values from that table"). So a long context and a cache hit are one question: whether a replay block may read values another block computed, and what the guarantee then charges it (the fan-out F, GLOSSARY.md:362-364).
- **What I read tonight. Check each before you use it:**
  - `CLASS_G_LIMITS` and `CLASS_G_SERVED` hold `max_model_len` 131,072 (`adapters/vllm/verity_vllm/engine/profiles/engine_classes.py:33-36`).
  - The long row's mask Program spans 63 classes, 4,096 to 131,072 keys in steps of 2,048; its radix select runs over 32 + 17 bits; `RowsPad`'s pinned ranges end at 131,072 (CIR-186's comments).
  - The index logits are 86,944 + 837,038·N ANDs a row, in 18 of 78 layers (57 share an index); the select is about 193 ANDs a key (CIR-186, circuits' counts).
  - The unit that merges the selected positions exactly fills the largest block of 2^27 bits at 131,072 keys (vLLM's goal `vllm/served-128k-window-committed`; relayed).
  - One rank's compile took 28,360 s and 788 GiB at 131,072 tokens, and 110 to 150 min and 253 GiB at 40,000 (circuits' goals).
- circuits is asked the circuit's side on CIR-244 (Thu 2 PM PST); vLLM the serving side on VLLM-377 (Thu 10 AM PST).
