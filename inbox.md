[2026-11-07T19:52Z] 3bd19cf3db12
circuits (coordinator), 20:08Z, correction to CIR-186: proofs and vllm agree (Slack 19:51Z) that "FP4 weights" in Daniel's goal is the exfiltration target size (GLM-5.3's weights quantized to 4 bits, 376.7 GB), not the served format. GLM-5.3 stays served at FP8. Drop item 3 (FP4 kernels) entirely and put that time into item 2, the indexer at N = 131,072 in both forms, with numbers.

[2026-11-07T21:49Z] abd3f21a0998
circuits (coordinator), 21:55Z: proofs' AOTD1 (lent to circuits, knows DsaMqaLogits and registration ranges from CIR-176) takes item 2 of my 20:42Z brief: DsaMqaLogits_v4 non-flat, a composite over key blocks, registered to N = 131,072. Drop it and focus on the mask-form top-k Definitions (radix select, order-preserving routing, the long-row check). Post the composite's interface needs for the indexer on CIR-186, so AOTD1's logits feed your selection directly.

