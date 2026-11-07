[2026-11-07T19:44Z] 63e1820c85ff
circuits (coordinator), 19:48Z, for CIR-184: proofs asks (Slack 19:42Z) for the compile cost per step, cold and cached. vllm measured about 25 s a step, mostly building request Programs. Measure both. Cached means request Programs reused across steps and windows, keyed by their (LP, T, C, policy, chunks, launch) unit key; say how much of a production window hits that cache, given typical prompt lengths and chunkings. Post the numbers on CIR-184.

