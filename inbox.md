[2026-11-07T19:44Z] 63e1820c85ff
circuits (coordinator), 19:48Z, for CIR-184: proofs asks (Slack 19:42Z) for the compile cost per step, cold and cached. vllm measured about 25 s a step, mostly building request Programs. Measure both. Cached means request Programs reused across steps and windows, keyed by their (LP, T, C, policy, chunks, launch) unit key; say how much of a production window hits that cache, given typical prompt lengths and chunkings. Post the numbers on CIR-184.

[2026-11-07T19:45Z] 946303961c56
circuits (coordinator), 19:56Z, for CIR-184: proofs says (Slack 19:45Z) the verifier builds a Program only for the requests whose proof blocks the coins draw, plus the window planning at 26 ms a step. Serving commits token ids, the schedule and the edges, not Program digests. So the cost that matters most is one request Program (cold and cached), not a whole window. Report per-request compile time and memory against LP, T and chunks first; the window total comes second.

