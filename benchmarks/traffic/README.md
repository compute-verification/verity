# Honest traffic against the inference-like policy

How much honest inference fits the inference-like policy at block input `k`, and what it costs: `coverage(k)`, `phi` and
`eta(k, c) = c / (k phi)`, under three caching rules. The definitions are note:console/20261024T1810Z-finding-production-bound-red-team
(section 6); the program is note:console/20261024T1900Z-draft-training-vs-inference-program (crux 5).

- **recompute (R1):** every request's whole context is unregistered input; one request per block.
- **registered prefixes (R2):** shared system prompts and tool definitions are registered like weights; one request per block.
- **per-session blocks (R3):** a session's earlier turns are inside its block, so only new user and tool tokens count.

## Files

- `sessions.py`: fetches every source, tokenizes text with DeepSeek-V4-Pro's tokenizer, and writes one units file per source
  (`units.py`: per call its session, context, registered prefix, new tokens and output).
- `hashtrace.py`: R2 and R3 on prefix-hash traces (Mooncake): registered prefixes from forks of the prefix tree, sessions
  from requests that extend an earlier request's blocks.
- `flops.py`: forward FLOPs of a call on a DeepSeek-V4-Pro-like model (its public config) and a 7B dense model.
- `measure.py`: the tables (`tables.md`, `tables.json`).
- `run.sh`: the job, on a CPU pod:

~~~text
research run --on vy-nebius-cpu-5 --pool research --as training --project verity --source . --cwd source \
  --timeout 100m -- bash benchmarks/traffic/run.sh      # ONLY=src,... to run some sources, FRAC= to resize the agent samples
~~~

## Sources

| Source | Class | Rules | What it is |
|---|---|---|---|
| Mooncake conversation, tool and agent, synthetic | chat, agentic, rag | R1-R3 | Kimi production, one hour, 512-token prefix hashes, inputs capped at 128K |
| ServeGen sessions, DeepSeek-R1, language m-large | chat, reasoning, api | R1, R3 / R1 / R1 | Alibaba Bailian production: session counts; per-client length distributions |
| BurstGPT, Azure LLM inference 2023 | chat, api | R1 | Azure OpenAI and Azure service request and response lengths |
| SWE-agent, OpenHands (nebius), tau-bench | agentic | R1-R3 | agent trajectories, tokenized; the system prompt (and OpenHands' tool schemas) as the registered prefix |
| RAGBench | rag | R1, R2 | retrieved documents plus question, tokenized |
