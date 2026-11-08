[2026-11-08T00:57Z] 3d2e6100d69a
top (Sat 4:58 PM PST): Daniel's idea: make verity_vllm generic over all protocols and expose hooks, with each protocol's adapter registering from its own side, so the public integration imports no protocol. Keep cutting tests and scripts as briefed. Before changing any library code under verity_vllm/{protocols,pipeline,commit,program}, write me a pros/cons assessment of that hooks design, in your final message or a notes report by 9 PM PST. Cover:
- the hook interface it would need (the events and data: window open/close, commit, advice, memory map, verdicts);
- each current call site into the certifier, remote_bandwidth, PoUS and tools/research, and which of them the served path of record (the remote demo's SPOT + Pearl+ run) executes;
- the size and risk of the change, and how it fits AGENTS.md "Integrations are frontends".
Then give a cheaper option that works for Sunday: move the adapters private where the served path doesn't need them. Recommend one option.

