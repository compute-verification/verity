[2026-11-08T00:57Z] 3d2e6100d69a
top (Sat 4:58 PM PST): Daniel's idea: make verity_vllm generic over all protocols and expose hooks, with each protocol's adapter registering from its own side, so the public integration imports no protocol. Keep cutting tests and scripts as briefed. Before changing any library code under verity_vllm/{protocols,pipeline,commit,program}, write me a pros/cons assessment of that hooks design, in your final message or a notes report by 9 PM PST. Cover:
- the hook interface it would need (the events and data: window open/close, commit, advice, memory map, verdicts);
- each current call site into the certifier, remote_bandwidth, PoUS and tools/research, and which of them the served path of record (the remote demo's SPOT + Pearl+ run) executes;
- the size and risk of the change, and how it fits AGENTS.md "Integrations are frontends".
Then give a cheaper option that works for Sunday: move the adapters private where the served path doesn't need them. Recommend one option.

[2026-11-08T00:59Z] 5c17b1c5e382
top (Sat 5:00 PM PST): Daniel moved the public split: the public tree must be ready to publish by Tue 10 Nov 6 PM PST. He creates the repository and handles outreach. New dates, replacing those in your brief:
- Sat 9 PM PST: the dry-run export report (SPLITX) and the hooks assessment (SPLITV) to top, unchanged.
- Mon 10 AM PST: all cut PRs marked.
- Mon 6 PM PST: a rehearsal export with --build at next's tip builds clean.
- Tue 2 PM PST: a final export at main's tip is clean: no allowed edge left except the ones Daniel keeps, no lane text or art: ids on the public side, a README at the public root, and LICENSE files (Daniel to pick; top recommends MIT OR Apache-2.0).
Push hard regardless: work in small PRs so each lands early.

