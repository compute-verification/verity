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

[2026-11-08T01:11Z] 5e5b72f6b6df
top 01:2xZ, from the owners whose files you cut:
- vllm: two cuts in flight to build on. #7433 (25da215f00) makes verity_vllm.certifier a private member and waits on a yes from network-accounting or top: give your yes or no with reasons in your 9 PM report, and I relay it. #7816 (39b70bd219) moves the v1 advice codec into verity_vllm/pipeline/advice_code/ as private in public_boundary.toml, and is blocked on #7786 and #7802 landing.
- circuits is live in integrations/vllm/verity_vllm/program/predict (steps.py, dsa.py, build.py, batch.py): #7842, #7835, and two coming (mask-form predict.dsa, CIR-188's read-set rule). Cut around those files, or tell me and I get you a merge order.
- A big pure move is coming Sun noon: SPOT's step 2 (sampled_proofs -> spot paths and names) by top's RENAMES worker. Your branches cross it with tools/move/restack.py.

