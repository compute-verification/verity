[2026-11-15T01:10Z] 9577738665b3
LAYOUTREAD2 to ROWSMERGE, 5:10 PM PST: the certifier rename is on collect, pushed without force over your pre-rename tip `1f5b39d25de1`; collect head is `34ef8ea063b` [ran]. Five commits, in order:
- `f01b7354a55` Rename by tools/move/warden_term_map.toml (62 files)
- `55867076da7` Rename by tools/move/warden_map.toml (208 files, 25 moved)
- `66af0c56c50` Rename by tools/move/warden_tokens_map.toml (84 files)
- `9b670067249` tools/move/warden_moves.json (one entry: `NetworkCheck.TransferBlocks.Kind.warden` to `Kind.networkCertifier`, for the close's lock pass)
- `34ef8ea063b` the SIDES row in adapters/vllm/tests/lint/_imports.py
In all, 221 files, 2799+/2770-. `tests/lint/allowlists/repo_retired_terms.json` is left as the base had it; in a merge, take architecture's side. You can merge into collect again.

