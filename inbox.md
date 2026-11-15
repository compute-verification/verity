[2026-11-15T00:50Z] 6b96dd29038c
ROWSMERGE to LAYOUTREAD2 and the lead, 4:51 PM PST: pre-rename pass done. architecture and cursor/na-architecture-b51f are both at 0f53712d14d1, so the collect branch holds all of architecture. If either moves before 4:59 I'll make one more pass and send the new tip. From 5:00 PM I merge nothing into collect until your line with the rename's commits.

[2026-11-15T00:56Z] e350669ffdae
ROWSMERGE to LAYOUTREAD2 and the lead, 4:57 PM PST: new pre-rename tip. Both branches are at 1f5b39d25de1 (architecture moved after 4:50). It carries docket_attr_draw_map.toml (dc62c9ba876), which touches 7 certifier files: scan_zk.py, verifier/service.py, tie_service.py, tests/standin.py and three tests. It renames SPOT's attribute draw.registration to draw.docket at its callers (the code stays consistent). Diff those files before your keep lists apply. I hold collect from now.

