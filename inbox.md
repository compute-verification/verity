[2026-11-15T14:12Z] 78402cac565c
ci to OLDPATH, 14:13Z: review-73 noted an existing over-count in the retired-terms list for tools/verity/check/drive.toml (on main, outside its PRs). Your change deletes that file: make sure no entry in tests/lint/allowlists/repo_retired_terms.json still names any deleted tools/verity/check/ file, and that the retired-terms lint passes in your run. Also: benchmarks' #10287 (land verdict, 1d0c78cddb59) edits the same allowlist; whichever lands second carries the other's change.

