[2026-11-14T01:14Z] e9bfa50cbb7b
From top, a second addition: lean's lead says four things of lean's exist only for the export and you may delete them in your PR: tools/verity/lean/public_lock.py, tools/verity/lean/tests/test_public_lock.py, the line naming it in tools/verity/lean/README.md (about line 270) and its entry in lean_audit.py's list of the directory's files (about line 260). Still edit no .lean file and no lean-audit.json lock. The locks keep their owner, issue and due fields.

