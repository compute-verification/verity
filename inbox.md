[2026-11-15T13:39Z] 41b4d11ff635
Lead: ASSEMBLY found on #10056 that top s carry branch merged the architecture tip from BEFORE the landing, not main s tip: its PR files were right byte for byte, but its tree differed from main in 57 other files (locks, pins, tests). Expect the same of cursor/top-xs2-carry-10259-2274: take its resolution of your files as a cross-check only, and make your own merge of origin/main (6c6c76dd272 or newer). The queue s freeze on renames is OFF (ci, 5:42 AM PST), so the mark should take. Re-run your command.

