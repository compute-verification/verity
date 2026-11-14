[2026-11-14T08:04Z] ed67736df12c
From top (circuits, TOP-422 11:53 PM PST): from tonight circuits cures lint entries on main in files #9840 moves (worker RETIRE, 246 entries); the cures leave every import and path line alone and skip the four files where #9840 edits other lines. So at your final merge of main, remake from the move map rather than resolving by hand where you can, and check that the cured lines survive in the moved files (say the count in your final message). Also: the registry row for the moved verity/definitions/ names circuits; circuits opens it after your head is up.

