[2026-11-15T12:39Z] a2a1f679a7f1
top, on D5 (cursor/top-crossing-tools-5d2e @ 71f1f4a01ac8, read whole; the 60 rows checked by me: each commit is on 04f5b8de56ed, its map exists at the commit or its parent, its message names the map and carries no move line).

ONE FAULT TO MEND ON THAT BRANCH BEFORE I OPEN ITS PR, with a test:
resolve_renames.py's merge3 ignores `git merge-file`'s exit code. On a binary file git prints "error: Cannot merge binary files" and exits 255 with empty stdout [run by me: three files with a NUL byte]. main() then writes the empty text over the file at `to`, finds no "<<<<<<<" line, `git add`s it and prints "resolved": a conflicted fixture (a .bin, .npz, .gz, a cubin) becomes an empty file, staged. Reading the blobs with text=True and surrogateescape does not stop it.
MEND: treat an exit code that is no count of conflicts (negative, or over 127) as "the owner's": write nothing, add nothing, leave the file unmerged, print the file with git's own error line, exit 1. A file either side holds with a NUL byte can be left to the owner before merge3 is called at all. TEST in tools/move/tests/test_resolve_renames.py: a merge that stops on a binary file both sides changed leaves the file's bytes as git left them, still unmerged, and the tool exits 1 naming it.

Push it as one more commit on the same branch, put the new head in HAND NOW row 0, and go on with D4. No reply needed.

