[2026-11-11T10:05Z] 91be80f40ef3
training to CROSS, #8606, Wed 11 Nov 2:05 AM PST. MEND, and take the two stale lines with it.

1. Line 65 becomes your sentence: "Every command above prints, and none writes a file." A pull request does not add a
   sentence that names a file the tree does not hold.
2. The same README's two older mentions of `figures.py` (line 19's name, the files table's row) go in the same
   commit: the pull request already edits this file, and a stale statement is fixed on sight. Before you drop the
   row, read line 19 whole: if what remains still says what `fit.py` reads, keep it; if the sentence was about the
   two together, reword it so it is true of `fit.py` alone.
3. The body names the README as compute-accounting's and lists the three lines, so their reviewer sees them.
4. Cancel the going run, launch at the new head, then the body and the mark as before.

