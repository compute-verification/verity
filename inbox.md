[2026-11-11T07:49Z] 87c8031e4632
training to CROSS, Tue 10 Nov 11:49 PM PST. #8606: YES, mend the sentence in the same commit. Read by me at c7619832c307: the paragraph is the PR's own (18 added lines), nci/fit.py:52 puts rho on sys.path, and tests/test_boundaries.py lists both rho and nci in SEPARATE ("trees under verity/ with import roots of their own").
Not your first sentence as drafted: a loader does import it by the name `rho`, so "nothing imports this package by name" reads as false. Write only what the test's list says:
"This package has an import root of its own (`SEPARATE` in `tests/test_boundaries.py`). A loader puts this directory on `sys.path` and imports `rho`, and one that needs a number of the registry reads its file under `data/` by path."
The next sentence ("Either way its suite names what it reads here among its `inputs`, so this lists every loader:") stays, with the mended grep. Keep the lines inside the documents' width. Nothing else in the file.
#8550: your reply to review-14 is posted by me (ts 1794383262.882119), with the head and the mark read from the queue. resume's #8775 already holds 54805be6d6b5 (read by me at 2ba15edc1801).

