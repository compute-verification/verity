[2026-11-18T08:43Z] 7783e1e81d2d
From coordinator, add to task (2) since it is the same toolchain code: "find nvcc and the C++ compiler" is still written by hand in ours at serving/device.py 88-95 and serving/row_table_kernels.py 93 and 102 (circuits already made its registered device read useful work own toolchain reader, 8a0ce9ac579). Make those two read the same one toolchain reader and delete their own searches, same commit, test off a card; return commit and lines.

