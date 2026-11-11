[2026-11-11T16:45Z] e289fcbe1e63
kernels lead, Wed 16:45Z: proofs said yes to #8953 (HASH-115, 16:43Z): cores stays the script's affinity. One change to cores()'s docstring, in proofs' words, is in the newest section of inbox/KSTEADY.md. After the review's fixes are pushed, merge main into #8953, make that change, run its suites and say the new head. It stays unmarked until #8679 is on main. The tenth check is taken over cores, so a row whose verifiers sit on another NUMA node than its provers fails it.

