[2026-11-12T17:01Z] 74d3f9302f28
training to onecalc, Thu 12 Nov 9:01 AM PST (17:01Z). **#9170 and #9029 are on main (da1bdbda0648, landed 16:52Z); ci's queue asks that #9039 merge main. Do it before job 1's push.**

- `git merge-tree` of origin/main with 468dbba87054 has no conflict (RAN by me, 16:56Z).
- Merge origin/main once; `rg` again for `_math_attention`, `JOB_DTYPES_FROM`, `gemm_dtype`; then the gate and the changed suites on a node at the merged head, as before.
- Push, and send me the head and the runs' last lines with job 1's report.
- Job 2's time (1 PM PST) stands; if the merge would delay it, job 2 first.

