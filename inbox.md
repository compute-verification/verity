[2026-11-09T23:01Z] f55765ab1c8f
From the vllm coordinator, 23:03Z Mon. Daniel wrote again at 2:49 PM PST, and it changes your task. Read the note lanes/vllm/20261109T2300Z-draft-layout-candidate.md (research notes sync pulls it; the same text is VLLM-265, where every lead answers by 01:30Z). His words, shortened: it is not obvious to him what "format" is; it is not clear that core and protocols should be separate; he is unsure about kernels; research workloads are "using Verity" rather than "part of Verity", so maybe top-level directories beside verity/; adapters should not say their type, and engine, workload and frontend are interfaces defined inside verity. Then: "This should be highly principled!"

What I need from you, in your draft note, synced, by Tue 01:30Z, then end your turn so I can resume you for the full draft (still due Tue 14:00Z):
1. Score the candidate's five rules against the tree on main 7af4d24fca3 by the note's test: how many path rules each decision takes today and under the candidate (imports: 31 allow rows and 36 exceptions today; public or private: 132 rules today; owners; the trusted set). Say how you counted.
2. What does not fit: units or files that no rule places, or that two rules place differently. Name each with its path and line count.
3. Kernels: place the kernels lane's census (note kernels/20261109T2153Z-report-kernel-census, 52 files) both ways, beside each Definition and in one verity/kernels, and count the includes that cross a directory each way. Answer its two questions for the architects (where attention's kernel sits; which files are behind the row that says the integration builds 4.5k lines of CUDA by path, or correct that row).
4. A rival set of principles, if you find one that scores better by the same test. Do not defend the candidate; try to break it.
Reading only, no moves, no PR. Reply with the line: inbox ok 23:03.

