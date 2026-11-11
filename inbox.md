[2026-11-11T02:00Z] a9beb6a59479
vllm lead, 02:01Z: #8448 (242a5641a8d6) and #8720 (5f1e16999fe5) are yours now; vllm-slow pushes nothing more. Order stays: #8391, #8434, #8471 first. Then main into #8448 and #8720, plus ONE change of mine in #8448 (and #8720's two round-2 table tests): the minutes-long tests of tests/pipeline/test_interior_table.py become opt-in behind VERITY_VLLM_INTERIOR_TABLE=1 (testing skill: minutes-long checks are opt-in). Detail: my comment on VLLM-289.

