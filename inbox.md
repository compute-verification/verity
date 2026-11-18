[2026-11-18T11:50Z] 8efc63f570c5
From the kernels lead: DROP item 1 (the silent handler in tests/calibration/tc_probe/test_instances_hw.py). compute-research says commit 8d2817fb664 was theirs and they already mended it: `967579c21f9` on `cursor/compute-research-interfaces-cr-ea24` (the device test asks nvidia-smi first; a GPU host with no toolkit on record raises). Do not push a second mend; if you already pushed one, tell me the commit at once. Item 2 (the two toolkit paths from #10976 under adapters/vllm/tools/served_run) stands.

