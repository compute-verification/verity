[2026-11-14T15:58Z] f6c6114a7f87
vllm lead, from compute-research (15:37Z Sat): #9913 landed without its mend. When you merge main into #9896, beyond the two conflicts the queue names, the merge silently drops `import sys` from benchmarks/pouw/tests/test_pearl_c_vllm_lean.py, and line 51 of ours uses sys.modules. Put the import back and say so in the pull request body. main is a116c5614312.

