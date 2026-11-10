[2026-11-10T21:30Z] 06e8c5ff22a0
vllm lead 21:30Z: the gate PASSED at bc024c2abc51 (memory-accounting, MEM-223 21:26Z): r20261110-194647-dad8, SERVE=tiny, accepted. Put that run and its command lines in #8696's body as the GPU check, say on VLLM-290 that the body is ready, and I mark at once (I told memory-accounting: marked by 23:00Z). SERVE=row is red on main too (KeyError 'layout', commit.py:1547): a fact for the body; its fix is VLLM-294, not yours.

