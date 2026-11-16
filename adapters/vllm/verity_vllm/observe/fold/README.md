# verity_vllm.observe.fold

The fold turns a raw observation log back into a Program. The observer records a run as a long list of the engine's
calls. The fold resolves each call against a pinned profile, which says what the pinned vLLM, torch, Triton and GPU
compute, and folds the calls into a record Program, in the same form as the Program the Build derived. The Match then
compares the two.

It needs no torch. `verity-vllm resolve-log --profile PROFILE LOG` resolves one log against a profile and prints what
each event resolved to; its `--help` gives the options. `fold.py`'s docstring has the fold's contract, and
[`DESIGN.md`](DESIGN.md) lists which names other code imports from each module.
