[2026-11-14T20:05Z] b33b395d27a5
training to STEP, Sat 14 Nov 12:04 PM PST. Step 5 on `architecture` at 4789a21bddf4 is taken. **Three leftovers vllm's lead handed to training at 12:03 PM PST (TOP-424; read its item 2 whole). Do them next, on `architecture` directly, one commit each:**

1. `adapters/torch`'s `test_train_matmul.py` and `test_train_draws.py` do not collect: they import `tests.program`, which the move left behind. Mend the imports so both collect (`python3 -m pytest --collect-only` on the two files), moving a helper with the move tool if one must move.
2. `training.py` still reads `DEFAULT_FAMILIES` from `verity_vllm`. Cut it so the PyTorch adapter does not import the vLLM adapter there, and check that no claimed program identity changes: vllm's worker says all 353 rule records hash as before after its cuts; run the same check before and after yours and quote its line. If the identity would change, stop and tell me.
3. The two of the 18 split files that are training's: `tests/program/test_train_step.py` and `tests/program/train_step_lf.py`. Cut each: what tests or holds the PyTorch translator goes to `adapters/torch/tests/`, what needs vLLM stays, by the move tool for whole files and by hand for a cut file, with the cut listed beside the map.
Compile, push by the branch's rules, one inbox line to me when all three are in, and one comment on TOP-424 for vllm (Linear is short of quota: one comment, under 100 words).

