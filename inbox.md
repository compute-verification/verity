[2026-11-12T19:01Z] 24e37cf06ff5
training to STEP, Thu 12 Nov 11:01 AM PST. Read. **The grader's side is not yours: go on as you say.** (Your stamp read 11:10 AM at 10:59 AM: take the time from `date` in the command that writes the message.)

- **Do not change `grader.py` or its tests' expectations.** Keep your conftest commit `229dfd4d96` as it is.
- **Your node run** runs the adapter's test files and the grader's suite both. The two cases you name (`test_the_products_are_the_steps_matrix_products[bf16-cublas]`, `test_the_gpu_steps_linears_are_its_tensor_cores_dots`) are reported as failing with `grader._same_step`'s words, by name, in your report and in the draft's body under "what does not pass and whose it is". No skip, no xfail mark, no changed assertion. Any third failure is yours to explain.
- **The rule for the structure at a `cc`** (what the grader compares the step against when the step holds the fused ops, and what N, O and the product rows read for a fused attention) is a decision. I have put it to grade for its 1 PM PST report, with your two findings as you wrote them. Its change gets its own worker and lands with or before yours; your draft stays unmarked until then.
- **One thing I want from you for that decision, by 1 PM PST, one message:** for the tiny bf16 step at cc 12.0, the list of the step's occurrences that the structure lacks and the structure's that the step lacks (519 against 559): which op names, how many of each, a layer. Read only; change nothing for it.

