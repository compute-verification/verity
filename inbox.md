[2026-11-13T06:57Z] ece8aaf44ae2
THREADS, Thu 12 Nov, 10:57 PM PST — TRN-153, #9608 merged with main

**#9608 is merged with `main` (`99cc585fa8f2`) at head `616b3c319e13e7df7c618ac24a4e80da5044959f`, VERSION 4.25.0, and the four suites it touches pass on a node. #9039 made three of its own grade tests fail under the pin, and I fixed them in one commit.**

- **Head.** `616b3c319e13e7df7c618ac24a4e80da5044959f`. Two commits on top of `14b6b448755a`:
  - the merge, `329663414242`;
  - the test fix, `616b3c319e13`.
- **The merge (RAN).** `__init__.py` takes VERSION 4.25.0. `test_run.py` had two conflicts, both in docstrings; both sides are kept, with the fused kernel named a job's on a GPU. I also rewrote #9039's README sentence "a job runs them under a fused kernel" to say a CPU job runs the math kernel, whose attention rows name float32, as the trace's do.
- **Run.** RAN `r20261113-061612-3055`, `suites.py --changed 99cc585fa8f2`, at `616b3c319e13`. Its last line: "4 suites: 4 passed (0 cached), 0 failed; 26.6 min of test time". By suite:
  - `benchmarks/training_slowdown`: 647 passed;
  - `benchmarks/training_slowdown/grader`: 28 passed, 3 skipped;
  - `repository`: 392 passed;
  - `verity-graph`: 246 passed.
- **Gate.** RAN `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD`: "ratchet gate 5d132e70ed76..616b3c319e13: 1 change(s), 0 failed". `main` has moved to `5d132e70ed`; the run is against the commit I merged, as you said.
- **Why the second commit.** The first run, at the merge commit (RAN `r20261113-053823-e353`), failed 5 tests in `benchmarks/training_slowdown`, all in `test_grade.py`. #9039's TF32 and one-price tests assumed a CPU job's attention `bmm` rows name bf16. Under the pin they name float32.
  - The TF32 refusal of a Shampoo or LoRA CPU job now names `(bwd bmm, fwd bmm, other mm)`, not `(other mm)` alone.
  - The one-price test now finds attention float32 in both the record and the trace.
  - M holds no attention product, so the ρ those tests grade is unchanged. RAN `pytest tests/test_grade.py` on the VM: 39 passed.
- **On the VM, by name (RAN).**
  - `pytest tests/test_run.py tests/test_resume.py`, which includes the two new tests: 74 passed, at `329663414242`. `616b3c319e13` changes only `test_grade.py`.
  - `ruff check` on the changed files: all checks passed.
- **Step 3 (READ #9039's diff of `run.py`, `flops.py`, `README.md` and `tests/synthetic.py`, and its `test_run.py` docstrings):**
  - **The pin** still holds a CPU job's attention at the math kernel in both processes. With dtypes named, it now does more: the math kernel casts attention's bf16 operands to float32, so a CPU job's attention rows name float32 and are priced at float32's gates. Its count is the same in MACs, higher in gates than fused bf16 rows, and equal to its trace's circuit. #9608's body said "the pin changes no count"; it now says "in MACs" and states the gates.
  - **The guard** means what the body says. It keys on the op `_scaled_dot_product_flash_attention_for_cpu` (forward or backward) and the thread count held, never on a dtype.
  - **`cpu_attention`** means what the body says: the kernel the route's process held (`"math"` on a CPU, null elsewhere). In a record that names dtypes it also explains why that record's attention rows read float32.
- **Body.** Updated with `research queue edit 9608`: `Goal training/stopped-job-resumes` under `Refs TRN-153`, VERSION 4.25.0, the merge, the step-3 lines, and the new head's runs, with what I ran apart from what I read.
- #9608 is still a draft and not marked.

