[2026-11-12T05:32Z] 449643f8a6d2
QAT (TRN-107): I built (b) and stopped at your second condition. The slow test over OUTPUTS fails for qat for a reason that is not qat's. Nothing in run.py changed.

Where things stand:
- #8910's head is b4d02a5a6f: the merge 4f5d27c65a, then your two approved commits. My VM was recreated between turns, so the held 8849711037 and 7451b335cb were lost; I recreated them as c33333074f and b4d02a5a6f, with the same content and messages.
- (b) is commit 3f217ad834. It is NOT on #8910; it is on the side branch cursor/training-qat-resume-fb49 only, so a VM reset can't lose it again. It holds qat.py's hook (rebinds `start` only when the loaded dict lacks `prefix + "v"`, with one docstring sentence), the identity assertion in test_routes, and the qat row and `bare` in test_resume.
- r20261112-053026-0004, the harness's suite --fresh at b4d02a5a6f, is running.

What passes with (b) [RAN, VM]: the quick test over OUTPUTS, qat row included, passes `assert_continues`. test_resume, test_routes and test_run pass in the quick tier (70 passed).

What fails: the slow test `test_a_stopped_job_of_a_route_with_an_output_continues_and_writes_the_unbroken_jobs_record[qat]`. Every count, checkpoint, bpb and E is the unbroken job's. Only `trace` differs: the continued job's is `{"failed": "TypeError: Muon.__init__() missing 1 required positional argument: 'lr'"}`, while the unbroken job's trace is taken.

The cause, which is not qat's [RAN, probes]:
- torch's `Optimizer.load_state_dict` calls `__setstate__`, which does `self.defaults.setdefault("differentiable", False)` (torch/optim/optimizer.py line 453, torch 2.14).
- The trace's `capture/trace.py::_fake_copy` rebuilds each optimizer as `type(opt)(groups, **opt.defaults)`. `routes/_optimizers.Muon.__init__` takes no `differentiable`, so that raises, and the fallback `type(opt)(groups)` raises "missing lr".
- So after any kept state's load, any optimizer of `_optimizers` (Muon, Lion, Adafactor, Shampoo: none takes `differentiable`) loses the continued job's trace. AdamW takes it, which is why ema and lora pass.
- Shown on a route that is not qat: subset with opt = "muon", stopped and continued, loses its trace the same way, and its unbroken job's is taken.
- With Muon accepting `differentiable` (a scratch edit, reverted), both tests over OUTPUTS pass for all three rows: 6 passed.

Where a fix could go (I built none):
- (i) `_fake_copy` passes only the defaults the class's `__init__` takes. That is the capture's code, the carrier's.
- (ii) the `_optimizers` classes take `differentiable` (and ignore it or refuse True). That is the routes' shared module.
- (iii) not run.py, per your ruling.

Your call: (i) or (ii), and whether it goes in as a fourth commit on #8910. Then I push 3f217ad834 to #8910 and run the four checks at the final head.

