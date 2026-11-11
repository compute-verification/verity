[2026-11-11T05:22Z] 41462ab274f8
From TRN-97's worker (bc-831fdbe0), Tue 10 Nov 9:22 PM PST. STEP 3 IS OPEN, NOT MARKED: #8816, branch cursor/rho-injected-held-to-core-e2e5, head 62b53e40ad24 (two commits on main 5c095bb4ab70). https://github.com/compute-verification/verity/pull/8816

Your 9:11 PM message is done: the test is in a file of its own, verity/protocols/inference_only/rho/tests/test_held_to_core.py (test_the_injected_bits_are_held_to_cores_bound_for_one_bernoulli_level); tests/test_envelopes.py is main's, untouched. My two docstrings (Detection.injected in rho/envelopes.py, Incorrect.capacity in rho/operating.py) sit in no hunk of #8813: its diff against main changes neither module (read).

What holds, each run by me at 62b53e40ad24 on my VM: the calculator's suite 603 passed, none failed (602 at main); the test takes 0.01 s; sampled / core from 0.999999998137 to 0.999999998450 over 27 corners x 6 rows, the same as at the old main; default periods.py output byte for byte the same as main's (sampled remote reference: one datacenter 1, 1, 1; the world 1, 1.47506793508, 2.79716747033, a week, a month, a year). Boundaries (read at 5c095bb4ab70, and tests/test_boundaries.py passes, run): the calculator is a SEPARATE tree, no list restricts its import of the profile, rho/params.py already has the edge and the suite's inputs already hold profile.py. No new edge, no new input.

One thing beyond your brief, for you to keep or strike: a second, sharper lower assertion. core / (1 + 3 * MARGIN) is one margin (9.3e-10) below core's exact optimum, because core rounds up by two margins. So the specified interval alone shows "at most one margin below core's exact bound". I added: sampled >= core's optimum before its rounding (step_bits ln(1/delta) / rho(1)), to 1e-12. A count 5e-10 smaller passes the specified interval at all 27 corners and fails the added line at all 27 (run). The body says this under "What the test asserts", item 2.

Deltas: pinned as they are today (p_evade = WINDOW_DELTA = 2^-40 for injected; incorrect.delta = 1e-6 for operating.py's row, which is held to core at that delta too). None changed.

Node run at this head: r20261111-051647-2a5c on vy-nebius-cpu-9 (suites.py --changed 5c095bb4ab70 --quick; 49 suites touched, the root suite among them), waiting for a slot since 9:16 PM PST. I mark #8816 when it passes with none failed and write again. An earlier run, r20261111-050940-8588 at the first commit b9f7c0e18887, never took a slot; I cancelled it when the head changed.

