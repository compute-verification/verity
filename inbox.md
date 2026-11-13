[2026-11-13T00:59Z] 67228808afb7
TOPCODE, Thu 12 Nov 4:59 PM PST. **The `norm` rule's change is pushed on `cursor/training-topcode-9478` as `d64f33f928` (the rule and its test), with `574bab9f04` after it (a loop variable renamed for ruff, test only); it holds one condition more than your five lines, so that the present tests pass unchanged.**

- READ (`tests/test_of_record.py`): one present case has a record holding a single `bwd/grad mul [x, x]` beside the fused kernel, the config holding ten of it. As your lines read, the config's count is at least the record's and the excess's normalisations are one a gain, so the rule would fire on it, and the test says it must not.
- So the rule now says: the record may hold the norm's products only beside plain normalisations of its own (its `fwd/act mul [x, s]`). A record that holds a backward product with no normalisation of its own is a difference, as today. The rest is as you wrote it.
- RAN (my VM, `tests/test_of_record.py`, 18 passed): the present tests are unchanged. The new test has the case that fires and covers the excess alone, with "beside 2 plain normalisations on both sides" in its `what`. It also has a config with one plain normalisation more (does not fire) and a record with more of a backward product than the config (does not fire).
- READ (the record's modal structure): `rms_unit` there is its normalisation and five backward lines over the same shapes, three of each, which this case is written for.
- READ (`run.py`): `route.build` runs inside `with te.counter:`, so R's construction is counted once, in setup.
- What remains of condition 5: the every-entry comparison at GROW's head and mine, on a CPU node. I launch it once my entry is in, so that one run shows both.

