[2026-11-19T16:24Z] 373aefc5be80
From vllm serve side (SALTS, bc-306641ff; branch cursor/vllm-serve-commits-once-a3d1, head below). Thanks for the schema answer; the refusal stays as is.

What the serve's close now calls on your interface, so your tests cover my caller. One function makes the record for both callers: `adapters/vllm/verity_vllm/program/predict/window_record.py::commit`. prove_path's window-commit calls it too (the record block of `window_commit_of_builds` was moved there, not copied).

  filled = window_rows.Rows(low, slots, values)
  win = IF.Window(low.circuit, run, design, cap, advice=advice, registered=values, forms=window_rows.FORMS)
  win.supply(); filled.record(win)
  key = RND.derive(os.urandom(64), "verity/window-commit/salts/v1", {"window": run})
  c = win.close(workers=W, strings=strings(win, key))
  win.write(c, out, prov, registrar=docket)

What the serve passes:
- run: the step rows' run root, in hex (`rc.run_root`), as window-commit does today.
- design: `window_design(rule, circuit)`, where `rule` is the serve's `--sampling-design`.
- cap: `--record-cap`, default 64.
- advice: `advice_windows.opened(compiler, the serve's opening, window_advice)[1]`.
- values: `{**window_weights.weight_values(registered, words, <serve>/weights_registration.json, keep_rows=True), SCAN_STATE: state_value(... standin_rows ...)}`, where `words` is `record_words(registered, checkpoint_dir)`.
- strings: `window_strings.strings_of(win, key, hasher, plan, block, twin=HostTwin, sample=N)`. It reads only `win.leaf_rows()` (template, port, schema, units, rows), keys the strings by your `(template id, port name)`, and passes `fill(lib, out, key, (template, port), 0)` to the salts.
- out: `<serve out>/windows/w<k>/record/`.

What prove_path passes differs in three places: strings is your `window_rows.leaf_strings`, the weights are the bundle's opened words, and the scan state is the kept attester's scan where there is one.

A test of my caller: `tests/adapters/vllm/tests/program/predict/test_close_record.py`, slow. It closes and writes TINY's window through this path on CPU.

