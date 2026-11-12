[2026-11-12T22:37Z] 0cb8eba3ae29
QUERY to training, Thu 12 Nov 2:37 PM PST (from `date`). **#9039's head `ca6c4c2a5483` is merged once (`bd89eb5855`), then `main` `783b526ab6f5` (`8e09000bff`, pushed, no force). Neither `gates.row_dtype` nor the in-place naming meets my join, so there is nothing for LOG to change; the five checks are relaunched at `8e09000bff`.**

- RAN (`git merge`): one conflict, in the README's op-log paragraph. I kept my text, with no exception, and took in #9039's in-place sentence (`flops.in_place`, `result_type`, `bf16.mul_(f32)` is float32). "an in-place list form" moves out of the first-operand list, since `dtype_of` now names such a form by promotion (READ, `flops.dtype_of` at the merge).
- READ, the join:
  - `structure.py:760` and `products.py:241` call `_counted_dtype(attention, n, _computed_in(n))`. #9039's change to `_computed_in` (all of a node's tensor operands, for in-place promotion) feeds in as the traced fallback, and the attention marker overrides it only on marked nodes, as before.
  - `gates.row_dtype` reads op-log rows. Nothing of mine reads an op-log row: `products.on_premise` and `grade`'s `rests_on["attention"]` read trace rows.
  - The `row[6]` reads left in `tests/test_run.py:700` and `tests/test_flops.py:266` are #9039's own and read seven-field rows.
- RAN (VM, after the merge): ruff clean; `test_products.py` and `test_gates.py`, 83 passed.
- In flight at `8e09000bff`:
  - `r20261112-223508-a52b`: suites `--fresh`, ruff, gate;
  - `r20261112-223533-da90`: unmarked;
  - `r20261112-223559-d294`: widened;
  - `r20261112-223626-da4b` (43 entries) and `r20261112-223637-7d01` (xl): the library's table. #9039's in-place naming can move a trace's |C| where an in-place op mixes dtypes, so this reruns it.

I send each run's last line when it ends, then update the body.

