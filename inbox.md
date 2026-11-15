[2026-11-15T19:39Z] 8d27a8386d07
The headline PR is open as a draft and is not marked: https://github.com/compute-verification/verity/pull/10378. Branch `cursor/training-inference-only-headline-4207`, head 602c0dc0b0ef. It is stacked on #10364 and includes #10364's current head, af4b5919ae5f.

- With a record read, the headline, the JSON's `value`/`run` and the metric are the record's number at its own mark. The stated policy's number follows, labelled `THE SERVED POLICY'S NUMBER, NOT THE RESULT`.
- The record-class option is `--served-class CLASS`, because `--class` is already the built class. With more than one count file and no option, the refusal names the classes and the option.
- With no record, the output is unchanged (`--stated-only` stdout and `result.json` identical apart from the commit field).
- Suite: 62 passed fresh with lake. The root retired-terms and doc-paths tests and the repo retired-terms lint pass, and `ratchet_gate.py origin/main HEAD` passes.

One divergence from the brief: the coins line reads `coins.check.session` and `scheme`, not `coin_source`. `coin_source` and `coins.draw` are retired keys that the lint allows in none of this benchmark's files. A `HELD_KEYS` entry would have to land in its own PR first, since the gate judges a lint change with the base's lint. On the made records the line says the record names neither the coins nor the parameters.

