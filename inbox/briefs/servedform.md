# SERVEDFORM: which circuit a served statement proves for the 45 kinds kernels timed on the census's own Definition

**Why.** Kernels timed Flock at steady state on 64 kinds of GLM-5.3 blocks (HASH-98). Its finding of Thu 12 Nov 07:18Z, read from `boolean_version` and with no circuit built: 15 kinds were timed on their Boolean version (rows 1 and 2 of its table among them); 4 have none registered (rows 3, 5, 13); **45 were timed on the census's own Definition where the tree registers another Definition as its Boolean version** (rows 4, 6 to 12, 14, 15, 16). Those 45 are 30.9% of the cost of proving every block, and at most 8.5% of SPOT's figure. One example it gives: row 15's Boolean version on `main` is `SparseMlaCombine_v2{…}`, 1,404,134 ANDs, where the word form it timed has 1,676,431.

Benchmarks' rule is: time the form a served statement proves. Which form that is, is ours to say, **by Thu 12 Nov 8 AM PST (16:00Z)**. Kernels holds two cards of node A for Thursday 9:30 AM to 1 PM PST on the answer, and Friday's figure reads the rows.

**The question.** For each of those 45 kinds: when vLLM serves GLM-5.3 and a unit of that kind is drawn, which Definition's circuit does the statement that Flock proves and the Lean verifier checks name: the census's Definition as kernels timed it, or the registered Boolean version (`verity.definitions.boolean.boolean_version`)? If the answer differs by kind, give it by row.

**What I have read, as a start and not as an answer** [read by me on `origin/main` `d6075b75d289`]:
- `adapters/vllm/verity_vllm/pipeline/chunk_rows.py:118-136`, `zk_refusal`: `verify --zk` refuses a chunked template's statement when it reads a table in lookup slots and returns its outputs from a tail stage ("Only a word template's reads are in slots"), and the reason names the Boolean version "to name in its place". `zk_check` (below it) refuses a `zk: true` partition's window at the first such template.
- The design rule (AGENTS.md, "Everything zero-knowledge"): every protocol is zero-knowledge, with no clear mode. So a form `--zk` refuses is not a served statement of record.
- `verity/definitions/boolean.py:1-30` and `:136`: what `boolean_version` is, the `opt_in` exception, the made-on-demand primitives.
- `adapters/vllm/verity_vllm/pipeline/serving_templates.py:492`, `benchmarks/one_stage/a11.py:553-565` (`top = boolean_version(spec) if boolean else spec`), `benchmarks/one_stage/a11_kinds.py:216-222`, `adapters/vllm/verity_vllm/program/predict/block_counts.py:17`.

**Tasks.**

1. **The 64 kinds and their rows.** Read HASH-98's comments (the Linear tools are read-only for you) and kernels' files they name (`ksteady-jobs/h98/…`; find where they are: the notes, the evidence store, or kernels' branch) until you have, for each of rows 4, 6 to 12, 14, 15 and 16: the kinds in it, the Definition id kernels timed, and the id `boolean_version` gives for it on `main`. Say where each came from.
2. **What the served path names, by read.** Follow one drawn unit of each row from the served Program to the statement: which code picks the Definition the unit's template names (`serving_templates`, `chunk_rows.statement`, the partition, the target profile's `ir`), what the registration commits for it, what `flock-verify verify --zk` is given, and what the sampled audit's proved scope takes (`one_stage.proofs.ks_form_why`, `Flock.ProvedScope.scopeOkH`). For each row: the Definition id the statement names, whether it is Boolean (`verity.circuits.boolean.is_boolean`), and the file and line that decides it.
3. **Check it on the code, not only by read**, where a call of under a minute and 1 GB does it: for instance build the template of one kind of each row as the serving path does and print its `fn`'s id, `is_boolean`, and `zk_refusal(tp)`. Anything heavier (staging a circuit) goes through `research run --on auto --cwd source --as proofs`, and only if the answer turns on it. Kernels has already staged rows 1 and 15 from `main`; do not repeat that.
4. **What the figure's code assumes.** `verity bench prove` and the plan it prices (`benchmarks/prove_path/`, the steady rows' reader): for these rows, which Definition id does a row's label name, and would a row timed on the Boolean version be read in its place with no code change, or does the label have to change? One paragraph.
5. **The line for HASH-98**, ready for me to post: the answer in one bold sentence, then by row where it differs, each with its file and line, and what kernels should time on Thursday.

**Your verdict line** is one of: "served statements prove the registered Boolean version for all 45 kinds", "… the census's Definition for all 45", "it differs by row: …", or "not decidable by read: needs …".

**Not yours.** Timing anything on a GPU. Changing code. Whether the census of record itself should change. The head blocks (rows 1 and 2; our worker HEADSPLIT's #9087 changes how they are served: say in one line if what you find meets it, and no more).

**Done when.** The entry with tasks 1 to 5 and the verdict line is in my inbox file.

**Hours (UTC, Thu 12 Nov; take the time from `date -u`).** 09:30Z: a first entry with the verdict line as far as you have it, for the heaviest rows first (kernels can still start on time with an answer by 17:00Z, but the earlier the better). 12:30Z: the full entry.
