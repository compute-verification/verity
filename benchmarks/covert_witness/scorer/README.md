# scorer: the training-efficiency search's ledger, scores, promotions and digests

The scorer reads only the evidence store and writes the ledger and scores under `$TE_SCORER_STATE` (default
`~/.research/te-scorer`), all of it rebuilt from the store by `collect`. It writes its board lines to its own file,
`islands/scorer/BOARD.md` on its branch (`cursor/te-scorer-516a`), which the coordinator merges into `BOARD.md` on
`cursor/te-digest-4207`, and edits nothing of the fixed side or te_core.

~~~text
python scorer.py collect                               # new manifests -> index.jsonl, index.md, runs.json, starts.json
python scorer.py score                                 # score.json, score.md, submissions.json
python scorer.py board --promote --digest [--append]   # this hour's PROMOTED and DIGEST lines (--append pushes them)
python scorer.py post TAG TEXT                         # one line onto islands/scorer/BOARD.md, pushed
python scorer.py final starts|launch ID...|collect     # final-set reruns (below)
~~~

- **Final scoring** (`final.sh`, `scorer.py final`): an entry is rerun at fresh seeds (101, 102, 103 on the `--seed` and
  `--data-seed` flags its script takes) by `research run --source` of its own commit, `final.sh` sent with the run, so
  the island, the fixed side and the train args are the ones its record names. `final.sh` trains as `candidate.sh` does
  and scores the submission (and its curve) on the search and the final set; `final starts` scores the starts on both.
  `final collect` reads each ended run's `out/` into `$TE_SCORER_STATE/final/results.jsonl`.

- **Records**: evidence/v1 artifacts with `meta.covert_witness.class` (launch.py's, one per candidate) and te_core
  ledger records (`meta.schema` te-core/submission/v1), written since `TE_SCORER_SINCE`. The start registry's bpbs and
  tensors come from launch.py's start-scores record. A record's agent is the `cursor/te-<direction>-4207` branch holding
  its commit (`candidate.json`), else its class.
- **ρ**: te_core's (`TE_CORE_PATH`, its `score.score_grid`, `w_star`, `honest`, `effective_slowdown`, `frontier.TARGETS`).
  A launch.py record has no te_core count, so its annotation is analytic (`scorer.analytic`): U is every forward
  multiplication whose weight operand is trained (a fully trained tensor's matmul or norm gain, both factors of an
  adapter; a frozen W·x is θ's), a node's upstream is the updated MACs per token of the nodes before it in (layer, stage)
  order, |C| is the training FLOPs over 2, p = 16 and c = 4096 times the start's forward MACs per token (weight matmuls,
  norm gains and 2·d·ctx attention MACs a layer). Each curve point is a submission (its prefix of the run).
- **Counterexample credit** (`scorer.credit`): an E whose decoder rebuilds a step's updated weights from θ within c is
  a T1 family for that step. A seed log (`E.format` seedlog) is one: its prefix E_t rebuilds step t in (t + 1)·n MACs
  for n trained values, so it covers the steps with (t + 1)·n ≤ c, at β_t = bits(E_t) / updated MACs per token (bounded
  by the smallest snapshot's E), and te_core's option (a) takes that β. Adapter and full E's rebuild no step's values
  bit for bit and get none; an adapter's ρ is low because its φ is.
- **Cells**: mechanism (the note's first word) x |E| range (decades from 10 kB) x φ range (te_core's `PHI_EDGES`, on
  the witness's φ) x start. A cell is thin at two valid records or fewer.
- **Promotions** (hourly): each direction's top third by relative bpb drop from its start, and the best record of each
  thin cell, among valid records launched from 08:30Z; re-screens (a mechanism's note, less rates and seeds, on a class
  and start another direction screened first, or its own direction swept three times already) are flagged.
