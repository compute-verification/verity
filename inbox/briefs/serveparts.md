# SERVEPARTS: the serving cost's target prices the three parts it leaves out (a model, a pull request; no GPU)

**Why.** The serving cost's target is 2.05 points of throughput lost against stock vLLM. It is our cost model's: the certified tick 1.63, the fixed top-k order 0.25, the snapshot at a phase edge 0.17 (0.163 the edge's pause plus 0.006 the carried pages' hash). The served engine turns on three things the model does not price: **the step record, the two-stage boundary commitment, and the history tables.** Until it prices them, the vLLM lane's benchmark prints its number beside the target with "not comparable". The vLLM lead asked proofs to take this (PRF-456, 9:36 AM PST) and I said yes with these dates.

**The run you must not read before you predict.** The vLLM lane's run of record is on the eight H200s today at 7 PM PST (03:00Z Thu): GLM-5.3, one engine across eight cards, the chat load at 384 requests at once, with one arm for the step record and the boundary commitment and the history tables each as the difference of two arms (`benchmarks/serve_tax/serve_tax.py:201-239` at their branch's `1b2750860c60`). Its measured shares come on PRF-456 by Thu 12 Nov noon PST. **Your three figures are predictions: posted before that run's shares exist, from the model and from earlier measurements only.** Say in the entry which runs you read and that the run of record is not among them. A prediction that misses is a finding, not a failure; a figure fitted after the fact is worth nothing here.

**What to build**
1. **Each part's price, in points of throughput at that run's configuration, with a low and a high**, from what it does:
   - what it hashes or copies a step, a request and a replay block (bytes, from the code: `adapters/vllm/verity_vllm/pipeline/boundary_rows.py`, `history_rows.py`, `verity/commitments/windows/history.py`, the step record's writer in `serve_commit.py`);
   - where that work runs (on the engine loop, off it on a host thread, on the card) and what it therefore takes from serving;
   - the rate it runs at, from a measured run you name (the `record`, `tokens`, `boundary` and `kv` arms on Moonlight on one H200: `r20261106-175939-df52`, `r20261106-193235-5170`; the hash lane's rates under `benchmarks/hash/`), and how you carry it to GLM-5.3 on eight cards.
   - Also: `boundary-rows` turns prefix caching off (my reading of Tue 10 Nov, `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, search "The boundary commitment: no"). Say whether the run of record serves with it off, and price that change as a fourth line if it does; the model's own price for it at the long-context plan is `PREFIX_OFF` in `benchmarks/spot_figures/breakeven/sweep.py`.
2. **The model reads them.** The serving cost's card gains the three names ("step record", split from the edge snapshot; "two-stage boundary commitment"; "history tables"), and `Case.tax` in `sweep.py` reads them, so the target is one sum of named parts each with its source and status (modeled, simulated, measured). A test holds the sum to its parts. No figure of the exfiltration capacity may move silently: say which rows of the cost model change and by how much.
3. **After Thu noon PST:** the measured shares beside your predictions, part by part, with the run's own interval (the vLLM lead says two passes of the same work differed by 6.0%, so a share whose interval holds zero is "not told apart from zero"). Then the target restated: which parts stay modeled, which become measured.

**Deadlines (UTC).**
- 20:30Z today: first entry: where each part's work runs and the bytes, read from the code.
- **Thu 03:00Z (Wed 7 PM PST): the three predictions**, as your entry and as a comment on PRF-456 (`research msg linear comment PRF-456 --as proofs --worker SERVEPARTS --file F`, to @vllm).
- Thu 12 Nov 02:00Z Fri (Thu 6 PM PST): the pull request open, quick tier run, marked by you (it changes no statement, premise or verifier).
- Fri 13 Nov 20:00Z (noon PST): the comparison on PRF-456 and the goal `proofs/serving-cost-three-parts` updated; end your turn with the three lines for me.

**Words.** The vLLM lead and Daniel read your comment: Glossary terms and ordinary words, every number with what it measures, its configuration and whether it was run. "The serve tax" is our name for it; write "serving cost".
