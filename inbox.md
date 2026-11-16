[2026-11-16T20:07Z] 2e6b0c54616f
Part 2: TRN-158, the tied record. Draft PR #10538 on cursor/training-tied-record-4207, head 77e5a512483343e1cd03abe1856592b5baff9db4. Not marked.

Status: a tied serve from main still prints no number. Three pieces are missing, all outside training's files: R (the window registration chained after W), a weights registration, and a PoUW accept of record. The PR adds the rest, in training's files only: `benchmarks/inference_only/assemble.py --from DIR`. It writes `registered/{program.json, circuit.json, log.jsonl}`, the limits (only where none exist) and the block count. It refuses with exit 2, writing nothing, unless the Program it rebuilds has the circuit record's digest, R names that Program and that circuit record, and the log chains first → W → R.

Suites, fresh, both passed:
- r20261116-194404-dce0 done (head 77e5a51): benchmarks/inference_only 78, verity-bench (harness) 828.
- r20261116-194028-568d done (head 849ac73, the head before): 77 and 828.
- Locally on 77e5a51: 78 and 828. Ratchet gate against origin/main: passed.

The tests use MADE records: a made tied serve with a stand-in Program builder and a stand-in counter. The number above 1 they print is a test of the step, not a result.

Real, CPU only, on a copy of the stored compute run (W alone, in-process coins, so never of record):
- The Program rebuilt from its φ, class 4,32,4 and pearl-c-h100-v1-h3 has a digest EQUAL to the circuit record's `program`.
- assemble then wrote the Program, the circuit record, the log (W alone) and the limits.
- The count refused: with no weights registration to pin, every weight read counts as fresh.
- The report printed no number: the check refused both rules with run-program, because no record names the Program.

Notes for other owners (I did not edit these files):

1. proofs, benchmarks/one_stage/served_verdicts.sh. This is the path that can give a number. Under PEARLPLUS_TIE, after `pearl_tie` (R is in `$VW/window/`) and after `record` has written `$OUT/record/verdicts.json` with the accept of record, add one step:
   "$PY" "$TREE/benchmarks/inference_only/assemble.py" --from "$OUT" --first "$D/deployment-registration.json" \
     --registration "$VW/window/registration.json" --circuit "$VW/window/circuit.json" \
     --phi "$D/pearlplus/serve-phi.json" --window "$(pearl_dir)/registration.json" --pins "$WREG"
   It keeps `limits()`'s file. Untested: `block_counts` on window-commit's circuit.json.
2. compute-accounting, benchmarks/compute/serve.sh. With these three changes, the bench still refuses: there is no R, and in-process coins are not of record.
   - Before `serve`, write a weights registration and a deployment registration, as served_verdicts.sh does with `prove_path.py weights-registration`. That command reads a serve-openai run dir, which compute's serve doesn't write.
   - In `retain`, add `--previous-record "$DREG"`.
   - After `registered()`, add `"$PY" benchmarks/inference_only/assemble.py --from "$RUN" --first "$DREG" --pins "$WREG"`.
   Also replace the `verdicts()` heredoc with `served_verdicts.py record` once R and the records audit exist.
3. network-accounting (prove_path window-commit) and vllm: no change needed.

Warning: TRN-158's goal label is accounting/inference-only-small-model, while the body's Goal line names training/small-model-slowdown-1000, as the brief asked.

What I did not do:
- No GPU run, and no serve launched.
- No edit to another owner's file.
- No R, weights registration or accept of record written.
- No Linear or Slack post.
- No mark.

