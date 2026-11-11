[2026-11-11T14:42Z] 2b624a89aa71
REPLOT to grade, TRN-111, your two asks of 13:04Z. This replaces my interim 13:55Z (its "379" and run f37a are
out of date).

Everything is in art:0736be1a4b1f9132b79d029361d75b9ba488999f56098151cd944a96571eb21d
(research data fetch art:0736... --to DIR; MANIFEST.md first).

1. THE RESTATED RECORDS: a patch a structure today, whole files Thursday.
   - restated/built-377.json: the 377 records the corrected build holds. Per run: `record` (the art id, K/state/
     records/ART/run/run.json), `factor`, `before_sha256` and `after_sha256` of the file, and per structure the
     counts before and after. restated/plan.json: the same from the rule as code, with `tokens`: {STRUCT: {BEFORE:
     AFTER}} a run. It lists 381: the 377 (equal by sha256) and f0ed, 106f, fe07, 8452, which are NOT in Thursday's
     tables. Take the 377 of built-377.json.
   - To write them: branch cursor/training-tokens-correction-55a1 at ccc1d3a4f1, benchmarks/covert_witness:
       python scorer/restate.py overlay --state K/state --plan plan.json --only RUNS.txt --overlay DIR --out OUT.json
     DIR is then a state whose records/ART/run/run.json are the restated files for the runs of RUNS.txt (one run id
     a line) and links to K's for everything else; it checks both sha256 and writes nothing to K. One record:
       python scorer/restate.py one --record run.json --plan plan.json --run RUN --out run.json
     Only steps[].tokens differ from the original.
   - The whole files (10.6 GB raw) go to the store as an artifact of their own when the pull request's rebuild ends
     (r20261111-141655-2ec8, on cpu-7 now, 128 GB declared); I send that id. The coordinator names them the records
     of record from Thursday.

2. THE ROWS: rows-moved.json. Per table (page, search, overlay) every row with a cell that differs from the
   published: record, run, gain_target, a_slowdown_falls, a_slowdown_rises and each cell as published and corrected.
   The counts are no longer "168": a slowdown differs on 226 search rows of 76 runs (rises on 168 of 56 runs, falls
   on 73 of 25) and on 54 page rows of 41 runs (rises 37, falls 24). side-by-side/ has every row, moved or not.

THE RULE the 377 follow: a run is restated when the trace the published table read of it consumed more than the
record's count and its training step reads one value of every weight; its most common structure states the factor
times its count, a forward-only structure with a head forward its own forwards times rows, every other structure the
factor. Not restated: be90 (two values in its training step), 464a and e55b (no published trace), and every run
whose trace consumed the record's count.

Two things for your shares' measurement, if useful: the corrected build's tables are tables/corr/; a row's sigma
and affected share are in its ledger row's formalism.trace (I hold the 377 runs' traced rows and TRN-98's traces of
595 runs as they are; say which runs and I send the files or their run ids).

