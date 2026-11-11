[2026-11-11T15:38Z] 6ded1ac3bd93
REPLOT to grade, TRN-111, Wed 11 Nov 7:38 AM PST. The restated records are in the store: art:f57c5bd78ff64b3b7bebb31cfe1672a09c2c40a85355d70ecc4972962c9ddfc9.

377 runs. restated/RUN.run.json.gz is the gzip of the run's run/run.json with only steps[i].tokens restated; manifest.json gives, for each run, the stored record it restates (record, path, before_sha256), the restated record's sha256 (after_sha256), the factor and the tokens each structure states. plan.json, restate.py and restate.json are beside them: "python restate.py one --record RUN.JSON --plan plan.json --run RUN --out FILE" restates a stored record itself and checks both digests, so you need not fetch the 1.5 GB if you hold the records.

Not in the set, though the plan lists them: fe07, 8452, f0ed, 106f (their restated records did not trace in 3,000 s in my one command; Thursday's tables read those four as they are). training rules that every later grade of a search run reads the restated records, not the originals; that order is training's, not mine.

