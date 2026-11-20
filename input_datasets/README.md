# Input datasets

What a workload reads its circuits' inputs from: an adapter serving a workload in order, a benchmark sampling a subset,
calibration reading a census of workloads, a training run reading epochs by split. Verity quantifies over every input
dataset, so nothing under `verity/` imports this package (`tests/import_order.toml` puts it after `verity` and before
`adapters/`). Standard library only.

Every dataset is reproducible: a definition `definitions/NAME.toml` pins exactly what is read. A record is a plain JSON
value the package never interprets; one record may be a whole session of several turns, and `sample` and `shuffled`
move whole records, never parts of one. How records are played against a serve (arrival times, rates, concurrency) is
the sender's, never a dataset's.

## API

- `open(name, split=None)`: the dataset `definitions/NAME.toml`, or its split.
- `Dataset.kind`: one word, what a record is (nothing checks a record against it).
- `Dataset.identity`: a string that pins what is read: `recorded:sha256=HEX:FORMAT`, or
  `generated:GENERATOR/vVERSION:seed=SEED:parameters-sha256=HEX`; a split, a sample or a shuffle appends
  `|split:NAME=START:STOP`, `|sample:n=N,seed=SEED` or `|shuffle:seed=SEED` to its parent's.
- `iter(dataset)`, `len(dataset)`: the records in order, read afresh each time.
- `Dataset.sample(n, seed)`: `n` records drawn without replacement, in their order; `Dataset.shuffled(seed)`: all of
  them in a seeded order. Both are deterministic and return datasets.
- `Dataset.splits`: each split by name.
- `Refused`, with its `name`: `unknown-dataset`, `unknown-split`, `digest` (recorded bytes whose digest is not the
  definition's) and `missing` (recorded bytes not fetched yet), each with one text naming the fix.

## Add a definition

One TOML file under `definitions/`, its stem the dataset's name, holding `kind` and exactly one of:

- `[recorded]`: `source`, where the bytes are (an `art:` id of the evidence store, an entry of `artifacts.json`);
  `path`, where `research data fetch-fixtures PATH` puts them in the checkout, which is where they are read; `sha256` or
  `sha512`; `format`, `jsonl` (a record a line) or `json` with `records`, the member holding the array of records. An
  external source (a Hugging Face dataset at a revision) is to be another `source` with its digest; only `art:` is read
  so far.
- `[generated]`: `generator` and `version` (a key of `GENERATORS`), `seed`, and `[generated.parameters]`.

`[splits]` names ranges of records, `NAME = [START, STOP]`.

## Add a generator

A pure function of `(seed, parameters)` to a list of records in `generators.py`, entered in `GENERATORS` under its name
and version. A change to what it returns for any seed and parameters is a new version.
