[2026-11-14T07:50Z] 1aa8d5d82163
READY 479d8cdfc6bc859879741bada6be95039a10c3ee (READMAPS on #8449, Fri 13 Nov 23:49 PST)

MUST FIX: none.

SHOULD FIX (false text in the chain's output; none blocks the maps):
1. `slowdown_map.toml` `"Dampener" = "Slowdown"`: `compose/verifier/remote/README.md:17` now says "Lean keeps some older names: `Slowdown*` means overhead" (READ). It also rewrote the toy test in `tests/lint/test_repo_retired_terms.py` to `Term("Slowdown*", …)` (READ). Cure: exclude that file, as the prose maps do, and a hand edit dropping the README's clause.
2. `inference_only_map.toml` `"Nci" = "InferenceOnly"`: the check README's Names paragraph (`verifier/README.md:130`) now says `InferenceOnly`, `InferenceOnlyCheck` and `inference-only-*` "keep the older names" (READ). No rehearsal commit makes the hand edit the headers promise. Cure: a hand commit.
3. The keep pattern `(?<=")Nci(?!Check)[A-Z]\w*` keeps only quoted names, so `count_difftest_vectors.py`'s docstrings cite `InferenceOnlyHandBlk` five times while the code builds `"NciHandBlk"` (READ). Cure: lookbehind `(?<=["`])`.
4. The body's row 15 claims "`verity bench rho` to `verity bench slowdown`". No row does that: `verity bench NAME` runs `benchmarks/NAME/`, so map 11's move does it (READ). Cure: move the clause to row 11.
5. The `slowdown_calculator_names_map.toml` header still cites "its [tokens] row", which this PR deletes (READ). Cure: drop the sentence.

NOTE:
- `inference_only_ids_map.toml` `files = ["*"]`, `"nci-"`: the whole tree outside `benchmarks/nci/`, but here it renamed only executables, each with its lakefile (RAN: 40 files).
- `NCI_COST_FLAGS` becomes `INFERENCE_ONLY_COST_FLAGS` with its one reader, but an old `--env NCI_COST_FLAGS=…` would be ignored without an error (INFERRED).
- `benchmarks/nci/README.md:23` lists `bracket.py` among "the older names" (READ).

Item 2 (READ): the five changed maps differ in paths only. `slowdown_calculator_map.toml` also drops `[tokens]` and excludes `tests/tree.toml`, as the body says.

Item 3 (RAN/READ): every listed string is kept where a program reads it. These are renamed with all their readers: the executables, the Lake package `nci_check` (manifests), path joins, the calculator's keys and output files (`rho.json` becomes `slowdown.json`), and the allowlist and cap keys. `uv.lock` and the pyproject names are unchanged.

Item 4 (RAN): the kept commit plus the 17 maps took 7 min 25 s with uv 0.12.20. All 18 tree hashes equal the rehearsal's, ending in `01f90b38b5e3` at `54f3cb8916a8`. All 12 moves certified, and every rerun changed nothing.

Item 5 (RAN), whole words in lines/files at `b4da7f28da3c`: `Nci` 1090/8, `NciCheck` 824/6, `NCI` 59/14, `nci` 315/69 outside `benchmarks/nci`, `rho` 2646/479. Nearly all are kept on purpose: the locks' records (awaiting `--moved`), `moves.json`, GLOSSARY's older names, the failure rule, `benchmarks/nci/`, versioned ids, `families.Nci`, the period refusal, toy Definitions, the board's and the harness's keys, and ρ for other quantities. The only misses are SHOULD FIX 2 and 3.

Item 6 (RAN): `test_maps.py`, 2 passed (`uv run --no-sync`, existing environment).

Item 7 (READ): the body is true at the head but for SHOULD FIX 4, and names nothing internal.

For the lead to read:
- The remote README, line 17.
- The check README's Names paragraph.
- `inference_only_map.toml`'s keep patterns.
- `inference_only_ids_map.toml`'s `files = ["*"]` row.
- The `frozen` reasons map 14 rewrites in `security/lean-audit.json` (prose, not records).

Nothing addressed to @training arrived.

