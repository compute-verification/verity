# proof-format-v3 vectors

Semantic-equivalence vectors for statement format 3 (`veritor/proofs/statement/v3`, range
positions) and witness formats 2 (multiproof, shared sibling paths) and 3 (packed leaves), against
the per-position reference (statement format 2, witness format 1). PROOF-FMT lane, GUIDANCE
§57.9 change **A**: *the optimized path must decide exactly what the reference decides* — same
accept, same reject, same phase — on wrong values, roots and positions; missing or substituted
obligations; invalid ranges and slot maps; and values supplied through a shared table that nothing
authenticated.

`vectors.json` (schema `veritor/proof-format-v3/vectors/v1`) was **generated** by veritor's
`test_format3_equivalence.py::test_write_vectors` (the Python reference, which decided every
vector; `expected` is its verdict) and is frozen; it is **replayed** by
`archive/sp1/common/tests/format3_equivalence.rs` on the guest kernel. Gate set
`veritor.word-arithmetic@2` at width 16; packed leaves of 8 `u16le` words under
`verity-vllm/leaf/v1` framing; frame-v3 (`u16`) leaves for the multiproof family.

## Rerun

~~~sh
# Python, in a veritor checkout (the reference; regenerates vectors.json and asserts it is unchanged)
.venv/bin/python -m pytest -q -p no:cacheprovider tests/veritor/protocol/proofs/test_format3_equivalence.py
# Rust (the guest kernel; every path that can take a vector must agree exactly)
cd archive/sp1 && CARGO_BUILD_JOBS=2 cargo test -p veritor-zk-common --test format3_equivalence -- --nocapture
~~~

## Fields

| field | meaning |
| --- | --- |
| `statement_v3` | format-3 statement bytes (hex), the optimized description |
| `statement_v2` | format-2 statement bytes (hex), the reference; `expand(v3) == v2` by definition |
| `witness` | witness bytes (hex); `witness_format` 3 = packed, 2 = multiproof |
| `witness_openings` | (multiproof family) the format-1 per-opening witness of the same statement, where one exists |
| `plan` | (multiproof family) the honest tree-table plan (`wire.encode_plan`), which `plan_of` must reproduce |
| `words` | (accept family) the words each obligation's relation receives, read straight off the leaf bytes by an oracle that does not use `expand` or any table |
| `expected` | `accept`, `reject:<VerificationCode>` (the Python code; the Rust error must be in the same phase), or `reject:parse` |
| `tampered` | what was changed relative to the honest vector |

## What each Rust path must satisfy

* `accept` / `reject-witness`: `check_batch_ranged(v3, w)` **==** `check_batch_packed(expand(v3), w)` **==**
  `check_batch_packed(parse(v2 bytes), w)` as `Result` values (same `CheckError`, same
  obligation/position attribution); decoded words equal on both passes; `verify(v3 bytes)` and
  `verify(v2 bytes)` give the same verdict, each over its own digest; `verify(v3 bytes, w, plan)` is
  refused (a format-3 statement takes no plan).
* `reject-parse`: `parse_statement_v3` and `parse_any_statement` fail; `verify` returns verdict 0.
* `multiproof`: `check_batch_multi(v2, w2)` **==** `check_batch_planned(v2, w2, plan)`; where a
  format-1 witness exists, `check_batch(v2, w1)` has the same verdict and phase; `plan_of(v2)` equals the
  recorded plan; `verify` agrees with and without the plan.

## Vectors

| name | witness fmt | tampered | expected (Python == Rust phase) |
| --- | --- | --- | --- |
| `accept/count-one-ranges` | 3 | – | accept |
| `accept/one-tree-range-plus-single` | 3 | – | accept |
| `accept/stride-three` | 3 | – | accept |
| `accept/stride-two-alias-lifted` | 3 | – (lifted node in a 5-leaf tree; a word named by a range and a single) | accept |
| `accept/three-trees` | 3 | – (ranges under commitment indices 0, 1, 2) | accept |
| `accept/two-trees-shared-leaf` | 3 | – (one leaf opened by two obligations: authenticated once, reused) | accept |
| `accept/whole-leaf-and-singles` | 3 | – | accept |
| `reject/tampered-leaf-value` | 3 | one byte of the opened leaf flipped | reject:INVALID_OPENING |
| `reject/wrong-root` | 3 | statement's commitment root flipped | reject:INVALID_OPENING |
| `reject/path-of-another-rank` | 3 | honest leaf, path of rank+1 | reject:INVALID_OPENING |
| `reject/authenticated-leaf-of-another-rank` | 3 | a fully authenticated (id, value, path) of another rank in the slot | reject:INVALID_OPENING |
| `reject/statement-moved-to-unopened-rank` | 3 | range moved to a rank the witness does not open | reject:MALFORMED_TRANSCRIPT (shape: 2 ranks, 1 leaf) |
| `reject/witness-extra-leaf` | 3 | one extra leaf in tree 0 | reject:MALFORMED_TRANSCRIPT |
| `reject/witness-duplicated-leaf` | 3 | tree 0's first leaf repeated | reject:MALFORMED_TRANSCRIPT |
| `reject/witness-missing-leaf` | 3 | tree 0's first leaf dropped | reject:MALFORMED_TRANSCRIPT |
| `reject/witness-extra-tree` | 3 | a tree appended | reject:MALFORMED_TRANSCRIPT |
| `reject/witness-missing-tree` | 3 | tree 0 dropped | reject:MALFORMED_TRANSCRIPT |
| `reject/leaf-id-substituted` | 3 | canonical id changed (id is hashed into the leaf) | reject:INVALID_OPENING |
| `reject/leaf-value-short` | 3 | leaf bytes truncated by one word | reject:INVALID_OPENING |
| `reject/range-shifted-by-one-word` | 3 | `start+1`: authenticates, relation reads other words | reject:RELATION_REJECTED |
| `reject/pinned-word-differs` | 3 | `expected` on the single set to `00 00` | reject:PUBLIC_IO_MISMATCH |
| `parse-reject/count-zero` | – | range count 0 | reject:parse |
| `parse-reject/stride-zero` | – | range stride 0 | reject:parse |
| `parse-reject/overflow-start` | – | start 2^64−1, count 2 | reject:parse |
| `parse-reject/overflow-span` | – | count 2^64−1, stride 2^64−1 | reject:parse |
| `parse-reject/beyond-leaf` | – | start 7, count 2 on 8 words | reject:parse |
| `parse-reject/start-at-word-count` | – | start 8, count 1 on 8 words | reject:parse |
| `parse-reject/beyond-leaf-stride` | – | start 2, count 3, stride 3 (word 8) | reject:parse |
| `parse-reject/slots-overrun-u32` | – | count 2^32 (caught by the leaf bound first) | reject:parse |
| `parse-reject/descriptor-with-index` | – | `…:u16le:8:0` (a schema, not a descriptor) | reject:parse |
| `parse-reject/descriptor-frame-v3` | – | `u16` | reject:parse |
| `parse-reject/descriptor-no-endianness` | – | `…:u16:8` | reject:parse |
| `parse-reject/slot-outside-expansion` | – | inputs (0, 3) for 3 slots | reject:parse |
| `parse-reject/slot-total-overruns-u32` | – | two in-bounds ranges of 2^32−1 words each | reject:parse ("too many positions") |
| `multiproof/honest` | 2 (+1) | – | accept |
| `multiproof/tampered-value` | 2 | value[0] flipped under the shared path | reject:INVALID_OPENING |
| `multiproof/values-swapped-between-ranks` | 2 | two authenticated values exchanged | reject:INVALID_OPENING |
| `multiproof/tampered-sibling` | 2 | siblings[0] flipped | reject:INVALID_OPENING |
| `multiproof/sibling-dropped` | 2 | first sibling removed | reject:INVALID_OPENING |
| `multiproof/sibling-added` | 2 | a sibling appended | reject:INVALID_OPENING |
| `multiproof/value-added` | 2 | one value too many | reject:MALFORMED_TRANSCRIPT |
| `multiproof/value-dropped` | 2 | one value too few | reject:MALFORMED_TRANSCRIPT |
| `multiproof/shared-path-reused-for-unopened-rank` | 2 (+1) | statement opens rank 3 instead of 2, same-shape multiproof of (1,2,6) | reject:INVALID_OPENING |
| `multiproof/shared-path-with-honest-value-of-unopened-rank` | 2 | as above with rank 3's true value ("trust the table") | reject:INVALID_OPENING |
| `multiproof/position-moved` | 2 (+1) | rank 2 at position 5 | reject:INVALID_OPENING |
| `multiproof/wrong-root` | 2 (+1) | root zeroed | reject:INVALID_OPENING |
| `multiproof/two-obligations-share-a-leaf` | 2 (+1) | – (rank 2 authenticated once, used twice) | accept |
| `multiproof/authenticated-values-fail-the-relation` | 2 (+1) | honest witness, `add` relation over values that do not satisfy it | reject:RELATION_REJECTED |

Nothing in this directory is a historical record; regenerate it when the corpus in the Python
test changes (the test asserts the file equals what it would write).
