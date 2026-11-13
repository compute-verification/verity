Lane circuits
Refs CIR-274
Goal circuits/request-built-as-proof-blocks

## What changed and why

Class G's request Programs are batched (`prune.batched_form`). Once a prompt has two or more ids, the norm after the embedding all-reduce reads the all-reduce's output in a batch over the prefill's rows. `tp_binding.rank_sites` refused that read by name ("a combine's output is read by calls"), so no TP window of a class-G request with more than one prompt id could be built. This PR takes such a read, and lowers it in the TP window.

- `program/tp_binding.py`:
  - A batch node reading a combine's output is one reader. Its run is stated as a call's: `(node, argument, leaf, lo, hi)`, the batch node's argument, whose mapped refs are its members' slices concatenated.
  - The batch's count and each argument's axis are the rank Program's. They ride outside the record in `RankSites.batches` and `RequestBinding.batches` (compare=False, like `fns`), and the record binds them through `ranks`. `record()` and `BINDING_TAG` (`v1`) are unchanged.
  - `batches` also names a batch producer, a batch whose output is a collective's own partial.
  - Still refused by name:
    - a combine's output read inside a scan;
    - a batch reader that produces a collective's partial;
    - an argument batched on an axis other than 0;
    - as before, a collective that is a batch.
- `pipeline/tp/wires.py` (functions touched: `_Window.consumer` and `sites_circuit`; added: `_members`, `_member_reads`, `_member_runs`):
  - `_Window.consumer` places n units for a batch of n. A batch of token selects is refused by name: the token select's split is lowered for a call only.
  - `sites_circuit` reads a mapped argument with `spread` (laid unit after unit) and a shared one with `same`.
  - It hands `grain.lower` each mapped read cut at the members (`_member_reads`), so the token row is a member's (6,144 words in class G) and not the batch's (n × 6,144).
  - It cuts a batch producer's run at its members (`_member_runs`), each piece at its member's leaves, as a call's partial is lowered. Without this, class G at 2 or more prompt ids refused next at the embedding batch (`EmbeddingShard_v1`, "neither its output words' units nor one whole output of it", run r20261113-014939-1c60).
  - `grain.py` is unchanged.
- `program/prune.py`: one docstring sentence. `batched_form` is not changed.

## Terms, each with its test

- a. Every Program `rank_sites` took before keeps its record byte for byte: a call's reader stays `(node, argument, leaf, lo, hi)`, and `BINDING_TAG` stays `v1`.
  - `tests/program/test_tp_binding.py::test_the_programs_whose_combines_calls_read_keep_their_binding_records_byte_for_byte` pins today's digests (prefixes) of the toy's three bindings and the w3, w4 and c3 bindings.
  - `tests/pipeline/test_tp_wires.py::test_the_window_of_combines_read_by_calls_is_the_circuit_it_was` pins the TP windows of the toy, w3, w4, c3 and the fused f3, read on `main` (e86d9204) before the change.
- b. A batch node is one reader, not one for each iteration. Its count and axes are in the rank Program, which the record's `ranks` bind.
  - `test_tp_binding.py::test_a_batch_reading_a_combines_output_is_one_reader_its_count_and_axes_the_rank_programs` covers mapped and shared arguments: the flat body has two readers, the batched one has one, `batches` holds `(2, axes)`, and the record has no batch field.
  - `test_tp_wires.py::test_a_batch_reading_a_combines_output_is_a_unit_a_member_wired_as_its_calls_are`, for mapped and shared arguments: the batch's window is the calls' window (templates, units, cuts, input bits, every wired row), with one range of 2 units.
  - `test_tp_wires.py::test_a_batch_producing_a_partial_is_its_members_units_as_its_calls_are` does the same for a batch producer.
- c. What the change leaves out still refuses by name, and a read listed as unwired is not a refusal.
  - `test_tp_binding.py::test_a_combines_output_read_in_a_scan_a_producing_batch_or_on_another_axis_is_refused_by_name` covers a scan, a batch that produces a partial, axis 1, and a collective that is a batch.
  - `test_tp_wires.py::test_a_batch_of_token_selects_is_refused_and_a_batch_read_the_grains_dont_fit_is_listed_unwired` covers the token-select batch refusal. It also checks that a batch reading half a fused row is listed in `Sites.unwired`, one entry per rank, with no error.
- d. `prune.batched_form` is unchanged: the diff has no hunk in it. `tests/program/test_batched.py` passes unchanged, and the term b and c tests build their batches with it.
- e. The sentences that state the refusal change with it. In `tp_binding.py`'s module docstring, the "What doesn't pair" sentence (now: a collective that isn't a call, a combine's output read inside a scan) and the "Each site also records" paragraph. In `prune.py`'s docstring, the "A collective stays a node of its own" sentence. In `wires.py`'s docstring, the units list. There is no test for prose; the term c tests hold what the sentences state.

## Runs

The head is b9e5c4a0: the circuits coordinator's carry of 1b640168 across this morning's layout moves (`tools/move/restack.py --kept --no-pins`, run r20261113-155530-28ef, result 0e6e13c3 onto `main` 4ae17f7e, `clean-skipped`), plus a plain `git merge` of `main` at c68135b4 (clean; no move landed after 4ae17f7e). Against `main` the change is the same 5 files, 264 insertions, 27 deletions as before the carry. Every run below is at b9e5c4a0 unless it names another commit.

- `pins.py --regen` at the head's root (r20261113-165911-4d7b): nothing stale before or after, nothing rewritten.
- `python3 -m research.carry 1b64016886e6 b9e5c4a088c0`: exit 0, `"kind": "merge-only", "carries": true` (60 commits, 24 moves rerun, nothing generated).
- `git merge-tree --write-tree` with `main` at fa3e619e: rc 0.

Collect runs, class G, two outputs, each from the drivers on the scratch branch copied into a clone of b9e5c4a0. Each gives empty `unwired`, `proof_blocks` and backward wires, and equals the same case at 3ec2a17b (the old head's library text, before the carry) in every result field: workload digest, sites, combines, units, template ids, wires, unwired, proof blocks, backward.

| prompt ids | layers | driver | run at b9e5c4a0 | before the carry | sites | wires |
|---|---|---|---|---|---|---|
| 1 | 4 | `scratch/cir271/collect43.py` | r20261113-170146-0752 | r20261113-033036-b53c | 20 | 768 |
| 2 | 4 | `collect43.py` | r20261113-170216-6bc5 | r20261113-033106-2ef2 | 26 | 1420 |
| 8 | 4 | `collect43.py` | r20261113-170247-e1da | r20261113-033134-59d5 | 62 | 1421 |
| 43 | 4 | `collect43.py` | r20261113-170320-a629 | r20261113-033148-defe | 20 | 724 |
| 2 | 78 | `scratch/cir274/collect_par.py` | r20261113-170350-6d72 | r20261113-033217-fd7e | 470 | 2144 |
| 8 | 78 | `collect_par.py` | r20261113-170421-24ba | r20261113-033248-273e | 1394 | 2144 |
| 43 | 78 | `collect_par.py` | r20261113-170451-7688 | r20261113-033320-3992 | 316 | 937 |

Suites: r20261113-170031-5017, `suites.py --changed c68135b4 --quick --fresh` (the suites the change reaches): 24 suites, 24 passed, 0 failed; 28,197 tests passed, 0 failed, 525 skipped. Among them:
- `verity-vllm`: 17,164 passed, 0 failed, 480 skipped (every test of `tests/program/test_tp_binding.py` and `tests/pipeline/test_tp_wires.py` among them);
- `repository`: 357 passed, 0 failed, 0 skipped.

Before the carry: r20261113-050213-ed4b at 3d00e506 and r20261113-033011-70e8 at b90c374b (`verity-vllm` 14,603 passed, `repository` 388 passed).

## Lint

- `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD --json` with `origin/main` at fa3e619e: `failures: []`.
- mypy as the lint runs it (`suites.py verity-vllm --fresh -- tests/lint/test_mypy.py`, its slow list checks included), r20261113-182209-33c4: 5 passed, 0 failed.
- No new entry in any of the adapter's lint lists.
- `ruff check --config ruff.toml` is clean on the five files.

## Not shown

- At 78 layers, `collect43.py` itself did not run. Its eight serial rank compiles and the uncached `sites_circuit` together take longer than a 30-minute run. An earlier driver at LP8, with the ranks in parallel but no cache, timed out (r20261113-023719-bb08). `collect_par.py` takes the same steps (`compile_steps` at each rank, `tp_binding.pair`, `TpWorkload`, `wires.sites_circuit`, then the test file's `in_order` and `proof_blocks`). It differs in three ways:
  - it compiles the ranks in forked processes;
  - it returns their Definitions by reference;
  - it computes `interface.layout` once per Definition.

  On `main`'s `sites_circuit`, `interface.layout` was 163 of 171 s at LP8, 4 layers (profile in r20261113-031049-1a2b). That cost is on `main`, not added here, and this PR leaves it alone.

  `collect_par.py` is shown equal to `collect43.py`, every result field, at 4 layers only: LP2 (r20261113-023621-37c1 against -015817-75b2) and LP8 (r20261113-031735-fd9e against -015844-09ea). At 78 layers, LP8 without the cache (r20261113-031018-5344, 3,042 s, 2,481 s of it in `sites_circuit`) equals LP8 with it (r20261113-033248-273e) in every result field. This shows the cache changes nothing. It does not show that the parallel compile and by-reference Definitions equal `collect43.py`'s serial ones at 78 layers.
- No proof: `proof_blocks` is the composer's `check_proof_block` under the window's cuts. No C-Flock prover or verifier ran on these windows.
- `circuit-check` did not run. The PR adds or changes no Definition, template or Boolean lowering: the window's units are existing Definitions.
- Only the quick tier of the suites the change reaches ran (`--changed --quick`), not the whole `verity check`.
- The read by token (a token select reading a combine's output inside a batch) is refused by name here. Its lowering is not in this PR, nor is `serving_templates.py`.

