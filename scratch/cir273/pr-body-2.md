Lane circuits
Refs CIR-273
Goal vllm/production-vllm-reproducible

Stacked on #9693 (`cursor/circuits-cir264-sampled-accept-6f04`), and lands after it. This PR's base branch is #9693's, so the diff here is this PR's alone. Each statement below is tagged: [run] is from a run whose output I read, [read] is from reading the tree or a record, [inferred] is my conclusion from those.

## The head

The head is `e57b2921aa722a6ef13fa0de56c557bd4648924a`. With `--first-parent`, from the head this PR had before the carry (`6ee3b8d3b9`), it holds the following:
- `9d7e426678` [read]: the carry of `6ee3b8d3b9` across the layout moves onto `main` at `4ae17f7eaf`, by `tools/move/restack.py --no-pins`, fast-forwarded onto this branch. The carry was not my run.
- `79c818fa45` [run]: the merge of #9693's final head `8e96f9e2a3`, which holds `main` at `af9b134406`. The head's `main` is therefore `af9b134406`, of which `4ae17f7eaf` is an ancestor [run].
  - The merge had one conflict, one hunk: the import line in `verity/definitions/registry/sampled_accept.py`. I took this branch's `from verity.definitions.registry import dead_free as DF, sampling as SA, topp_stages as TS`; #9693's line has no `topp_stages`.
  - `adapters/vllm/tests/program/test_sampled_accept.py` merged by itself, taking #9693's style commit for its import blocks.
- `7dd7e11362` [read]: my docstring commit, in the module docstring's paragraph on `halves_split`. It says that the 1.4e-6 a request is over the rows where the draft token holds 50% to 99.9% of the mass, and is a mean over those rows, not a ceiling.
- `9eee018145` [read]: the paragraph's rates cite their record, `evidence.json`'s new `sampled_accept_split_rate` (`run_record` `art:fc64796c…`, `harness` `37eeb6d81`). The record of #9693's rates is cited in this change, after ci's review-58 on #9693. That id is the run record of the run behind the rates: it is preserved, its source commit is `37eeb6d818`, and its stdout holds the rates' figures [read].
- `e57b2921aa` [run]: the two sidecars' reference helpers (`call`, `mask`) now sit beside `_model_roots`, not inside it. Inside it, their `return`s made the catalog read the whole function as one unit, and circuit-check's `test_the_catalogs_units_are_its_root_functions_and_entries` failed on the carried tree (`r20261113-212217-c645`). With the helpers outside, it passes; with them back inside, it fails [run, locally, by name].

**#9693's docstring sentence on the 1.4e-6 a request is qualified in this change** [read]. `_CONF` is unchanged: its text hashes the same before and after these two commits [run].

The carry checks [run]:
- `git diff --stat 8e96f9e2a3d3 HEAD` names only this PR's nine files, as of `79c818fa45`. The two docstring commits touch `sampled_accept.py` and `evidence.json`, both among the nine.
- `git diff 8e96f9e2a3d3 79c818fa45` against `git diff 66f957e940d7 6ee3b8d3b9d5`, the diff before the carry: every added and removed line is the same. Apart from `index` and `@@` lines, the only lines that differ are four unchanged context lines, `from verity.definitions.ops.{fp32,mufu,prims,scalar} import …` → `from verity.circuits.ops.… import …`, which the moves rewrote.
- #9693's eight docstring lines stand verbatim at `79c818fa45`. My two commits then reflow that paragraph, and the citation sits inside its parenthesis.
- There is no conflict marker in the tree.

## What it adds

- **The unit that writes the masked row.** `TopPMaskChunk_v1` (`topp_stages.py`) writes one row of the top-p masked logits. `mask_plan(V)` lists the top-p pipeline's units with the mask chunks in place of the select chunks. The mask's work is `mask_lanes`, which both the Definition and its reference run. The catalog id is in `catalog_ids.json` [read].
- **The recut accept units.** `AcceptTileHead_v1`, `AcceptTileSum_v1`, `AcceptRowEnd_v1` and `AcceptResample_v1` read a row's lanes as rows of `ROW = TS.SELECT_N` lanes, so each read is one mask chunk's output row, whole. `pass_plan(K, V)` is a pass of K + 1 rows: each row's top-p pipeline and mask chunks, then the accept's units, with every read of a masked row a wire from the chunk that writes it [read].
- **The resample on 32-bit stream keys.** `AcceptResample_v1` keys each row's position (`GumbelStreamKey(seed, pos[p])`) and selects the 32-bit key with `SelectI32`, where it selected the 64-bit position with `SelectI64_v1`, which `Q_call v1` refused as wider than its X = 32 [read]. The Call's ports are those at #9693's head [run, at `6ee3b8d3b9` against `66f957e940`; the merge changed none of them, inferred from the diff checks above].
- **The numpy references.** `TopPMaskChunk_v1` is compared against the twin's `mask_chunk`, and `SampledAccept_v1` against `sampled_accept` (`accept_kernel`). `evidence.json`'s `sampled_accept_capture.replay_run_record` is this branch's replay record, which `_CONF` names [read].

## Runs at this head (`e57b2921aa`)

- circuit-check `r20261113-231251-c271` [run], `--update-pins` in the run's clone: `git status` and `git diff` were empty after it, so it wrote no pin. Every target's line matches `r20261113-083534-e2a9`'s by script (`compare_cc.py` on the first, uncached pass). The second pass reused those checks. `Q_call v1` refuses none.
```
circuit-check: vectors a realization compared with the reference on: 9 random, 0 edge-only, 25 none
circuit-check: 35 target(s), 0 failure(s), 33 unchecked (warnings, which never fail) on 33 Definition(s), report …/cc-sampled_accept.json
circuit-check: vectors a realization compared with the reference on: 5 random, 0 edge-only, 8 none
circuit-check: 13 target(s), 0 failure(s), 9 unchecked (warnings, which never fail) on 9 Definition(s), report …/cc-topp_stages.json
circuit-check: vectors a realization compared with the reference on: 13 random, 0 edge-only, 0 none
circuit-check: 13 target(s), 0 failure(s), 15 unchecked (warnings, which never fail) on 9 Definition(s), report …/cc-boolean_topp_stages.json
```

- Replay `r20261113-224628-6189` [run], off the card, with #9693's script and capture tar (sha256 `4ca7772a558b…`). The counts equal `r20261113-081538-2873`'s:
```
cases 410 (requests of 1 + 5 rows; 2460 rows), inside the domain 409, of them 0 differ; words compared 80226, differ 0
definition: 11 cases, 264 words compared, 0 differ
```

- Checks `r20261113-224512-e9e5` [run], with BASE `af9b13440693`, the commit of `main` the head holds:
  - `pins.py --stale` printed nothing.
  - `suites.py --changed af9b13440693 --quick`: `62 suites: 62 passed (11 cached), 0 failed`. Among them, `verity-vllm` 17,221 passed with 479 skipped, `verity-flock` 1,470 passed, the circuit-check suite 383 passed, and `verity/definitions` 1,058 passed.
  - `repository` whole: 361 passed.
  - `ratchet_gate.py af9b13440693 HEAD --json`: `"failures": []`.

No pin, catalog id or `_CONF` text moved: circuit-check's clone diff was empty and `pins --stale` named nothing [run]. `_CONF` hashes the same before and after the docstring commits [run]. `evidence.json` gained one record, `sampled_accept_split_rate`, and nothing else of it changed in these commits [read].

## Runs at earlier heads (superseded)

Also at `e57b2921aa`: two circuit-check runs (`r20261113-224550-71be`, `r20261113-230750-f188`) were cancelled while waiting for a check slot on vy-nebius-cpu-14, so neither check started [run]. `c271` ran on vy-nebius-cpu-9, named with `--on`.

At `9eee018145`:
- circuit-check `r20261113-212255-cd19` and replay `r20261113-212343-3511` [run]: the same lines and counts.
- Checks `r20261113-212217-c645` [run]:
  - `pins --stale`: nothing.
  - `repository`: 361 passed.
  - Ratchet gate: `failures: []`.
  - The `--changed --quick` set: 60 of 62 suites passed. The circuit-check suite failed on the test fixed in `e57b2921aa`. `verity-flock` failed 7 tests in `test_lean_verifier.py` with `ModuleNotFoundError: No module named 'pytest'`: they run a bare `python3`, and my job called `.venv/bin/python` rather than `uv run`, so the system Python came first on `PATH` [read]. The job now runs under `uv run`, as #9693's job did, where `verity-flock` passed [read].

At `7dd7e11362` (before the record commit):
- circuit-check `r20261113-205759-c821` [run]: the same report lines as `r20261113-083534-e2a9` by script, and no pin written.
- Replay `r20261113-205829-17e0` [run]: `definition: 11 cases, 264 words compared, 0 differ`.
- Checks `r20261113-205731-35a9`: cancelled when the head moved [run].

At `6ee3b8d3b9` and `0d2d60aba3`, before the carry: circuit-check `r20261113-083534-e2a9`, replay `r20261113-081538-2873`, suites `r20261113-084943-f574` (`repository` and `verity/definitions`), and the whole `--changed` set `r20261113-015516-01e6` at `f65b556ae5` [run].

## What was not run at this head

- The `--changed` set's slow tests: the set ran with `--quick` [run]. The slow tests of modules the change reaches ran whole, as the run reports for `test_sampled_accept.py`, `test_topp_stages.py` and `test_boolean_topp_stages.py` [read]. The rest of the slow tests last ran at `f65b556ae5`, before the carry.
- `main` after #9693 landed (`cfce541acc17`) is not merged: when step 4 was checked, `8e96f9e2a3` was not yet on `main` [run].
- lean-agreement and the Lean check: this PR changes no Lean and no Flock [read].

## What I trust least

- The mask chunk's reference shares `mask_lanes` with the Definition. The comparison therefore checks the Definition's compilation into gates, not the masking logic both run [read].
- The nine `AcceptResample_v1` bindings, like every accept unit but the root, are compared on no vector (`vectors/none`) [run]. They are held only through the root's reference at K=2 and the replay at K=5 [read].
