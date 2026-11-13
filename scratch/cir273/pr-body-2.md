Lane circuits
Refs CIR-273
Goal vllm/production-vllm-reproducible

Stacked on #9693 (`cursor/circuits-cir264-sampled-accept-6f04`), and lands after it. This PR's base branch is #9693's, so the diff here is this PR's alone. Each statement below is tagged: [run] is from a run whose output I read, [read] is from reading the tree or a record, [inferred] is my conclusion from those.

## The head

The head is `HEAD_SHA`. With `--first-parent`, from the head this PR had before the carry (`6ee3b8d3b9`), it holds the following:
- `9d7e426678` [read]: the carry of `6ee3b8d3b9` across the layout moves onto `main` at `4ae17f7eaf`, by `tools/move/restack.py --no-pins`, fast-forwarded onto this branch. The carry was not my run.
- `79c818fa45` [run]: the merge of #9693's final head `8e96f9e2a3`, which holds `main` at `af9b134406`. The head's `main` is therefore `af9b134406`, of which `4ae17f7eaf` is an ancestor [run].
  - The merge had one conflict, one hunk: the import line in `verity/definitions/registry/sampled_accept.py`. I took this branch's `from verity.definitions.registry import dead_free as DF, sampling as SA, topp_stages as TS`; #9693's line has no `topp_stages`.
  - `adapters/vllm/tests/program/test_sampled_accept.py` merged by itself, taking #9693's style commit for its import blocks.
- `7dd7e11362` [read]: my docstring commit, in the module docstring's paragraph on `halves_split`. It says that the 1.4e-6 a request is over the rows where the draft token holds 50% to 99.9% of the mass, and is a mean over those rows, not a ceiling.
- `9eee018145` [read]: the paragraph's rates cite their record, `evidence.json`'s new `sampled_accept_split_rate` (`run_record` `art:fc64796c…`, `harness` `37eeb6d81`). The record of #9693's rates is cited in this change, after ci's review-58 on #9693.

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

## Runs at this head (`HEAD_SHA`)

CC_SECTION

REPLAY_SECTION

CHECKS_SECTION

## Runs at earlier heads (superseded)

At `7dd7e11362` (before the record commit):
- circuit-check `r20261113-205759-c821` [run]: the same report lines as `r20261113-083534-e2a9` by script, and no pin written.
- Replay `r20261113-205829-17e0` [run]: `definition: 11 cases, 264 words compared, 0 differ`.
- Checks `r20261113-205731-35a9`: cancelled when the head moved [run].

At `6ee3b8d3b9` and `0d2d60aba3`, before the carry: circuit-check `r20261113-083534-e2a9`, replay `r20261113-081538-2873`, suites `r20261113-084943-f574` (`repository` and `verity/definitions`), and the whole `--changed` set `r20261113-015516-01e6` at `f65b556ae5` [run].

## What was not run at this head

- The whole `--changed` set without `--quick`: its slow tests, `verity-vllm`'s among them, did not run at this head [inferred from the flag]. They last ran at `f65b556ae5`, before the carry.
- lean-agreement and the Lean check: this PR changes no Lean and no Flock [read].

## What I trust least

- The mask chunk's reference shares `mask_lanes` with the Definition. The comparison therefore checks the Definition's compilation into gates, not the masking logic both run [read].
- The nine `AcceptResample_v1` bindings, like every accept unit but the root, are compared on no vector (`vectors/none`) [run]. They are held only through the root's reference at K=2 and the replay at K=5 [read].
