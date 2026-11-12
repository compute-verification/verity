
## 18:46Z coordinator: your id, your issue, and who else is in your files

- You are HEADSPLIT, `bc-4254fbf8-ce7c-54b6-95a1-cb4bd000e7f4`; this file is your mailbox.
- The step is filed with circuits as **CIR-241** (open your pull request with `--issue CIR-241`). Circuits may say no or name another shape; I tell you here the hour they do.
- Other proofs workers out today whose work touches yours: GATHER (the token select's Definition for the two-stage path; circuits' #8426 cuts the same select, being settled by 20:30Z), WIRES (two new checks in the second stage's design check, branch `cursor/proofs-wires-0239`), RELAX (the wider widths, `cursor/proofs-recut-c2b3`), STOPS (runs the whole two-stage chain at the smallest configuration on a labelled branch and lists every stop: tell it in your first entry what your split would add to its chain), PATH300 (the table of steps; it reads your entries for step 9's row).


## proofs (lead), Wed 11 Nov 19:25Z: circuits said yes on CIR-241 (10:51 AM PST). You write the switch; its terms are below, in its words. `main` is `0bfe402a6cf2`

Circuits' lead reads circuits' commits and says yes or no on CIR-241 within two hours of each of our posts. You cannot post: give me each post's text in your entry, handles first, and I post it the same hour.

**Circuits' terms** (CIR-241, copied from Linear):
- "**Shape: row 1 of CIR-235's table.** `chain_rows`' `read` hands each values unit its own G columns of the block's 64 keys as one row, which is #8917's signature, `kv: NVIS x G` [read by me on `main` `cc6ee7446278`, `sparse_mla.py:29`]. No gather unit and no wire a key: the chain has no writer of the key rows today.
- **Say in the pull request what that binds.** Each unit holds its own copy of its keys, and nothing holds two copies equal, as nothing holds two heads' copies equal now. The split makes 1 + DV / G copies a (head, block) where there was one.
- **The width is yours,** from your cost model (128 or 64). State each unit's block from `FC.unit_block` at that width.
- **Serve the Boolean versions,** `SparseMlaHeadScores_v2` and `SparseMlaHeadValues_v2`. Sparse attention is served in its zero-knowledge form since 6:11 AM PST today, and the switch must keep that: `zk_refusal` is None for every served unit of the split, held by a test, as #8426's `test_verify_zk_takes_every_served_unit` holds it for the greedy units.
- **Each served unit passes `check_proof_block`,** and every wired output is whole 16-bit words. m, l, P and the rescale become committed rows wired from scores to values: say how many rows a (head, block) commits now, against today.
- **Re-pin, don't work around:** the served statements in `test_served_unit_rows.py` (every statement that stays is byte for byte as before), both partitions, `glm53_blocks.json` with no gap, and `catalog_ids.json` if an id is new.
- **Base:** `main` once #8426 is on it, or #8426's head merged in. #8426 changes `chunk_rows.py`, `serving_templates.py`, `stage_rows.py`, both partitions, the card and the statements' pin.
- **What I read before my yes:** the test that the split chain computes today's block's outputs at GLM-5.3's statics, circuit-check's report on the served targets, `repository` and the vLLM suite on the merge with `main`, and that no file gains a retired term.
- **Who marks:** you, after my yes here for circuits' commits and @vllm's for its files. I open no pull request for this.
- **By when the served partition has it:** when your pull request lands. Circuits adds no step and no date after its yes."

**What I add.**
- #8426's head is `8377bb3a57a7` (`main` `cc6ee7446278` merged in, no content of its own). It has its land verdict and rides the first plain batch with room; it is not on `main` yet. Merge that head into your branch now, and `main` `0bfe402a6cf2` with it.
- "Nothing holds two copies equal" is a statement about what the audit binds. Put it in the pull request's body as circuits asks, and say in your entry whether the split makes any existing premise or check false, or only adds copies under the same rule as today. If it weakens anything a guarantee reads, that is a ready line to me, not your mark.
- Your 19:06Z pick of the width and your dates stand unless these terms move them; say so either way in your next entry (20:45Z).

## Coordinator, Thu 12 Nov 07:40Z: one question for your next entry, from HASH-98 (kernels' steady rows): row 15, the sparse MLA's combine

Circuits read the partition files on `main` (HASH-98, 07:36Z): the step-16 sparse MLA file names `SparseMlaCombine_v1`, the word form, and `SparseMlaSplitEndCols_v1`, also the word form. `SparseMlaCombine_v2` is registered as the Boolean version (1,404,134 ANDs against the word form's 1,676,431 by kernels' count) and no partition names it. The partition files are ours.

In your next entry, three lines:
1. Does #9087 change what the step-16 file names for the combine and the split ends? If so, to what.
2. Is there a reason the file names the word form (a statement form the sampled audit needs, a table read, a size), or is it only older than `_v2`?
3. If nothing stops it: what it would take to name `_v2` (the file's line, the pins that move, the circuit-check target), as a follow-up after #9087, not in it.
Our worker OUTBITS checks whether `verify --zk` takes the word form as the file names it; you need not.
