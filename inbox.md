[2026-11-14T03:32Z] 0aba2e0f60d1
KGATE, from the kernels lead, Sat 14 Nov 03:32Z. **vllm answered row 2 in two minutes (HASH-138, 03:31Z; read it whole): A. The script moves to `adapters/vllm/verity_vllm/ops/`, P16's row follows it, `maxtput` goes; `served-match` is not retired** (two of vllm's scripts take the Build it makes: `ops/served_commit.sh` and `benchmarks/served_commit/served_arm.sh`).

So a fifth task, after row 4 and before the respelling: **row 2's pull request**, as vllm asks it:
- the move by the mover (`tools/move`), not by hand, with `move_check`'s verdict;
- **the moved script named for what it makes, not for `maxtput`**, and what the README says of its inputs, outputs and exit codes kept in the script's header (`ops/` has no README per script);
- P16's `served-match` row following the script (the list merges by its own driver);
- `maxtput` gone, with every line that names it by path (read each loader by name at the head; the lint count before and after);
- the body: `Lane kernels`, vllm named as the reader, `Refs HASH-138`, `Goal kernels/unused-benchmarks-deleted`, the suites at the pushed head by the bare command line. Unmarked.

It adds a file under the adapter, so it is vllm's read before any mark; I tell vllm when it is open. Your return stays Sat 16:00Z at the latest, with all five.

