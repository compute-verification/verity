[2026-11-12T07:21Z] d9576202fdb6
CERTNEG, NET-340, Wed 11 Nov 11:22 PM PST. The head for Thursday's launch, with both reruns at it.

HEAD: `32e18ebcbcca4df4ce2f444898ea123f05d9c4d6` on `cursor/network-accounting-served-cheats-06b2`, pushed, no force. Two commits on `dc0c3b5c4769`; `main` not merged again. #9153 open, MERGEABLE, not marked [read].

1. `20e34fad02`: `served_cheats.py` and `tests/test_served_cheats.py` deleted. `timing` gave nothing more, so nothing moved to `benchmarks/certifier_timing/` [read]. The relay's run on a virtual clock, which the deviation tests took from the file, is now their own helper. `git grep served_cheats` is empty, the README included [ran]. `pyproject.toml` is `main`'s again, so the pull request edits two held files, not three: `README.md` and `warden_serve.py` [ran].
2. `32e18ebcbc`: the README's list says the two deviations not tried, each as itself. `deviations.py` and `deviations.sh` did not change.

RUNS at the head, each launched once through `--on auto` [ran]:
- `r20261112-065605-1236`, the `FAKE=1` job at the command's schedule: done, rc=0. "6 tries in 963.4 s"; every row "as its check says"; "guard: no address in out/ (9 files); exit 0"; the table by id is the same bytes. bucket: 64 planted, 29 read right, 0.0 bits.
- `r20261112-065529-8669`, `suites.py --quick --changed d6075b75d289` (where the branch left `origin/main`): done, rc=0. "7 suites: 7 passed (0 cached), 0 failed; 11.8 min of test time". Seven where there were eight: without the `pyproject.toml` change, `research` is no longer touched. `verity-network-traces` 596 passed, 603 less the deleted file's seven.

The pull request's body and NET-340's one comment name this head and these runs [ran]. The pull request now adds 1567 lines and removes 9, in five files.

NOT DONE: no merge of `main`; the two deviations not built; the forward's success path still not run.

