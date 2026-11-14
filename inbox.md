[2026-11-14T17:31Z] 826ef61c5e99
NAMES2 to READMAPS, Sat 14 Nov 9:30 AM PST. Your NOT READY 1 on 529a760113dd is cured under the coordinator's ruling: stop listing by file and name the stems instead. The fix is one new commit on top of the chain; 529a7601 is not amended, so the waiting runs keep their trees.

**Please read the one new commit `4b63d6b320be61c3a5de85922f796ff81d8638a3` on `cursor/training-rename-2-a32b` (#10164). It is now the head, on top of `1f03e873b722188465a9f92b84e5047508b7acd4`.**

- `benchmarks/nci/README.md`: the sentence names the stems, each with its current term in the Glossary's words, at most two examples a stem, and the look-alikes. RAN: each stem and example was grepped at the head.
  - `rho` is the slowdown: `bracket.py`'s `"rho"`.
  - `efficiency` or `eta` is η, the term the slowdown replaces: `tiers.py`'s `efficiency`, `exposure.py`'s `eta`. I first wrote "training efficiency" here, but the maintained-docs test flags it as a retired term, so I said η.
  - `residual` is free compute: `residual_share`, `memory_caps_residual`.
  - `unmonitored` is unregistered hardware's share, ε_unreg: `tiers.py`'s key `unmonitored`.
  - `dampener` or `D` is the slowdown: `fit.py`'s `dampener`, the keys `D_*`.
  - Not older names:
    - `tiers.py`'s `rho_T`, `rho_PoS` and `rho_NCI`, which are rates;
    - `exposure.py`'s `wipe_residual`, the memory the fill misses;
    - `kappa.py`'s `RESIDUAL`, the model's residual stream;
    - `reference_efficiency`, the reference-efficiency assumption, which the README names a few lines above. I added this one beyond the coordinator's list.
- `verity/protocols/compose/verifier/remote/README.md`: "overhead" becomes "slowdown" at lines 28, 396 and 397. Each names `InferenceOnly.Guarantees.SlowdownRemote` or `SlowdownRemotePeriod`, so all three mean the slowdown. They were the file's only "overhead"s, so none was left. READ
- At the head, `tests/lint/test_repo_retired_terms.py`, `tests/test_retired_terms.py` and `tests/test_doc_paths.py`: 51 passed, 7 bookkeeping warnings, none naming either file. RAN

