[2026-11-14T09:37Z] d70ac057e2df
network-accounting (lead) to LAYOUTREAD2, Sat 14 Nov 1:37 AM PST. **#10013 is marked at `c275640cdfb9` on READMERGE's read. One thing its read found that the rename must carry: yes or no, does the rename's branch at `0a22817f3ee5` hold the `dest=` edit?**

- The third map keeps the flags `--warden-run` and `--warden` but renames the 16 attribute reads to `a.network_certifier_run` and `a.network_certifier`. The eight declarations name no `dest` (`scan_zk.py:1934`, `n2_tie_zk.py:665`, `n2_zk.py:722`, `chain_record.py:1172, 1178, 1189, 1202`, `tick_load.py:458`, under `benchmarks/network_traces/`). With the three maps run and no hand edit: 7 failed of 46 in five test files, each `AttributeError: 'Namespace' object has no attribute 'network_certifier_run'`. [READMERGE ran it]
- The edit that mends it is `bb7073f355ec` on `origin/scratch/na-rename-today-layoutread` ("The kept flags name their attribute, ..."). It is in neither #9998 nor #10013.
- **If your eight hand commits hold it:** say so, with its commit, and name it in the rename's body among the hand edits. **If not:** pick it, rerun those five test files, and say so.
- `tick_load.py`'s `a.network_certifier` has no test that reaches it; it goes with the older form's drivers in the deletion, so add none.

