[2026-11-19T02:50Z] 4b78973caecc
From your lead: CHANGE to the follow-up. Do NOT touch circuits window_cut.py or its test, in any commit (circuits has one writer in that file tonight and takes TR in itself after your merge). If you already edited it, revert that edit. Everything else stands: merge origin/one-circuit in, resolve the two conflicts, check their builder still builds with TR = TM, report the runs.

