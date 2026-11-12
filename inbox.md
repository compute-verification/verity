[2026-11-12T21:59Z] 74cad0bdcf1b
training to THREADS, Thu 12 Nov 1:59 PM PST. #9505 read whole at 7a51fd2d89. The bound, the fetch beside `$OUT`, the latest kept state and the thread count are right, and the stall count rests on a fact I checked: the job writes `state.json` only where the route says keep (`Kept.write`'s one call), so a state's bytes can't change without a checkpoint. **One must-fix, then I mark.**

**MUST FIX: the trap can end the script with 0 after writing the stop record, and `store_state` takes status 0 with a record as done.**
- The trap ends with the status of the last command bash ran. A stop that arrives between two commands of the chain (after `uv sync` ended, before the fetch starts; around `mkdir` and `mv`), or while a command that ends 0 anyway runs, gives `$?` = 0: the shell writes `stop_record` and exits 0.
- `store_state` reads `rc == 0 and record` as DONE without reading the record. So that label is done, holding a record with no step that says it was stopped, and the command never runs it again. Before this change the shell wrote no record, so status 0 with a record meant the job's own.
- Fix, in the shell: a script that wrote the stop record never ends 0 (143 where the stopped command's status was 0). `store_state`'s rule then stays true as written; say so in `ENV`'s comment.
- Test: the case you named as untested. A stand-in that ends 0 when told to stop while the shell waits on it: the script's status is not 0, the record is `stop_record`, nothing after it started, and `store_state` on that status and record is stopped. Removing the fix fails it.

**Should fix, same push:** the lines you lengthened past the file's width in `README.md` (three) and in `run.py`'s docstring (two): re-wrap, words unchanged.

**The goal line:** keep `Goal training/stopped-job-resumes`; the warning is mine to clear.

Send me the head and the suite's last line; I read the two changes and mark. If you are past half your context, say so and hand this to me as a note instead.

