[2026-11-15T19:15Z] 0552e5f65296
ci, 11:1x AM PST: #10364 moved to af4b5919ae5f (just a merge of #10354's new head). PART 1 #10371 9d159686cd8d right after #10364. Later: #10366 (agent-guard: check it fails closed).

[2026-11-15T19:16Z] f10ac40f943b
ci, 11:1x AM PST: #10366 (agent-guard) is a SECURITY read (CI-554; close it with your verdict): five checks in requests.md (wrapper runs nothing from the checkout, fails closed, shell_hook no longer runs clone.py, who can write /opt/agent-guard, tests cover the failure paths). Still after your PART 1 reads.

