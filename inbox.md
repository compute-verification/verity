[2026-11-17T01:45Z] f93a7f7338a8
From the coordinator: lean asks that NO Lean lands on main after 12:00 AM PST tonight (08:00Z Tue), so its renaming pass can be built. #10700 is a Lean change and our goal is to have it on main by 11:00 PM PST (07:00Z). The moment the records run ends and the records are committed, return at once with the moved records (do not wait for 02:30Z); if the run is still going at 02:15Z tell me its state then.

