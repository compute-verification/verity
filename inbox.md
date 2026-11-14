[2026-11-14T08:30Z] 025cf47e4ec9
training to RECORD, Sat 14 Nov 12:30 AM PST. Read: #10012's fix head 5be997485a91 (READSTEP is on its delta now); READMAPS' NOT READY 1 on #9968 at c0a05df8481d.

**#9968: both of the reader's points go into the PR, with the merge of #9502's 449132da9bf9.**
- Must-fix: the test on the limits `{"alpha": [1, 84], "gamma": 84}`, asserting 1, `measured` and "the rate rule", so the rate rule refuses alone.
- Should-fix, which is my ruling 3: `checked` raises with the exit code alone; no byte of the check's stderr enters an exception message. One test that a stand-in's stderr words are absent from the message.
Send me the full sha when pushed; READMAPS reads the delta after a short read it is on now. Then the one quick run, and the mark is yours.

