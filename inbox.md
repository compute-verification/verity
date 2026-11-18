[2026-11-18T04:32Z] 0ae06568ba05
From the proofs lead, a CAUTION on #10995: ci posted at 04:26Z (Linear TOP-435) that #10995 is 'back in a candidate with #10884 (275efdd8fd2e)', while the queue's status at 04:31Z says it is refused for a conflict with main. Before pushing to #10995's branch, find out which is current (queue status; ci's newest comment): if it is in a candidate being checked, do NOT push: tell me its state instead. The same check for #10966.

