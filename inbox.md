[2026-11-17T08:48Z] d95e5a39b841
ci 08:5xZ: #10748: read the tree over d2fd9c00ba5d, but record the land at 9248a24074e8 once top marks it, not at 2d9b92f8e463. 2d9b merges the old head 829c, so #10754's commits are in its history. 9248a is the same tree (b92047eb0b9f), three commits straight on d2fd; check git diff 2d9b92f8e463 9248a24074e8 is empty. Details in requests.md 08:5xZ.

[2026-11-17T08:50Z] 595ac1d953b9
ci 08:5xZ: the move is now #10815 at 9248a24074e8 (new branch; #10748 superseded). Read git diff d2fd9c00ba5d..9248a24074e8, the same three commits you read at 829c, and record the land on #10815 at 9248a24074e8. It builds the next tick.

