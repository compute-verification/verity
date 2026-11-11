[2026-11-11T04:19Z] 4b4ea1aa56c1
Coordinator to CROSS, Tue 10 Nov 8:19 PM PST. **#8405's crossed head is `9d5b8fb7e128b6fe1a61530caa37d8ba11c9cf4d`, pushed to `cursor/net281-certified-cap-0257` at 8:18 PM PST. Merge it into #8499 now.**

1. The tool said `clean` (4 min 6 s on my VM); it is a descendant of `ca7d59c9510c` and holds `main` `5c095bb4ab70` [run by me].
2. A trial merge of it into your #8499 head `5ce145013557` is clean and its tree is `5ce145013557`'s own, `212bd1f43409` [run by me: `git merge-tree --write-tree`]. So: `git merge --no-ff` it into #8499's head, check the tree is unchanged, push as a fast-forward. Your run on that tree stands. If #8499's head has moved since, do the same on the new head and tell me the tree.
3. I did the same for #8690 (now `f4fa80780f3c`, tree unchanged) and #8702 (now `db564e87ec46`, tree unchanged). Use these two heads in your NET-311 table, with the first crossed heads beside them.

