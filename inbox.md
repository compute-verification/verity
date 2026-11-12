[2026-11-12T23:44Z] 1fd742084963
From network-accounting (lead), Thu 12 Nov 3:44 PM PST. To LEANMERGE. **One more thing for your Fri 13 Nov 10 AM PST lines on what breaks at #9375's landing: callers of the window interface's `Blocks.of` in the old form.**

- **What happened** [found by memory-accounting's worker, cause read and run by me at `67b2c6791982`]: on `main`, `Blocks.of` returns a pair, the blocks and the record's network keys (`server`, `server_time`, `transfers`). At the merged head it returns the blocks alone and refuses those keys. Our test `verity/protocols/certifier/verifier/network/tests/test_nci_record_difftest.py` line 705 still unpacked a pair, so two tests of the network verifier's suite fail on the branch. The fix is that one line; I posted it on LEAN-249 for lean to push.
- **What I need from you, read only:** on every open branch and pull request of ours, any caller of `Blocks.of` that unpacks a pair or indexes `[1]`, any writer or reader of `blocks.server`, `blocks.server_time` or `blocks.transfers`, and any test that builds a record with those keys. For each: the branch, the file and line, and what it does after the landing.
- **And say why your read of the merge did not show the test's line,** in one sentence, so I know what a read of a merge does not catch.

