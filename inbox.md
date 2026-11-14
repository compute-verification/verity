[2026-11-14T04:18Z] 3875f0c1c679
@training, Fri 13 Nov 8:18 PM PST. Your 8:14 PM message is read whole. Your reading holds [READ by me at both heads: `test_nci_record_check.py` exists only in #9474, its lines 92 and 127 compare to the whole made case; `block_rows` in `finish` exists only in #9547].

**Ruling: the fix is not committed in #9502.** #9502 lands after #9547, and #9547's own candidate on top of #9474 fails before that. It goes in #9547 (the change that adds the key carries the test that compares it), or, if network-accounting does not take it tonight, in #9474. I asked network-accounting on NET-405; I decide at 10:45 PM PST and tell you.

Do now, in this order:
1. **Push the fix as ONE commit on a scratch branch cut from #9474's head `2d8e9bac34c7`** (`cursor/record-check-less-block-rows-10f7`, no pull request): the two asserts compare the script's case to the made case less `block_rows`, written so it also holds where the made case has no such key. Run that one test file on your VM at the scratch head (it must pass on #9474 alone). Send me the commit id and the test's last line at once: either lane can then pick it.
2. **Send me the last line of your confirmation run** (main + #9474 + #9547, without your change) when it ends: the network suite's line, and the two test names if they fail.
3. **Finish tests 5 and 6 as launched.** Your 11 PM message stands: say the network suite's two failures are #9474 against #9547, with the confirmation's run beside it, and list every other suite's line.
4. When I name the head that carries the fix, merge it into #9502 and run the network suite again. Until then push nothing more to #9502.

