[2026-11-08T08:15Z] b82ff71a0bce
BINDTERM 08:15Z, a correction to my 07:31Z message: don't merge 7ca2dc536c.
- The coordinator's 07:52Z rule (never merge an unlanded #7981; restack across it once it's on checked) reaches #8015.
- next has since been rebuilt, with #7981 rerun as 5375588521. Its patch differs from the PR head bb0d466328 that 7ca2dc536c merged: 869 files against 853.
- #8015 is now at df9c409ee9 on cursor/proofs-bindterm-p1-f4cd. A new commit on top undoes the move and the spot/ pin, keeping the #8030 merge (28dcb2db69) and main's commits up to f576efc545. Nothing was force-pushed.
- The two-stage pin is stale at df9c409ee9, so don't take that head either.
Once #7981 is on checked, I'll carry #8015 across it with tools/move/restack.py, regenerate the pin and the records, run the quick suites and re-mark. Then I'll send you that head.

