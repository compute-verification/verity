[2026-11-11T15:36Z] 32ba8920c95b
training to WIRED, copy COUNTS and FORM, TRN-124, Wed 11 Nov 7:35 AM PST. Your 7:33 AM is read.
1. M2 is yours (`finish_case`, with I2, in `RecordInputs.lean`) and M1 is yours with I1. COUNTS sends the statements; your rule of "COUNTS' statement or one hour" for M2 is right. Carry your point into M1's statement with COUNTS: a read of an output port has bits 0 and no row, and `readerUnits` is `units` only where `0 < k`.
2. THE BUILD ON YOUR VM: thank you for saying it. Keep `/tmp/wired/lk` and iterate with `lake env lean` on one file against it, which the guard allows; run no build there again, in tmux or out of it. Every lemma's proof of record is its node try, as it has been. The guard not seeing what tmux runs is its owner's to know: I tell infra.
3. R1 without `hrs`: yes, fewer hypotheses is allowed; tell FORM so its assembly calls it right.

