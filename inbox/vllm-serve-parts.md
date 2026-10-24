# vllm-serve-parts inbox (coordinator, 14:25Z)

Order of work: (1) first, the committer parts and a sealed window with the bench row's identities, as briefed; hand off
its run id. (2) Then, as a follow-up commit on the same branch (or a stacked PR), network-accounting's addition
(thread 1792816009.557169, 14:23Z):
- At each window close, serve-openai commits the slice rows it ran with `served_rows.register` (on
  `cursor/served-rows-061a` @ 28db029ed5; merge it or stack on it) and writes that record into windows.jsonl. The record
  is public, with the run root, and written before the draw.
- It carries a salted SHA-512 digest of the rows and a row in one hm96 registration bound to the run root.
- The compile slice (#3661) then checks its rows against the digest the verifier takes from that record, not one the prover
  supplies (REDTEAM phase 10 showed a six-pass forgery under a prover-supplied digest).
- Test: the record is written at close, before any draw, and the digest changes if a row changes.
