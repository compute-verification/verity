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

## 15:05Z (coordinator): one more item for the served-rows follow-up
PROVE-PATH's tie unit #3745 (and IoTieRequest) read each step's committed stream in this layout: the step's input ids first,
padded to the step's token budget, then its positions, padded the same way. Today the eager stream starts with the ids
unpadded and the positions sit elsewhere. Make serve-openai's committer write that layout, on the same branch as
`served_rows.register`; the exact interface is in PROVE-PATH's lane note (network-accounting's lanes, prove-path). Keep the
declared Commit of record's layout unchanged unless PROVE-PATH's note says both must move; if so, say so in the PR body.
Also: G-A's prove-path stage PASSED on your sealed w5 (r20261024-141912-8eb1).

## 15:35Z (coordinator)
#3747's head is now a1e9862964: served-rows @9c90546165 merged in, and PROTOCOL.md set to main's (the served-rows bullet lands with network-accounting's PR). Stack your third PR on a1e9862964 (merge it into your branch). Don't touch verity/core/protocols/certifier/PROTOCOL.md: it conflicts with next until drive rebuilds next on the current main.
