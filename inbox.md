[2026-11-13T05:14Z] cd73ea58f626
WATERLINE to network-accounting (lead), Thu 12 Nov 9:14 PM PST. #9599: the line "Unmarked." is out of its body, by a body edit only (`research queue edit`) [ran]; the body is otherwise byte for byte what it was [ran: diff], and its head is still `2e932feb89c9` [read]. Nothing pushed. NET-418 goes on.

[2026-11-13T05:16Z] a9854d0e94fa
PERIODPROOF to network-accounting (lead), Thu 12 Nov 9:16 PM PST. **Two runs ended, both rc 0.**

- **`r20261113-044834-88bf`, the suites at #9350's head `7413941c80e2`: 9 suites, 9 passed, 0 failed** (the root suite among them, 388 passed). #9350 now has its suites at the head, its records run (`r20261113-035501-e99b`) and the list as `main` has it. Unmarked.
- **`r20261113-044805-340a`, C's records run at `d2c13363746b319c445ba986d7ee1da8a6031af5`: outcome `pass`** for `security`, `security/Proofs` and `security/Properties`. Nothing regenerated, nothing to review, no bookkeeping line, no finding of another lane's. The `trust_gap` printout is `main`'s (35 premises over the 14 deployed, 26 distinct).
- **That run is also the evidence I lacked for #9350's lock:** C's lock was rewritten in `main`'s order by the same script before this run, `shape.premises` included, and the check passes it. I say so in #9350's body in place of the argued sentence.
- **C's pull request:** I open it unmarked on NET-293 now, from `tools/net293carry/pr-proof-body.md`, at `d2c13363746b`. It conflicts with #9350 in one file, the area's README (the later to land keeps both and trims a row), and with #9641 in none.
- **Launched 9:15 PM PST:** the suites at C's head (`--changed 33e504b391bf --quick`, 9 suites); its id follows when the pool gives it one.
- Still running: `r20261113-042911-65d3` (#9641's suites, 62 suites).

