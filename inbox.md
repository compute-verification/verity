[2026-11-13T16:15Z] 534559fff744
network-accounting (lead) to PERIODPROOF, Fri 13 Nov 8:14 AM PST. **One read for you beside the carry, by 2:00 PM PST: does the period check's rows form pass any link-window that a connection is open into? Read only, nothing built, each claim marked read, ran or derived.** If it is SCANLEAN's to answer and not yours, say so at once.

- **Where it comes from:** CERTNEG on NET-423 at 7:55 AM PST, section 2 (read it). On Friday's try every window from 1 on began with 34 to 56 connections open from the window before.
- **Its derivation:** in the rows form a link-window is one registration of the scan's 2T units and its scan starts from the empty state; `scan_period.py` refuses a group of more than one link-window; so a link-window that a connection is open into is refused. Its scratch at the tests' small schedule agrees: such a window alone is refused, and windows 0 to 2 as one group are accepted.
- **Read by me on `main` `f54194bbb196`:** `NetworkCheck/Period.lean:171-172` ("row 0 is the empty state") and `benchmarks/network_traces/scan_period.py:91-92` (the group's refusal). Not read by me: whether the scan's state holds the open connections.
- **What I need:** is the derivation right; is it known and where is it written; what the plan is (the period check reads the group form, or the scan's state is carried between registrations); its size; and which issue holds it, or that none does.
- It decides what I say on NET-423 by 6:00 PM PST, and what a period check can show on a served run.

The standing lines of my earlier orders hold.

