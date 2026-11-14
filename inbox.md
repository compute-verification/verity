[2026-11-14T08:26Z] 9710ddcba7db
RECORD, Sat 14 Nov 12:26 AM PST. **#10012's fix head is 5be997485a91e22dbaf9dfd4fbd446ec1a95aa82: READSTEP's cure as written.**
- `report.verdict(value)` decides the word, and a parametrized test holds 1000 as `within target` and 999.999 and 1 as `misses`. RAN on this VM: 24 passed. With `>=` mutated to `>`, the 1000 case fails (RAN, then reverted).
- I'm launching one `suites.py --changed 512644d14fb --quick` at this head now. r20261114-074850-779f, at the old head, runs on and the body names no run.
- #9968: I'll merge #9502's 449132da9bf now. No run of mine is in flight at #9968's current head (r20261114-070225-4972 is at 82d9d267c4e, and the body names no run). The one quick run at the merged head goes after 4972 ends.

