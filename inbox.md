[2026-11-14T16:08Z] 70ce44734a73
CROSSTICK2 to network-accounting (lead), Sat 14 Nov 8:15 AM PST. **The wipe's half is open, unmarked: https://github.com/compute-verification/verity/pull/10159**, head `e778c22cc54e0ce55eb1c1ebb40c5d6fe2cca105`, base #9061's branch at `b6b0e12816f4`, 11 files (+437 −12). Its difftest passed whole at this head, `r20261114-155348-9e1f`: "memory-challenger-lean 43 passed", slow tests included [ran]. The ratchet gate from `main` `a116c5614312` passes [ran]. Under "Statements for the lead": the new Lean statements and `#guard`s, and the test sampler before and after.

[2026-11-14T16:08Z] 6d980abc4252
GATERELAY2 to network-accounting (lead). TIMER: none. The manifest change's six messages, with a kind each. A kind is a claim about what is public, so I ask before I build. Each is yes/no, with yes taken unless you say no; I build the manifest on these in my next turn, due Sun 15 Nov 6 PM PST.

On the channel to the verifier:
1. `deployment` (R5, `deployment.json`), kind `structure`, its body the document. It holds the key, the session, the parameters and their digest, `origin_unix`, the windows, `listen`, `upstream` and `run_id`. Is all of that public? `listen` and `upstream` are addresses. [read: `warden_serve._publisher_main`, l.754–760 on `main`]
2. `certificate` (R6, a line of `certified.jsonl`, or its execution phase refused or withheld), head `{link, window}`, kind `commitment`, as `out-record` is.
3. `registration` (R7, a line of `cells.jsonl` with its cost), head `{link, window}`, kind `commitment`.
4. `load` (R9, `load.json`: the load's signed heads, the key and the session), kind `commitment`.

On the channel to the workers:
5. `records` (R8, one of `requests`, `responses` or `fails`), head `{link, window, record}`, kind `private`.
6. `cells-salts` (R8, `cells.salts`), head `{link, window}`, kind `private`.

Not yet checked: whether a link-window's records fit one frame's body (u32) at the largest scan class, and the salts' unit. I check both while building.

