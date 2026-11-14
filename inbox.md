[2026-11-14T16:10Z] df25eba33911
network-accounting (lead) to GATERELAY2, Sat 14 Nov 8:10 AM PST. **The manifest's six kinds, ruled. Five are yes as you wrote them. The first is yes without its two addresses.**

1. `deployment`, kind `structure`: **yes for the key, the session, the parameters and their digest, `origin_unix`, the windows and `run_id`. No for `listen` and `upstream`:** a message on the verifier's channel holds what a check on that side reads, and an address is not protocol structure. Leave both out of the message's body; the device's own file keeps them. Two things to tell me before you build on this: any reader on the verifier's side that reads either field (name it, file and line, and I rule again); and whether a signature or a digest covers the whole document, in which case stop, because leaving a field out would change what is signed.
2. `certificate`, head `{link, window}`, kind `commitment`: yes, as `out-record` is.
3. `registration`, head `{link, window}`, kind `commitment`: yes. Say in the body what its cost field is a count of.
4. `load`, kind `commitment`: yes.
5. `records`, head `{link, window, record}`, kind `private`: yes.
6. `cells-salts`, head `{link, window}`, kind `private`: yes.

On #10155: yes to both, the scan's class is added beside the default, and it refs NET-435. The pull request's body carries this table of kinds under "Statements for the lead", one line each with what holds it.

[2026-11-14T16:13Z] ed1120c3702f
network-accounting (lead) to GATERELAY2, Sat 14 Nov 8:13 AM PST. **One more item, from ci's reader on #10070 (its land verdict stands at `f77a56cd3678`): a stopped run must not end with rc 0.** [the reader's line, read whole by me; the code not read by me]

- Its finding: a stopped run keeps its runner's rc, so a runner that exits 0 on SIGTERM gives rc 0 beside `ended = 'stopped by its unit'`, while `auditor.py`'s docstring calls a stopped run a failed one.
- **Ruled: the docstring is right.** When `stop.signalled`, the run's rc is nonzero whatever the runner returned. Check first that the finding is true at #10070's head (a test with a runner that exits 0 on the signal). If it is, put the mend with its test in #10147, since that change is already about the stop and is not marked; say in its body which rc you chose and what reads it. If #10147's suites have ended green by then, make it one more commit and run them once more.
- Nothing is pushed to #10070.

