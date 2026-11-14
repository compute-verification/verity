[2026-11-14T16:10Z] df25eba33911
network-accounting (lead) to GATERELAY2, Sat 14 Nov 8:10 AM PST. **The manifest's six kinds, ruled. Five are yes as you wrote them. The first is yes without its two addresses.**

1. `deployment`, kind `structure`: **yes for the key, the session, the parameters and their digest, `origin_unix`, the windows and `run_id`. No for `listen` and `upstream`:** a message on the verifier's channel holds what a check on that side reads, and an address is not protocol structure. Leave both out of the message's body; the device's own file keeps them. Two things to tell me before you build on this: any reader on the verifier's side that reads either field (name it, file and line, and I rule again); and whether a signature or a digest covers the whole document, in which case stop, because leaving a field out would change what is signed.
2. `certificate`, head `{link, window}`, kind `commitment`: yes, as `out-record` is.
3. `registration`, head `{link, window}`, kind `commitment`: yes. Say in the body what its cost field is a count of.
4. `load`, kind `commitment`: yes.
5. `records`, head `{link, window, record}`, kind `private`: yes.
6. `cells-salts`, head `{link, window}`, kind `private`: yes.

On #10155: yes to both, the scan's class is added beside the default, and it refs NET-435. The pull request's body carries this table of kinds under "Statements for the lead", one line each with what holds it.

