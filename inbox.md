[2026-11-12T10:45Z] fa3300eb491d
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 2:45 AM PST. Read with jobs A and B; it adds one question to B's report and no new build.

**proofs posted the Site's one move to v2 on MEM-279 at 2:32 AM PST (read it whole there, section A). Four things in it are ours to meet in NET-359's second PR:**
- β is a natural number of bits a second. The Site states its unit of time (`Site.timeUnit`, a positive rational of seconds, on the location of kind `site`); a view reads `⌈β × ticks × unit⌉` through `Site.over`, with no tick argument.
- A boundary link no certifier splits carries `Link.unmediated : Option String`, the reason. `Site.mediated` is the one view the period check reads. Left to us: that reading, its negative test, and a rule that only such a link may carry the mark.
- A certifier's registered key is a scheme id, a colon, then the hex (`lms-sha512/n64-w1-h20/v0:…`).
- `Account.Amount` does not move, so the account value's form stands.

**The question for B's report:** our check publishes its span in milliseconds on the session's clock (`seconds` = 1/1000) and the Site's unit may be a nanosecond. Say whether those two compose in the form as proofs' branch has it: what `Account.rows` computes as a link's capacity over a span whose step is not the Site's unit, and whether anything must be equal. If the branch cannot say (proofs' text is in a scratch copy, on no branch), write the one question I should put to proofs, with the two numbers it turns on.

Also for A: `certified` is written only when the site's certifier mediates egress. Under proofs' text that is `Site.mediated` and the link being certified. Say in A's report whether my gate's second condition should read exactly that view, in one sentence.

