[2026-11-12T13:00Z] 39b644abaccb
WATERLINE, from the network-accounting coordinator, Thu 12 Nov 5:00 AM PST. **Your plan for NET-372 stands: build on it. Six things with it.**

1. **The kept run is not a served run, and nothing calls it one.** It is the stand-in front, the certifier and the load on one machine's loopback. The body and the README say that in those words. I change the issue's last condition to match. A capture on a real link between two machines is a second step that I ask of infra; it is not yours.
2. **Sequence numbers wrap at 2^32.** Count a direction's stream across a wrap, or refuse a flow that wraps and say why. Either way, with a test.
3. **Pairing by byte counts:** print how many flows paired and how many did not, each side. Two connections with equal counts pair as a multiset and no line says which is which. The word "agrees" is for the sums and the multiset, never for one connection.
4. **A hole is never agreement,** as you say. The table has a column for holes in bytes, and a direction with a hole prints the difference as a bound, not a number that looks exact.
5. **Refuse what you do not read, by name:** pcapng with the conversion, IP fragments, any encapsulation or link type outside your list. Do not read pcapng now. I ask what the operator's tap hands over.
6. **`Goal demos/onsite-demo`** in the body: the reader is for the operator's site.

**Addresses:** the fixture's are documentation addresses, as the issue says. The kept capture goes to the evidence store only, and no address of ours is printed in the table you put in the body.

Your date stands: the pull request open and unmarked by Fri 13 Nov 6 PM PST. No answer needed.

