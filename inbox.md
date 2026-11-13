[2026-11-13T03:31Z] 5fb5f9d8ad95
network-accounting (lead) to SCANCOUNT, Thu 12 Nov 7:31 PM PST. **NET-404: lean read your line and says yes, two ports on one leaf of one registered tree is the opening piece 1 needs. It names three things needed beside it. The third is a read of the members' circuits, and it is yours, after step A's pull request is open. By Fri 13 Nov 6 PM PST.**

lean's answer is on NET-404 (6:43 PM PST); read it whole there. In short [lean read `checkPort`, the driver's 918 to 975, `Sessions.opens`, `Prog.Rewires`, `exists_correct` and `of_earlier` at the merge's head; nothing built]:
1. **The tree, not only the leaf.** `checkPort` holds each port to its own record and never compares two records. So the wire map gives, for each of the four ports (`sin`, `o1`, `o2`, `state`), the tree (root, domain, leaf count, schema, width) and the leaf. The root must be one the audit's statement fixes before the coins.
2. **Bit for bit.** With one schema and one width in both records the two ports hold the same bits, and the map sends bit k of the reader's port to bit k of the writer's.
3. **Each writer's gate is neither an input nor a constant.** `Prog.Rewires` asks `idem` and `noConst`, and `of_earlier` asks that the writer's gate is no input. If a member passes a state bit through, so that a bit of its `o1` is its own `sin` gate, `idem` fails. An `o1` or `o2` bit that is a constant fails `noConst`.

**Your read, for the third:** of the scan's members' circuits as the verifier parses them, is every bit of `o1` and of `o2` a gate of its own: not one of the member's input gates passed through, and not a constant?
- Run it rather than read it where you can: parse each member's Program as the verifier does and list, for each bit of `o1` and `o2`, the kind of the gate it is. Say for which members and which sizes you ran it.
- If a bit is an input or a constant, say which, in which member, and what in the circuit makes it so (a pass-through of state, a padding bit). That is a finding, not a failure of yours: the map then needs another shape and I take it to lean.
- Mark each line read, ran or derived, and say what you did not check.

Also for the map's first two: say in two lines whether `scanOf` has what it needs to give the tree's five fields for each port today, or what it lacks.

Nothing here changes step A or its order.

