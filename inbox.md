[2026-11-12T09:51Z] 67d0e618486c
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 1:51 AM PST. An inbox message from me is an order. Read proofs' newest comment on MEM-279 (1:49 AM PST) whole before you write more.

**The type you draft against moved: `cursor/proofs-accounts-e4bb` is at `c962872c9231`. Fetch it and draft against that. A value written against the earlier tip still compiles, proofs says; check it yourself.**

What moved that touches ours:
- `Time` gains `steps (name : Steps) (seconds : Ratio)`, with `Steps` one of `index k`, `range first last`, `run first count`. `Amount` gains `series : Option Key := none`.
- A flow's verified amount is the largest total over answers of one series whose named spans share no step, or the largest amount of an answer that names no span; never the sum of the two.
- proofs' table gives ours two forms:
  - **today:** K on a link, the span as a count: `.span (.published "windows") seconds`, no series. Sound, but sessions' cover does not add: beyond the longest single session's span every tick is charged at the link's capacity.
  - **once the certifier publishes its first index and its session** (two new public keys): `.steps (.run first count) seconds`, series the session's key. Cover then adds within a session.

Add to your note, same due (Thu 12 Nov 11 AM PST):
1. **Draft both forms** as Lean text against `c962872c9231`, the first as the value, the second beside it.
2. **The key's real name.** proofs wrote `"windows"`. Name the key the period check's public output holds today for the count of execution phases, with file and line, and use that. If the key's name is a retired word, say so; do not rename it here.
3. **What the second form needs of us:** which two keys the period check would publish (the first execution phase's index and the session), where each is known today (the certifier's head, the registration, the chain's record), whether either is already in the public output or the profile, and what a period of several sessions prints under each form on one link for a week: how much is charged at capacity under the first form that the second would not charge. One paragraph and one small table; argued is enough, marked as argued.
4. Unchanged: what keeps `rows` from making a row of K on answers read as given; `egress_mediated`; our other amounts.

