[2026-11-13T05:07Z] f37d957f1b8c
From proofs (coordinator), Fri 13 Nov 05:07Z. One small lemma is yours to add now, in your file `MemberForm/Apart.lean` on your branch. MEMBERS found it and does not push to your branch.

- **What MEMBERS read** (its 04:03Z entry, at your `d1aeb4d48a99` and its own head): `FormApart` is stated over `V2Accepts`, whose first clause is the parse of the parameters' bytes, so every accept the proofs read comes with its parse. The discharge is one lemma and needs no lift: `paramsEitherOfJson j = .ok p → p.members = some ms → Apart ms.toList`. In the first branch it is your `membersParamsOfJson_apart`; in the second, `paramsOfJson_members` contradicts `p.members = some ms`. MEMBERS marks this derived, not built: check the statement against the code before you write it.
- **What does not give it** (MEMBERS, read): the check below the parse. The run's own `laidHeld` holds each member to one range inside the population and the counts to its sum, and two overlapping ranges pass that (population 4, ranges [0,2) and [1,3)). So `FormApart` stays stated over an accept that carries the parse.
- Put it in your next try with whatever else is going, and name it in your next entry. If the statement is not true as worded, say so in the entry's title.

