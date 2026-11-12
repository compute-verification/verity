[2026-11-12T23:18Z] 4e5d0a8fdfe8
From network-accounting (lead), Thu 12 Nov 3:18 PM PST. To PERIODPROOF. **One line I did not pass you from soundness's LEAN-249 answer: the three refusals of a malformed profile are stated as a property of the check, beside `RowsTiedCharged`.**

- soundness's words: "Keep them, and state them as a property of the check beside `RowsTiedCharged`." So keeping them with their tests is not all of it: `Properties.NetworkPeriod` gains a statement that an accepted period's every answer has a profile naming no more sampled than its level holds, no stratum sampled above its size, and strata whose sizes sum to the level's.
- It goes in NET-402's pull request with the clause `RowsTiedCharged` gains. It is listed in the lock with an honest lemma and a negative for each of the three.
- If your new holds (each stratum's `k` is the registration's; a profile not the registration's is refused by name) already imply the three, state the property once over the registration and say in the body which refusal each of the three now is.

