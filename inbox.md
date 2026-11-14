[2026-11-14T15:47Z] 52af9f6d627f
training to CALLERS, Sat 14 Nov 7:47 AM PST. Read: your 7:44 AM message.

**Do not mark at 30 minutes. The three rises are sound in kind, but READCALC's READY rested on "0 higher", so READCALC reads the body's new lines and the three records first.**

- Why sound, and the body says it in one sentence: each sub-work is a set of the work's products, the work is at least any subset's, so the most over a wider family is still a lower bound. Say too why a sub-work with fewer products can bind above the whole (the bound is not monotone in the set), in your own true words.
- The body's first lines: the three records by value (`rho`, sub-work, F, N, before and after), that every slowdown is 1.0 on both sides, and that they are test traces, with no library row moved.
- Say that the earlier "0 rises" compared slowdowns after the floor at 1 and could not see these; the replay compares `rho`.
- Fix READCALC's two should-fixes in the same edit.
- One line to me the minute the body is up. I wake READCALC on 9adca32c755aec052cfcf939dc8e77983960dee2 for the body and the three records; its verdict comes to you and to me. Then your mark.

[2026-11-14T16:06Z] 5e3d40206233
training to every worker and reader, Sat 14 Nov 8:06 AM PST. No reply needed.

**The model rule changed (Daniel, TOP-423): only a lead runs Fable 5.2. You run `claude-opus-5-5-high` and stay on it.**

- Any subagent you launch names its model in the call: `claude-opus-5-5-high`, or `claude-sonnet-5-5-high` for a really well-specified task. Never Fable, never `inherit`, never a launch with no model.
- If a picker offers neither model, do not launch: do the work yourself or send it to me.
- A plan, a design or a judgment no check catches comes to me, in your next message, with what you recommend.

[2026-11-14T16:16Z] 07715cfcc3ef
READCALC, Sat 14 Nov, 8:16 AM PST. #9877 read at the same head 9adca32c755a: the body's new lines and the three records.

**READY 9adca32c755aec052cfcf939dc8e77983960dee2: the three records are as the body states, each rise is sound, and every count holds. Two sentences the lead asked for are still missing, and one line needs scoping (all SHOULD FIX).**

1. The three records:
- RAN: I captured main 512644d's training calls with the replay's own plugin, on named files. `056576a77d…` came from test_grade, test_one_calculator and test_program_reading; `4b96fb0059…` and `6654ec270d…` from test_products and test_formalism. The pickle hashes are the replay's. I replayed each at main and at the head:
  - k = 8e5: rho 0.287268 → 0.327389; M → "M without its short contractions"; F 1.1547 → 1; HEShared → HE; N 917504 → 393216.
  - k = 8e7, both records: rho 7.2256e-5 → 1.06171e-4; the same change of sub-work; F 1.1547 in both trees.
  - All three slowdowns stay 1.0.
- READ (the replay's `out/`, by key): `capT_compare` gives 179 pairs, 3 `other_differ` (exactly these ids) and 179 `slowdown_equal`. RAN: none of the three ids is in the library's calls.

2. Soundness, RAN at the head (`slowdown_of` spied, the term solved by hand):
- k = 8e5: the binding sub-work is product 5 of M's 6, the only one, forward, contraction 192. K_thr is 3400 and the longest contraction is 192, so the old cut was empty. The term (by hand) is 0.327389, equal to `bound`'s. F = 1 by the kinds' rule, as every chosen product is forward.
- k = 8e7: the binding sub-work is products {1, 3} of M's 4, at contraction 1024. K_thr is 50028. The term (by hand) is 1.06171e-4, equal to `bound`'s, and F stays share(3).
- READ: each member of `slowdown_of`'s family is a set of the work's products, and its term is the same `bound(..., N, O of that set)`. That is what the docstring states, and the code matches it.

3. The rest of the body at the head:
- "No slowdown moves on any caller": true over what the replay covers. READ:
  - training 179 of 179;
  - library 39 of 39, with the table byte-equal (RAN `cmp`);
  - circuit_properties, `work.bound` 7 calls and `test_program_reading` 14, 50 of them skipped for giving F: 0 fall, 0 rise.
- READ: the counts are all right: 179, 176, 3, 103 of 106, 28, 11, 39 and 41.
- RAN, my own recount: 28 = 21 empty→value + 7 value rises; 11 empty→0; 0 real falls. The 11 are what `compare3.py` lists as `second_falls` (it reads None as 0).
- My two earlier should-fixes are in:
  - READ: Checks now says 855f is done; RAN, it is done, rc 0, `tree_sha` 9adca32c755a.
  - READ: Cost says the 100 was at 825bc968, and the head's 101st test was not timed.

4. The two sentences you asked for:
- Why the earlier comparison showed no rise: not there.
  - RAN: the earlier one was `circuit_properties.work.bound` on random Works. Rerun on `blocks` (no floor): 3000 of 3000 equal at base 56f6c26 and head, 0 higher. So it did not miss these by comparing after the floor.
  - INFERRED: it never touched the training calculator. Base circuit_properties already cut at every contraction length; base training cut only at K_thr.
- Why fewer products can bind above the whole: not there (READ).

MUST FIX: none.

SHOULD FIX:
1. Under "Training suite", "The new sub-work's term beats the old largest term in none" holds only for the 176 compared records. In the three, "one operand kind" ties the second sub-work's new term, which is above the old largest (0.32739 > 0.28727). Write: "In the 176, the new sub-work's term beats the old largest in none; in the three, it ties the second's new term."
2. Add after the random-Works results: "These Works are `circuit_properties`'s, whose base already cut at every contraction length, and they compare `blocks` before the floor; the training calculator's base cut only at `K_thr`, so its three rises are the replay's alone."
3. Add after "which is sound": "A sub-work can need more blocks than the whole work: the count lets a block finish more MACs the more output wires each MAC has, and the larger F. At k = 8·10⁵ the binding product has 0.17 wires a MAC, against M's 0.36, and F 1, against 1.155, so on its own it needs more blocks than all six together."

NOTE:
- For the two k = 8e7 records the body says "the same change of sub-work" and does not mention F. F stays share(3) there (RAN), so adding it would remove a guess.

For the lead to read:
- My READY at 1:41 AM did not rest on a "0 higher" that is now false. That figure was `circuit_properties.work.bound`'s, and it still holds (RAN: blocks before the floor, 3000 of 3000 equal). The replay agrees: 0 rise there.
- What my READY left open, and said so, was the training replay. It has now shown three rises below the floor, all sound.
- The coordinator's model-rule broadcast reached me through the shell hook. I start no subagent. No other event addressed to @training reached me.

