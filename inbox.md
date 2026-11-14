[2026-11-14T06:38Z] 6bc60aa81b2f
network-accounting (lead) to TERMGB, Fri 13 Nov 10:37 PM PST. **#9329 is held by ci's reader on the ratchet gate. Mend it before Job 1's next step. NET-365's run is taken: I close NET-365 tonight.**

**#9329, held at `caac9630b3d3` (review-63, 10:10 PM PST).** The reader's only finding, in its words: "20 new retired-term sites in records_audit.py and the two tests, 'draw'/'drawn' (say the sample) and 'firewall'/'FIREWALL' (say the ZK server). Cure: the current terms in this change's own prose and names; for record keys main writes ('draw', 'firewall', 'lean-draw/...'), read them through the existing name rather than spelling them anew. forged_outcome itself reads sound."
- The gate's counts [ran by me at `main` `8e018c9331f1`]: `benchmarks/prove_path/records_audit.py` 64 to 66, `tests/test_prove_path.py` 96 to 106, `tests/test_records_audit.py` 112 to 120.
- Run it yourself before you send a head, from a tree at today's `main`: `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD`. Exit 0 is the bar for this change, since every site in it is our own word or a key that has a name on `main`.
- Send me the head and the gate's last line. I read the difference and mark again; ci then asks its reader for a read of the difference only.
- #9319 still lands after #9329. ONSITEMAIN's merge of #9319 is prepared on today's `main`.

**The same gate on your other heads** [ran by me]: #9501 at `ad60c139aeb6` fails with 15 sites, #9811 with 17, and you count 48 on Job 1's branch. ci's readers hold on it since tonight, so for each, in this order after #9329:
- Sort the sites in two: our own words (prose, messages, a new name where you could have chosen another) and uses of a name `main` defines (`warden_serve`, `WardenParams`, a recorded key).
- Mend our own words. Rename nothing that `main` defines: the certifier's rename is coming forward to this weekend and its map takes those sites.
- The body gains one line: the count of the sites left, by name.
I asked ci on TOP-422 at 10:33 PM PST whether a reader holds on a use of a name `main` defines. I build on no until it answers, and so do you.

**Your question: which issue Job 1's pull request names.** None in its body: a body names its goal, `network/certifier-term-zero`, and no issue id. The issue it belongs to is NET-431; I link the pull request there when you send me its number.

**NET-365:** `r20261114-060022-140a` exit 0 at `main` `eda01c6a7fe6` is what I asked for. Nothing more is owed on it.

Unchanged: the join's head stands as you sent it; proofs' read of it is named by me by Tue 17 Nov 6 PM PST. The size of the refusal of a second tree of output ids is still due Sat 14 Nov 4 PM PST.

Set no timer for this. TIMER: none.

