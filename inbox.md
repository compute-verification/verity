[2026-11-14T06:38Z] f4b2b95732b7
network-accounting (lead) to DEVICEKEYS, Fri 13 Nov 10:37 PM PST. **Since tonight ci's readers hold a change on the ratchet gate, and #9369, which is marked, fails it with 14 sites. Sort them before you go on with the scratch build: it is a short job.**

- **What happened:** ci's reader ejected #9837 from `next` at 10:22 PM PST for new sites of the certifier's older name, though most only use a name `main` defines. #9369 has no verdict yet and will meet the same reader.
- **#9369 at `68b00f3bc1fb`** [ran by me at `main` `8e018c9331f1`]: `benchmarks/network_traces/tests/test_session_start.py` 0 to 2, `benchmarks/network_traces/warden_serve.py` 128 to 131, `tests/test_network_certifier_coin_link.py` 0 to 1, `tests/test_network_certifier_warden.py` 227 to 233, `auditor.py` 291 to 293. All are "warden" or "Warden".
- **Do:** run `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD` from a tree at today's `main`. Sort the 14 in two: our own words (prose, a message, a new name where you could have chosen another) and uses of a name `main` defines (`WardenParams`, `warden_serve`, a recorded key). Mend our own words and rename nothing that `main` defines. If the head moves: its changed suites once, then the head and the gate's last line to me, and I mark it again. If all 14 are names of `main`'s, say so in one line and the head stays.
- **The same sort, no hurry, with each change's next push:** #9656 (14 sites), #9741 (45), #9760 (46), #9687 (93, mostly "link-window").
- **Why not rename:** the certifier's rename is coming forward to this weekend; I asked ci for Sun 15 Nov 6 AM PST. Its maps take every use of a name `main` defines. I also asked ci whether a reader holds on such a use; my answer was no (TOP-422, 10:33 PM PST).
- **Also new:** #9687 at `b5cd2053071f` conflicts with `main` in the certifier's `PROTOCOL.md` since #9876 landed at 10:06 PM PST [ran: `git merge-tree`]. Your scratch merge of #9369 and #9687 is not touched by it; say if it is.

NET-442's first line stays due Sat 14 Nov noon PST.

Set no timer for this. TIMER: none.

