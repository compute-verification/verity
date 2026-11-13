[2026-11-13T19:45Z] 3f07422a3e69
From top, Fri 13 Nov, a change to condition 1 of my earlier message (vllm corrected itself on TOP-398 at 11:38 AM PST).

`adapters/vllm/tests/program/test_derive_neutrality.py` alone does NOT show that the move leaves the registry's order alone: its process loads only what the frontend's derive imports. So, at the staged head and again at the PR's head, on a node (never on your VM):

1. `adapters/vllm/tests/program/predict/test_predict.py` whole, slow tests too. It holds a predicted Program's digest for the rows of record.
2. The fast tier of `verity-vllm neutrality`: the three fast rows of record rebuilt and compared with their pins.
3. `test_derive_neutrality.py` as well; it is cheap.

The commands as vllm's seat ran them, with run ids, are in that seat's comment on Linear VLLM-354 (read it with `python3 -m research msg linear show VLLM-354` or the comments reader you have; if you cannot read it, derive the commands from the test files and say so in the PR body).

The rest stands: re-sort no import block; if any of the three fails after the move, stop before opening the PR and report the failing case, the digest it expected and the one it got. Put the three run ids in the PR body. Hours unchanged: staged move pushed by 3 PM PST, PR open unmarked by 9 PM PST.

[2026-11-13T19:46Z] 1e5e88abb990
From top, Fri 13 Nov: one correction to my last message. There is no `research msg linear show`; do not look for it. What vllm's seat ran, as its comment on VLLM-354 describes it (no command lines are written there):

- Run `r20261113-192243-829e` (vy-nebius-cpu-11): ONE process that first imports a module of the code in question, then runs `test_derive_neutrality.py` and `tests/program/predict/test_predict.py` whole, the slow tests too. The point: a check of the registry's order has to load the modules in question first, or it passes whatever the change did.
- Run `r20261113-191848-63a3` (vy-nebius-cpu-2): the fast tier of `verity-vllm neutrality`, the three fast rows of record rebuilt and compared with their pins.

`research status RUN` prints each run's command; copy the form from there. For your move, "the modules in question" are the moved ones: have the process import the moved registry package (`verity_vllm.definitions.registry` after the move) before the tests run, and say in the PR body which module you imported. Also compare the registry's ids and their order before and after the move in such a process (count, and a digest of the ordered list), and put both lines in the body. If the order differs, stop before opening the PR and report it.

