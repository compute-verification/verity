[2026-11-13T03:24Z] 951252239e66
infra (your coordinator) on INF-775, Fri 13 Nov 03:26Z. The cause is found, by proofs' worker [told to me by proofs on INF-767 at 02:46Z; not run by me]: the two tests fail on vy-nebius-2 while the file /etc/vy/auto-hold exists there. The probe they start reads auto.HOLD_FILE and answers "held", and the test dies with KeyError: 'error'. It is neither CHECK_SLOTS nor /etc/vy/runs-via-slot.

proofs has the two-line cure in pull request 9596 (tests only: each of the two tests points auto.HOLD_FILE at a path that does not exist, as the test near line 705 does). proofs marks it itself. Do NOT write those two lines again.

What is left of INF-775 for you:
1. Show the cause on your machine: with auto.HOLD_FILE pointed at a file that exists, each of the two tests fails with the node's words. Say so with the command and the output.
2. The one fixture the issue asks for: every test of tools/research/tests/test_auto.py that starts the probe runs free of the machine it is on (HOLD_FILE, and whatever else the probe reads from its machine: list what you find), by ONE fixture and not a copy in each test. Build it so it merges cleanly with 9596's head (read that head first: `gh pr view 9596 --json headRefOid,files`), or on top of it, and say which.
3. If, with 9596, nothing else of the probe reads the machine, say that and hand back the small change or none.

The due date and what to hand back are unchanged.

