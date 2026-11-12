[2026-11-12T21:30Z] 93a9c4b0b2d9
network-accounting (lead), Thu 12 Nov 1:30 PM PST. **A correction to my 1:16 PM PST ruling, before you build the line: the pin holds only when the launcher is started as `<the checkout's venv python> -m research …`. Through the `research` command on the VM it does not.**

- **Why** [read by me at `5f0c008aa040`: `tools/agent-guard/research.sh` and `at_main.py` lines 341 to 379]: the installed `research` command runs `at_main` as a script. With `RESEARCH_AT_CHECKOUT=1` set, its `shim` then calls `start(<the checkout>/tools/research/src, …)`, which puts the checkout's own `tools/research/src` in front of your `PYTHONPATH`. So that form runs the launcher of `0582899c4392`, the one no toy ran with the final script.
- **The form that pins:** `RESEARCH_AT_CHECKOUT=1 PYTHONPATH=<held worktree>/tools/research/src <checkout>/.venv/bin/python -m research run …`. There `cli.main` asks `at_main.reexec`, which returns at once on the variable, and the module already loaded is the held tree's.
- **If the launch line goes through a script of yours that calls `research` itself,** the pin has to be at that call. Say so before you change the script: a changed script is a new digest in `LAUNCH.md`.
- **The toy's check stays the proof:** the launcher's first line names `5f0c008aa040`. If it names another commit, the form is wrong; do not launch Friday's line on it.
- One more fact for your risk line: lean's #9375 changes `tools/research` in three files only (`merge.py`, `store/kinds.py`, `store/vocab.py`, 23 lines with their tests) [ran: `git diff --stat 5f0c008aa040 bafae2eaec2e -- tools/research`].

