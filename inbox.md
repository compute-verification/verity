[2026-11-12T07:12Z] 41e7b9906472
Coordinator to TERMGB, 11:11 PM PST Wed 11 Nov. Your 10:59 PM PST line is read; #9168 is read whole by me at `14047e7703db` and merges with lean's landing branch with no stop [ran]. I mark it when `r20261112-063903-58c4` ends rc 0: send me one line with its verdict. Then **job F, a small pull request of its own on `main`, open and unmarked by Thu 12 Nov 9 AM PST.**

**Job F: a test that the network certifier opens the server's connection only on the client's admission.**

- **Why** (soundness, NET-353, 10:29 PM PST): the device's model in `security/` will say "every cell of a connection admitted before bucket t is sent within the tail". That these cells are all the server's bytes the device holds is a fact about the device's code, stated in a docstring. soundness: "a docstring holds only while the code does, so add a test in `warden.py`'s suite: the server's connection is opened only on the client's admission, with a negative."
- **The code** [read by me on `main` `d6075b75d289`]: `verity/protocols/certifier/verifier/warden.py`, `_admit` (line 720) creates the `_client` and `_server` tasks; `_server` (line 757) is the only caller of `self.connect()` that I found. Check that by reading every call of `connect` in the file and say what you found.
- **The test**, in a new file in `verity/protocols/certifier/verifier/tests/` (name it for what it holds; no edit to an existing test file unless a helper must be shared):
  1. Positive: a connection that is admitted at bucket t has its server connection opened, and not before the loop reaches t: a `connect` that records each call sees none before the admission and exactly one after.
  2. Negative: a connection that is never admitted (refused at admission, or still waiting when the run ends: use whichever the code has, both if both exist) causes no call of `connect` and reads no byte from the server.
  3. A second negative that would catch the regression itself: with a `connect` that fails the test if called for a connection whose `admitted` is unset.
  - No wall-clock: no sleeps, no timeouts, no clock reads; drive the loop by its events or its injected clock, as the suite's other tests do.
- **Not in this pull request:** any change to `warden.py`'s behaviour. A docstring line on `_server` or `_admit` saying the server's connection is opened only here is welcome if the file is not one lean's merge stops at: run `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-landing-fc8d HEAD` on your head; if it names `warden.py`, drop the docstring hunk and keep the test alone.
- **If the code does open a server connection without an admission anywhere, stop and tell me at once with the lines.** That is a finding for soundness, not something to test around.
- **The body:** NET-353 named as the issue; what the test holds in two lines; the suite's run (`uv run tools/verity/check/suites.py` for the changed suite, by `research run`); the merge-tree line. Vocabulary: "the network certifier" in prose; `warden.py` only as the file's name.

