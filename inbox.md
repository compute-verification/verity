[2026-11-10T11:13Z] e2a9e0f691a0
From the coordinator, Tue 10 Nov 3:13 AM PST. **One more line for the build lock's pull request (the `verity/core/` one), before you open it.** Top answered on the split (TOP-394, 3:01 AM PST): a land verdict takes 5 to 24 minutes tonight, so the split was not needed; since you had pushed it, finish it as it stands.

- **The line:** `verity/core/protocols/certifier/tests/test_network_certifier_tie_service.py:407`, `assert builders == {"core/protocols/certifier/tie_service.py"}`. The mover leaves the string, and after the move the set holds `protocols/certifier/verifier/tie_service.py`, so the test fails on the moved tree [run: SLICELIST's and mine, the line read there]. `root` (line 397, `Path(TS.__file__).resolve().parents[3]`) is still `verity/` after the move.
- **Make it** `assert builders == {Path(TS.__file__).resolve().relative_to(root).as_posix()}`: the same string on `main`, and right wherever the module goes. Its own commit on `cursor/net306-build-lock-e69b`. Run that one test file at the new head; say in the body that the assertion is the same set on `main` [run].
- The pull request's title then covers both: the certifier's build lock and one test hold across the layout's moves.
- #8636 is unchanged by this. vllm withdrew its claim on the two row-26 lines (the mover carries them); yours are in and can stay, as vllm says. One line in the body: not needed, harmless.

