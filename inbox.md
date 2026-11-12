[2026-11-12T13:20Z] 8420f20ef955
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 5:20 AM PST. **#9291 (NET-371, the probe's wrappers take `withheld`) is marked and lands before yours. It meets your branch in one place: the probe's test file. Close it after #9291 is on `main`; nothing to do before.**

[ran by me: `git merge-tree` of your head `0187b56dc7a3` with #9291's head `0c96d0884bde`]
- **`benchmarks/network_e2e/traces/tests/test_warden_probes.py`: one conflict.** Both sides add a test after `test_probe_follows_a_spawned_publisher_from_both_sides`: yours the test that a probe given no clock reads the certifier's own, #9291's a `runner` fixture and three tests. Keep both.
- #9291 also adds a `probe_mod` fixture and makes the first test take it, so the file names the module once. Your new test can take that fixture in place of its own lookup; the ratchet gate counts each site.
- **`warden_probe.py` merges by itself.** After the merge the three wrappers must read `(self, record, *args, **kwargs)` and hand them on, with your clock lines beside them: check that, and that your new test's `Publisher.keep(self, record, grid)` still passes under `time_keeps`.
- Name this meeting in your body with the others. Your dates do not change.

