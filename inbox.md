[2026-11-10T17:13Z] f039cba2e40f
From the coordinator, Tue 10 Nov 9:13 AM PST. **The hand mend for your branch at the crossing, from CROSS (NET-311), rehearsed on the eb0b staging on a scratch copy of e42dd30d70 (its 5c7f4baf96, not pushed: the branch is yours).** Do it when the layout is on `main`, not before, and tell me the crossed branch's name.

- On the branch's head, BEFORE `git merge BASE`:
  `mkdir -p verity/protocols/certifier/verifier && git mv verity/core/protocols/certifier/connection_rows.py verity/protocols/certifier/verifier/connection_rows.py && git commit -m "connection_rows.py goes where the layout puts its siblings, ahead of the move"`
- Then top's two commands (TOP-394: `--dry-run --json`, then `--push-to` a new branch). Nothing to fix after: the move's script rewrites the importers itself (`from verity.core.protocols.certifier import X as C` becomes `from verity.protocols.certifier.verifier import X as C`).
- What CROSS saw [its run]: crosses clean in 4.5 min; both importers rewritten; `pins.py --stale` exit 0; the result holds no `core/protocols/certifier` beyond the staging's own (`git grep -c 'core[./]protocols[./]certifier'` on result and staging, equal). Check the same two on your crossed branch, on the real base, before you run suites there.
- If your head has moved past e42dd30d70 with another new file under a retired directory, the same `git mv` applies to it: the map is closed and has no row for a file `main` does not hold.

