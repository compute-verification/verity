[2026-11-10T16:15Z] b1b84db716eb
From the coordinator, Tue 10 Nov 8:15 AM PST. **Your branch will not cross the layout as it is: it adds `verity/core/protocols/certifier/connection_rows.py`, a new file under a directory the layout retires file by file, and the map has no row for it** [CROSS's dry run of the day's commands on your head e42dd30d70 against top's staging: `script-failed`, "tracked files under a retired path with no row in the map"]. The map is closed, so no row is coming.

- **The mend, on your branch, when you cross** (this is what CROSS does for #8459, which has the same fault with `continuation.py`): in a commit before the crossing, `git mv` the file to where its siblings go, `verity/protocols/certifier/verifier/connection_rows.py`; cross; then fix its importers by hand on the moved tree (the map rewrites no import of a module it has no row for) and run your suites there.
- CROSS writes its commands for #8459 in its table at 10:30 AM PST; I send them to you then. If you reach the crossing first, do it as above and tell me what you did.
- Nothing changes before the layout is on `main` (about 11:15 AM PST): finish step 1 as briefed.

