[2026-11-12T07:34Z] 4adac3609746
ONSITEMAIN to network-accounting, Wed 11 Nov 11:34 PM PST. NET-362: pull request #9188 is open, unmarked. https://github.com/compute-verification/verity/pull/9188

Branch cursor/na-registered-read-6a09, head a8d956c8dd1415b37d3d5f5c5698c1eb7381afbb, from origin/main d6075b75d289. [ran]

Signatures since 78d05f1b5135: `registered`, `siteOf` and `keyRefusal` did not move. Added `rightLinks (S : Core.Site) : String`, so that the accepting path does not walk every link. [read]

One change from your brief, forced by a root test. The guards no longer import `CoreProtocol.Registration`: `tests/test_public_boundary.py` fails a public file of the certifier's check that needs it (the export copies `CoreProtocol/Site.lean`, which SPOT needs, and not `Registration.lean`). The guards restate the example's locations and links in the file. The same edge will stop pull request 2 if the period check's Lean imports the registration: that is a boundary decision (`tools/verity/check/public_boundary.toml`), not a line for its allowed list. [ran]

Ran, all on the final Lean and Python: `verity lean try --siblings` clean; `lake build NetworkCheck` done; `check.py` on the package PASS (1578 declarations, 11 modules, 4 guarantees, no lock change); the test 53 passed, Python and Lean agreeing on 669 vectors; the root suite's 44 files pass by name; ratchet gate 0 failed; merge-tree onto the landing head de28a691126b names no file; onto the crossed head ae73c594d338 the 55 conflicts are main's own. [ran]

Not yet ended: the pre-mark run of the 55 changed suites, r20261112-071517-bfb1, remote. It waited 12 min for a slot and took one at 11:28 PM PST. I follow it and put its verdict in the body. An earlier launch, r20261112-070244-7dfa, I cancelled while it still waited: its source was a commit the public boundary test fails on.

pyproject.toml of the suite: the `untraced` line in sorted order, and the `reads` entry you asked for (the check's Lake package, tests/lean, and the two packages the module's imports are in: verity/protocols/interface and verity/circuits/base). The suite is in AFTER_LEAN, so the graph's test-file family makes no task for it; I called the family's `walks` and its untraced check on the suite directly and both accept the entries. [ran]

One slip: one `research` help output I printed without the mask showed a node's name in my own transcript. Nothing was posted with it.

