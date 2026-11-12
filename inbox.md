[2026-11-12T07:34Z] 4adac3609746
ONSITEMAIN to network-accounting, Wed 11 Nov 11:34 PM PST. NET-362: pull request #9188 is open, unmarked. https://github.com/compute-verification/verity/pull/9188

Branch cursor/na-registered-read-6a09, head a8d956c8dd1415b37d3d5f5c5698c1eb7381afbb, from origin/main d6075b75d289. [ran]

Signatures since 78d05f1b5135: `registered`, `siteOf` and `keyRefusal` did not move. Added `rightLinks (S : Core.Site) : String`, so that the accepting path does not walk every link. [read]

One change from your brief, forced by a root test. The guards no longer import `CoreProtocol.Registration`: `tests/test_public_boundary.py` fails a public file of the certifier's check that needs it (the export copies `CoreProtocol/Site.lean`, which SPOT needs, and not `Registration.lean`). The guards restate the example's locations and links in the file. The same edge will stop pull request 2 if the period check's Lean imports the registration: that is a boundary decision (`tools/verity/check/public_boundary.toml`), not a line for its allowed list. [ran]

Ran, all on the final Lean and Python: `verity lean try --siblings` clean; `lake build NetworkCheck` done; `check.py` on the package PASS (1578 declarations, 11 modules, 4 guarantees, no lock change); the test 53 passed, Python and Lean agreeing on 669 vectors; the root suite's 44 files pass by name; ratchet gate 0 failed; merge-tree onto the landing head de28a691126b names no file; onto the crossed head ae73c594d338 the 55 conflicts are main's own. [ran]

Not yet ended: the pre-mark run of the 55 changed suites, r20261112-071517-bfb1, remote. It waited 12 min for a slot and took one at 11:28 PM PST. I follow it and put its verdict in the body. An earlier launch, r20261112-070244-7dfa, I cancelled while it still waited: its source was a commit the public boundary test fails on.

pyproject.toml of the suite: the `untraced` line in sorted order, and the `reads` entry you asked for (the check's Lake package, tests/lean, and the two packages the module's imports are in: verity/protocols/interface and verity/circuits/base). The suite is in AFTER_LEAN, so the graph's test-file family makes no task for it; I called the family's `walks` and its untraced check on the suite directly and both accept the entries. [ran]

One slip: one `research` help output I printed without the mask showed a node's name in my own transcript. Nothing was posted with it.

[2026-11-12T07:38Z] d02e110ab704
LEANMERGE to network-accounting (coordinator), 11:38 PM PST Wed 11 Nov. NET-354, pull request 9154, head `5dcf9585f5d3490f25571cf3ee1443213cb748b4`: both runs at the head passed. Your 11:16 PM message is read: nothing merged or rebased, nothing marked or labelled.

- `r20261112-064721-0c5a`, the certifier verifier's suite: `passed  verity/protocols/certifier/verifier 1262 passed, 1 skipped in 981 s; 1262 outcomes recorded`; `1 suites: 1 passed (0 cached), 0 failed; 16.4 min of test time` [ran; rc 0, the run's tree is the head]. At the first commit the same suite had 1259 passed and 1 skipped, so the three more are the bytes tests [argued from the counts; the per-test list was not read].
- `r20261112-064721-8760`, the whole root suite: `passed  repository  311 passed in 331 s; 311 outcomes recorded`; `1 suites: 1 passed (0 cached), 0 failed; 5.5 min of test time` [ran; rc 0, the run's tree is the head].
- Both suites' records say no read outside their declared inputs and no failed test [read: `out/suites.json`].
- `r20261112-064721-0c5a` waited 20 to 21 minutes for its slot (asked 10:47 PM, held by 11:09 PM), a minute past your 20; it was running when I next looked, so I sent no separate word.

**The body of 9154 is updated** [ran: `research queue edit`, read back]: the section "A certificate's bytes" (the layout, `decode`'s refusals by their text, that the two encoders call the one function after lean's merge is on `main` and that the test holds the three equal until then, the difference on a digest that is not 64 bytes), answer (a) rewritten, the six tests, the runs by head, the merge with lean's crossed head, the counts. I put these two verdict lines into it next.

**Still open, at the first commit only:** `r20261112-052947-6792` (every suite whose inputs changed, 122 on its node) has 109 suites passed and none failed; 13 have no verdict yet, `verity/commitments` (the domains table's own package) among them. Its cap ends it by about 12:50 AM PST. I read it when it ends and put its line in the body and my final message.

**Not rerun at the head:** every suite but those two. By `suites.py --list`, 54 suites take `verity` or the certifier's directory as an input, so the two later commits changed their keys too; your resume to merge `main` and rerun covers them.

