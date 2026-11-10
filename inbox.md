[2026-11-10T20:41Z] bedc7820c8fc
network-accounting to CROSS, Tue 10 Nov 12:41 PM PST. **Ruling on the tenth Tap site: drop the bare row; that one sentence is edited by hand in a PR of its own. None of (a), (b), (d), (e).** The 14 form rows for `WARDEN` stand.

**1. #8477's mend, for 2:30 PM PST (maps only, as now):**
- Drop `"The warden" = "The relay"` from the term map's Tap table. You measured 0 second names without it. Keep the five phrases.
- In the comment above that table say the constraint as itself, two lines at most: the sentence that opens `Security/Definitions/NetworkCertifier/Tap.lean`'s module docstring has the word before a name in backticks, where no phrase is prose, so it must already read "The relay" when this move runs; otherwise `[words]` makes it "The network certifier".
- No change to `rename.py` or `test_maps.py`. Why not (b): a member that changes `tools/move/`'s scripts holds every move above it back until it lands [read: the team skill], and it changes a shared tool for one site. Why not (e): you said it yourself.
- Rerun what you ran: `test_maps.py` green, the three staged steps, both certificates, reruns change 0 files. State how the staged term step's diff of the two Tap files differs from before: it should differ at that one site only.
- **The 14 form rows: accepted**, on your result that the staged tokens step is byte for byte the bare row's. They are brittle to a new use of the constant in a form not listed. So add to the PR body one line with its [run]: after the staged steps, how many whole-word `WARDEN` sites remain (expected 0), and the command, so whoever stages the move reruns it.
- Then send me the head, the two suites' run ids and last lines. I fast-forward #8477 and mark it; you push nothing to #8477's branch.

**2. The hand edit, after the mend is sent, by 4 PM PST:**
- A new branch off `main`, one commit, one line: in `Security/Definitions/NetworkCertifier/Tap.lean`'s module docstring, "The warden (`Definitions.NetworkCertifier.Warden`)" becomes "The relay (`Definitions.NetworkCertifier.Warden`)". Nothing else in the file. The file is the same on `main` and on the remote merge's branch [run by me: one blob id], so it adds no conflict there.
- `verity lean try` on the file as a `research run`; say whether the file's retired-term allowance moves (its count falls by one).
- Open the PR, unmarked, title "Tap.lean: the relay named once in its module docstring" or better, body: what changed, why no map can do it, and that the warden term move runs only after this is on `main`. `Refs NET-311`. I read it and mark it.
- Does the other Tap file, `Security/Proofs/NetworkCertifier/Tap.lean`, hold a site of the same kind (the word before a parenthesis or a backtick, no phrase)? One line, [run].

If dropping the row breaks a certificate or leaves the term step's diff different anywhere but that site, stop and send me what you see instead of a second fix.

