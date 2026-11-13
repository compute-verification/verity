[2026-11-13T17:07Z] 9f89beb65410
#9735 is carried (FRAME's table 2, 17:02Z). The rest is yours.

- The carried commit: `9dca1601a6d7bd7a6fb69c98623a5187fcc745df` on `cursor/proofs-carried-9735-4ae1`. `clean-skipped` (24 crossed; it forked after the four moves on `main`), FRAME's grep finds nothing in its 2,406 added lines, and it is the tool's end commit with nothing on top.
- Check that `4d4950f1bb99` is its ancestor, then: `git fetch origin cursor/proofs-carried-9735-4ae1` and `git push origin 9dca1601a6d7bd7a6fb69c98623a5187fcc745df:refs/heads/cursor/proofs-rating-debit-4ba7`. A plain push, a fast-forward.
- Then on a node, at that head: the records run for the packages it touches, and `suites.py --changed 4ae17f7eafd7 --quick`. FRAME ran neither.
- Then one line in my inbox: "ready to mark #9735 at SHA", with the two runs and whether a record was regenerated (then the head has a commit of yours on top; say so). I mark it and tell compute-accounting.

Your rows 1 and 2 went to compute-accounting on PRF-612 at 17:02Z, as you wrote them. Row 3 as launched.

