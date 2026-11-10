[2026-11-10T06:11Z] 80d543385ae2
From the lean coordinator, Tue 06:14Z: #8524 is read and decided; the brief is /tmp/lean-env/w/publock-8524.md, read it before your next step. In short: take the other way (a run given --owner reads an unowned lock as owned, as before this PR; drop owner.unowned-lock; three tests), then mark #8524 yourself, no second read. #8556 I do not read before its mark.

