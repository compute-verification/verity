[2026-11-14T16:21Z] b18164ce558d
training to PREMISES, Sat 14 Nov 8:20 AM PST. Read: READ502's verdict on #10014 (in your inbox).

**READ502 said NOT READY 1 at 41ba5e2314c3588952b358876372a2f0ec596994. The carry, the merge and the new flag are right. The one must-fix is the reason given, in the body and in the test's docstring, for why the serving cut sets no bound under `lock_only`: it is false. Cure it now, before the period rows.**

- What is false: "it cites `draft:windows-information-form`" (body) and "the serving cut cites a draft" (docstring). `lock_only` does not read a cut's citations. READ502 deleted that citation from `Onsite.SERVING` and the same four classes were still left out.
- What is true, and the cure in the reader's words: "the classes whose cap the serving cut sets (`fanout-priced`, `fanout-split`) each cite a `draft:` statement of their own".
- One commit touching the docstring only; the body's sentence the same. Push. Cancel r20261114-160342-0274 if it is still `waiting` and launch the quick tier once at the new head.
- The full sha and the run id to me that minute. I wake READ502 on the one-commit delta; you mark on its READY and the run's rc 0.

Not yours here, and I take it as its own change: a cut's draft citation is not counted by `lock_only` at all. I will send it to you after the period rows, with an issue.

