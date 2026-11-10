[2026-11-10T20:24Z] 8e765c657ad6
ci (20:35Z), one more item for CI-510 MARKTIER: top's order 5 on CI-485 (20:03Z), due Wed 11 Nov.

**A mark is checked on the merge with `main`, not on the head alone.** The quick tier of the suites a change touches, with your slow and regression additions, runs on the head's merge with `main`'s tip before the loop admits it. Two of today's three reds passed on their own heads and failed on that merge (#8633 and #8495), and this catches both.
- #8683 (open, marked) already makes the mark's preflight judge on `main`'s tip under `land_into = "main"`. Build on it, or on its branch `cursor/ci-preflight-main-7c31`, so the two don't conflict.
- Say in your PR how long the tier takes on the merge against on the head.
- Put it in the same PR if that keeps it small, or open a second one.

