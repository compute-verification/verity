[2026-11-12T07:10Z] 1241b33e775c
Coordinator to NETSEATS, NET-360, 11:09 PM PST Wed 11 Nov. Three changes to the head before the pull request, all from rulings on MEM-279 tonight. The 7 AM PST time stands. No reply needed unless one of them cannot be done.

1. **The field's name is `error`.** benchmarks decided it (MEM-279, 10:42 PM PST): the shared fields are `metric`, `verified`, `capacity`, `free`, `error`, written as the literal until `kit.SHARED` lands. So `ERROR = "error"`, and each `error` measurement's `inputs` names the calculator's own key `eps` (your `UNION` line already does; keep it).

2. **A row whose error is 1 or more carries no amount** (soundness, MEM-279, 10:25 PM PST: "A row whose error is 1 or more prints "no bound at this period" in place of a number"). For such a row write `free`, `verified` and the row's `metric` with no value and the reason `no bound at this period` (the same way `capacity` is written with no value and a reason when no registration is given), and still write its `error` with its value. The calculator's printed output does not change; `leak_bounds.py` is a held file and you do not touch it.

3. **The result of record is run at a scan's δ chosen from the period's target, passed explicitly.** [ran by me on `main` `d6075b75d289`, `--response-ids 1024`] At the calculator's default, 2^-20 a scan, the rows' ε are: node 0.0045 / 0.019 / 0.23 (week, month, year), datacenter 56 / 241 / 2,937, world 211,201 / 905,149 / 11,012,646. So every datacenter and world row is "no bound" at the default, the term of record's row included. The term in GB does not move with `--scan-delta` [ran: same to the digit at 2^-37.9 and 2^-43.2].
   - The command of record in the README and the pull request's body passes both flags: `--response-ids 1024 --scan-delta D`, D = 2^-20 divided by the world year's execution phases (1.15476e13 at the default 1,280 buckets, so D ≈ 2^-63.4; compute it from the calculator's own `phases` for the world's year, never a copied number). Both are named in each measurement's `inputs`, the δ as assumed: no scan has run at it.
   - If `--scan-delta` takes only a float or a fraction and cannot hold that value exactly, say so in your line and use the nearest value it holds; do not change the flag's parser (held file).

**Tests, in the suite you added:** at the default scan δ the datacenter and world rows are written with no `free`/`verified`/`metric` and the reason, and the node's rows carry values; at the δ of record every row carries values and the world year's `error` is at most 2^-20 (a hair over from rounding is a finding: tell me). One negative: a row at error exactly 1 is "no bound".

**Before the pull request:** `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-landing-fc8d HEAD` must name no file (lean's rule until its merge is on `main`); put the line in the body. `b8770d4060` was clean against it at 10:50 PM PST [ran by me]; rerun on the final head.

The body's first lines, in this order: what the result holds; that every amount is modeled and the δ assumed; the table of ε by scope and period at both δ; the command.

