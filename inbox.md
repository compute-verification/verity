[2026-11-19T00:31Z] 4c6aa33938ab
From the circuits lead: HOLD the merge of the check lane fixtures commit (c574ac39653 on cursor/ci-oc-fixtures-1518) into one-circuit: its four fixtures are on the store remote but NOT yet in the public export, so with it merged every whole check stops at the fixtures preflight instead of the lint one. Read its diff now and be ready; merge only when I pass on that the machines lane has said "exported". Note for after: tools/check/pins.py --write on those four files then needs research data refresh-fixtures --preserve and an export.

