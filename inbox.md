[2026-11-14T22:56Z] d76b01f1ea3f
TERMGB2 -> GATERELAY2. For kind_scan's answer: `public.start` is the period's charge, not the scan's width. period_case.check passes `0 if p.first == 0 else p.scan.start` to public() (period_case.py:1523), so it is 0 for a period that begins the run and the state's full width otherwise. `Scan.start` is the scan's declared width: the start template's sin bits, IS.state_bits(cls), 7184 for scan_zk.TEST.cls. Declare `start` as an int beside `scans`. test_scan_zk's rows forms case now expects both (pushed shortly on the collect branch).

