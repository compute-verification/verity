[2026-11-14T19:24Z] 11dbdad19d70
LEAD: compute-accounting says the call is on architecture at 7eac32fbf4f: run_matmul(run, site, x, weight, bias=None) with MatmulRun; observe_matmul(run, site, x, weight, y, *, x_scale=None, weight_scale=None, z=None) with MatmulObserver for H1; the lead goes through run_matmul with x=QuantizedRows(rows, scales). Read their TOP-424 comment of 11:52 AM PST whole, then make the adapter side now (before finishing row 4b), on a branch cut from origin/architecture, pushed there by merge. Write your read to notes, not Linear (quota).

