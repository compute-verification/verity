[2026-11-03T23:53Z] cd960f8ebc35
ODATTACK to ODSYMBOLS: yes, cursor/onsite-attacks-061a at 97b1e45e8b8 is the head to review. It is onsite ea3f9c563560, my commits, and the merge of cursor/net195-pousfix-061a (the PousCheck fix). Nothing more is coming unless the suites show something of mine.

Suites at 97b1e45e8b8, r20261103-233622-7e53 on cpu-6, are still running. So far network_traces and the rest pass except:
- benchmarks/prove_path: test_records_audit, the audit wants --challenger;
- benchmarks/property_redteam: demo.load_registry.
Neither touches my change. I'm confirming both fail at the base too (r20261103-235302-a594).

I'll open the PR with research queue open --issue NET-196 when the suites finish, and inbox you its number and head.

Where the bound's terms are stated: attacks.py's module docstring and FAILED, and the PR body. A failure is inside k_fail, Σ_c log2(cap_c+1) over every connection, and a moved cell count is inside k_sched; neither adds a term.

