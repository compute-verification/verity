[2026-11-17T06:48Z] 17104365a928
PPL1B: entry-7 ppl C4 run started on node A card 5 (cores 60-71) from origin/main 144b3aecad0; row booked 06:50Z-08:00Z

[2026-11-17T06:50Z] a6d549b1c983
MODEL: Claude Opus 5.5. DRAWMAP: G2 at 5e6d854b69a r20261117-052504-48a1: repository 377/377 passed; verity-vllm 24040 passed 548 skipped 2 FAILED, both tests/adapters/vllm/tests/protocols/test_credit_gate.py (test_a_period_open_without_draw_k_is_refused, test_a_window_presented_outside_its_periods_close_or_out_of_order_is_refused). The map leaves that file and all Python code unchanged (pure-Python PHC period checks); no main baseline, so repro of the 2 on main r20261117-064951-b68e vs head r20261117-065031-5b8a running.

