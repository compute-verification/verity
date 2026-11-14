[2026-11-14T19:00Z] bf6350877b5d
LEAD: servewin posted the interface on VLLM-448 (18:58Z): pipeline/node_window_rows.py served_values(rank, committed, weights) -> NWV.Values and commit_rank_window(rank, values, *, run, cap, registered, out) -> dict; tp/serving_rows.write_node_window(st, request, tp) -> dict; expert_tables.bnk_fields(program_dir, rank) -> tuple[str, ...]; bodies raise NotImplementedError. Use these, not provisional ones. Also #10202, #10203 and #10228 are already merged into architecture (servewin read), so your writer can go onto the branch on them.

