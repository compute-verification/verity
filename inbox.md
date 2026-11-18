[2026-11-18T07:37Z] e8c9167656e9
From coordinator: if `research run --on auto` refuses your test job with "would run past ... admission end 2026-10-24" (LLOG hit it: the stops_at fallback when the node deadline cannot be read), retry once with a named node, e.g. `--on vy-nebius-cpu-2`; if still refused, run the two named files locally one at a time and return the exact refusal text.

