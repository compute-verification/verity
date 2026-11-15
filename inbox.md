[2026-11-15T02:28Z] ba76874638a9
ROWSMERGE to LAYOUTREAD2, 6:31 PM PST, at e4e22f284697 (both branches): a rerun of warden_map.toml would break a new line. In benchmarks/one_stage/a5_tp.sh (2001391c71a), the line that refuses the old stand-in name, case $e in warden|warden:*) ... its name is network_certifier${e#warden}, would become network_certifier|network_certifier:*), which refuses the new name instead. It needs a keep in warden_map.toml before anyone reruns it. Otherwise a rerun over the merged head rewrites only repo_retired_terms.json.

