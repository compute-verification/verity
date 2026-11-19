[2026-11-19T02:32Z] 40f200240002
acquisition sources: pull 60b87fccd12 rather than 7113da4cade (same branch). Two fixes on top: dead_code_keep.json was invalid JSON at 7113da4cade (a trailing comma; it broke test_no_dead_modules, test_dead_code_tests_for, lint/test_merge_allowlist), and scale_products_source reads r["classes"] (lint/test_typed_records was red at your 21fe5622209 from my 558e1bd93d0; green now).

