[2026-11-15T14:18Z] 6134e1994c4f
ci to OLDPATH, 14:19Z: infra says 'watchers installed' (TOP-424, 6:16 AM PST): nothing installed on the control pod runs tools/verity/check/check.py any more. DELETE the check.py shim in this change too. Only merge_map.py and merge_rows.py stay under tools/verity/check/ (until infra's merge-driver re-registration is deployed); their docstrings say so, and 'tools/verity/check/' stays in lowrisk_excludes while they remain.

