[2026-11-14T02:37Z] 6a01858ee5fb
Coordinator: Daniel cancelled the public export; top's #9876 deletes tools/verity/check/public_boundary.toml (and public_export.py, public_boundary.py, public_lock.py, their tests). When you rebuild #9696 on main, leave out any edit to those files (nothing reads them once #9876 lands). Same for #9535 if it touches them.

