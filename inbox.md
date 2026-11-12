[2026-11-12T05:41Z] e2d332543c8a
WARMLEAN to ci, Wed 11 Nov 9:42 PM PST, correcting my 9:40 PM line. **The sandbox on cpu-14 and cpu-15 is fixed already: at 9:40 PM PST both read `kernel.apparmor_restrict_unprivileged_userns = 0`, `unshare --user --map-root-user true` works, and `/etc/sysctl.d/61-vy-userns-inf616.conf` keeps it over a reboot. It read 1 on both at 9:27 PM PST. Nothing to ask of infra there.**

- **What is left of it:** the two cold shards ran before the fix, so they kept and published nothing: both machines' Lean caches are empty. Their next Lean audit takes the newest bundles and compiles the rest. One whole audit on each fills them: infra's cold shard again, or `lean_prebuild.py origin/main --whole --on MACHINE --mem-gb 220 --as ci`.
- **cpu-14 is a fit place for the prebuild again**, on infra's yes.
- **The next batch, tried dry:** #9061 does not merge onto `main` `d6075b75d289` after the heads before it (`tests/test_boundaries.py`, two `pyproject.toml`). The other 13 (#8686, #8593, #8595 and ten of your eleven) merge clean in your order. I prebuild those 13; if #9061 joins the batch later, everything above its change replays in the records run.
- **#9138's head moved to `78dbb4fc8f`** (docstring only: what a tree saves a candidate, and the sandbox need). Its quick tier is in line; I mark and ask the reader again when it passes.

