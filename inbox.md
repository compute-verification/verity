[2026-11-14T09:04Z] 8fa4044ba70d
From top (mine, by inbox.py), 1:06 AM PST Sat. Your run r20261114-082820-da07 stands in vy-nebius-cpu-2's research line, and that pool is shut (the machine is under its free-inode floor), so it cannot start there. Cancel it and launch the same command again with --on auto; if auto places it on cpu-2, cpu-4 or cpu-5 again, name another machine with --on (cpu-6's line was the shortest at 08:58Z: 13 runs). Then re-run the command this refused.

