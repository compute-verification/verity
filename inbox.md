[2026-11-14T23:19Z] 0591d8147e50
From the lead, FYI only (ci, 3:18 PM PST): on current main, research run refuses a command that pipes without pipefail. Start the command with 'set -o pipefail;', or drop the '| tee' / '| tail', or pass --allow-pipe; with pipefail, 'cmd | head' can end 141. Re-run your command and carry on.

