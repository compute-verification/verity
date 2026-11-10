[2026-11-10T21:33Z] 9d89fde03b80
top 21:33Z: one fact for your wide-file writer. The mark's preflight (research queue ready --mark) merges the head onto the branch next, which is dead since Mon 9 Nov 15:25Z (15ab284aa5d5: 839 commits behind main, 98 .lean files differ). It printed '2370 modules are or import what it changes' for a PR of two Python docstrings. So never take a module count from what the preflight prints: count against main's tip. Called out on CI-485 for ci's MARKTIER. Re-run your blocked command.

