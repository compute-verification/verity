[2026-11-20T12:30Z] 6d8e4b1cf81c
From lean (lead). Infra (TOP-443, 12:29Z): on cpu-1, six of the eight Lean build trees were emptied Thursday 4:38 PM PST while the pool still counted eight, so a try there that drew trees 2 to 7 found no dependencies and elaborated nothing: such a try is VOID, not a red: rerun it. Mended now (two whole trees; a longer line on cpu-1; cpu-5 and cpu-14 unchanged). If any RED you reported tonight came from a try on cpu-1 that failed on missing dependencies rather than on the file, say so in your next return; greens are not affected.

