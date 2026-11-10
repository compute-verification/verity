[2026-11-10T04:13Z] ba91292c84d0
L3MEASURE, TOP-394 count done (read-only). Remote 99c8914848 against main 7af4d24fca3, paired through the 13 maps whose moves landed since the base 38f6228a3a7 (name rules run on the side's names) and PoUW's gap MOVES.json (dfa8e918d3b). No map is left unapplied.

1. Only on the remote side: 366, all proved there (full name resolves to a declaration under proved_in), none listed and open. 119 new on the side; 223 main unlisted or retired (#7194: 203, #7668: 20); 24 renamed or removed by hand on main.
2. Different statement record: 560 of 1,199 pairs. 313 equal after main's renames, 155 edited on main only, 31 on the side only, 33 on both lines, 22 added on both differently, 6 not established.
3. Only on main: 332.

| area (item 1) | total | new | unlisted/retired | other |
|---|---:|---:|---:|---:|
| Accounting | 84 | 4 | 72 | 8 |
| FlockSoundness | 59 | 2 | 56 | 1 |
| Nci | 52 | 12 | 40 | 0 |
| Pouw | 38 | 6 | 32 | 0 |
| Properties.Pouw | 26 | 26 | 0 | 0 |
| NetTiming | 22 | 22 | 0 | 0 |
| Properties.RemoteAudit | 21 | 21 | 0 | 0 |
| FlockVBridge | 17 | 0 | 17 | 0 |
| Pous | 16 | 12 | 0 | 4 |
| other 10 areas | 31 | 14 | 6 | 11 |

| owner | item 1 (new) | item 2 (both lines edited) |
|---|---:|---:|
| @proofs | 111 (28) | 77 (5) |
| @compute-accounting | 94 (0) | 172 (11) |
| @memory-accounting | 58 (37) | 183 (14) |
| @network-accounting | 35 (33) | 17 (2) |
| @training | 30 (12) | 29 (0) |
| @lean | 26 (0) | 73 (1) |
| @compute-research | 8 (6) | 7 (0) |
| @soundness | 4 (3) | 2 (0) |

| slice (by area) | item 1 (new) | item 2 (both lines edited) |
|---|---:|---:|
| none (accounting, network) | 141 (58) | 221 (20) |
| shared serving path | 85 (2) | 56 (4) |
| 7 compute, served_passes | 64 (32) | 145 (9) |
| 9 inference_only | 52 (12) | 62 (0) |
| 8 memory, served_piece | 19 (12) | 49 (0) |
| 10 served_verdicts | 5 (3) | 27 (0) |

Ten largest groups of 1: compute accounting's TS headline floors (49, unlisted); NCI containment and Loomis–Whitney bounds (35, 12 new); Pearl+ checker properties (26, new); C-Flock partition audit (26); network timing witnesses (21, new); PoUW NCP lemmas (18); C-Flock verifier bridge (17); training's cover bounds (14, retired); rate audit witnesses (13); PoUS period-held coins (12, 10 new).

Not established: hand renames on main with no map (in the 24 and in "main only"); proof by source name resolution, not a build of 99c8914848; slice by area, not by what each driver reads.

Note: lean/20261110T0415Z-finding-top394-remote-only-lean. Names: art:6c2ef818f967f81b86f8226c12875176fd99b8904f5549dee23c73094ff48330. TOP-387 comment 7273df53. Restage on 6f7ddc8d55's network map: stages 1-2 of 6 pushed (0d57e899d92), stage 3 in records; suites next.

