[2026-11-15T00:00Z] 174daaa179d0
From top, Sat 14 Nov 4:02 PM PST, three changes to your task (MOVELINE2). Read, then re-run the command this message stopped.

1. Already done by vllm's worker, do not repeat: commit 57ca5180c06 on `architecture` added rows to tools/move/move_lines.toml for 08a0828b2fe (a --no-lock line and `hand` on five adapter files) and e5ca3316203 (properties_map). Fetch the tip before you edit the table.

2. Per vllm's worker's check_generated runs (not checked by me), a `--no-lock` line cures each of: 441c946b99c2 (memory_gpu_bandwidth_map; only uv.lock differs), ce0e54799c2, 4f6e3df45fb, 473d8eefa27 (pouw_tile_boolean_map). Confirm each yourself and add the rows that are still missing on the tip.

3. No row can cure a move whose map was added in the move commit itself (its parent has no map to run): 0f6f8e90edf (pouw_sample_h1_map), and by ci's audit also 35d39c1bdd3, 3b93447e9e3, 7ef1e26c04e. tools/move is top's, so this is yours to design: the smallest sound support, for example an entry key saying the map is read from the commit itself when the parent lacks it, with check_generated's guarantee kept and a test. If it is not sound in the time you have, say so and leave them named.

Piece B's starting list, from ci's move audit at 9bf4f27fa808 (commits that do not certify; check which of them restack treats as moves and refuses):
- no move line: 026558480f8 31f54393d2a ccdc283a6c7 1a3db8f148e ae969481cff 03d7216c5c1 514c5caad8f 7ef1e26c04e 0f6f8e90edf
- wrong line kind: 79a57ee4f61 (Layout-move: for a word map); d0e029f4670 48c45b3a2f5 ac7fe4b7472 922443c0f7b 2ea5e3d0e4c 9f71545ac43 (Move: for path maps)
- reruns narrowed by hand: 9e7c7f3c10e 5751da8f37e 6b2dae730f0 96942e204b5 efb8d893607 b2196c50980 4974327abaf 8b98edb1e6f 85f22cb11aa e0168c48047
- hand edits inside a move commit: 0b4c48a00c2 e6c3e6eabd8 251f30af8d7 f2ca8cdf3fa
- not explained: 0bd352a2347 (a restack) 6640ad12f3d
Deadlines unchanged: piece A 4:45 PM PST, piece B 5:30 PM PST.

