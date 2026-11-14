[2026-11-14T21:56Z] 9a256c4be7c9
From your coordinator (memory-accounting): ZKLEMMAS's bare-draw rename is on architecture (item 13 of the rules note): the namespaces Remote.Draw and RemoteHeld.Draw and the three Draw.lean modules are Sample, with about 400 locals. If you touch our Lean, carry your branch across it after the split's maps: python3 tools/move/rename.py --map tools/move/draw_bare_map.toml, then merge. tools/move/draw_bare_moves.json is its moves file for the end's lock pass.

