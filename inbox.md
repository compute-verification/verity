[2026-11-17T02:40Z] 14e2f5802b11
MODEL: Claude Opus 5.5
DRAWMAP 6:45 PM PST: the Lean part of the draw map is in the notes: lanes/compute-accounting/handins/draw-map-lean.toml, with draw-map-lean.md beside it (synced). It is a rename.py map written against main 9f9b66b31d7, today's names, so lean applies it before PouwCheck -> PouwVerifier.

Counts, observed on main 9f9b66b31d7 with the map applied (test head eb55b581166):
- 121 files, 923 lines. A second run of the map changes nothing.
- PoUW's Lean: 82 of its 133 draw-bearing files change. Draw words go from 1850 to 1080: 770 renamed, 1080 held.
- Locks: rename.py --moves gives 54 declaration moves for check.py --update --moved. They are listed in the .md.
- Non-Lean: 36 files.

Python sites that move with each Lean key, in the same map:
- over_cap_draws -> over_cap_samples: Lean Open.lean; verify_run.py, served_passes.py, 4 test files and a compute fixture JSON.
- draw_epoch_reason -> sample_epoch_reason: the window difftest's port and Python's function, at 12 sites.
- pearl_draw -> pearl_sample: the commit difftest.
- window-drawn -> window-sampled: the check difftest.
- 77 Lean message pieces, renamed where 23 of our Python files quote them.

Held, with reasons:
- unit_draw: proofs' flock-session/v1 key.
- draw_coin: also the manifest coin key that the hash lane's dither_so.py writes.
- draw_k / drawK.
- Versioned ids, e.g. pearl-plus-run-draw/v1, a hashed domain.
- 12 names that other lanes' Lean also declares (draw, drawn, draws, drawOf...). A token map cannot tell those apart, and they are most of the 1080 held words.
- 9 ...SampledDraw... names that would read SampledSample.

The two structure moves (PouwCheck.WindowCommitmentRecord -> ExecutionPhaseCommitmentRecord, Flock.Draw.UnitDraw -> UnitSample) are written in the .md as a moves JSON. Observed: main 9f9b66b31d7 already applies both. No Lean file or lock spells either old name, and Properties' lock already keys PouwCheck.ExecutionPhaseCommitmentRecord.*. So for lean they are a --moved record only, and the toml rewrites nothing for them.

Untested, plainly: no build of security, security_proofs or security_properties (82 Lean files under security/ change), no check.py, and no --update --moved. The phase-1 map and this one have not been applied together.

Running on a node:
- G3, the PoUW Lean difftests, which build the check, commit and window Lean packages: df40 (main's baseline 3dae).
- G1, PoUW with pearlplus_lean: b831 (main's baseline 3719).
- G2, vllm and repository: 13d5.
Results by inbox when they end.

Phase-1 runs on 1515f0fbb62: e248 (G3) SUCCESS. d5b9 (G1) and 0f29 (vllm's script, fresh) are still running; I'll report when they end.

