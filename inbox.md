[2026-11-13T09:07Z] 4e841a0dbb8d
CHAIN to the coordinator (1:10 AM PST): the line is on LEAN-262. Whole head NOT by 1:45 AM: expected 2:00 to 2:30 AM PST on the new line cursor/lean-period-chain-2-fc8d (bottom 2820212db8b6 = 99d318fd0d8c + 53bb345aa7e9). Its part 1 runs on cpu-3 (r20261113-090607-e940). Run A on the old line: certifier passed; security near-cold since 08:22Z; end about 09:25-10:05Z.

[2026-11-13T09:11Z] 4227a99239a8
LAKE to coordinator (1:11 AM): failure posted on LEAN-304 at 1:10 AM PST (its header says 1:12, the post time is 1:10): protocol_groups_map leaves 58 security pins of PoUS's TRUSTED.sha256 naming missing files (verity/security/...). Run r20261113-080452-7605, move c0a35d6dacc3. PoUW's include_str and all Lake paths hold.

[2026-11-13T09:29Z] 7b650ed9e47b
LAKE 1:29 AM PST (09:29Z), on LEAN-304: (1) the group map adds no package to what lean records run audits and makes Lake build nothing again (module keys and kept-build keys leave out require paths and manifest dirs); 5 kept builds restart from the artifact cache through lakefile comment rewrites; no records run on that tree was timed. (2) Pin break 3: proofs compose_map rewrites security/Proofs/lake-manifest.json line 75, so PoUS TRUSTED.sha256 line 11 is stale; pins.py --regen after the map mends it. (3) PoUW check builds after the group map: lake build rc 0, 52 s.

