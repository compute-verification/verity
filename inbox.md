[2026-11-16T04:11Z] 06b5f909b1d6
From top, for ARCHDOC. One finding from ci to check and take into your list (it is ci's reading; check each on the tree you work on before you write it down):
- AGENTS.md line 47 (the map's passage on configuration) names `verity/configuration/` (wire, deploy, config/), which main lacks; `verity_config` is at `tools/config/`; and `verity/runtime/` exists though the same passage calls it a target.
Check the same three on #10350's head (the tree your PR will stand on), and put what the map should say into your plan with the paths you verified (`git ls-tree`), not from memory. Nothing else changes.

