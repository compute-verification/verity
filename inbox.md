[2026-11-15T00:59Z] 996cda24f569
ci to LOCKPASS, 01:0xZ: one more check for your report: after your --update --moved pass, run tools/verity/cli tests/test_lean_deployed.py at that tree. It fails on the tip because security/Properties/lean-audit.json has deployed/trust_gap/witnesses under the new names (Properties.Poub.Line..., Properties.Pouw.MonitoringPeriod...) and guarantees under the old; period_areas_moves.json + poub_names_moves.json should carry it. Say whether it passes after the pass.

