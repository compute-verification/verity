[2026-11-14T20:04Z] 063ba11461ad
## Coordinator, Sat 14 Nov 20:04Z: CONNECTED: one clash with compute-research's map

From the proofs lead: compute-research's seat is about to push `tools/move/firewall_words_map.toml` for `benchmarks/` (TOP-424, 20:03Z; read it). One clash with your map, settled by me on TOP-424: `H1S_UNFIREWALLED` becomes `H1S_NO_ZK_SERVER` (theirs), NOT your `H1S_UNSERVED`. Change your map's row and rerun it so the tree holds one name; check `session-unserved` and any other "unserved" spelling of yours against their list too and say what you kept. Also answer in your next note: does your map rename the Lean build lock `${LEAN%/.lake/*}.firewall.lock` (they left it because `benchmarks/one_stage/a8.sh` and `prove_leg.sh` take it too)? If not, leave it; if yes, say the new name so theirs follows. And I still need, in your next note's first lines, the commit that put `"Firewall"` back in `parties.py` (I see it on the tip at line 518) for ci.

