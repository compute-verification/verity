[2026-11-14T07:51Z] c186c3ffede6
PREMISES, Fri 13 Nov 11:51 PM PST. **Gap PR head pushed: cursor/serving-boundary-term-ed00 at e79000b4b709722636acd726721e219f9a9d5d64 (on bd87c4b01bfa); one printed figure moves: onsite-b200-kimi's lower bound 866 -> 865.**

- RAN evaluate_named over all 41 registry scenarios before (bd87c4b) and after: 105 values differ at full precision, 4 at 3 significant figures, all onsite-b200-kimi's lower bound, 865.5007 -> 865.4999 (the boundary term is about 1e-9 of a server block's weight; 866 was on the rounding edge).
- RAN run.py before and after at PYTHONHASHSEED=0 (two seeds on the base agree): beside that 866 -> 865 and its shares (unregistered 86.6% -> 86.5%, monitored 13.4% -> 13.5%), the serving-cut scenarios print a new "boundary 0.0%" term; regime.onsite.wipe_s and regime.onsite.intake, equal at 3 figures, swap order in some tables; onsite-h200-llama405b's corner says "10 more" parameters under 0.5% (family.full.per_param now moves it), its value unchanged at 1.19.
- Your two conditions are tests: a lower bound a serving cut sets reads structure.<name>.mu and .omega_max (test_scenario), and classes.structure refuses a structure without mu (test_classes).
- Also: cover now cites BlockFits and CapBelowDepth, which test_classes requires in data/premises.toml. CapBelowDepth: supported (upstream_share picks D with k < b D). BlockFits: assumed (no data states the widest gate or theta's width). cover's premise status at small k goes supported -> assumed; no printed status moves (UpdateHardness is contested at every binding k). Recommended as is; say if BlockFits should read supported.
- Issue: Refs TRN-167, as you said.

