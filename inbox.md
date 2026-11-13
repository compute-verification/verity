[2026-11-13T18:09Z] 6e54bbd602f6
Coordinator: #9579 adds `from verity.hardware.cards import load, number`. compute-research #9770 (marked, CMP-437) moves those readers into verity/device.py: `load` -> `verity.device.load_card`, `number` -> `verity.device.number` (cards.py and nodes.py are deleted). If #9770 lands before #9579, rewrite that import (and any other verity.hardware.cards/nodes use) to verity.device, rerun, and re-mark. If #9579 lands first, nothing: #9770 rebases over it.

