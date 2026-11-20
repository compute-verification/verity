[2026-11-20T06:26Z] d92665dfe76c
From your coordinator, 06:26Z: circuits merged a change to laying `pack` (one-circuit at 248e32870ac): it returns no proof block that holds a unit and a writer of its input. This can change how the tile blocks are packed (Seed/Line/GramRow writers vs the core that reads them), so the block counts by Definition, the fan-out and the strata in your credit run must be taken AFTER merging this; say in your return if the population changed against your earlier census.

