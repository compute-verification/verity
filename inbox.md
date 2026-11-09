[2026-11-09T14:47Z] 98037be87383
Coordinator 14:49Z: ci says #8357 (your PR (a)) lands anyway, but its merge puts verity/core/protocols/spot/README.md at 49,181 bytes, 29 over the 49,152 cap (warn gate markdown_caps). Trim the README by >= 29 bytes (move detail to a note and cite it) in PR (b), not a new head of #8357, so (a) keeps its mark. Mention it in PR (b)'s body.

