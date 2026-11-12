[2026-11-12T23:37Z] ed6451f54fc1
KEY, from training, Thu 12 Nov 3:37 PM PST. You read this on a new machine: everything you need is here.

**The `--format` option goes in #9502 itself, not through PROBE: #9474's head stays b23c853a35bc so its runs stand.**

1. #9502 is at ea620a89cf on origin; the option is at efd15349eb on `cursor/trn147-served-cases-10f7` (ea620a89cf plus the script's 33 lines). When the twelve served cases at efd15349eb end (r20261112-233108-21af, -233205-6c00, -233259-86da), add that one commit to #9502 with a test over the 15 made cases that state formats, and put the twelve's table and the command a reader runs from the head in the body.
2. Then run only what read the script: `uv run tools/verity/check/suites.py` picks the suites whose inputs changed. The Lean runs at ea620a89cf stand if no Lean input moved; say so in one line with the diff's file list.
3. Send me the head with: the five suite groups' last lines (r20261112-232245-e582, -232311-85ba, -232338-3ab1, -232351-a95f, -232403-d24c), `RecordRead.lean` and `RecordTemplates.lean` (r20261112-232515-0fba, -de9b), the security proofs package's check (r20261112-232445-9cb4), and the ratchet's line (exactly 3, all #9474's).
4. Your slips are noted; reading named fields only is right. Commit and push before you end every turn.

