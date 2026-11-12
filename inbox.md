[2026-11-12T23:41Z] aecf79596751
BACKWARD2, Thu 12 Nov 3:42 PM PST.

**The third launch stopped in the CLI before it sent anything (no run id, nothing reached the node): `--pool` needs a lane. I've stopped; on your word I add `--as training` and nothing else.**

- RAN: `research run --on vy-nebius-2 --project verity --pool research --source ... --mem-gb 16 -- ...`. Its own line: "research run: --pool needs a lane: --as LANE or $RESEARCH_LANE".
- The next launch would be the same command with `--as training`. I read the CLI's help for nothing further, so I can't promise it is the last flag the pool asks for.
- Running at `a809ac0598`: circuit-check r20261112-233604-6905 and suites r20261112-233635-4858. The suites at `1178eb5439` (r20261112-212535-7f3f) are still running. I cancelled the superseded suites run at `31fdc1c38b` (r20261112-214623-31a3).

