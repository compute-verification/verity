[2026-11-14T20:06Z] d0942d00e2bc
LEAD ANSWER (build-match, T201): YES, add `cli.write(text)` yourself now (one stdout write in pipeline/cli.py, door named in the commit), with a test, and route your commands reader output through it. Engine (four prints), predict (7 T201) and others will use the same writer: say its name and signature in notes handins/build-match.md so they can. Nothing on Linear.

