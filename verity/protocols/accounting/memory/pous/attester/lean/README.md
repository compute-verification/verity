# The memory attester's verdicts in Lean

The memory attester signs what it saw; this package decides whether what it signed passes, as executable Lean behind
the core protocol interface (`Core.Protocol`): `memory-challenger`, PoUS's timed rounds, whose public output is the
verdict alone, and `wipe-protocol`, the memory wipe as `verity/wipe/v0`. Both take the core protocols' command line:
`id`, `check PARAMS COMMITMENT EVIDENCE` or `serve`. It is code, not security analysis: it imports nothing of
`security/`, whose `Properties/Pous/` states its properties against these functions, and its difftests hold each verdict
to the memory attester's Python, one directory up.

~~~sh
uv run tools/check/suites.py verity/protocols/accounting/memory/pous/attester/lean
~~~
