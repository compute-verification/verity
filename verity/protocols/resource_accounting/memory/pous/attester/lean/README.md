# The memory attester's verdicts in Lean

The memory attester signs what it saw. This package decides whether what it signed passes: the memory attester's two
verdicts, on PoUS's timed rounds and on the streamed-label wipe, as executable Lean, with their record formats and
crypto, and each verdict behind the core protocol interface (`Core.Protocol`).

It is code, not security analysis. It imports Lean core, Flock's verifier for its hashes and canonical JSON, and
core's randomness, LMS, attest, protocol interface and difftest packages; no Mathlib and nothing of `security/`.
`security/Properties/Pous/` states its properties against these functions. Its difftests hold each verdict to the
memory attester's Python, one directory up, which is the reference.

## Build and test

~~~bash
lake build                                       # here: the library and every executable
# from the root:
python3 tools/verity/lean/check.py --build verity/protocols/resource_accounting/memory/pous/attester/lean   # the Lean check (lean-audit.json)
uv run tools/check/suites.py verity/protocols/resource_accounting/memory/pous/attester/lean          # the difftests
~~~

| Executable | What it decides |
| --- | --- |
| `memory-challenger` | The timed rounds' verdict, as a core protocol. Its public output is the verdict alone. |
| `wipe-protocol` | The streamed-label wipe, as the core protocol `verity/wipe/v0`. |

Both take the core protocols' command line: `id`, `check PARAMS COMMITMENT EVIDENCE`, or `serve`.
[`DESIGN.md`](DESIGN.md) gives every entry point, what each publishes, and the Python callers.
