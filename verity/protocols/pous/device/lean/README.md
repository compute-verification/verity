# The memory challenger's verdicts in Lean (`core_memory_challenger`)

This Lake package is code: the memory challenger's two verdicts as executable Lean, their record formats and crypto, and
each verdict behind the core protocol interface (`Core.Protocol`). It imports Lean core, Flock's verifier for its
hashes and canonical JSON, and core's randomness, LMS, attest, protocol interface and difftest packages; no Mathlib and
nothing of `security/`. `security/Properties/Pous/` states its properties against these functions (the area's README,
one directory up).

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `wipe-protocol COMMAND` (the Lean executable) | The streamed-label wipe as a core protocol, `verity/wipe/v0` (`MemoryChallenger.WipeProtocol.protocol`, `MemoryChallenger/WipeProtocol.lean`), on the core protocols' command line: `id`, `check PARAMS COMMITMENT EVIDENCE`, or `serve`, one case a line in and one answer a line out. An accepted, covered run publishes `foreign_bits` at `eps = 2^-128` (`certify`), and an unsigned one is undecided. |
| `memory-challenger COMMAND` (the Lean executable) | The timed rounds' verdict as a core protocol (`MemoryChallenger.protocol`, `MemoryChallenger/Protocol.lean`), on the same command line. Its public output is the verdict alone. |
| `verdict.py`'s `verdict(signed: mc.SignedSession, commitment: mc.SessionCommitment, openings: Sequence[mc.CoinOpening], expected: Callable[[int], bytes], *, trusted: frozenset[bytes] \| set[bytes], schemes: Mapping[str, Any] = mc.SCHEMES) -> mc.Verdict` | `memory_challenger.verdict` computed by the Lean, behind the Python's signature: the verdict on a record as its caller holds it (`MemoryChallenger.Judge.judge`, `MemoryChallenger/Judge.lean`, the executable `memory-challenger-verdict`). It raises `ValueError` where the Lean refuses the record. The file is loaded by path: this directory is no Python package. |
| `wipe_verdict.py`'s `judge(signed: sw.SignedWipe, commitment: sw.WipeCommitment, *, trusted: frozenset[bytes] \| set[bytes], schemes: Mapping[str, Any] = sw.SCHEMES, decide: str = VERDICT) -> tuple[sw.Verdict, dict[str, dict[str, Any]]]` | The wipe's verdict on a record as its caller holds it, by the side `decide` names (`lean`, `both` or `python`), with each side's verdict as it gave it (`MemoryChallenger.StreamedWipe.judge`, `MemoryChallenger/WipeJudge.lean`, the executable `streamed-wipe-verdict`). Loaded by path too. |
| `MemoryChallenger.Judge`, `MemoryChallenger.Records`, `MemoryChallenger.StreamedWipe`, `MemoryChallenger.WipeProtocol` (Lean modules) | What another Lake package imports. `MemoryChallenger.verdict` (`MemoryChallenger/Verdict.lean`, which `Judge` imports through `Protocol`) is the check of a signed run of timed rounds against the commitment released with it; it fails closed, and its reasons are the Python's word for word. `MemoryChallenger.StreamedWipe.verdict` (`MemoryChallenger/StreamedWipe.lean`) is the wipe's check of a signed run; it refuses a batch of more than one read, which the guarantee doesn't cover. `MemoryChallenger.Records` and `MemoryChallenger.SignedMessage` are the coin tree, the run's commitment, the round record, the records root and the message the device signs. `WipeProtocol.protocol` and `WipeProtocol.check` are the wipe's protocol and its decision. |

Internal: `MemoryChallenger.Difftest` (`memory-challenger-difftest`), the functions `tests/` holds to the Python, and
`memory-challenger-test` and `wipe-protocol-test`, the two protocols under ids of their own that accept only the test
signers' schemes, for the difftests.

## Build and test

~~~bash
lake build                                       # here: the library and every executable
# from the root:
python3 tools/verity/lean/check.py --build verity/protocols/pous/device/lean   # the Lean check (lean-audit.json)
uv run tools/verity/check/suites.py verity/protocols/pous/device/lean          # the difftests in tests/
~~~
