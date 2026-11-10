# PoUS's Lean check (`pous_check`)

PoUS's verifier as executable Lean, the check of record: what the verifier checks at setup, a run composed of the setup
and a memory challenger's signed session, the timed audit's verdict and the verdict proof block's gates. It reads the
formats the Python reference writes (`verity/protocols/pous/verifier/`), and its suite holds each part to that
reference. It is code: it requires no package of the security analysis, whose `security/Properties/Pous.lean` states
its properties.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `pous-setup COMMAND` (a Lean executable) | The setup protocol of record, `verity/pous/p2/v4/setup` (`PousCheck.SetupProtocol.protocol`), on the core protocols' command line: `id`, `check PARAMS COMMITMENT EVIDENCE`, or `serve`, which reads one case `{"commitment", "evidence", "params"}` a line and writes one answer a line. It accepts `C*` when it decodes to the committed `W`, and publishes `blocks`, `commitment`, `root` and `salt`. |
| `pous-run-protocol [--test] COMMAND` (a Lean executable) | The run protocol of record, `verity/pous/p2/v4/run` (`PousCheck.RunProtocol.protocol`), on the same command line. It accepts a run when setup accepts `C*` and publishes the run's commitment, and the run the memory challenger signed passes against it. It publishes `certified`, `eps` and `s_bits`. After `--test` it accepts the test signers' schemes and only them. |
| `pous-timed-audit READY.json LEAVES.bin AUDIT.json` (a Lean executable) | The verdict of record on a strict timed audit's record (`PousCheck.TimedAudit.verdict`), printed. `timed_audit.py`'s `lean_verdict(ready: Any, leaves: bytes, doc: Any) -> dict[str, Any]` runs it on a record held in memory; the memory challenger's `timed_audit.py` loads that file by its path. |
| `pous-run TRUSTED.json SETUP.json DIGESTS.bin SESSION.json SESSION_ID` (a Lean executable) | `PousCheck.Run.check` on a recorded run, with `TRUSTED.json` the verifier's own list of memory-challenger keys. |
| `PousCheck.SetupProtocol`, `PousCheck.RunProtocol`, `PousCheck.Run`, `PousCheck.TimedAudit`, `PousCheck.Verdict`, `PousCheck.Merkle` (Lean modules of the Lake package `pous_check`) | What another Lake package imports: the two protocols (`SetupProtocol.protocol`, `RunProtocol.protocol`), the run's check (`Run.check`), the timed audit's verdict (`TimedAudit.verdict`), the verdict proof block's gates (`Verdict.prog`), and the store's leaves and paths (`Merkle.packageLeaf`, `Merkle.verifyPath`). The properties' package, onsite Verity's composite and PoUB's Lean verifier require the package. |
| `uv run tools/verity/check/suites.py verity/protocols/pous/verifier/check` | The suite, where `lake` is: each Lean function against the Python reference through the difftest harness, the two protocols against the run in Python (`verity/protocols/pous/verifier/period.py`), and the verdict's gates against the compiled Definition (`verdict_circuit.PousVerdict`). |
| `python verity/protocols/pous/verifier/check/interop.py --all [--out FILE]` | `p2-audit-wire/v4` both ways, byte for byte, on every vector the P2 vector files hold: the Lean (`PousCheck.Wire`, `PousCheck.Audit.check`) against the Python reference, `benchmarks/pous/p2_v1/wire.py`. It exits 0 iff nothing differs. |
| `python verity/protocols/pous/verifier/check/timed_audit.py (--generated \| --replay RUN_DIR...) [--out FILE]` | The timed audit's verdict both ways, the memory challenger's Python and `pous-timed-audit`: on generated records with their negative controls, or on the audit documents a run kept. It exits 0 iff the two agree. |

`lake build` here builds the executables. The three difftest executables (`pous-setup-difftest`, `pous-audit-difftest`,
`pous-verdict-difftest`) and `scripts/difftest_vectors.py` are internal: the suite drives them.

What this directory reads from `benchmarks/`, each loaded by its path: P2's reference, wire and vector files
(`benchmarks/pous/p2_v1/`), which `interop.py` and the vector tests hold the Lean to, and the harness's responder
(`benchmarks/pous/band_gpu/harness.py`), which one test of the timed audit drives.
