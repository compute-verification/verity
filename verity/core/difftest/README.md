# verity/core/difftest: the Lean against the Python that mirrors it

Lean is the definition and the verifier of record (Daniel, 18 Oct; root `AGENTS.md`, "Design rules"), and Python mirrors
it for the prover's speed. Each piece of core in executable Lean (`verity/core/<core path>`, a Lake package of its own) and
its Python are held together by a differential test: Hypothesis draws inputs, invalid ones included, and each goes to the
Python (the harness's `reference`) and to the Lean executable, which must agree exactly. Until a protocol's verifier of
record switches to its Lean (the 8 Oct procedure, `.agents/skills/friction/SKILL.md`), its Python verifier is what the Lean
is checked against; after, the Python is held to the Lean.

- `lean/` is the Lake package `verity_difftest`, lib `Difftest`: `Difftest.serve` answers requests `{"fn", "input"}` one
  JSON line at a time with `{"output"}`, `{"reject"}`, or `{"error"}` for a protocol fault (an unknown `fn`, a bad
  request, an output with no canonical form). Output is canonical JSON, written by C-Flock's `Flock.canon` (required by path
  from `verity/core/service/flock`). `difftest-example` is the port the harness's own tests drive.
- `python/verity_difftest` is the Python side: `Port`, `run`, `replay`, `Server`, `server_fixture`, `canonical`. `run`
  keeps one server open, compares outputs as canonical JSON, requires reject parity (both reject or neither; reasons may
  differ), treats an undeclared exception or a protocol fault as a difference, and fails a vacuous run (no output, or no
  reject unless waived). A failure is a `Mismatch` naming the minimal input and the seed that finds it again. Runs are
  derandomized, with no example database and no deadline.
- Protocol mode drives a protocol's own `serve` executable, the interface's `Protocol.serve`
  (`verity/core/protocols/interface/README.md`): a `ProtocolPort` sends each case line and compares the
  outcome's `verdict`, `commitment`, `public` and `profile` exactly and its `reason` not at all, through the same server,
  comparison and shrinking as `Port`. `verity_difftest.protocol` is the interface's framing in Python (`answer`, `unhex`,
  `canonical_json`, `nat`), so a reference writes only its protocol's decoders and check. A run fails as vacuous unless
  some case is accepted and, unless waived, some is not.
- `verity_difftest.settings` declares every environment variable core's difftests read (`SETTINGS`, read through `get`):
  opt-ins to the long or recorded runs, never a value a comparison depends on.

## Using it from a port

~~~python
# PORT/lean/lakefile.toml: [[require]] name = "verity_difftest", path = "<relative path to this lean/>"
# PORT/lean/DifftestMain.lean:  def main : IO Unit := Difftest.serve (Difftest.table [("encode", Port.encode)])
# PORT/lean/lakefile.toml: [[lean_exe]] name = "port-difftest", root = "DifftestMain"
# PORT/pyproject.toml: a suite in AFTER_LEAN (tools/verity/check/suites.py); inputs list what lake reads outside PORT
import shutil, pytest, verity_difftest as D
from hypothesis import strategies as st
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
PORT = D.Port("port-difftest", "encode", reference=encode, inputs=st.one_of(VALID, INVALID), package=LEAN, rejects=(ValueError,))
server = D.server_fixture(PORT)
def test_encode(server): D.run(PORT, examples=500, server=server)
~~~

A protocol port (`tests/test_protocol_port.py` is the worked example, `CoreProtocol.Example`):

~~~python
# PROTOCOL/lean/lakefile.toml: [[lean_exe]] name = "my-protocol", root = "Main"
# PROTOCOL/lean/Main.lean:      def main (args : List String) : IO UInt32 := My.protocol.main args  (the interface's CLI)
from verity_difftest import protocol as P
def reference(case):  # the interface's order: case, params, commitment, digest, evidence, check
    return P.answer("verity/my/v0", case, params=decode_params, commitment=decode_root, evidence=decode_evidence,
                    digest=digest_of, check=check)  # decoders raise D.Reject; check gives ("accept", public, profile)
# cases: valid ones, plus one malformation each: hex Flock.unhex refuses (uppercase, a space, odd length),
# a field missing or not a string, params and evidence that don't decode (non-canonical, keys out of order, floats)
PORT = D.ProtocolPort("my-protocol", reference, inputs=st.one_of(VALID, MALFORMED), package=LEAN)
server = D.server_fixture(PORT)
def test_my_protocol(server): D.run(PORT, examples=500, server=server)
~~~

The harness never skips. When `lake` is missing or the build fails it raises. The port's test module skips on `lake`
itself, so `tests/test_no_wall_clock.py` requires its suite to be among those `check` runs after its Lean steps. A reference
returns JSON values (integers, never floats) or raises one of `rejects` (default `verity_difftest.Reject`).

Run the suite with `uv run tools/verity/check/suites.py verity-difftest`.
