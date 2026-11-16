# tools/difftest: design

## The two sides

- `verity/circuits/difftest/` is the Lake package `verity_difftest`, lib `Difftest`: `Difftest.serve` answers requests `{"fn", "input"}` one
  JSON line at a time with `{"output"}`, `{"reject"}`, or `{"error"}` for a protocol fault (an unknown `fn`, a bad
  request, an output with no canonical form). Output is canonical JSON, written by `Flock.canon` (`Base.Canon`, required by path
  from `verity/circuits/base`). `difftest-example` is the port the harness's own tests drive. `annot-difftest`
  (`AnnotDifftest.lean`) serves Flock's annotation check, `Flock.Annot`, and its per-gate reference, for the tests that
  hold `verity.circuits.query_ast` to them; `verity_difftest.annot` is its Python side (`port`, `server`, `instance`,
  and `INPUTS`, the files a calling suite declares), and `tests/tools/difftest/test_annot.py` holds it to `query_ast` on small Programs.
  `flock-json-difftest` is the JSON parser's port: `Flock.JsonParse.parse` (`Base.JsonParse`)
  and Lean core's `Lean.Json.parse` against one Python mirror, value and refusal text alike (`tests/tools/difftest/test_flock_json_port.py`).
  `tests/tools/difftest/test_canonical_json_port.py` holds `Core.canonicalJson`'s answer under each of the two to one Python mirror the
  same way; its port, `core-canonical-json-difftest`, is in the interface's package (`verity/protocols/interface`).
  The package requires the base and the circuit format (`core_base`, `core_circuits`) and nothing of Flock's package:
  the port of the honest worker's requests, which reads the ZK server, is in `verity/protocols/proofs/recursion/verifier/zk_server`.
- `python/verity_difftest` is the Python side: `Port`, `run`, `replay`, `Server`, `server_fixture`, `canonical`. `run`
  keeps one server open, compares outputs as canonical JSON, requires reject parity (both reject or neither; reasons may
  differ), treats an undeclared exception or a protocol fault as a difference, and fails a vacuous run (no output, or no
  reject unless waived). A failure is a `Mismatch` naming the minimal input and the seed that finds it again (under
  `shrink=False`, the first input the sides differ on, as sampled). Runs are derandomized, with no example database and
  no deadline.
- Protocol mode drives a protocol's own `serve` executable, the interface's `Protocol.serve`
  (`verity/protocols/interface/README.md`): a `ProtocolPort` sends each case line and compares the
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
# PORT/pyproject.toml: a suite in AFTER_LEAN (tools/check/suites.py); inputs list what lake reads outside PORT
import shutil, pytest, verity_difftest as D
from hypothesis import strategies as st
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
PORT = D.Port("port-difftest", "encode", reference=encode, inputs=st.one_of(VALID, INVALID), package=LEAN, rejects=(ValueError,))
server = D.server_fixture(PORT)
def test_encode(server): D.run(PORT, examples=500, server=server)
~~~

A protocol port (`tests/tools/difftest/test_protocol_port.py` is the worked example, `CoreProtocol.Example`):

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
