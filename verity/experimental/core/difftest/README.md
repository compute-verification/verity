# verity/experimental/core/difftest: Lean ports against today's Python

Each port of core to executable Lean (`verity/experimental/core/<core path>`, a Lake package of its own) is held to the
Python it replaces by a differential test: Hypothesis draws inputs, invalid ones included, and each goes to the Python
reference and to the port's Lean executable, which must agree exactly.

- `lean/` is the Lake package `verity_difftest`, lib `Difftest`: `Difftest.serve` answers requests `{"fn", "input"}` one
  JSON line at a time with `{"output"}`, `{"reject"}`, or `{"error"}` for a protocol fault (an unknown `fn`, a bad
  request, an output with no canonical form). Output is canonical JSON, written by C-Flock's `Flock.canon` (required by path
  from `verity/core/service/flock`). `difftest-example` is the port the harness's own tests drive.
- `python/verity_difftest` is the Python side: `Port`, `run`, `replay`, `Server`, `server_fixture`, `canonical`. `run`
  keeps one server open, compares outputs as canonical JSON, requires reject parity (both reject or neither; reasons may
  differ), treats an undeclared exception or a protocol fault as a difference, and fails a vacuous run (no output, or no
  reject unless waived). A failure is a `Mismatch` naming the minimal input and the seed that finds it again. Runs are
  derandomized, with no example database and no deadline.

## Using it from a port

~~~python
# PORT/lean/lakefile.toml: [[require]] name = "verity_difftest", path = "<relative path to this lean/>"
# PORT/lean/DifftestMain.lean:  def main : IO Unit := Difftest.serve (Difftest.table [("encode", Port.encode)])
# PORT/lean/lakefile.toml: [[lean_exe]] name = "port-difftest", root = "DifftestMain"
# PORT/pyproject.toml: a suite in AFTER_LEAN (tools/verity/check/check.py); inputs list what lake reads outside PORT
import shutil, pytest, verity_difftest as D
from hypothesis import strategies as st
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
PORT = D.Port("port-difftest", "encode", reference=encode, inputs=st.one_of(VALID, INVALID), package=LEAN, rejects=(ValueError,))
server = D.server_fixture(PORT)
def test_encode(server): D.run(PORT, examples=500, server=server)
~~~

The harness never skips. When `lake` is missing or the build fails it raises. The port's test module skips on `lake`
itself, so `tests/test_no_wall_clock.py` requires its suite to be among those `check` runs after its Lean steps. A reference
returns JSON values (integers, never floats) or raises one of `rejects` (default `verity_difftest.Reject`).

Run the suite with `uv run tools/verity/check/suites.py verity-difftest`. A bare `uv run` inside this directory stops its
workspace search at `verity/experimental/pyproject.toml` and makes a venv of its own.
