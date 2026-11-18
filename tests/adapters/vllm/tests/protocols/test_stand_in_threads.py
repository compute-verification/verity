"""A stand-in challenger serves its challenge phase in a thread, and that thread is a daemon.

A test that fails with the challenge phase open never ends it. The thread holds the stand-in and so the client's socket,
which nothing then closes, and it waits for the client for good. Python ends only once every thread that is no daemon
has, so a run with one such failure would print its failures and then never end."""
import ast
from pathlib import Path

import pytest

#: the modules that hold a stand-in challenger: `Host`, `_GateSession` and `_Session`
STAND_INS = ("challenger_host.py", "test_pouw_pearl_c_served.py", "test_rt4_h1_challenger.py")


@pytest.mark.parametrize("module", STAND_INS)
def test_a_stand_in_challengers_thread_is_a_daemon(module):
    made = [n for n in ast.walk(ast.parse((Path(__file__).parent / module).read_text()))
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "threading.Thread"]
    assert made, f"tests/protocols/{module} makes no thread: take it out of STAND_INS"
    for call in made:
        daemon = next((k.value for k in call.keywords if k.arg == "daemon"), None)
        assert isinstance(daemon, ast.Constant) and daemon.value is True, (
            f"tests/protocols/{module}:{call.lineno}: a thread made without daemon=True. A test that fails with its "
            "challenge phase open would hold the process at its exit: make it threading.Thread(..., daemon=True)")
