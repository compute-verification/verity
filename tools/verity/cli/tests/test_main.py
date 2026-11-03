"""`verity`'s dispatch: an unknown command is named on its own line, then the usage."""
from __future__ import annotations

import pytest

import verity_cli


def test_an_unknown_command_is_named_then_the_usage_follows():
    with pytest.raises(SystemExit) as e:
        verity_cli.main(["nosuch"])
    first, rest = str(e.value.code).split("\n", 1)
    assert first == "verity: no command 'nosuch'"
    assert rest == verity_cli.__doc__
