"""`verity_vllm.program.frontend.rules.common.known`: a rule reads a matched operand's static shape through it, and an operand
with none is a TypeError there, as indexing or unpacking the absent shape was."""
from __future__ import annotations

import pytest

from verity_vllm.program.frontend.rules.common import known


def test_known_hands_back_the_shape_it_is_given():
    assert known((2, 3)) == (2, 3)
    assert known(()) == ()


def test_known_raises_where_the_operand_has_no_static_shape():
    with pytest.raises(TypeError, match="no static shape"):
        known(None)
