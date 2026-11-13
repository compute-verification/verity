"""What the Match does with a value that only a caller hands it, and that no loader of the package produces:
`verity_vllm.check.match.per_request._compare_component`, given a component's directory and no loaded component, fails where it
reads the component's digest."""
from __future__ import annotations

import pytest

from verity_vllm.check.match import per_request


def test_a_component_directory_with_no_loaded_component_fails_where_its_digest_is_read():
    out: dict[str, list] = {"fails": []}
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'digest'"):
        per_request._compare_component(None, "dag", False, {"request_id": "r0"}, {}, None, None, "a-dir", 3, out)
    assert out == {"fails": []}
    assert per_request._compare_component(None, "dag", False, {"request_id": "r0", "T_bound": 4}, {}, None, None, None, 3, out) is out
    assert out["fails"] == [(per_request.SUBCIRCUITS, "r0: no component Program in the global Program for shape (LP 3, T_bound 4)")]
