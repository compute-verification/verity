"""The one rule for where a suite's tests are and what a suite is called (`research.suite_layout`), which `suites.py`, the queue,
`suite_size` and the graph read: on today's layout, where a suite's tests sit in its own directory, and on the mirrored one,
where the tests of `X` are at `tests/X`."""
from __future__ import annotations

import pytest

from research import suite_layout as L

DIRS = {"tools/research": "research", "adapters/vllm": "verity-vllm"}


@pytest.mark.parametrize(("path", "project", "name"), [
    (".", {"name": "verity"}, "repository"),
    ("./", {}, "repository"),
    ("tools/research", {"name": "research"}, "research"),
    ("tools/verity_graph", {"name": "Verity_Graph"}, "verity-graph"),
    ("verity/tests", {}, "verity/tests"),                                  # a suite with no distribution: its path
    ("adapters/vllm/tests", {"name": "verity-vllm-tests"}, "verity-vllm"),   # the tests of a member, in its own directory
    ("tests/adapters/vllm/tests", {"name": "verity-vllm-tests"}, "verity-vllm"),   # and mirrored
    ("tests/adapters/vllm", {"name": "verity-vllm-tests"}, "verity-vllm"),
    ("tests/tools/research", {"name": "research-tests"}, "research"),
    ("tests/verity", {}, "verity"),                                        # mirrored with no distribution: what it mirrors
    ("tests/verity/protocols", {"version": "0"}, "verity/protocols"),
    ("tests/x", {"name": "unrelated-tests"}, "unrelated-tests"),             # a `-tests` distribution of no member keeps its name
    ("tests/tools/research", {"name": "verity-vllm-tests"}, "verity-vllm-tests"),   # nor of another member's directory
])
def test_a_suite_is_named_by_its_distribution_its_path_or_what_it_mirrors(path, project, name):
    assert L.suite_name(path, project, DIRS) == name


def test_the_members_are_read_from_the_workspace_with_their_names_normalized():
    tomls = {".": {"tool": {"uv": {"workspace": {"members": ["tools/research/", "adapters/vllm", "virtual"]}}}},
             "tools/research": {"project": {"name": "Research"}}, "adapters/vllm": {"project": {"name": "verity_vllm"}},
             "virtual": {}}
    assert L.member_dirs(lambda d: tomls.get(d, {})) == {"tools/research": "research", "adapters/vllm": "verity-vllm"}


def test_tests_names_the_repository_suite_and_a_mirrored_suite_is_named_by_what_it_holds_the_tests_of():
    assert {"tests", "tests/", "repository", "."} <= set(L.names(".", "repository"))
    assert "tests" not in L.names("verity/tests", "verity/tests") and "tests/" not in L.names("verity/tests", "verity/tests")
    assert set(L.names("tests/adapters/vllm/tests", "verity-vllm")) == {
        "verity-vllm", "tests/adapters/vllm/tests", "adapters/vllm/tests", "adapters/vllm"}
    assert set(L.names("tools/check", "verity-check")) == {"verity-check", "tools/check"}


def test_a_path_covers_its_tests_under_the_roots_tests_where_they_exist():
    there = {"tests/verity", "tests/tools/research", "tests/README.md"}
    assert L.mirrors(["verity", "tools/research/", ".", "", "tests/verity", "tools/check", "README.md"], there.__contains__) == {
        "tests/verity", "tests/tools/research", "tests/README.md"}
    assert L.mirrors(["verity", "tools/check"], lambda p: False) == set()


@pytest.mark.parametrize(("path", "own", "stands"), [
    ("tests/test_boundaries.py", True, None),
    ("tests/conftest.py", True, None),
    ("tests/lint/test_repo_ruff.py", True, None),
    ("tests/verity/core/test_claims.py", False, "verity/core/test_claims.py"),
    ("tests/tools/research/test_queue.py", False, "tools/research/test_queue.py"),
    ("tests/linted/test_a.py", False, "linted/test_a.py"),
    ("tests/tests/test_a.py", True, None),
    ("verity/tests/test_a.py", False, None),
    ("testsuite/test_a.py", False, None),
])
def test_the_repository_suites_own_under_tests_are_its_files_and_own_directories(path, own, stands):
    assert L.root_owns(path) is own and L.stands_for(path) == stands


@pytest.mark.parametrize(("path", "dirs", "held"), [
    ("verity/core/a.py", ("verity/core/",), True),
    ("tests/verity/core/test_a.py", ("verity/core/",), True),
    ("tests/verity/core/test_a.py", ("tests/verity/",), True),
    ("tests/verity/circuits/test_a.py", ("verity/core/",), False),
    ("tests/verity/corex/test_a.py", ("verity/core/",), False),
    ("tests/test_boundaries.py", ("tests/",), True),
    ("tests/lint/test_a.py", ("tests/",), True),
    ("tests/verity/core/test_a.py", ("tests/",), False),
    ("tests/lint/test_a.py", ("lint/",), False),
    ("tests/tests/test_a.py", ("tests/",), True),
    ("verity/core/a.py", (), False),
])
def test_a_directory_holds_its_tests_under_tests_and_tests_holds_the_repository_suites_own(path, dirs, held):
    assert L.held(path, dirs) is held
