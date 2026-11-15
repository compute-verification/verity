"""`lint_counts.py`'s arithmetic and output on made-up findings and entries, with no lint run: a file's `past` sums each
key's excess over its entry, so a key under its entry never cancels one past its own; keys come back from the child's
JSON as the lint made them; and each line is five tab-separated columns, sorted, an uncounted list's and a list that is
no ratchet's each marked apart."""
from __future__ import annotations

import json
from collections import Counter

import lint_counts as C


def test_a_key_past_its_entry_is_counted_beside_one_under_its_own() -> None:
    found = Counter({("a.py", "E1"): 3, ("a.py", "E2"): 0, ("b.py", "E1"): 1, "c.py": 2})
    allowed = Counter({("a.py", "E1"): 1, ("a.py", "E2"): 2, ("b.py", "E1"): 1, "d.py": 4})
    assert C.by_file(found, allowed) == {"a.py": (3, 3, 2), "b.py": (1, 1, 0), "c.py": (2, 0, 2), "d.py": (0, 4, 0)}


def test_files_a_path_names_alike_are_summed() -> None:
    found, allowed = Counter({("x/a.py", "k"): 2, ("a.py", "k"): 1}), Counter({("x/a.py", "k"): 1})
    assert C.by_file(found, allowed, lambda f: f.removeprefix("x/")) == {"a.py": (3, 1, 2)}


def test_a_key_comes_back_from_json_as_the_lint_made_it() -> None:
    keys = ["a.py", ("a.py", "E1"), ("a.py", "module", "<module>", "lines"), ("a.py", ("x", "y"))]
    assert [C.key(k) for k in json.loads(json.dumps(keys))] == keys


def test_each_line_is_five_sorted_tab_separated_columns() -> None:
    rows = [("p10_size", "b.py", "1", "0", "1"), ("p10_size", "a.py", "2", "2", "0"), ("repo_ruff", *C.NOT_COUNTED),
            ("repo_python_verdicts", *C.NO_RATCHET)]
    assert C.lines(rows) == ("p10_size\ta.py\t2\t2\t0\n"
                             "p10_size\tb.py\t1\t0\t1\n"
                             "repo_python_verdicts\t-\tn/a\tn/a\tn/a\n"
                             "repo_ruff\t-\t-\t-\t-\n")
    assert len(C.NOT_COUNTED) == len(C.NO_RATCHET) == 4 and C.NOT_COUNTED != C.NO_RATCHET


def test_only_a_named_list_is_no_ratchets_and_any_other_without_allowed_is_not_counted() -> None:
    why = "tests/lint/test_x.py has no allowed(), so its list is no ratchet's"
    assert C.unread("repo_python_verdicts", None, why) == (C.NO_RATCHET, why)
    columns, said = C.unread("repo_env_reads", None, why)
    assert columns == C.NOT_COUNTED and "NO_RATCHETS does not name it" in said
    assert C.unread("repo_ruff", "no ruff", None) == (C.NOT_COUNTED, "no ruff")
    assert C.unread("repo_env_reads", None, None) == (None, "")
