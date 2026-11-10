"""The repository suite's bookkeeping tests (`BOOKKEEPING`) report what they find as a `Bookkeeping` warning and pass:
the tree stays landable, and a later PR fixes the finding.  Bookkeeping is what can be put right after
landing without any claim being false meanwhile: lane names in code text, retired terms, dead doc paths, naming conventions,
stale allowlist entries, Markdown over its cap.  A scan's own tests (that it reads each form) stay plain tests and fail, and
so does a bookkeeping test that raises anything but a failed check (its scan broke), and anything that can make a claim false
or can't be undone (a secret, a blob over its cap, an import across the boundaries): ruff's errors among them, which
`test_repo_ruff.py`'s `test_no_ruff_error` keeps failing.  The ratchet gate calls the lints' `test_no_new*` functions itself,
outside pytest, so it still refuses what they find.  `tests/test_bookkeeping.py` holds the list to the tree."""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: {test file: the tests of it whose findings warn}
BOOKKEEPING = {
    "tests/lint/test_repo_ruff.py": ("test_no_new_findings",),
    "tests/lint/test_repo_lane_text.py": ("test_no_new_lane_text", "test_pinned_files_hold_none",
                                          "test_the_list_is_well_formed_and_an_entry_moves_with_its_file"),
    "tests/lint/test_repo_retired_terms.py": ("test_no_new_retired_terms",),
    "tests/test_retired_terms.py": ("test_no_retired_term_in_prose", "test_no_retired_term_in_the_maintained_docs",
                                    "test_no_retired_word_in_a_new_lean_name", "test_the_glossary_records_the_retired_terms",
                                    "test_spot_s_older_name_only_falls"),
    "tests/test_doc_paths.py": ("test_every_path_a_document_names_exists", "test_not_paths_are_current"),
    "tests/test_documented_commands.py": ("test_every_documented_command_line_parses",
                                          "test_known_lists_only_lines_that_still_fail"),
    "tests/test_glossary_lean_names.py": ("test_every_row_names_its_lean_or_says_it_has_none",
                                          "test_every_lean_name_in_the_symbol_table_exists",
                                          "test_every_lean_name_in_the_security_argument_exists"),
    "tests/test_lean_vocabulary.py": ("test_every_hit_is_canonical_or_listed", "test_known_shrinks_by_count",
                                      "test_known_reasons_name_their_owner", "test_canonical_entries",
                                      "test_mirrors_name_their_port"),
    "tests/test_module_strings.py": ("test_the_exemptions_are_current",),
    "tests/test_cli_conventions.py": ("test_no_new_synonym_flags", "test_known_list_has_no_stale_entries"),
    "tests/test_provider_names.py": ("test_no_provider_name_in_a_security_statement",),
    "tests/test_refusal_remediations.py": ("test_every_command_a_refusal_names_is_remediated_or_known",
                                           "test_the_lists_are_current"),
    "tests/test_repository.py": ("test_tracked_markdown_is_maintained_documentation", "test_no_report_genre_basenames",
                                 "test_oversize_skills_only_shrink", "test_markdown_size_caps",
                                 "test_allowlists_name_tracked_files", "test_owned_paths_exist", "test_every_path_has_owner"),
}


class Bookkeeping(UserWarning):
    """What a bookkeeping test found: a later PR fixes it."""


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "bookkeeping: what it finds warns (`Bookkeeping`) and a later PR fixes it")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if isinstance(item, pytest.Function) and item.path.resolve().is_relative_to(ROOT) and \
                item.originalname in BOOKKEEPING.get(item.path.resolve().relative_to(ROOT).as_posix(), ()):
            item.add_marker(pytest.mark.bookkeeping)


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function):
    if pyfuncitem.get_closest_marker("bookkeeping") is None:
        return (yield)
    try:
        return (yield)
    except (AssertionError, pytest.fail.Exception) as e:
        warnings.warn(Bookkeeping(f"{pyfuncitem.nodeid}: {e}"), stacklevel=1)
        return True
