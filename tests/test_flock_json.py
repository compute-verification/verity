"""C-Flock's Lean, sampled proofs', the interface every protocol implements and the proofs of `Verity`'s properties read
JSON only through `Flock.JsonParse.parse`, the total parser the kernel can evaluate.

`Lean.Json.parse` is a `partial def`, an opaque constant to the kernel, so no proof can show that a text it reads parses;
`JsonParse.parse` reads the same language, with the same values and refusal strings (`Flock.Test.JsonParse` holds the
two to each other). A scan of the Lean under `verity/protocols/proofs/flock/verifier/`, the base packages it requires
(`verity/circuits/base/`, `circuits/` and `commitments/`), `verity/protocols/proofs/spot/verifier/lean/`,
`verity/protocols/interface/` and `security/Proofs/Properties/Verity/` refuses a call of `Lean.Json.parse`, written
`Json.parse` under `open Lean` or in full, outside the module that mirrors it, the tests that compare the two, and the
declarations `PENDING` names.
"""

from __future__ import annotations

import re

from test_repository import ROOT, _tracked

TREES = ("verity/protocols/proofs/flock/verifier/", "verity/circuits/base/", "verity/circuits/lean/",
         "verity/commitments/lean/", "verity/protocols/proofs/spot/verifier/lean/", "verity/protocols/interface/",
         "security/Proofs/Properties/Verity/")
NOT_READ = ("verity/protocols/proofs/recursion/verifier/firewall/", "verity/protocols/proofs/recursion/verifier/vstar_cli/")  # Lake packages of their own, not held to the total parser yet (proofs)
ALLOWED = {
    "verity/circuits/base/Base/JsonParse.lean": "the total parser, whose docstring names the source it mirrors",
    "verity/protocols/proofs/flock/verifier/Flock/Test/JsonParse.lean": "the test holding the total parser to Lean.Json.parse",
    "verity/protocols/interface/CoreProtocol/CanonicalJsonDifftest.lean":
        "the difftest holding Core.canonicalJson's answer to the one it gives reading with Lean.Json.parse",
}
INTERFACE = "verity/protocols/interface/CoreProtocol/"
#: (file, declaration) still calling `Lean.Json.parse`, each with the switch that deletes its entry. An entry whose file
#: is in the tree names a declaration that still calls it; one whose file isn't yet arrives with that file.
PENDING: dict[tuple[str, str], str] = {
    (INTERFACE + "Protocol/Canonical.lean", "strValue"):
        "the in-place reader's switch: a string holding an escape is read from its own slice of the text",
    (INTERFACE + "Protocol/Canonical.lean", "refusal"):
        "the in-place reader's switch: the reason Core.canonicalJson gives for a text it refuses",
    (INTERFACE + "Example.lean", "refusal"): "the example's guards' switch: an answer line read back",
    (INTERFACE + "Example.lean", "profileRefusal"): "the example's guards' switch: a profile read back",
}

#: `/- … -/` and `-- …`, blanked so that line numbers stay
COMMENT = re.compile(r"/-.*?-/|--[^\n]*", re.S)
CALL = re.compile(r"(?<![\w.])(?:Lean\.)?Json\.parse\b(?![?'\w])")
DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|partial|unsafe)\s+)*"
                  r"(?:def|theorem|lemma|abbrev|instance|structure|inductive)\s+([^\s:({\[]+)", re.M)


def _calls(path: str) -> list[tuple[int, str]]:
    """(line, the declaration it's in) of each `Lean.Json.parse` call in a Lean file, comments aside."""
    text = COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), (ROOT / path).read_text())
    decls = [(d.start(), d.group(1).rsplit(".", 1)[-1]) for d in DECL.finditer(text)]
    return [(text.count("\n", 0, m.start()) + 1, next((n for at, n in reversed(decls) if at < m.start()), ""))
            for m in CALL.finditer(text)]


def test_lean_reads_json_through_the_total_parser() -> None:
    bad = []
    for path in _tracked():
        if not (path.startswith(TREES) and path.endswith(".lean")) or path in ALLOWED or path.startswith(NOT_READ):
            continue
        bad += [f"{path}:{line} ({decl or 'no declaration'})" for line, decl in _calls(path) if (path, decl) not in PENDING]
    assert not bad, ("Lean.Json.parse is partial, so no proof reads through it: call Flock.JsonParse.parse (JsonParse.parse "
                     "inside namespace Flock), which reads the same language:\n  " + "\n  ".join(bad))


def test_allowed_entries_exist() -> None:
    assert all((ROOT / p).exists() for p in ALLOWED), "an ALLOWED entry names a file that no longer exists"


def test_pending_reads_still_call_it() -> None:
    tracked = set(_tracked())
    done = [f"{p} {d}" for p, d in PENDING if p in tracked and d not in {decl for _, decl in _calls(p)}]
    assert not done, "these declarations read through the total parser now: delete their PENDING entries:\n  " + "\n  ".join(done)
