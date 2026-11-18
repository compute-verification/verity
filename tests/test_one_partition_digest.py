"""One partition digest: the harness (`benchmarks/harness/`) computes no partition digest and names no partition rule
of its own. A circuit binds its partition under a rule the proof system defines, and the digest is the proof system's:
`verity.circuits.partition_object.digest` under `verity/partition/v1`, `verity_flock.circuit.partition_digest` under
the provisional cover, each rule's name read from there.

- No tracked source of the harness outside its tests writes a rule's name (`RULES`) as a literal.
- None hashes with SHA-512 a message that holds a zero byte and a JSON encoding (`json.dumps`, `canonical_json`): a
  rule, its zero byte and an object's canonical JSON, the partition digest's form.

Nothing is executed (AST only). A digest computed another way (another hash, a zero byte built at run time) is not
seen."""

from __future__ import annotations

import ast

from test_boundaries import _is_test, _tracked_python

HARNESS = "benchmarks/harness/"
#: the partition rules a circuit binds, as the proof system names them
RULES = frozenset({"verity/partition/v1", "flock-circuit/unit-cover/v0"})
ZERO = frozenset({b"\0", "\0"})
JSON = frozenset({"dumps", "canonical_json"})


def _name(e: ast.AST) -> str | None:
    return e.attr if isinstance(e, ast.Attribute) else e.id if isinstance(e, ast.Name) else None


def _second_definitions(text: str) -> list[str]:
    out = []
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Constant) and n.value in RULES:
            out.append(f"{n.lineno}: the rule {n.value!r} written again")
        elif isinstance(n, ast.Call) and _name(n.func) == "sha512":
            inner = [m for a in n.args for m in ast.walk(a)]
            if any(isinstance(m, ast.Constant) and isinstance(m.value, (str, bytes)) and m.value in ZERO for m in inner) and \
                    any(isinstance(m, ast.Call) and _name(m.func) in JSON for m in inner):
                out.append(f"{n.lineno}: SHA-512 of a zero byte and a JSON encoding, a partition digest computed again")
    return out


def test_the_harness_computes_no_partition_digest_of_its_own() -> None:
    found = {rel: hits for rel, text in _tracked_python().items()
             if rel.startswith(HARNESS) and not _is_test(rel) and (hits := _second_definitions(text))}
    assert not found, ("the harness defines a partition rule or digest again; import the proof system's "
                       "(`verity.circuits.partition_object.digest`, `verity_flock.circuit.partition_digest`, "
                       f"`PARTITION_V1`, `PARTITION_RULE`): {found}")


def test_the_check_sees_the_harness_s_former_digest() -> None:
    former = ('import hashlib, json\n'
              'PARTITION_V1, PARTITION_COVER = "verity/partition/v1", "flock-circuit/unit-cover/v0"\n'
              'def _digest(obj, rule=""):\n'
              '    return hashlib.sha512((rule.encode() + b"\\0" if rule else b"")\n'
              '                          + json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()\n')
    assert [h.split(":")[0] for h in _second_definitions(former)] == ["2", "2", "4"]
    assert _second_definitions('import hashlib\nhashlib.sha512(b"proof").hexdigest()\n') == []
