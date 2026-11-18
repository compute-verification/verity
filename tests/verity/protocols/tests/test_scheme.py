"""The resource scheme interface and its registry (`verity.protocols.resource_accounting.scheme`)."""

from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.protocols.resource_accounting import scheme as S

EPS = Fraction(1, 2**40)


def _toy(name: str = "toy-bandwidth", **kw) -> S.Scheme:
    return S.Scheme(**{
        "name": name, "protocol": f"verity/{name}/v0", "resource": "read",
        "blocks": (S.Block("ToyReceipt_v1", lambda shape: 8 * shape["bytes"]),),
        "assumptions": ("ToyNoCompression",),
        "log": (S.Order("payload", "key"),
                S.Order("receipts", "seed", within_ns=1_000_000, assumption="ToyNoCompression")),
        "charges": (S.Charge("on-chip", "on_chip_bits"), S.Charge("rebuilt", "rebuilt_bits")),
        "settings": {"refresh": False}, "requires": {"coins-at-rate": "its own challenger"},
        "delta_scheme": Fraction(0), "over": "the session's credited bits, cleared of (1 - γ)",
        "under": "the tier's reads, in bits", "premises": (), **kw})


@pytest.fixture(autouse=True)
def _empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(S, "SCHEMES", {})


def test_picked_by_name_its_module_imported_on_first_use(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from verity.protocols.compose import schemes

    (tmp_path / "toy_declaration.py").write_text(
        "from verity.protocols.resource_accounting import scheme as S\n"
        "S.register(S.Scheme('toy', 'toy', 'memory', S.Lacks('a stand-in'), (), (), ()))\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(schemes, "DECLARATIONS", {"toy": "toy_declaration", "absent": "toy_declaration"})
    assert "toy" not in S.SCHEMES
    assert schemes.get("toy") is S.SCHEMES["toy"]
    with pytest.raises(KeyError, match="declared: absent, toy"):
        schemes.get("pouw")
    with pytest.raises(KeyError, match="registers no scheme 'absent'"):
        schemes.get("absent")
    with pytest.raises(KeyError, match="registered already"):
        S.register(_toy("toy"))


def test_a_shell_reads_one_field_of_a_declaration(monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path) -> None:
    """`python3 -m verity.protocols.compose.schemes NAME FIELD`, what the benchmarks' shell scripts read instead of a
    literal of their own."""
    from verity.protocols.compose import schemes

    (tmp_path / "toy_shell_declaration.py").write_text(
        "from verity.protocols.resource_accounting import scheme as S\n"
        "S.register(S.Scheme('toy', 'verity/toy/v0', 'memory', S.Lacks('a stand-in'), (), (), (), settings={'reads': 5, 'on': True},\n"
        "                    option='toy:at-v1'))\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(schemes, "DECLARATIONS", {"toy": "toy_shell_declaration"})
    for field, printed in (("protocol", "verity/toy/v0"), ("option", "toy:at-v1"),
                           ("settings", '{"on": true, "reads": 5}')):
        assert schemes.main(["toy", field]) == 0 and capsys.readouterr().out == printed + "\n"
    assert schemes.main(["pouw", "protocol"]) == 2 and "declared: toy" in capsys.readouterr().err
    assert schemes.main(["toy", "name"]) == 2 and "protocol|option|settings" in capsys.readouterr().err


def test_settings_toggle() -> None:
    s = _toy()
    assert S.settings(s, {}) == {"refresh": False}
    assert S.settings(s, {"refresh": True}) == {"refresh": True}
    with pytest.raises(ValueError, match="no setting"):
        S.settings(s, {"lifetime": 3})
    with pytest.raises(ValueError, match="is a bool"):
        S.settings(s, {"refresh": 1})


def test_a_settings_values_are_checked_by_its_declaration() -> None:
    s = _toy(settings={"challenges": 320},
             check_settings=lambda v: None if v["challenges"] >= 1 else f"challenges is at least 1, not {v['challenges']}")
    assert S.settings(s, {}) == {"challenges": 320} and S.settings(s, {"challenges": 1}) == {"challenges": 1}
    with pytest.raises(ValueError, match="toy-bandwidth: challenges is at least 1, not 0"):
        S.settings(s, {"challenges": 0})


def test_block_cost_and_unit() -> None:
    s = _toy()
    assert s.unit == "bits" and s.blocks[0].cost({"bytes": 4096}) == 32_768


def test_bound() -> None:
    s = _toy()
    b = S.bound(s, Fraction(1, 100), 65_536, {"on-chip": 10, "rebuilt": 6}, epsilon=EPS)
    assert b.excess == 65_552
    with pytest.raises(ValueError, match="charges are"):
        S.bound(s, Fraction(1, 100), 0, {"on-chip": 0})
    with pytest.raises(ValueError, match="δ lies"):
        S.bound(s, Fraction(1), 0, {"on-chip": 0, "rebuilt": 0})
    with pytest.raises(ValueError, match="the audit's error ε .* is given by the audit"):
        S.bound(s, Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0})


@pytest.mark.parametrize("bad, match", [
    ({"name": "PoUB"}, "configuration's name"),
    ({"resource": "bandwidth"}, "resource is one of"),
    ({"requires": {"fiat-shamir": "x"}}, "not requirements"),
    ({"charges": (S.Charge("on-chip", "a"), S.Charge("on-chip", "b"))}, "charge twice"),
])
def test_refusals(bad: dict, match: str) -> None:
    base = {"name": "toy", "protocol": "verity/toy/v0", "resource": "read", "blocks": S.Lacks("a stand-in"),
            "assumptions": (), "log": (), "charges": ()}
    with pytest.raises(ValueError, match=match):
        S.Scheme(**{**base, **bad})


def test_hooks() -> None:
    assert _toy(hooks=("memory", "declared")).hooks == ("memory", "declared")
    with pytest.raises(ValueError, match="not hooks of a served run"):
        _toy(hooks=("memory", "pouw-gemm"))


def test_time_only_through_an_assumption() -> None:
    with pytest.raises(ValueError, match="only by an assumption"):
        S.Order("receipts", "seed", within_ns=1)
    with pytest.raises(ValueError, match="only by an assumption"):
        S.Order("value roots", "seeds", "after", assumption="SAKMatmul")
    with pytest.raises(ValueError, match="positive time"):
        S.Order("value roots", "seeds", "after", within_ns=0, assumption="SAKMatmul")
    per_session = S.Order("receipts", "pass seed", within_ns="deadline_ns", assumption="SquaringCeiling")
    assert per_session.within_ns == "deadline_ns"
    with pytest.raises(ValueError, match="names its logged parameter"):
        S.Order("receipts", "pass seed", within_ns="10 s", assumption="SquaringCeiling")
    with pytest.raises(ValueError, match="not among its assumptions"):
        _toy(log=(S.Order("value roots", "seeds", "after", within_ns=30 * 10**9, assumption="SAKMatmul"),))


def test_order_direction() -> None:
    after = S.Order("value roots", "seeds", "after", within_ns=30 * 10**9, assumption="ToyNoCompression")
    assert _toy(log=(after,)).log[0].direction == "after"
    assert S.Order("start", "first-tick").direction == "before"
    with pytest.raises(ValueError, match="before or after"):
        S.Order("start", "first-tick", "beside")


def test_scheme_term_carried() -> None:
    s = _toy(delta_scheme=Fraction(1, 2**64))
    b = S.bound(s, Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0}, epsilon=EPS)
    assert b.delta_scheme == Fraction(1, 2**64) and b.epsilon == EPS
    assert b.probability == Fraction(1, 100) + EPS + Fraction(1, 2**64)
    assert S.bound(_toy(), Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0}, epsilon=EPS).delta_scheme == 0
    with pytest.raises(ValueError, match="δscheme lies"):
        _toy(delta_scheme=Fraction(1))


def test_a_said_term_says_both_sides() -> None:
    assert _toy().over and _toy().under
    with pytest.raises(ValueError, match="says its two sides"):
        _toy(under="")
    assert _toy("silent", delta_scheme=S.UNDECLARED, over="", under="").over == ""


def test_a_named_term_is_evaluated_at_the_parameters_its_caller_gives() -> None:
    pearl = S.Term("Pouw.PearlC.δPearlC", ("q", "N"), lambda q, N: Fraction(q + N, 2**128))
    b = S.bound(_toy(delta_scheme=pearl), Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0}, {"q": 2**64, "N": 4096}, epsilon=EPS)
    assert b.delta_scheme == Fraction(2**64 + 4096, 2**128) and b.term == "Pouw.PearlC.δPearlC"
    assert b.parameters == {"q": 2**64, "N": 4096}
    with pytest.raises(ValueError, match=r"δPearlC is evaluated at \['q', 'N'\], and \['q'\] were not given"):
        S.bound(_toy(delta_scheme=pearl), Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0}, {"N": 4096}, epsilon=EPS)
    with pytest.raises(ValueError, match="has no parameters"):
        S.bound(_toy(delta_scheme=pearl), Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0}, {"q": 1, "N": 1, "P": 1}, epsilon=EPS)
    with pytest.raises(ValueError, match="outside"):
        S.bound(_toy(delta_scheme=pearl), Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0}, {"q": 2**128, "N": 0}, epsilon=EPS)
    with pytest.raises(ValueError, match="takes no parameters"):
        S.bound(_toy(), Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0}, {"q": 1}, epsilon=EPS)
    sketch = S.Term("ResourceAccounting.Guarantees.Remote.SketchBoundDeployed", ("Q", "P", "N", "ε"),
                    lambda Q, P, N, ε: Fraction((Q + P) ** 2, 2**257) + N * ε)
    four = {"Q": 2**40, "P": 2**10, "N": 64, "ε": Fraction(1, 2**80)}
    assert S.bound(_toy(delta_scheme=sketch), Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0}, four, epsilon=EPS).parameters == four
    with pytest.raises(ValueError, match="once each"):
        S.Term("Pouw.PearlC.δPearlC", ("q", "q"), lambda q: Fraction(0))


def test_scheme_term_not_said_refuses_a_bound() -> None:
    s = _toy("silent", delta_scheme=S.Lacks("its term has no Lean constant"))
    assert S.register(s) is S.SCHEMES["silent"]
    with pytest.raises(ValueError, match=r"silent: its delta_scheme \(Verity.Scheme.δscheme\) is not said"):
        S.bound(s, Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0})


def _dither_circuit() -> list[tuple[str, list[list[int]]]]:
    from verity.circuits.defs import bind
    from verity.commitments.periods import interface as I
    from verity.protocols.experimental.dither.prover import dither as D
    from verity.protocols.proofs.spot.verifier.one_stage import partition as P

    program, query = I.Circuit("toy", ((bind(D.PouwDitherBlock, L=32, FIRST=True), 3),
                                       (bind(D.PouwDitherLast, K=64, L=64), 1))).program()
    return P.strata(program, query)


def test_credited_resolves_a_declaration_in_a_registered_circuit() -> None:
    from verity.protocols.experimental.dither.scheme import DITHER

    strata = _dither_circuit()
    assert [t for t, _ in strata] == ["PouwDitherBlock_v1{L=32,FIRST=true}", "PouwDitherLast_v1{K=64,L=64}"]
    assert S.credited([DITHER], strata) == frozenset({0, 1, 2, 3})
    assert S.credited([], strata) == frozenset()
    with pytest.raises(ValueError, match="toy-bandwidth: its block ToyReceipt_v1 resolves to no block"):
        S.credited([DITHER, _toy()], strata)
    rival = _toy("rival", blocks=(S.Block("PouwDitherLast_v1", lambda shape: 0),))
    with pytest.raises(ValueError, match=r"PouwDitherLast_v1\{K=64,L=64\} would be credited twice"):
        S.credited([DITHER, rival], strata)


def test_unbounded_charge_refuses_a_bound() -> None:
    s = _toy(charges=(S.Charge("on-chip", "on_chip_bits"), S.Charge("padding", "Network.PaddingBound", bounded=False)))
    with pytest.raises(ValueError, match="nothing bounds yet: padding"):
        S.bound(s, Fraction(1, 100), 0, {"on-chip": 0, "padding": 0})
    assert S.unbounded(s).terms == ("padding",) and S.unbounded(_toy()) is None


def test_ports() -> None:
    port = S.Port("scan state", "input", "ToyReceipt_v1", "state")
    assert _toy(ports=(port,)).ports == (port,)
    with pytest.raises(ValueError, match="not its blocks' Definitions"):
        _toy(ports=(S.Port("scan state", "input", "IngressScanBucket_v1", "state"),))


def test_a_field_no_declaration_fills_is_refused_by_its_name() -> None:
    toy = _toy(record="toy", symbol=S.Lacks("it certifies nothing yet"))
    assert S.declared(toy, "record") == "toy"
    with pytest.raises(LookupError, match="toy-bandwidth declares no symbol: it certifies nothing yet"):
        S.declared(toy, "symbol")
    with pytest.raises(LookupError, match="toy-bandwidth declares no parties$"):
        S.declared(toy, "parties")


def test_every_declaration_fills_each_field_or_says_why_not() -> None:
    probe = ("import json\n"
             "from verity.protocols.compose import schemes as C\n"
             "from verity.protocols.resource_accounting import scheme as S\n"
             "C.load()\n"
             "F = ('parties', 'channels', 'channel_schemas', 'colocated', 'needs', 'depends', 'record', 'symbol',\n"
             "     'published', 'check', 'reader', 'premises', 'tables', 'output_format', 'product_family',\n"
             "     'y_committed', 'salt_open_root', 'kernels', 'answers')\n"
             "print(json.dumps(sorted([n, f] for n in C.DECLARATIONS for f in F if getattr(C.get(n), f) == S.UNDECLARED)))\n")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == [], "fill each field a declaration leaves unset, or say why not with S.Lacks(why)"


def test_an_answers_entry_is_module_colon_attribute_and_a_lacks_is_refused_by_name() -> None:
    for entry in ("toy.answers", "toy:", ":ANSWERS", "toy answers:ANSWERS", "toy:A.B"):
        with pytest.raises(ValueError, match="answers is an entry, module:attribute"):
            _toy(answers=entry)
    with pytest.raises(LookupError, match="toy-bandwidth declares no answers: it keeps no session$"):
        S.answering(_toy(answers=S.Lacks("it keeps no session")))


def test_the_bandwidth_audit_is_answered_through_its_entry_in_a_process_with_no_exchange(tmp_path: Path) -> None:
    probe = ("import json, sys\n"
             "from pathlib import Path\n"
             "from verity.protocols.compose import schemes as C\n"
             "from verity.protocols.resource_accounting import scheme as S\n"
             "A = S.answering(C.get('bandwidth'))\n"
             "out = A.missing(Path(sys.argv[1]), 3, 'no session ran')\n"
             "print(json.dumps({'line': A.line(out), 'held': A.held, 'session': dict(A.session),\n"
             "                  'loaded': sorted(m for m in sys.modules if m.startswith('verity_vllm') or 'exchange' in m)}))\n")
    out = json.loads(subprocess.run([sys.executable, "-c", probe, str(tmp_path)], capture_output=True, text=True,
                                    check=True).stdout)
    assert out["line"].startswith("BANDWIDTH: undecided: no session ran; "), out
    assert json.loads((tmp_path / "answer.json").read_text())["verdict"] == "undecided"
    assert out["held"] == "held-case.json" and out["loaded"] == [], out
    assert out["session"] == {"windows": 6, "passes": 4, "samples": 1024, "window_s": 20.0, "deadline_s": 1.0}


def test_a_component_that_charges_nothing_is_a_scheme_bound_and_credited_refuse() -> None:
    nothing = S.Lacks("charges nothing: it checks a policy")
    free = _toy(name="toy-check", resource=nothing, blocks=nothing, record="toy")
    assert S.declared(free, "record") == "toy"
    with pytest.raises(LookupError, match="toy-check declares no resource: charges nothing"):
        S.bound(free, Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0}, epsilon=EPS)
    for field in ("resource", "blocks"):
        with pytest.raises(LookupError, match=f"toy-check declares no {field}: charges nothing"):
            S.declared(free, field)
    with pytest.raises(LookupError, match="toy-check declares no resource"):
        free.unit
    placed = _toy(blocks=(S.Block("Toy_v1", lambda shape: 1),))
    assert S.credited([free], []) == frozenset()
    assert S.credited([free, placed], [("Toy_v1", [(0, 2)])]) == {0, 1}
    with pytest.raises(ValueError, match="its resource is one of"):
        _toy(resource="disk")
    with pytest.raises(ValueError, match="places no block"):
        _toy(blocks=())
    with pytest.raises(ValueError, match="charges nothing says why it has no blocks"):
        _toy(name="toy-check", resource=nothing)


def test_the_records_unit_is_an_exact_factor_and_schemas_are_of_carried_messages() -> None:
    from verity.channel import Schema
    from verity.protocol import Channel

    assert _toy(published=("bytes", Fraction(1, 8))).published == ("bytes", Fraction(1, 8))
    with pytest.raises(ValueError, match="published is"):
        _toy(published=("bytes", 0.125))
    ch = (Channel("prover", "verifier", ("Root",)),)
    assert _toy(channels=ch, channel_schemas=(Schema("Root", 1, "commitment"),)).channel_schemas[0].type == "Root"
    with pytest.raises(ValueError, match=r"channel_schemas name \['Open'\]"):
        _toy(channels=ch, channel_schemas=(Schema("Open", 1, "commitment"),))


def test_a_ports_layout_at_a_shape() -> None:
    def rows(shape):
        n = shape["slots"]
        return S.Layout(16 * -(-(8 + 4 * n) // 16), {"head": (0, 8), **{f"{j}.x": (8 + 4 * j, 4) for j in range(n)}},
                        {"slots": n})

    toy = _toy(ports=(S.Port("state", "input", "ToyReceipt_v1", "state", layout=rows),
                      S.Port("bare", "input", "ToyReceipt_v1", "state")))
    got = S.layout(toy, "state", {"slots": 3})
    assert (got.bits, dict(got.fields), dict(got.counts)) == (32, {"head": (0, 8), "0.x": (8, 4), "1.x": (12, 4),
                                                                   "2.x": (16, 4)}, {"slots": 3})
    assert S.Port("bare", "input", "ToyReceipt_v1", "state").layout is None
    with pytest.raises(ValueError, match="port bare declares no layout"):
        S.layout(toy, "bare", {})
    with pytest.raises(KeyError, match="no port 'missing'; its ports: state, bare"):
        S.layout(toy, "missing", {})


@pytest.mark.parametrize("fields,match", [
    ({"a": (0, 8), "b": (4, 8)}, "field b at bit 4 overlaps field a, which ends at bit 8"),
    ({"a": (0, 8), "b": (8, 9)}, "field b at \\(8, 9\\) runs past the layout's 16 bits"),
    ({"a": (-1, 2)}, "field a's offset and width are integers >= 0"),
])
def test_a_layout_refuses_an_overlap_and_an_overrun_by_name(fields, match) -> None:
    with pytest.raises(ValueError, match=match):
        S.Layout(16, fields)
    assert dict(S.Layout(16, {"a": (0, 8), "pad": (8, 0), "b": (8, 8)}).fields)["b"] == (8, 8)


def test_every_scheme_is_a_line_of_the_one_table() -> None:
    """Each name maps to the module that registers it, and every module that registers one is in the table."""
    from verity.protocols.compose import schemes

    root = Path(__file__).resolve().parents[4]
    declaring = {str(p.relative_to(root).with_suffix("")).replace("/", ".")
                 for d in ("verity/protocols/resource_accounting", "verity/protocols/experimental",
                           "verity/protocols/proofs/spot", "verity/protocols/workload_classification")
                 for p in (root / d).rglob("scheme.py") if "register(" in p.read_text()}
    declaring.discard("verity.protocols.resource_accounting.scheme")
    assert declaring == set(schemes.DECLARATIONS.values()), "add each declaring module's names to compose.schemes"
    probe = ("import importlib, json, sys\n"
             "from verity.protocols.resource_accounting import scheme as S\n"
             "by = {}\n"
             "for m in sys.argv[1:]:\n"
             "    before = set(S.SCHEMES); importlib.import_module(m)\n"
             "    by.update({n: m for n in set(S.SCHEMES) - before})\n"
             "print(json.dumps(by))\n")
    out = subprocess.run([sys.executable, "-c", probe, *sorted(declaring)], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == dict(schemes.DECLARATIONS), "each name maps to the module that registers it"


def test_every_reason_a_list_cannot_attach_in_its_order_is_given_from_its_declarations() -> None:
    base, store = _toy("base", depends=()), _toy("store", resource="memory", needs=frozenset({"store"}), depends=())
    over = _toy("over", depends=("base", "absent"))
    work = _toy("work", resource="compute", depends=("base",))
    assert S.order_refusals([base, store, work]) == S.order_refusals([base, over]) == []
    assert S.order_refusals([over, work, base, store, _toy("unsaid")]) == [
        "order: over requires base, so protocols lists base before over",
        "order: work requires base, so protocols lists base before work",
        "depends: unsaid's declaration does not say which schemes it depends on (depends is undeclared)",
        "outer: store is storage (it needs the workload's store) and is listed after work, useful work (resource compute); "
        "where both wrap one site the earlier listed is the outer, so protocols lists store before work"]


def test_the_lookup_table_is_each_declarations_own_values() -> None:
    """`KEYS` holds, for every declared scheme, exactly the option, roles and port, framing and process names its
    declaration states: an entry the declaration lacks fails, and so does a declared value the table lacks."""
    probe = ("import json\n"
             "from verity.protocols.compose import schemes\n"
             "print(json.dumps({n: schemes.keys(s) for n, s in schemes.load().items()}))\n")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    from verity.protocols.compose import schemes

    declared = {n: {f: tuple(v) for f, v in k.items()} for n, k in json.loads(out.stdout).items() if k}
    assert {n: dict(k) for n, k in schemes.KEYS.items()} == declared, "write each declaration's values in compose.schemes.KEYS"


def test_a_lookup_by_a_declared_value_names_one_scheme_and_refuses_none_or_two(monkeypatch) -> None:
    from verity.protocols.compose import schemes

    assert schemes.named("option", "pous:p2-16448/v4") == "pous"
    assert schemes.named("provides", "admission") == "network-attester"
    assert schemes.named("ports", "scan-state") == "network-attester"
    assert "pous" not in S.SCHEMES
    with pytest.raises(LookupError, match=r"one scheme whose option is 'pous:absent', not 0: \[\]; declared: pou"):
        schemes.named("option", "pous:absent")
    with pytest.raises(LookupError, match="no lookup by 'name'"):
        schemes.named("name", "pous")
    monkeypatch.setattr(schemes, "KEYS", {"a": {"provides": ("cut",)}, "b": {"provides": ("cut",)}})
    with pytest.raises(LookupError, match=r"one scheme whose provides is 'cut', not 2: \['a', 'b'\]"):
        schemes.named("provides", "cut")


def test_a_scheme_attaches_nothing_by_default_and_registers_its_processes(monkeypatch) -> None:
    monkeypatch.setattr(S, "PROCESSES", {})
    assert _toy().requests(None) == () and _toy().attach(None, None) is None
    start = lambda **readers: readers  # noqa: E731
    monkeypatch.setattr(S, "SCHEMES", {})
    S.register(_toy(processes={"toy-beside": start}))
    assert S.PROCESSES == {"toy-beside": start}


def test_a_pieces_party_is_a_declared_developer_party_and_its_development_one_of_its_bindings() -> None:
    from verity.protocol import Role
    parties = (Role("prover", "developer"), Role("challenger", "auditor"), Role("verifier", "auditor"))
    ok = _toy(parties=parties, pieces={"prover": S.Launch(optional=("address",), development="address")})
    assert ok.pieces["prover"].development == "address"
    for pieces, why in (({"challenger": S.Launch()}, "declared developer-side parties' \\['prover'\\], not 'challenger'"),
                        ({"prover": S.Launch(development="host")}, "development binding 'host' is one of its binds"),
                        ({"prover": S.Launch(reaches="prover")}, "reaches a declared auditor party, not 'prover'")):
        with pytest.raises(ValueError, match=why):
            _toy(parties=parties, pieces=pieces)
