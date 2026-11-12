"""The response tie's verifier (`response_tie_service`) on the toy schedule's form-2 heads
(`test_network_certifier_content_tables`' loads): a tie's record at the head's count of a class's rows, which
registers the class's end and response tables as the head signs them and the compute side's rows as ports of its
log's records, is accepted, and its verdict is accepted and complete only as one-stage's answer says. Each refusal
fails for the reason it names: a record whose entry of a table has another root (`R7-certified`) or another domain
(`R7-domain`), by the sampled-proofs verifier's certified read; a population that is not the head's count
(`population`) and a class the head counts no row in (`no-rows`), by this verifier; and a registration that doesn't
read under the network certifier's key (`head`). The compute side's records here are stand-ins: one-stage records
of another Program, of which the tie reads a port's root, domain and rows."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

import signers
from verity.protocols.certifier.prover import response_tie as RT
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier import response_tie_service as TS
from verity.protocols.certifier.verifier.certificate import Issuer
from verity.protocols.certifier.verifier.tests import test_network_certifier_content_tables as TB
from verity.protocols.certifier.verifier.tests.test_network_certifier_warden import run
from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import partition as P, registration as R

RATE = "rate:1/2"
RUN = "r-response-tie-test"
#: the class under test in each load, with the rows the load files in it: two of 8 ids, three of 4
FIRST, SECOND = ("first", 2), ("second", 1)
#: the compute side's ports the tie's values are: its rows of ids, and in a stop class its stop words
EGRESS, STOP = "b.egress", "b.stop"
FORMS = pytest.mark.parametrize("stop", (False, True), ids=("nostop", "stop"))


@pytest.fixture(scope="module")
def heads(tmp_path_factory):
    """The network certifier's key, and each of the content tables' three loads as execution phase 0 of the form-2
    schedule, signed under it: by load, its registration as the verifier is handed it."""
    _, arrivals, answers = TB.mixed()
    device, _, _ = run(answers, arrivals, params=TB.CON, windows=3, client=TB.leaving(TB.GONE))
    handed = dict(device.lengths)
    loads = {"first": handed[0], "second": handed[1], "none": ((),) * TB.STRIDES}
    docs = {}
    for load, strides in loads.items():
        issuer = Issuer(signers.signer(tmp_path_factory.mktemp(load)), signers.SESSION)
        _, docs[load], head, _ = TB.read(issuer, 0, strides, TB.HELD)
        assert tuple(head.table(f"{CT.END}.{i}").rows for i in range(len(TB.CLASSES))) == TB.COUNTS[load]
    return issuer.signer.public_key(), docs


def tie(key, cls: int, stop: bool = True, **changes) -> TS.ResponseTie:
    """The verifier of class `cls` of execution phase 0 of the form-2 schedule, with the tie's Definition of its
    form at the class's ids."""
    unit = RT.block(RT.Shape(TB.HELD.content[cls][1], stop))
    return TS.ResponseTie(**{"key": key, "session": signers.SESSION, "params": TB.HELD, "link": TB.LINK, "window": 0,
                             "cls": cls, "unit": unit, "rate": RATE, **changes})


def compute(t: TS.ResponseTie, rows: int) -> tuple[list[dict], dict[str, dict[str, str]]]:
    """Stand-ins for the compute side's records a tie of `rows` proof blocks reads, and the tie's `outputs` naming
    them: a boundary record whose port `EGRESS` is its rows of ids and, where the tie reads a stop word, a stop
    record after it whose port `STOP` is its stop words. Each is a one-stage record of another Program."""
    other = RT.block(RT.Shape(CT.MIN_SIZE, False))
    program, query = P.population_program(other, rows), P.template_instance_query(other)
    log, outputs = [], {}
    for value, port in zip(t.registered, (EGRESS, STOP)):
        domain = R.window_domain("a0" * 64, "b0" * 64, port, TS.SCHEMA, rows, RUN)
        roots = [{"port": port, "root": hashlib.sha512(port.encode()).hexdigest(), "leaves": rows, "domain": domain}]
        log.append(R.record(program, query, {"law": "subset", "k": rows}, S.SCHEME, roots,
                            {"kind": "served", "run": RUN, "population": port},
                            previous=R.digest(log[-1]) if log else None))
        outputs[value] = {"record": R.digest(log[-1]), "port": port}
    return log, outputs


def record(t: TS.ResponseTie, doc: dict, log: list[dict], outputs: dict, rows: int | None = None) -> dict:
    """The honest tie's record on `doc`'s head, after `log`: the Program at the head's count (or at `rows`), its
    verdict words' root, the compute side's values as the log's ports, and the class's two tables as the head signs
    them."""
    head = t.read(doc)
    n = t.rows(head) if rows is None else rows
    by_digest = {R.digest(r): r for r in log}
    entries = []
    for value, words in t.registered.items():
        e = next(x for x in by_digest[outputs[value]["record"]]["roots"] if x["port"] == outputs[value]["port"])
        entries.append({"value": value, "root": e["root"], "leaves": e["leaves"], "words": words, "domain": e["domain"]})
    entries += [{"value": v, **e} for v, e in t.certified(head).items()]
    roots = [{"port": TS.PORT, "root": "0f" * 64, **t.ports(head)[TS.PORT], "leaves": n}]
    return R.record(t.program(n), t.query, t.law(n), S.SCHEME, roots,
                    {"kind": "served", "run": RUN, "population": "response-tie"}, previous=R.digest(log[-1]),
                    registered=entries)


def changed(rec: dict, value: str, **change) -> dict:
    """`rec` with its registered entry of `value` changed."""
    assert [e["value"] for e in rec["registered"]].count(value) == 1
    return {**rec, "registered": [{**e, **change} if e["value"] == value else e for e in rec["registered"]]}


def answer(rec: dict, **public) -> dict:
    """One-stage's accept of `rec` (`Core.Protocol.answer`'s form), its public outputs with `public`'s."""
    return {"commitment": R.digest(rec), "profile": {"delta": "1/1024", "levels": []}, "protocol": R.PROTOCOL,
            "public": {"bound": 1, "drawn": [0], "per_template": None, "proved": [0], **public}, "reason": None,
            "verdict": "accept"}


def case(heads, load: tuple[str, int] = FIRST, stop: bool = True):
    """A load's class under test: its verifier, the registration, the compute side's records, the tie's `outputs`
    and its honest record."""
    key, docs = heads
    name, cls = load
    t, doc = tie(key, cls, stop), docs[name]
    log, outputs = compute(t, TB.COUNTS[name][cls])
    return t, doc, log, outputs, record(t, doc, log, outputs)


@FORMS
@pytest.mark.parametrize("load", (FIRST, SECOND), ids=("first", "second"))
def test_an_honest_ties_record_is_held_to_the_head_and_its_verdict_says_accepted_and_complete_apart(heads, load, stop):
    """The tie's record at the head's count, registering the class's end and response tables as the head signs them
    beside the compute side's ports, passes the verifier in either form. Its verdict names the head and the record,
    and carries two bits: accepted on one-stage's accept of the record, and complete only where the accept's public
    `complete` is true, so an accept without the field, or with anything but true there, is accepted and not
    complete."""
    t, doc, log, outputs, rec = case(heads, load, stop)
    head = t.read(doc)
    assert rec["population"] == t.rows(head) == TB.COUNTS[load[0]][load[1]]
    assert sorted(t.registered) == (["egress", "stop"] if stop else ["egress"])
    assert set(t.certified(head)) == {f"{TB.LINK}/end.{load[1]}/0", f"{TB.LINK}/response.{load[1]}/0"}
    checked = t.check(rec, doc, outputs=outputs, log=log)
    assert checked.ok and checked.codes == ()
    whole = t.verdict(rec, doc, answer(rec, complete=True), outputs=outputs, log=log)
    assert whole == TS.TieVerdict(TB.LINK, 0, load[1], head.digest().hex(), R.digest(rec), True, True)
    assert whole.to_json() == {"format": TS.VERDICT, "link": TB.LINK, "window": 0, "class": load[1],
                               "head": head.digest().hex(), "registration": R.digest(rec), "accepted": True,
                               "complete": True, "codes": []}
    for public in ({}, {"complete": False}, {"complete": None}, {"complete": 1}, {"complete": "true"}):
        part = t.verdict(rec, doc, answer(rec, **public), outputs=outputs, log=log)
        assert (part.accepted, part.complete, part.codes) == (True, False, ()), public


def test_a_record_at_another_root_is_refused_by_the_certified_read(heads):
    """A record whose entry of the class's end table, or of its response table, has another root than the head
    signs is refused by the sampled-proofs verifier's certified read, `R7-certified`, and by nothing of this
    verifier's; its verdict is refused with that code, whatever the answer says."""
    t, doc, log, outputs, rec = case(heads)
    for value in t.certified(t.read(doc)):
        other = changed(rec, value, root="ab" * 64)
        assert t.check(other, doc, outputs=outputs, log=log).codes == ("R7-certified",), value
        verdict = t.verdict(other, doc, answer(other, complete=True), outputs=outputs, log=log)
        assert (verdict.accepted, verdict.complete, verdict.codes) == (False, False, ("R7-certified",))
        assert verdict.registration is None and verdict.head == t.read(doc).digest().hex()


def test_a_record_under_another_domain_is_refused_by_the_certified_read(heads):
    """A record whose entry of the class's end table, or of its response table, has another domain than the one
    the head signs, another execution phase's of the same table among them, is refused `R7-domain`."""
    t, doc, log, outputs, rec = case(heads)
    head = t.read(doc)
    for kind in TS.READS.values():
        e = head.table(t.table(kind))
        later = CR.domain(TB.HELD, signers.SESSION, TB.LINK, 1, e.name, e.rows, e.row_bytes).domain_id.hex()
        assert later != e.domain.hex()
        for domain in ("00" * 64, later):
            other = changed(rec, CR.value_name(TB.LINK, e.name, 0), domain=domain)
            assert t.check(other, doc, outputs=outputs, log=log).codes == ("R7-domain",), (kind, domain[:8])
            verdict = t.verdict(other, doc, answer(other, complete=True), outputs=outputs, log=log)
            assert (verdict.accepted, verdict.complete, verdict.codes) == (False, False, ("R7-domain",))


def test_a_population_that_is_not_the_heads_count_is_refused(heads):
    """A record of more proof blocks than the head counts rows in the class, or of fewer, is refused `population`,
    naming both numbers, though it is a well-formed record of the tie's Program at its own population; so is a
    record without one."""
    t, doc, log, outputs, rec = case(heads, SECOND)
    count = TB.COUNTS["second"][1]
    for rows in (count + 1, count - 1):
        other = record(t, doc, log, outputs, rows)
        assert other["population"] == rows
        checked = t.check(other, doc, outputs=outputs, log=log)
        assert checked.codes == (TS.POPULATION,)
        assert checked.detail[TS.POPULATION] == {"record": rows, "head": count, "table": "end.1"}
        with pytest.raises(S.Refused) as refused:
            t.held(other, doc)
        assert refused.value.verdict.codes == (TS.POPULATION,)
        verdict = t.verdict(other, doc, answer(other, complete=True), outputs=outputs, log=log)
        assert (verdict.accepted, verdict.complete, verdict.codes) == (False, False, (TS.POPULATION,))
    for other in ({k: v for k, v in rec.items() if k != "population"}, {**rec, "population": True}, None):
        assert t.check(other, doc, outputs=outputs, log=log).codes == (TS.POPULATION,)
    assert t.held(rec, doc) == t.read(doc)


def test_a_class_the_head_counts_no_row_in_has_no_tie(heads):
    """For a class in which the execution phase filed no release the head counts 0 rows: no tie is owed, and the
    verifier accepts none, naming the class's table (`no-rows`), whatever record and answer it is handed: one of
    another class of the same sizes among them. It makes no port for it either."""
    key, docs = heads
    t, doc, log, outputs, rec = case(heads)
    assert TB.HELD.content[3] == TB.HELD.content[2] and TB.COUNTS["first"][3] == 0
    for load, cls in (("first", 3), ("first", 1), ("none", 2), ("none", 0)):
        none = tie(key, cls, unit=t.unit) if TB.HELD.content[cls] == TB.HELD.content[2] else tie(key, cls)
        handed = rec if none.unit is t.unit else None
        checked = none.check(handed, docs[load], outputs=outputs, log=log)
        assert checked.codes == (TS.NO_ROWS,)
        assert f"no row in class {cls} (end.{cls})" in checked.detail[TS.NO_ROWS]
        assert "no tie is owed for it and none is accepted" in checked.detail[TS.NO_ROWS]
        verdict = none.verdict(handed, docs[load], answer(rec, complete=True), outputs=outputs, log=log)
        assert (verdict.accepted, verdict.complete, verdict.codes) == (False, False, (TS.NO_ROWS,))
        with pytest.raises(S.Refused) as refused:
            none.read(docs[load])
        assert refused.value.verdict.codes == (TS.NO_ROWS,)
        head, _ = CR.check_registration(docs[load], key, signers.SESSION, TB.HELD, TB.LINK, 0)
        with pytest.raises(ValueError, match=f"table 'end.{cls}' has no row"):
            none.reads(head)


def test_a_registration_that_does_not_read_under_the_certifiers_key_is_refused(heads, tmp_path):
    """The head is the one the registration's certificate signs under the network certifier's key in the session,
    for the link's execution phase under the schedule: under another key or session, for another execution phase,
    or with a table's root changed after signing, the verifier refuses the record `head`, with what fails."""
    key, docs = heads
    t, doc, log, outputs, rec = case(heads)
    other_key = signers.signer(tmp_path, b"another").public_key()
    moved = {**doc, "tables": [{**e, "root": "ab" * 64} if e["name"] == "end.2" else e for e in doc["tables"]]}
    for refusing, handed, why in ((tie(other_key, 2), doc, "doesn't verify under the network certifier's key"),
                                  (tie(key, 2, session=signers.OTHER_SESSION), doc, "has domain"),
                                  (tie(key, 2, window=1), doc, "the registration's window is 0"),
                                  (t, moved, "doesn't sign its head"), (t, None, ""), (t, {}, "format")):
        checked = refusing.check(rec, handed, outputs=outputs, log=log)
        assert checked.codes == (TS.HEAD,) and why in checked.detail[TS.HEAD], why
        verdict = refusing.verdict(rec, handed, answer(rec, complete=True), outputs=outputs, log=log)
        assert (verdict.accepted, verdict.complete, verdict.codes, verdict.head) == (False, False, (TS.HEAD,), None)


def test_an_answer_that_is_no_accept_of_the_record_is_no_accepted_tie(heads):
    """The verdict is accepted only on one-stage's accept of the record the verifier checked: a reject, an
    undecided, another protocol's accept, an accept of another record or no answer leaves a record that passes
    refused `answer`, and never complete, whatever the answer's public outputs say."""
    t, doc, log, outputs, rec = case(heads)
    ok = answer(rec, complete=True)
    other = answer({**rec, "window": {**rec["window"], "run": "another"}}, complete=True)
    for bad in ({**ok, "verdict": "reject"}, {**ok, "verdict": "undecided"}, {**ok, "protocol": "verity/another/v1"},
                other, {**ok, "commitment": None}, {k: v for k, v in ok.items() if k != "verdict"}, None, [ok], "accept"):
        verdict = t.verdict(rec, doc, bad, outputs=outputs, log=log)
        assert (verdict.accepted, verdict.complete, verdict.codes) == (False, False, (TS.ANSWER,)), bad
        assert verdict.registration == R.digest(rec)
    for accepted, complete, codes in ((True, True, ("x",)), (False, True, ("x",)), (False, False, ()), (1, False, ())):
        with pytest.raises(ValueError):
            TS.TieVerdict(TB.LINK, 0, 2, "aa" * 64, "bb" * 64, accepted, complete, codes)
    with pytest.raises(ValueError, match="names its head and its record"):
        TS.TieVerdict(TB.LINK, 0, 2, "aa" * 64, None, True, False)


def test_the_compute_sides_values_are_ports_of_the_ties_own_log(heads):
    """Every parameter of the proof block but its end row is a port of a record of the tie's log: the verifier
    refuses its own inputs where `outputs` name another set of values or anything but a record's port, and
    one-stage refuses first of all (`R7-output`) a record whose value is no port of the log's record it names, or
    a port of fewer rows than the head counts."""
    t, doc, log, outputs, rec = case(heads)
    for bad in ({}, {"egress": outputs["egress"]}, {**outputs, "end": outputs["egress"]},
                {**outputs, "stop": {"port": STOP}}, {**outputs, "stop": {"record": "aa" * 64, "value": STOP}}, None):
        with pytest.raises(ValueError, match="the compute side's values are"):
            t.check(rec, doc, outputs=bad, log=log)
    elsewhere = {**outputs, "stop": {**outputs["stop"], "port": EGRESS}}
    assert t.check(rec, doc, outputs=elsewhere, log=log).codes[0] == "R7-output"
    short, named = compute(t, 1)
    assert t.check(record(t, doc, short, named), doc, outputs=named, log=short).codes[0] == "R7-output"


def test_the_registered_ports_are_built_from_the_head_and_a_classes_tables_are_taken_by_name(heads, monkeypatch):
    """The verifier's `--registered` ports of the proof block's end row and response row are the head's: each names
    the class's table's value, its signed root and domain, and row j for proof block j. The class's tables are
    taken by name: a class that lists another table among its own has the same two."""
    t, doc, _, _, _ = case(heads)
    head = t.read(doc)
    reads, certified = t.reads(head), t.certified(head)
    assert set(reads) == {"end", "o1"} == set(TS.READS)
    for port, kind in TS.READS.items():
        e, got = head.table(f"{kind}.2"), reads[port]
        assert got == CR.port(head, f"{kind}.2") and got["value"] == f"{TB.LINK}/{kind}.2/0"
        assert (got["root"], got["domain"], got["reads"]) == (e.root.hex(), e.domain.hex(), list(range(e.rows)))
        assert certified[got["value"]] == {k: got[k] for k in R.CERTIFIED_KEYS}
    assert t.reads(head, 5)["end"] == CR.port(head, "end.2", 5)
    monkeypatch.setattr(CT, "names", lambda i: (f"request.{i}", f"words.{i}", f"end.{i}", f"response.{i}"))
    assert (t.table(CT.END), t.table(CT.RESPONSE)) == ("end.2", "response.2")
    assert t.certified(head) == certified and t.reads(head) == reads
    with pytest.raises(ValueError, match="not a table 'stop.2'"):
        t.table("stop")


def test_a_tie_is_of_a_form_2_class_its_own_definition_and_a_rate(heads):
    """The verifier is built only for a class of a form-2 schedule, whose head states a count, with a Definition
    of the class's interface (an end row, the compute side's rows, and the class's response row after a verdict
    word) and a rate: a form-1 schedule, a class the schedule doesn't list, a Definition at another class's ids
    and a selection that is no rate are each refused, naming what is wrong."""
    key, _ = heads
    assert tie(key, 2).ids == 8 and tie(key, 1, False).registered == {"egress": 8}
    for change, why in (({"params": TB.CON}, "form 2"), ({"params": TB.LEN}, "form 2"), ({"cls": 4}, "lists 4 classes"),
                        ({"cls": -1}, "lists 4 classes"), ({"unit": RT.block(RT.Shape(4, True))}, "no response tie's"),
                        ({"unit": None}, "its Definition"), ({"rate": "all"}, "sampled at a rate"),
                        ({"rate": "subset:2"}, "sampled at a rate"), ({"link": ""}, "named link"),
                        ({"session": "5e"}, "named link"), ({"window": -1}, "named link")):
        with pytest.raises(ValueError, match=why):
            dataclasses.replace(tie(key, 2), **change)
    assert dataclasses.replace(tie(key, 2), rate="rate:1/1").law(2) == {"law": "subset", "k": 2}


@pytest.mark.slow
def test_the_ties_stream_takes_its_record_only_at_the_heads_count_and_under_the_heads_entries(heads):
    """On the tie's own stream, whose log starts with the compute side's records, `register` takes the honest
    record with the verifier's port, the compute side's values and the head's certified entries, and the stream
    keeps those entries for its run of record. A record at another population is refused before the stream reads
    it (`population`), which the stream alone would not compare with the head's count; one at another root is
    refused by the stream (`R7-certified`); and a stream of another tie's Spec is not this tie's."""
    t, doc, log, outputs, rec = case(heads)

    def stream(of: TS.ResponseTie = t) -> S.Stream:
        return dataclasses.replace(of, proof_deadline=60.0).stream(
            log, budget=4, delta=2.0 ** -40, challenger="challenger-test", clock=lambda: 0.0,
            sampler=lambda *a, **kw: pytest.fail("no sample is read"))

    served = dataclasses.replace(t, proof_deadline=60.0)
    for other, codes in ((record(t, doc, log, outputs, 3), (TS.POPULATION,)),
                         (changed(rec, f"{TB.LINK}/response.2/0", root="ab" * 64), ("R7-certified",))):
        with pytest.raises(S.Refused) as refused:
            served.register(stream(), other, doc, outputs=outputs)
        assert refused.value.verdict.codes == codes
    on = stream()
    reg = served.register(on, rec, doc, outputs=outputs)
    assert reg.digest == R.digest(rec) and on.log == (*log, rec)
    assert on._entries[-1].certified == t.certified(t.read(doc))
    with pytest.raises(ValueError, match="not this tie's"):
        served.register(stream(dataclasses.replace(t, cls=3)), rec, doc, outputs=outputs)
