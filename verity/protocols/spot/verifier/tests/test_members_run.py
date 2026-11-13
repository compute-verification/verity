"""`one_stage.members.members_run` on what the stand-in files of its difftest never reach: sessions the verifier
accepts. A window of two members on a challenger's own release log, each session closed after its evidence; the
verifier's verdict is handed in (`verdict`), and the read of the auditor's ledger (`challenger.trusted_session_of`,
which its own tests hold) is taken as passed. The run credits each accepted session its member's units, refuses proved
units other than the evidence's and a verdict naming another program, rejects an accepted session whose public file is
not its member's slice of the registration, and accepts nothing: what a run of sessions that all accepted maps to is
the form's refusal."""

from __future__ import annotations

import hashlib
import json

import pytest

from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.types import Array, Value
from verity.commitments import coin_tree as CT
from verity.protocols.spot.device import challenger as C
from verity.protocols.spot.verifier.one_stage import closing as K, members as M, partition as P, proofs as PF

from test_challenger import entropy

XOR8 = PrimitiveDefinition("MembersRunXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("MembersRunPair", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, x[0], x[1]), register=False))
TRIPLE = bind(CompositeDefinition("MembersRunTriple", 0, (), lambda S: ((("x", Array(3, Value(8))),), Value(8)),
                                  lambda B, S, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
PROGRAM = P.mixed_population_program([(PAIR, 2), (TRIPLE, 3)])
QUERY = P.template_instances_query([PAIR.id, TRIPLE.id])
PARTITION = P.partition(PROGRAM, QUERY)
A, B = "a/c0", "b/c0"
UNITS = {A: [[0, 2]], B: [[2, 5]]}
SCHEME = "frame-v3-sha512/hm96-sha512"
SPEC = {"block": 8, "rounds_log": 1, "scheme": CT.SCHEME, "streams": [CT.SESSION, "circuit/rep0", "circuit/rep1"]}
ROOTS = {A: ("11" * 64, "aa" * 64), B: ("22" * 64, "bb" * 64)}
RECORD = {"population": 5, "scheme": SCHEME,
          "roots": [{"domain": ROOTS[m][1], "leaves": 4, "port": f"{m}/out", "root": ROOTS[m][0]} for m in (A, B)]}
SAMPLED = {A: [1], B: [0, 2]}


def header(member: str, **changes) -> dict:
    return {"frame_v3": {"domain_ids": {"out": ROOTS[member][1]}, "roots": {"out": ROOTS[member][0]}},
            "instances": sum(hi - lo for lo, hi in UNITS[member]), "scheme": SCHEME,
            "units": {"indices": [u for lo, hi in UNITS[member] for u in range(lo, hi)]}, **changes}


class Window:
    """The two members' files under `at`, their sessions on one challenger's release log, each tree's link, one round
    and its closing round before the next tree's first, and the per-member parameters and evidence over them. `headers`
    are the members' public files' headers (`header`'s by default)."""

    def __init__(self, at, headers: dict | None = None) -> None:
        self.at, self.files = at, []
        challenger = C.LiveChallenger("verity/test/coins/v1", 0, [("sample", 1)], entropy("members-run"))
        log = [json.dumps({"event": "commitment", "root": challenger.commitment.root.hex()})]
        statements, members = [], {}
        for k, name in enumerate((A, B)):
            tree, line = challenger.sample_tree(SPEC, bytes([k]) * CT.NONCE_BYTES)
            log.append(line)
            root, root_f, root_b, message = tree.root().hex(), bytes([7]) * 8, bytes([8]) * 8, bytes([9]) * 64

            def opened(stream: str, rnd: int, n: int, after: bytes) -> list[int]:
                coins, _, line = challenger.open_tree(root, stream, rnd, n, after)
                log.append(line)
                return coins
            points = opened(CT.SESSION, 0, 2, C.link_after(root_f, {"circuit": root_b}))
            coins = opened("circuit/rep0", 0, 3, message)
            record = json.dumps({
                "coin_tree": tree.record(), "unit_draw": {"units": SAMPLED[name]},
                "link": {"points": [[f"{c:032x}"] for c in points], "root_b": {"circuit": root_b.hex()},
                         "root_f": root_f.hex()},
                "streams": [{"stream": "circuit/rep0", "rounds": [{"coins": [f"{c:032x}" for c in coins], "k": 0,
                                                                  "msg_sha512": message.hex(), "n": 3}]}]}).encode()
            rep = f"proof of {name}".encode()
            opened(CT.SESSION, C.CLOSING_ROUND, 0,
                   hashlib.sha512(K.session_evidence(record, [("circuit.rep0.bin", rep)])).digest())
            statements.append({"member": name, "sessions": [{"name": f"s{k}", "record": self.put(record),
                                                             "reps": {"circuit.rep0.bin": self.put(rep)}}]})
            public = json.dumps((headers or {}).get(name) or header(name)).encode() + b"\nthe rows"
            members[name] = {"circuit": self.put(b"flock-circuit/v1 u\nMETA {\"vus_per_block\": 4}\n" + name.encode()),
                             "public": self.put(public), "registered": None, "units": UNITS[name], "vus_per_block": 4}
        self.params = {"challenger": {"key": {"W": 8, "H": 5, "public": "00" * C.PUBLIC_BYTES}, "label": "sample"},
                       "partition": PARTITION, "program": json.loads(json.dumps(codec.encode_program(PROGRAM))),
                       "query": QUERY, "window_root": "ee" * 64, "proofs": {"members": members}}
        self.evidence = {"coins": {"ledger": [], "main": [], "pins": [], "segment": {}, "session": {"log": log}},
                         "proved_units": [1, 2, 4], "sessions": ["s0", "s1"], "proofs": {"statements": statements}}

    def put(self, b: bytes) -> str:
        d = hashlib.sha512(b).hexdigest()
        (self.at / d[:16]).write_bytes(b)
        self.files.append(self.at / d[:16])
        return d

    def run(self, verdict=None, record: dict = RECORD, evidence: dict | None = None, params: dict | None = None,
            program=PROGRAM):
        return M.members_run(self.params if params is None else params, record,
                             self.evidence if evidence is None else evidence, M.blobs_of(self.files), program=program,
                             verify=lambda key, message, signature: False, verdict=verdict or accepting)


def accepting(st: M.Statement) -> dict:
    return {"accepted": True, "why": "", "program_sha512": st.partition["program"]}


@pytest.fixture(autouse=True)
def ledger(monkeypatch):
    monkeypatch.setattr(C, "trusted_session_of", lambda *a: None)


def test_a_run_of_accepted_sessions_credits_each_members_units_and_maps_to_the_forms_refusal(tmp_path):
    """Each session is verified on its member's own files, with the program and partition every member reads, and
    credited the units of its member's public file its record drew; the run accepts nothing, and what the audit above
    it answers is the form's refusal."""
    w = Window(tmp_path)
    seen = []

    def verdict(st: M.Statement) -> dict:
        seen.append(st)
        return accepting(st)
    run, refused = w.run(verdict)
    assert refused is None and run.all and run.proved == (1, 2, 4)
    assert [(f.session, f.accepted, f.units, f.program) for f in run.findings] == [
        ("s0", True, (1,), PARTITION["program"]), ("s1", True, (2, 4), PARTITION["program"])]
    assert M.run_reason(run, refused) == M.ALL_PASSED
    ms = w.params["proofs"]["members"]
    assert [PF.file_sha512(st.circuit) for st in seen] == [ms[A]["circuit"], ms[B]["circuit"]]
    assert [PF.file_sha512(st.public) for st in seen] == [ms[A]["public"], ms[B]["public"]]
    assert all(json.loads(st.program) == w.params["program"] and st.partition == PARTITION and st.registered is None
               and [n for n, _ in st.reps] == ["circuit.rep0.bin"] for st in seen)


def test_proved_units_other_than_the_evidences_and_a_verdict_of_another_program_are_refused(tmp_path):
    w = Window(tmp_path)
    assert w.run(evidence=w.evidence | {"proved_units": [1, 2]}) == (None, M.PROVED_UNITS)
    assert w.run(lambda st: accepting(st) | {"program_sha512": "ab"}) == (
        None, f"member {A}: session s0's verdict names program ab, not the partition object's "
              f"\"{PARTITION['program']}\"")
    got = w.run(lambda st: {"accepted": True, "why": ""})
    assert got == (None, f"member {A}: session s0's verdict names program none, not the partition object's "
                         f"\"{PARTITION['program']}\"")


def test_a_session_the_verifier_rejects_is_a_finding_and_the_run_proves_only_the_others_units(tmp_path):
    w = Window(tmp_path)
    run, refused = w.run(lambda st: accepting(st) if st.public.read_bytes().startswith(
        json.dumps(header(B)).encode()) else {"accepted": False, "why": "S16"})
    assert refused is None and not run.all and run.proved == (2, 4) and M.run_reason(run, refused) is None
    assert [(f.accepted, f.why, f.units) for f in run.findings] == [(False, "S16", ()), (True, "", (2, 4))]


@pytest.mark.parametrize("change, why", [
    ({"units": {"indices": [0, 1, 2]}}, M.MEMBER_UNITS),
    ({"units": {"indices": [1, 0]}}, M.MEMBER_UNITS),
    ({"units": {}}, M.MEMBER_UNITS),
    ({"scheme": "another"}, "public file: not the registration's: scheme"),
    ({"frame_v3": {"domain_ids": {"out": "aa" * 64}, "roots": {"out": "33" * 64}}},
     "public file: not the registration's: roots"),
    ({"frame_v3": {"domain_ids": {"out": "cc" * 64}, "roots": {"out": "11" * 64}}},
     "public file: not the registration's: domains"),
    ({"frame_v3": {"domain_ids": {"out": "aa" * 64}, "roots": {"out": "11" * 64, "x": "11" * 64}}},
     "public file: not the registration's: roots"),
    ({"instances": 5}, "public file: not the registration's: population"),
    ({"instances": 3, "scheme": 5}, "public file: not the registration's: scheme, population"),
])
def test_an_accepted_session_is_rejected_where_its_public_file_is_not_its_members_slice(tmp_path, change, why):
    """The first member's public file names other units than the member's, in another order or none, or another scheme,
    root, domain or count than the member's roots of the registration, ports bare: its session, accepted by the
    verifier, is rejected with the reason, and the other member's stands."""
    w = Window(tmp_path, {A: header(A, **change)})
    run, refused = w.run()
    assert refused is None and not run.all
    assert [(f.accepted, f.why, f.units) for f in run.findings] == [(False, why, ()), (True, "", (2, 4))]


def test_the_members_population_check_reads_the_registrations_strata(tmp_path):
    strata = P.strata(PROGRAM, QUERY)
    slice_ = {"population": 3, "roots": [{"domain": ROOTS[B][1], "leaves": 4, "port": "out", "root": ROOTS[B][0]}],
              "scheme": SCHEME}
    assert M.population_why(UNITS[B], strata, header(B), slice_) is None
    assert M.population_why([[2, 6]], strata, header(B), slice_) == M.MEMBER_STRATA
    assert M.population_why(UNITS[B], strata[::-1], header(B), slice_) == M.MEMBER_STRATA
    assert M.population_why(UNITS[B], strata, "no header", slice_) == M.MEMBER_UNITS


def test_the_registrations_roots_are_each_one_members_and_none_a_row_tables(tmp_path):
    w = Window(tmp_path)
    roots = RECORD["roots"]

    def refused(*more: dict, params: dict | None = None) -> str:
        run, why = w.run(record=RECORD | {"roots": [*roots, *more]}, params=params)
        assert run is None
        return why
    assert refused({"port": M.TABLE_PORT, "root": "00" * 64}) == M.ROW_TABLES
    assert refused({"port": "zzz/out"}) == ("roots: the registration's root zzz/out is no member's: a member's roots "
                                            "are named <member>/<port>")
    assert refused({"root": "00" * 64}) == "roots: a root of the registration names no port"
    ms = w.params["proofs"]["members"]
    nested = w.params | {"proofs": {"members": {A: ms[A], f"{A}/x": ms[B]}}}
    statements = [w.evidence["proofs"]["statements"][0], w.evidence["proofs"]["statements"][1] | {"member": f"{A}/x"}]
    run, why = w.run(record=RECORD | {"roots": [{"port": f"{A}/x/out"}]}, params=nested,
                     evidence=w.evidence | {"proofs": {"statements": statements}})
    assert run is None and why == (f"roots: the registration's root {A}/x/out is two members': a member's roots are "
                                   "named <member>/<port>, and a root is one member's")
    assert w.run(record=RECORD | {"roots": {}}) == (None, "roots: the registration's roots are not a list")
    assert w.run(record={"scheme": SCHEME, "roots": roots}) == (None, "the record names no population")


def test_what_the_run_does_not_decide_is_outside_and_never_a_pass(tmp_path):
    """Parameters or evidence the per-member parsers refuse, a program other than the parameters', and a registered
    file whose port is no registered port's."""
    w = Window(tmp_path)
    ms = w.params["proofs"]["members"]
    for params, evidence in ((w.params | {"proofs": {"circuit": "", "public": "", "registered": None}}, None),
                             (w.params | {"proofs": {"members": {A: ms[A] | {"units": [[2, 1]]}, B: ms[B]}}}, None),
                             (None, w.evidence | {"proofs": {"sessions": []}}),
                             (None, {k: v for k, v in w.evidence.items() if k != "sessions"})):
        with pytest.raises(M.Outside, match="per-member form"):
            w.run(params=params, evidence=evidence)
    with pytest.raises(M.Outside, match="partition object"):
        w.run(program=P.population_program(PAIR, 5))
    with pytest.raises(M.Outside, match="registered port"):
        w.run(params=w.params | {"proofs": {"members": {A: ms[A] | {"registered": {"ports": {"p": {}}}}, B: ms[B]}}})
