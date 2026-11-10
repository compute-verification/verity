"""`onsite_record.py` on toy served runs' $OUTs. A run whose network certifier, PoUW and composites are missing, as a
rehearsal's, records each of them missing with its why, and no bound and no period figure. A run whose network
certifier accepted in the rows form records K term by term in the notation table's symbols, each naming the public
fields it sums, with its overcount as a note, the seconds it covers, the line that published its verdict, the tree, and
the connections its schedule's caps count (other caps are the connections' error); its run's own period proved, and
each monitoring period's node proved where the run's period covers it and else scaled from the run, and the
datacenter's and the world's scaled from the node's, each scaled cell naming its rule, whose sources (the census lines
by SHA-512) the record states, beside its figure over one execution phase. A public whose terms don't sum to K, or in
the clock form, is the bound's error. The served run's advice charge stays out of the record, ℓ_{a,j} is stated with
its source (a scan's slot entries, charged once where every proof block of it is drawn, else its tied audits' free
inputs; no advice length and no cap), and a test run's measurement is stated beside the bound as that run's. The
figure of record, every period's cells included, is that K, the check's, and nothing added to it. It is proved only
where the lock (a toy one in Security's lock's place) lists every premise's discharge, taking no assumption but a
standing one, the coins are live and, where a phase is charged its tied audits, the verdict held both checks that
charge rests on; otherwise it is the same K, open, with what is open named in the figure and in each cell, none of
which then reads proved. A phase charged its scan's slot entries once rests on none of the tied charge's premises or
checks. ℓ_a, the cap of the run's published advice registration, is stated beside the bound and charged nowhere
(missing where the run published none; an audit's own registration binding an advice is the figure's error). The
record names no guarantee or discharge the tree's locks don't list. A scan's slot entries are charged in k_{skip,j}
only by a period check that charges none once; each cell states its share of the schedule's egress capacity,
and the served egress is missing. Sampled proofs' coins are live only where
every COINS line names the live challenger at a named peer, and the network certifier's chain draws only where each
takes everything; otherwise, or with none, live-coins is open by name (each chain draw by its own), the figure of
record is open on it whatever else holds, and it and its cells are conditional on the coins, and
so is the sampled-proofs verdict where its draw is the one open. While certifier-key is open an accepted network
verdict is not established and its bound conditional on it. It closes only where the lock holds PeriodLeakRegistered
at its pinned statement taking hA, hr and hkey as auditor-record and the run's check of the auditor's inputs holds:
its certifier-key record's key, LMS parameters and session are the period's, and its registration, which checks,
holds that key for a network certifier its log places. Where a circuit file the period lists isn't the one V3c's
published period names, or the scan's files are listed, an accepted network verdict is not established and there is
no bound; so it is, and the composite of record's verdict too, where a file remote's sampled proofs case registers
isn't, by SHA-512, the one its params name, or its public file the one V3c's audit pinned at staging, before the draw,
and where those params name no public file, sampled proofs' verdict is not established. Where V3c's audit's window
commits the scan's state in its activations, the network verdict and the composite of record's stand only where its
registered reads opened that state at the entry's value, root and domain (its published summary's registered.own), the
audit accepted and its record reported each member's reads; otherwise, its evidence missing included, neither is
established and there is no bound. The figure of record says which charge K applied each tied audit: stateFree only
where the lock holds RowsStateHeld at its pinned statement over stateHeldIn and auditStateEntries, else free, with no
figure where it is free and a tied audit states its circuit record. Sampled proofs credits only the drawn members whose
--registered files its audit completed and whose Lean verdicts reported exactly their reads, naming each other member
as not credited. PoUW's freed compute is never credited or claimed, whatever its row states, and neither is one
program a day, each open by name beside what the run's lines say. The run's `publish` leaves the record as written.
The run is of record only when every verdict, the onsite composite's included, accepted and every drawn member is
credited, PoUW's verdict aside: the record claims nothing of it, and lists it as its row states it. Memory's three
verdicts, PoUS's, the bandwidth audit's and the wipe's, are listed as their rows state them, missing where the run has
no such row, and whether the run is of record reads none of them: it is of record exactly when it is without them.
Where the run has
the network certifier's composite without PoUW, that is the composite of record, named by its id: the run is of record
over its verdict and its subprotocols', remote's aside too, its error has no PoUW term, and no verified compute is
credited.
Sampled proofs' guarantee is its window's ε at δ beside its pooled count over any number of windows, and that count for
each monitoring period's node, datacenter and world with its share of their proof blocks, the line stating both; a law
other than a k-subset has no pooled figure, a draw of several members is named refused, and an answer not accepted
states none. Its caller's serving settings, S and MAX_TOKENS, both or neither, label the record and its line, missing
where it states none; runs' records at two settings go beside each other, each run's figures labelled with its own
settings and ℓ_a, and a record of another format, or stating no settings, is refused by path."""
from __future__ import annotations

import dataclasses
import decimal
import hashlib
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import composite as CO  # noqa: E402
import onsite_record as OR  # noqa: E402
import served_cases as SC  # noqa: E402
import served_verdicts as SV  # noqa: E402
import leak_bounds as LB  # noqa: E402
import warden_chain as WC  # noqa: E402
from verity.core import site as S  # noqa: E402
from verity.core.protocols import registration as REG  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import audit as A  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402

NODE = "computeinstance-toy"
SCAN = {"delta": "1/1048576", "drawn": 126, "units": 126, "wrong_units": 0, "free": 9088, "faulty_sets": 0,
        "faulty_bits": 0, "faulty_entries": 0, "jitter": 63}
#: the schedule's 4 · 63 connections of cap 4 (⌈log2 5⌉ = 3 each, as cells and again as schedules) and one execution
#: phase's scan
PUBLIC = {"K": 756 + 756 + 9088 + 63, "audits": [{"delta": "1/100", "drawn": 2, "units": 1024, "wrong_units": 0}],
          "cells": 756, "eps": "1/1048576", "scans": [SCAN], "schedules": 756, "windows": 1}
K = PUBLIC["K"]
#: the period's one compute audit's registration, V3c's class window's, which binds no advice or compiler
NO_ADVICE = {"format": "verity/registration/v2", "window": {"circuit": "cd" * 64}}
#: the run's advice registration as `served_verdicts.py public-registration` publishes it, its advice committed padded to
#: its compiler's cap ℓ_a = 2^23 (`advice_windows.pad`)
PUBLISHED = SV.public_registration({"row_bits": 256, "rows": 2 ** 23 // 256,
                                    "binding": {"advice": {"root": "ab" * 64, "bits": 2 ** 23}},
                                    "compiler": {"name": "c", "digest": "d", "cap_bits": 2 ** 23}})
#: the public of a period check that charges a scan with every proof block drawn its slot entries once: the one
#: phase's 9088 bits as its scan's entries, no tied audit charged
ONCE = dict(PUBLIC, scans=[dict(SCAN, free=0, entries=9088, tied=[0])])
#: T = 63 buckets of 20 ms: an execution phase of 63/50 s
SCHEDULE = {"bucket_ms": 20, "cell_bytes": 512, "delay": 4, "egress_cells": 4, "every": 63, "ingress_buckets": 63,
            "ingress_cells": 4, "lanes": 4, "queue": 16, "slots": 8, "window": 63}
MEASURED = {"format": "verity/advice-charge/v2", "label": "measured on test run toy-test: …", "run": "toy-test",
            "commit_periods": [{"period": 0, "bits": 480}, {"period": 1, "bits": 2390}],
            "bits": {"count": 2, "min": 480, "median": 1435, "mean": 1435.0, "max": 2390, "total": 2870},
            "by_phase": [{"phase": 0, "bits": 2870, "commit_periods": 2}],
            "phase_bits": {"count": 1, "min": 2870, "median": 2870, "mean": 2870.0, "max": 2870, "total": 2870},
            "prompt_len": {"count": 5}, "output_len": {"count": 5}, "finishes": {"eos": 3, "length": 2, "abort": 0}}
TREE = "ab12" * 10
#: the certifier's published verdict line, after an earlier attempt's
LINE = 'THE NETWORK CERTIFIER {"rc": 0, "verdict": "accept"}'
#: SCHEDULE's egress capacity a phase: its 4 · 8 cells a bucket over 63 buckets exceed its caps' 252 · 4 = 1008 cells
CAPACITY = 1008 * 512 * 8
#: the auditor's record of its network certifier's key, as `certifier_serve.public_json` writes one, and session; and
#: the period's params stating them, as `scan_period.rows_case` does
CERT_KEY = {"params": [64, 8, 10], "public": "c3" * 64, "key_id": "1d" * 16}
SESSION = "5e" * 32
CERTIFIER = {"key": CERT_KEY["public"], "lms": {"H": 10, "N": 64, "W": 8, "toy": False}, "session": SESSION}


def key_record(key: dict = CERT_KEY, session: str = SESSION) -> dict:
    """The auditor's record of its network certifier's deployment (`warden_chain.RECORD`)."""
    dep = {"key": key, "session": session}
    return {"schema": WC.RECORD, "key": key, "session": session, "deployment": dep,
            "deployment_sha512": hashlib.sha512(json.dumps(dep).encode()).hexdigest()}


def site_registration(held: str = CERT_KEY["public"], placed: str | None = None,
                      composite: str = "remote-network") -> dict:
    """The auditor's registration of the deployment (``--site-registration``): the log's params, its auditor holding
    ``held`` for its network certifier cert0, and the log placing cert0, its key ``placed`` (``held`` unless given), on
    node0's uplink."""
    log = REG.chain(composite, [REG.site(
        locs=(S.Loc("site0", "site"), S.Loc("node0", "host", "site0"),
              S.Loc("cert0", "certifier", "site0", "auditor", key=held if placed is None else placed)),
        links=(S.Link("uplink0", ("node0", "cert0"), 100_000, 100_000),
               S.Link("uplink0-out", ("cert0", S.OUTSIDE), 100_000, 0)))])
    return {"params": REG.params(composite, S.Auditor(keys=(("cert0", held),))),
            "log": json.loads(REG.committed(composite, log))}


def auditor_args(out: Path) -> dict[str, Path]:
    """The auditor's inputs `served_out` writes beside ``out``, and remote's sampled proofs case under the serve dir
    beside it with V3c's audit's pin of its public files, as `OR.record` takes them."""
    return {"certifier_key": out.parent / "auditor" / "certifier-key.json",
            "site_registration": out.parent / "auditor" / "site-registration.json",
            "sampled_case": out.parent / "serve" / "private" / "sampled-proofs" / "case.json",
            "public_pin": out.parent / "serve" / "v3c" / "audit" / "window" / "public-pin.json"}


#: the circuit and public files remote's sampled proofs case registers, its first two blobs (`proofs.evidence_v2`), the
#: public file a header line and its body
FILES_REGISTERED = (b"toy circuit\n", b"toy public header\ntoy public body")


def sha512_hex(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


#: the window's layout as `pin_of` pins it, θ's proofs.members
LAYOUT = [["m0", 1]]


def pin_of(public: bytes) -> dict:
    """a5's pin at staging (`a5.public_pin`) of a window whose one member, of one unit, has public file ``public``."""
    body = sha512_hex(public.partition(b"\n")[2])
    return {"format": "verity/public-pin/v1", "leaf_layer": sha512_hex(canonical_json({"m0": body})),
            "members": {"m0": {"k": 0, "n": 1, "file": sha512_hex(public), "body": body}}}


def registration_of(pin: dict) -> str:
    """The case's commitment: a registration record, as canonical JSON in hex, holding ``pin``'s leaf layer."""
    return canonical_json({"format": "verity/registration/v1", "leaf_layer": pin["leaf_layer"]}).hex()


def sampled_case(out: Path, proofs: dict | None = None, evidence: dict | None = None) -> Path:
    """Remote's sampled proofs case beside ``out`` (`auditor_args`), as `served_cases.sampled_case` writes one: its blobs
    `FILES_REGISTERED` and a session's file, its params θ naming ``proofs`` (both files' SHA-512s unless given), its
    commitment the registration holding the leaf layer of a5's pin of the public file at staging, written beside it
    (`pin_of`), and its ``evidence``."""
    f, pin = auditor_args(out)["sampled_case"], auditor_args(out)["public_pin"]
    f.parent.mkdir(parents=True, exist_ok=True)
    pin.parent.mkdir(parents=True, exist_ok=True)
    blobs = [f.parent.parent.parent / n for n in ("circuit.txt", "pub-1.bin", "session.json")]
    for b, data in zip(blobs, (*FILES_REGISTERED, b"{}")):
        b.write_bytes(data)
    pin.write_text(json.dumps(pin_of(FILES_REGISTERED[1])))
    proofs = {"circuit": sha512_hex(FILES_REGISTERED[0]), "public": sha512_hex(FILES_REGISTERED[1]),
              "registered": None, "members": LAYOUT} if proofs is None else proofs
    f.write_text(json.dumps({"params": {"form": "one-stage", "proofs": proofs},
                             "commitment": registration_of(pin_of(FILES_REGISTERED[1])),
                             "evidence": json.dumps((evidence or {}) | {"form": "one-stage"}).encode().hex(),
                             "blobs": [str(b) for b in blobs]}))
    return f


def auditor_files(out: Path, key: dict | None = None, site: dict | None = None) -> dict[str, Path]:
    """`auditor_args`' files: ``key`` (`key_record` unless given), ``site`` (`site_registration` unless given) and
    `sampled_case`'s."""
    args = auditor_args(out)
    args["certifier_key"].parent.mkdir(parents=True, exist_ok=True)
    args["certifier_key"].write_text(json.dumps(key or key_record()))
    args["site_registration"].write_text(json.dumps(site or site_registration()))
    sampled_case(out)
    return args


def record(out: Path, **kw) -> dict:
    """`OR.record` of ``out``, with the auditor's inputs beside it (`auditor_args`) unless ``kw`` gives them."""
    return OR.record(out, **(auditor_args(out) | kw))


def main(argv: list[str]) -> int:
    """`OR.main` on ``argv``, with OUT's auditor's inputs beside it (`auditor_args`) where it names an OUT."""
    if argv and not argv[0].startswith("--"):
        a = auditor_args(Path(argv[0]))
        argv = [*argv, "--certifier-key", str(a["certifier_key"]), "--site-registration", str(a["site_registration"]),
                "--sampled-proofs-case", str(a["sampled_case"]), "--public-pin", str(a["public_pin"])]
    return OR.main(argv)


#: stand-ins for the theorems that discharge the premises `PREMISES` names no discharge of yet and no guarantee states,
#: and those premises as they are
TOY = {n: f"Toy.{n}" for n, p in OR.PREMISES.items() if not p["discharged_by"] and "stated_by" not in p}
UNNAMED = {n: OR.PREMISES[n] for n in TOY}
#: the guarantees of `RESTS_ON` as they are, and those no statement of which is pinned yet
RESTS_ON = {g: dict(r) for g, r in OR.RESTS_ON.items()}
UNPINNED = [g for g, r in OR.RESTS_ON.items() if r["statement"] is None]
ACCEPTED = "NetTiming.SecurityProofs.PeriodLeakAccepted"
#: the stand-ins that discharge the free-input count's premise, schedules-covered, held-performed and wires-fit
COUNTED, COVERED, HELD, WIRED = (TOY[n] for n in ("free-count", "schedules-covered", "held-performed", "wires-fit"))


def discharged(drop: tuple[str, ...] = OR.ENTRIES) -> dict:
    """``premises`` of a lock listing every discharge and each guarantee the figure rests on at the statement it
    names, none with an assumption, less the premises ``drop`` of a charge the public applies to no phase (the slot
    entries' unless given: `PUBLIC` charges its phase its tied audits)."""
    found = {n: {"is": p["is"], "discharged_by": list(p["discharged_by"]), "missing": [], "unpinned": {}, "assumes": [],
                 "takes": []} | ({"charge": p["charge"]} if "charge" in p else {})
             | ({"stated_by": p["stated_by"], "checked": p["checked"]} if "stated_by" in p else {})
             for n, p in OR.PREMISES.items() if n not in drop}
    return {"is": OR.DISCHARGE,
            "from": "Security/lean-audit.json and Security/Properties/lean-audit.json guarantees and reads",
            "premises": found, "standing": OR.STANDING, "counts": OR.COUNTS, "open": [], "unread": [],
            "rests_on": {g: {"assumptions": [], "statement": r["statement"], "takes": r["takes"]}
                         for g, r in OR.RESTS_ON.items()}}


def lock_doc(listed: dict[str, list[str]], stated: dict[str, str | None] | None = None) -> dict:
    """A lock listing each guarantee of ``listed`` with its named assumptions, each of `RESTS_ON` and `DISCHARGES` at
    the statement ``stated`` gives it, else at the one `RESTS_ON` or `DISCHARGES` pins, among its reads."""
    pins = OR.RESTS_ON | OR.DISCHARGES
    stated = {g: r["statement"] for g, r in pins.items()} | (stated or {})
    guarantees = {g: {"assumptions": a} | ({"signature": [f"{g} : {g}.Stated"]} if g in stated else {})
                  for g, a in listed.items()}
    reads = {"Definitions.Toy": {"definitions": {f"{g}.Stated": h for g, h in stated.items() if h is not None}}}
    return {"guarantees": guarantees, "reads": reads}


def write_lock(f: Path, given: dict[str, list[str] | None] | None = None,
               stated: dict[str, str | None] | None = None) -> None:
    """A lock at ``f`` listing every discharge of `PREMISES` and each guarantee the figure rests on with no named
    assumption, each guarantee of ``given`` instead with the assumptions given, or unlisted where None, and each of
    `RESTS_ON` at the statement it names unless ``stated`` gives another."""
    names = [t for p in OR.PREMISES.values() for t in p["discharged_by"]] + list(OR.RESTS_ON)
    listed = {g: [] for g in names} | (given or {})
    f.write_text(json.dumps(lock_doc({g: a for g, a in listed.items() if a is not None}, stated)))


@pytest.fixture
def verified(monkeypatch) -> None:
    """The side on which the period's check verifies the answers it reads (`ANSWERS_VERIFIED`): nothing stands beside K,
    and the record writes its periods' cells. It reads them as given, so every other test is on that side."""
    monkeypatch.setattr(OR, "ANSWERS_VERIFIED", True)


@pytest.fixture(autouse=True)
def lock(tmp_path_factory, monkeypatch) -> Path:
    """A lock in Security's lock's place discharging every premise, each of `UNNAMED` by its `TOY` stand-in, each
    discharge at a stand-in statement `DISCHARGES` pins, taking nothing, and listing each guarantee of `RESTS_ON` at the
    statement it names, each of `UNPINNED` restated, at a stand-in's."""
    for name, toy in TOY.items():
        monkeypatch.setitem(OR.PREMISES, name, dict(OR.PREMISES[name], discharged_by=(toy,)))
    for t in (t for p in OR.PREMISES.values() for t in p["discharged_by"]):
        monkeypatch.setitem(OR.DISCHARGES, t, {"statement": f"toy-{t}", "takes": {}})
    for g in UNPINNED:
        monkeypatch.setitem(OR.RESTS_ON, g, dict(OR.RESTS_ON[g], statement=f"toy-{g}"))
    f = tmp_path_factory.mktemp("lock") / "lean-audit.json"
    write_lock(f)
    monkeypatch.setattr(OR, "LOCKS", (f,))
    return f


def answer(public: dict | None, verdict: str = "accept") -> dict:
    return {"protocol": OR.NETWORK, "verdict": verdict, "reason": None, "commitment": "ab" * 64, "public": public,
            "profile": {"link": "served", "windows": 1, "placement": {"trusted-host": "the certifier on the gate's host"}}}


def network_row(public: dict | None) -> dict:
    return {"protocol": OR.NETWORK, "owner": "network-accounting", "verdict": "accept", "reason": None,
            "commitment": "ab" * 64, "quantity": public, "delta": None, "epsilon": None, "runs": [],
            "answer": answer(public)}


#: the sampled-proofs part of a COINS line naming the live challenger, as served_verdicts.sh says it under PP_CHALLENGER
GATE = "the challenger gate:research@toy-gate (peer toy-peer)"


def coins_line(said: str, lead: str = "") -> str:
    """The run's COINS line as served_verdicts.sh says it, its sampled-proofs part ``said``, led by ``lead``."""
    return (f"COINS {lead}sampled proofs: {said}; PoUW H1: not served (H1_GATE empty); the bandwidth audit's: "
            "vy-auditor-2's bw_party, inside PoUS's piece")


#: sampled proofs' one drawn member, and its audit's published record (`served_integrity.public_copy`) crediting it:
#: its --registered file completed and its Lean verdict of record reporting exactly its reads
MEMBER = "ToyGemm"
#: sampled proofs' window, N proof blocks, and its profile's δ
N, DELTA = 16, Fraction(1, 2 ** 40)
#: the run of record's draw, one member (subset:1), and a member of each of two templates' strata (stratified:1)
SUBSET1 = {"law": "subset", "k": 1, "units": [3]}
STRATIFIED1 = {"law": "stratified", "k": 2, "units": [3, 12],
               "strata": [{"template": "A", "units": [[0, 8]], "k": 1}, {"template": "B", "units": [[8, 16]], "k": 1}]}


def integrity_answer(draw: dict, bound: int) -> dict:
    """Sampled proofs' accepted answer as `served_integrity.answer` writes it: its audit's profile under ``draw`` over
    N proof blocks at DELTA, and ``bound`` incorrect proof blocks there."""
    profile = A.profile({"population": N, "partition": "toy", "program": "ab" * 64}, draw, float(DELTA)).to_json()
    return {"protocol": "verity/sampled-proofs/one-stage/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
            "profile": profile | {"delta": str(DELTA)},
            "public": {"bound": bound, "drawn": draw["units"], "per_template": None, "proved": draw["units"]}}


def audit_record(completed: tuple[str, ...] = (MEMBER,), reported: dict[str, bool] | None = None,
                 refused: tuple[str, ...] = ()) -> dict:
    """The audit's published record: the drawn members ``completed``, each of ``refused`` not, and each member's
    registered reads ``reported`` (each completed one's ok unless given)."""
    reported = {m: True for m in completed} if reported is None else reported
    return {"accepted": True,
            "checks_after_draw": {"registered_complete": {"ok": not refused, "codes": list(refused),
                                                          "members": list(completed)}},
            "registered_reads": {"ok": all(reported.values()), "chunks": {m: {"ok": ok} for m, ok in reported.items()}}}


AUDITED = audit_record()
CREDITED = {"credited": [MEMBER], "not_credited": {}, "from": str(OR.AUDIT), "is": OR.CREDIT}
#: the credit of sampled proofs where its verdict is not established: none
UNESTABLISHED = {"credited": [], "missing": OR.UNESTABLISHED_CREDIT, "is": OR.CREDIT}


def served(row: dict | None) -> str | None:
    """The server the toy window report names for ``row``, as `served_verdicts.record` does for a run whose every answer
    is about `NODE`: `NODE` for an answer, none for a row that is missing, dropped or absent. Its ``sources`` states
    each row's answer real, as the report does unless a piece says otherwise."""
    return NODE if row is not None and row["verdict"] not in ("missing", "dropped") else None


def served_out(tmp_path: Path, *, network: dict | None = None, work: dict | None = None, case: dict | None = None,
               composites: dict[str, dict] | None = None, coins: str | None = GATE,
               audit: dict | None = AUDITED, draw: dict = SUBSET1, bound: int = N - 1) -> Path:
    """A toy $OUT: sampled proofs accepted under ``draw`` with ``bound`` (`integrity_answer`), its audit's record
    ``audit`` (none where None), PoUW and the network certifier missing unless given, the given composites and
    certifier case, and its COINS line, its sampled-proofs part ``coins`` (none where None); and beside it the auditor's
    inputs (`auditor_files`)."""
    out = tmp_path / "out"
    auditor_files(out)
    rows = {"integrity": {"protocol": "verity/sampled-proofs/one-stage/v1", "owner": "compute-accounting",
                          "verdict": "accept", "reason": None, "answer": integrity_answer(draw, bound)},
            "work": work or {"owner": "compute-accounting", "verdict": "missing", "why": "POUW_CMD unset: toy"},
            "communication": network or {"owner": "network-accounting", "verdict": "missing",
                                         "why": "CERTIFIER_CMD unset: the network certifier in front of the serving "
                                                "endpoint"}}
    (out / "record").mkdir(parents=True)
    (out / "record" / "verdicts.json").write_text(json.dumps(
        {"format": OR.EXTENDS, "window": 3, "node": NODE, "run": "toy-run", "protocols": rows,
         "servers": {key: served(row) for key, row in rows.items()}, "sources": dict.fromkeys(rows, "real")}))
    for name, doc in (composites or {}).items():
        (out / "composite" / name).mkdir(parents=True)
        (out / "composite" / name / "composite.json").write_text(json.dumps(doc))
    if case is not None:
        (out / OR.CASE).parent.mkdir(parents=True)
        (out / OR.CASE).write_text(json.dumps(case))
    if coins is not None:
        (out / "verdicts.txt").write_text(coins_line(coins) + "\n")
    if audit is not None:
        (out / OR.AUDIT).parent.mkdir(parents=True)
        (out / OR.AUDIT).write_text(json.dumps(audit))
    return out


def chain_report(n2: str = "all", scan: str = "all") -> dict:
    """The network certifier's published chain report (`certifier_piece.PUBLIC`), its draws at these laws."""
    return {"steps": {s: {"eps_at_delta": {"law": law, "eps": "0"}} for s, law in (("n2", n2), ("scan", scan))}}


#: the chain report whose draws each take everything, and the record's chain draws read from it
ALL = chain_report()
ALL_DRAWN = {"laws": {"the period draw": "all", "the scan's draw": "all"}, "open": [], "from": str(OR.CHAIN_REPORT),
             "is": OR.CHAIN_DRAWS}


#: what ``of_record`` states of a run whose composite of record is onsite's
ONSITE_OF_RECORD = {"composite": OR.ONSITE, "verified_compute": "none"}
#: the charge the figure of record states of one audit with no circuit record where no lock lists RowsStateHeld
FREE_CHARGE = {"charge": "free", "is": OR.STATE_CHARGE, "tied": [0], "with_record": [],
               "unpinned": {g: None for g in OR.STATE_PINS}}
SAMPLED_V1 = "verity/sampled-proofs/v1"
REMOTE_IDS = (OR.INFERENCE_ONLY, OR.POUW_ID, SAMPLED_V1)
#: the id the network certifier's composite without PoUW is registered under
WITHOUT = "verity/remote-network-no-pouw/v0"
WITHOUT_IDS = (OR.NETWORK, OR.INFERENCE_ONLY, SAMPLED_V1)


def sha512_of(params) -> str:
    """The SHA-512 `composite.py` states of a case's ``params``, over their canonical JSON (`composite.canonical`)."""
    return hashlib.sha512(CO.canonical(params)).hexdigest()


def composed(out: Path, f: Path | None = None) -> str:
    """`sha512_of` the params of OUT's network certifier case (or the case at ``f``), "00" where there is none."""
    f = out / OR.CASE if f is None else f
    return sha512_of(json.loads(f.read_text())["params"]) if f.is_file() else "00"


def sampled_stated(out: Path) -> dict[str, str]:
    """What a composite states of remote's sampled proofs case beside OUT (`sampled_case`): its params' SHA-512 and its
    commitment, "00" where there is none."""
    f = auditor_args(out)["sampled_case"]
    return {"params_sha512": composed(out, f),
            "commitment": json.loads(f.read_text())["commitment"] if f.is_file() else "00"}


def with_composite(out: Path, name: str, product: str, ids: tuple[str, ...], verdict: str = "accept",
                   missing: tuple[str, ...] = (), inference_only: str = "accept") -> Path:
    """``out`` with ``product``'s composite.json in composite/NAME: a case of each of ``ids`` but those ``missing``,
    which it names, inference-only's answering ``inference_only`` and each other accepting, the network certifier's
    with the params of OUT's case and remote's sampled proofs' with the params and commitment of its case beside OUT
    (`composed`, `sampled_case`), and Lean's ``verdict``."""
    stated = {OR.NETWORK: {"params_sha512": composed(out), "commitment": "00"}, SAMPLED_V1: sampled_stated(out)}
    cases = [{"id": i, "verdict": inference_only if i == OR.INFERENCE_ONLY else "accept"}
             | stated.get(i, {"params_sha512": "00", "commitment": "00"}) for i in ids if i not in missing]
    (out / "composite" / name).mkdir(parents=True, exist_ok=True)
    (out / "composite" / name / "composite.json").write_text(json.dumps(
        {"product": product, "commitment": "00", "missing": list(missing), "exceptions": [], "cases": cases,
         "verdict": {"id": product, "verdict": verdict}}))
    return out


def remote_composite(verdict: str = "accept") -> dict:
    cases = [{"id": i, "verdict": "accept", "params_sha512": "00", "commitment": "00"}
             for i in (OR.INFERENCE_ONLY, "verity/pouw/period/v1", "verity/sampled-proofs/one-stage/v1")]
    return {"product": OR.REMOTE, "commitment": "00", "missing": [], "exceptions": [], "cases": cases,
            "verdict": {"id": OR.REMOTE, "verdict": verdict}}


def with_onsite(out: Path, sampled: str = "accept") -> Path:
    """``out`` with onsite's composite, accepting, its network certifier's case the params of OUT's (`composed`) and
    remote's sampled proofs case, answering ``sampled``, the params and commitment of the case beside OUT
    (`sampled_stated`)."""
    cases = [c for c in remote_composite()["cases"] if c["id"] != "verity/sampled-proofs/one-stage/v1"] + [
        {"id": SAMPLED_V1, "verdict": sampled} | sampled_stated(out),
        {"id": OR.NETWORK, "verdict": "accept", "params_sha512": composed(out), "commitment": "00"}]
    doc = remote_composite() | {"product": OR.ONSITE, "cases": cases, "verdict": {"id": OR.ONSITE, "verdict": "accept"}}
    (out / "composite" / "remote-network").mkdir(parents=True, exist_ok=True)
    (out / "composite" / "remote-network" / "composite.json").write_text(json.dumps(doc))
    return out


def commitment(*audits: dict) -> str:
    return json.dumps({"audits": list(audits), "windows": [{}]}).encode().hex()


def accepted(tmp_path: Path, public: dict = PUBLIC, windows: int = 1, caps: list[int] | None = None,
             schedule: dict = SCHEDULE, audits: tuple[dict, ...] = (NO_ADVICE,),
             registration: dict | None = PUBLISHED, coins: str | None = GATE,
             draws: dict | None = ALL, params: dict | None = None, onsite: bool = True) -> Path:
    """A run whose network certifier accepted ``public``, its case the period's of ``windows`` execution phases at
    ``schedule``, whose caps are the schedule's (`Schedule.caps`) unless given, the network certifier's key, LMS
    parameters and session the auditor's record's (`CERTIFIER`) and any other ``params``, and whose commitment holds
    ``audits``' registrations, with ``registration`` its published advice registration, none where None, ``coins`` its
    COINS line's sampled-proofs part (`served_out`), ``draws`` its chain report, none where None, and remote's
    composite and, where ``onsite``, onsite's (`with_onsite`), the composite of record that composed its case."""
    pouw = {"protocol": "verity/pouw/period/v1", "owner": "compute-accounting", "verdict": "accept", "reason": None}
    if caps is None:
        caps = [schedule["egress_cells"]] * (schedule["lanes"] * windows * schedule["window"])
    out = served_out(tmp_path, network=network_row(public), work=pouw,
                     case={"commitment": commitment(*audits), "evidence": "00",
                           "params": {"windows": windows, "cells": caps, "schedule": schedule, **CERTIFIER}
                           | (params or {})},
                     composites={"remote": remote_composite()}, coins=coins)
    (out / "machine.txt").write_text(f"ubuntu\n{NODE}\n{TREE}\n96\nserve dir /serve/d\n")
    with open(out / "verdicts.txt", "a") as f:
        f.write(f'THE NETWORK CERTIFIER {{"rc": 1, "verdict": "error"}}\nRECORD {{}}\n{LINE}\n'
                "COMPOSITE CASE verity/certifier/v0 accept\n")
    if registration is not None:
        (out / OR.REGISTRATION).write_text(json.dumps(registration))
    if draws is not None:
        (out / OR.CHAIN_REPORT).parent.mkdir(parents=True)
        (out / OR.CHAIN_REPORT).write_text(json.dumps(draws))
    return with_onsite(out) if onsite else out


def numbers(x) -> list:
    """Every number in a JSON document."""
    if isinstance(x, dict):
        return [n for v in x.values() for n in numbers(v)]
    if isinstance(x, list):
        return [n for v in x for n in numbers(v)]
    return [x] if isinstance(x, (int, float)) and not isinstance(x, bool) else []


def test_a_rehearsal_records_each_missing_part_with_its_why_and_no_figure(tmp_path, capsys):
    out = served_out(tmp_path)
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    v = rec["verdicts"]
    assert list(v) == sorted(OR.VERDICTS) and v["sampled proofs"] == {
        "protocol": SAMPLED_V1, "verdict": OR.NOT_ESTABLISHED, "answer": "accept",
        "from": "record/verdicts.json protocols.integrity",
        "why": f"{OR.V1_UNREAD}, and its case is read from {OR.ONSITE}'s composite, and the run has none",
        "credit": UNESTABLISHED}
    assert v["PoUW"] == {"verdict": "missing", "why": "POUW_CMD unset: toy", "from": "record/verdicts.json protocols.work",
                         "not_claimed": OR.POUW_OPEN}
    assert v["the network certifier"]["why"].startswith("CERTIFIER_CMD unset")
    assert v["remote"]["why"] == "remote's composite didn't run: the run has no composite/remote/composite.json"
    assert v["inference-only"]["verdict"] == v["onsite"]["verdict"] == "missing"
    assert rec["of_record"] == {"holds": False, "missing": ["PoUW", "inference-only", "remote", "the network certifier",
                                                            "onsite"], "not_accepting": [],
                                "not_established": ["sampled proofs"], "not_credited": [], "not_claimed": ["PoUW"],
                                "uncredited": OR.UNESTABLISHED_CREDIT, **ONSITE_OF_RECORD}
    assert rec["coins"]["open_draws"] == [] and "missing" in rec["coins"]["draws"]
    assert rec["bound"] == {"missing": "the network certifier's verdict is missing (CERTIFIER_CMD unset: the network "
                                       "certifier in front of the serving endpoint): only its accepted answer certifies K"}
    assert rec["periods"] == {"missing": "no bound"}
    assert rec["advice_charge"] == {"served": OR.ADVICE, "charged": {"missing": "no bound"},
                                    "cap": {"missing": "no bound"}, "measured": {"missing": OR.NO_MEASUREMENT}}
    assert numbers({k: rec[k] for k in ("bound", "periods", "verdicts", "advice_charge")}) == []
    assert (rec["composite_error"]["stated"], rec["composite_error"]["vacuous"]) == (None, False)
    assert rec["composite_error"]["terms"][3] == {"subprotocol": "the network certifier", "missing": "no accepted bound",
                                                  "from": f"{OR.EPS['source']}: its eps (`EPS`)"}
    assert "window" not in rec and rec["tree"] is None
    line = capsys.readouterr().out.strip()
    assert line.startswith("ONSITE ") and json.loads(line[7:])["K"] is None and json.loads(line[7:])["answers"] is None


def test_wherever_K_is_stated_the_answers_it_is_on_stand_beside_it_and_nothing_is_scaled_from_it(tmp_path, capsys):
    """K is on one-stage answers the driver verified, which the period's check read as given: the record's K, its
    figure of record and its line each state that beside the number, the record's own account of K doesn't call it a
    bound, and no period's figure is made from it: no cell, none proved, no share, no pooled eps, no year's cell."""
    out = accepted(tmp_path)
    assert main([str(out)]) == 0
    text = (out / "record" / "onsite.json").read_text()
    rec, line = json.loads(text), json.loads(capsys.readouterr().out.strip()[7:])
    b = rec["bound"]
    assert OR.ANSWERS_VERIFIED is False and OR.ANSWERS == "answers read as given" and OR.given() == {"answers": OR.ANSWERS}
    assert OR.GIVEN.startswith("K, on one-stage answers the driver verified; the check read them as given")
    assert (b["K"], b["answers"], b["is"]) == (K, OR.ANSWERS, OR.GIVEN)
    assert (b["of_record"]["K"], b["of_record"]["answers"]) == (K, OR.ANSWERS)
    assert rec["periods"] == {"missing": OR.NOT_SCALED} and OR.cells(rec["periods"]) == rec["periods"]
    assert "answers" not in rec and OR.read_as(rec) == OR.ANSWERS
    assert "read as given" in OR.NOT_SCALED and "labelled number" in OR.NOT_SCALED
    assert '"how"' not in json.dumps({k: rec[k] for k in ("bound", "periods")}) and '"how": "proved"' not in text
    assert not {"share", "share_served", "share_of_capacity"} & set(re.findall(r'"(\w+)":', json.dumps(rec["periods"])))
    assert (line["K"], line["answers"]) == (K, OR.ANSWERS)
    assert (line["year_datacenter"], line["eps_pooled"]) == (None, None)
    assert "no bound" in OR.GIVEN and "The bound" not in OR.__doc__
    assert "certifier's bound" not in OR.__doc__.split("\n\n")[0]


@pytest.mark.usefixtures("verified")
def test_once_the_check_verifies_its_answers_the_record_says_so_itself_and_writes_its_periods_cells(tmp_path, capsys):
    """The other side of `ANSWERS_VERIFIED`: the record says in its own ``answers`` that its answers were verified, no
    read-as-given label is in it or its line, and its cells are as `periods` makes them."""
    out = accepted(tmp_path)
    assert main([str(out)]) == 0
    text = (out / "record" / "onsite.json").read_text()
    rec, line = json.loads(text), json.loads(capsys.readouterr().out.strip()[7:])
    assert OR.given() == {} and OR.ANSWERS not in text and OR.GIVEN not in text and OR.NOT_SCALED not in text
    assert rec["answers"] == OR.VERIFIED == OR.read_as(rec) and text.count('"answers"') == 1
    assert (rec["bound"]["K"], rec["bound"]["of_record"]["K"], line["K"], line["answers"]) == (K, K, K, OR.VERIFIED)
    p = rec["periods"]
    assert p["run"]["bits"] == K and p["run"]["how"] == "open" and set(LB.PERIODS) <= set(p)
    assert all(p[n][w]["how"] == "scaled" for n in LB.PERIODS for w in ("datacenter", "world"))
    assert line["year_datacenter"]["bits"] == p["year"]["datacenter"]["bits"] and line["eps_pooled"] is not None
    assert OR.cells(p, OR.read_as(rec))["answers"] == OR.VERIFIED and OR.cells(p)["run"]["bits"] == K


@pytest.mark.parametrize("reading", [False, True])
def test_a_record_that_doesnt_say_its_answers_were_verified_is_read_as_given_whatever_the_reading_code_is(
        tmp_path, capsys, monkeypatch, reading):
    """The label a record is read with comes from the record, never from the code reading it. A kept record with its
    periods' cells and no ``answers`` of its own, as one written before the label is, prints `ANSWERS` beside its K and
    at the head of its cells, in its line and under ``--beside``, with `ANSWERS_VERIFIED` False and with it True; and
    so does one whose ``answers`` says anything but `VERIFIED`. One that says `VERIFIED` is read as it says, on both."""
    out = accepted(tmp_path / "run", registration=published(2 ** 23))
    with monkeypatch.context() as mp:
        mp.setattr(OR, "ANSWERS_VERIFIED", True)
        assert main([str(out), "--window-steps", "4", "--max-tokens", "8"]) == 0
    said = json.loads((out / "record" / "onsite.json").read_text())
    kept = {k: x for k, x in said.items() if k != "answers"}
    assert said["answers"] == OR.VERIFIED and "run" in kept["periods"] and '"answers"' not in json.dumps(kept)
    capsys.readouterr()
    monkeypatch.setattr(OR, "ANSWERS_VERIFIED", reading)
    for name, rec, label in (("kept", kept, OR.ANSWERS), ("other", kept | {"answers": "verified"}, OR.ANSWERS),
                             ("said", said, OR.VERIFIED)):
        assert OR.read_as(rec) == label
        line = OR.summary(rec)
        assert (line["K"], line["answers"]) == (K, label) and line["year_datacenter"] is not None
        f, to = tmp_path / f"{name}.json", tmp_path / f"{name}-beside.json"
        f.write_text(json.dumps(rec))
        assert main(["--beside", str(f), "--to", str(to)]) == 0
        (run,) = json.loads(to.read_text())["runs"]
        assert (run["K"], run["answers"], run["periods"]["answers"]) == (K, label, label)
        assert run["periods"]["run"]["bits"] == rec["periods"]["run"]["bits"] == K
        assert all(run["periods"][n][w] == {k: rec["periods"][n][w][k] for k in ("bits", "share", "how")}
                   for n in LB.PERIODS for w in ("node", "datacenter", "world"))
        (printed,) = json.loads(capsys.readouterr().out.strip()[14:])
        assert (printed["K"], printed["answers"]) == (K, label)


@pytest.mark.usefixtures("verified")
def test_an_accepted_rows_form_answer_is_K_term_by_term_with_its_overcount_and_connections(tmp_path):
    rec = record(accepted(tmp_path))
    b = rec["bound"]
    assert (b["K"], b["phases"], b["period_s"], b["statement"]) == (
        K, 1, "63/50", "NetTiming.Guarantees.PeriodLeakAccepted")
    assert "eps" not in b and b["delta_net"] == {"note": OR.UNVALUED, "terms": [
        {"symbol": "eps", "value": "1/1048576", **OR.EPS}, {"symbol": "ε_ks", "times": 1, **OR.ERRORS["ε_ks"]},
        {"symbol": "ε_bind", "times": 1, **OR.ERRORS["ε_bind"]}, {"symbol": "ε_tap", "times": 1, **OR.ERRORS["ε_tap"]}]}
    assert all(t["source"] for t in b["delta_net"]["terms"]) and "δ part" in OR.EPS["is"]
    assert (rec["run"], rec["tree"], b["verdict_line"]) == ("toy-run", TREE, LINE)
    assert [(t["symbol"], t.get("phase"), t["bits"], t["lean"]) for t in b["terms"]] == [
        ("k_{sched}", None, 756, ["schedules"]), ("k_{fail}", None, 756, ["cells"]),
        ("\\ell_{a,j}", 0, 9088, ["scans[0].free"]),
        ("k_{skip,j}", 0, 0, ["scans[0].faulty_sets", "scans[0].faulty_bits", "scans[0].faulty_entries"]),
        ("T_j \\log_2 d_{jit}", 0, 63, ["scans[0].jitter"])]
    assert sum(t["bits"] for t in b["terms"]) == b["K"]
    assert b["notes"] == [OR.OVERCOUNT] and "k_sched" in OR.OVERCOUNT and "k_fail" in OR.OVERCOUNT
    assert "answers" not in b and "is" not in b and "proved" not in OR.OVERCOUNT
    assert b["connections"] == {"count": 4 * 63, "from": OR.CONNECTIONS}
    assert rec["placement"] == {"trusted-host": "the certifier on the gate's host"}
    assert rec["advice_charge"]["charged"] == {
        "is": OR.CHARGED, "bits_per_phase": [9088], "entries_per_phase": [0], "share": 9088 / K,
        "lean": "scans[j].free + scans[j].entries", "link": OR.ADVICE_LINK}
    assert "no cap and no advice length" in OR.CHARGED and "audits[].free" in OR.CHARGED
    assert b["of_record"] == {"K": K, "tie_charged": [0], "held": [], "open": ["free", "circuit"], "is": OR.OPEN,
                              "premises": discharged(), "advice": {"is": OR.ADVICE_LINK}, "tight": OR.TIGHT,
                              "state_charge": FREE_CHARGE}
    assert b["skip"] == {"is": OR.SKIP, "bits_per_phase": [0], "entries_per_phase": [0]}
    assert rec["advice_charge"]["cap"] == {"ell_a": [2 ** 23], "is": OR.CAP}
    assert "A cap, in no term" in OR.CAP and "`CAP`" in OR.OPEN and "`DISCHARGE`" in OR.OPEN
    run = rec["periods"]["run"]
    assert (run["bits"], run["how"], run["open"]) == (K, "open", ["free", "circuit"]) and "periods_tight" not in rec
    assert rec["periods"]["egress"] == {"capacity_bits_per_phase": CAPACITY, "is": OR.EGRESS,
                                        "served": {"missing": OR.SERVED_EGRESS}}


@pytest.mark.usefixtures("verified")
def test_the_served_audits_cap_is_stated_and_charged_nowhere_whatever_is_open(tmp_path, capsys):
    """The served shape: one V3c audit, its class window binding no advice, tied to the one phase's scan, and the run's
    advice registration published padded to its compiler's cap ℓ_a = 2^23. With neither check held, the figure of
    record is still the check's K, open on both by name, and every cell is that figure's: no ℓ_a + 1 is added for the
    tied audit, in the record or its line, the cap is stated beside the bound, and the record says the link figure
    charges the advice through the entries and how the figure tightens in its place (`TIGHT`)."""
    out = accepted(tmp_path)
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    of = rec["bound"]["of_record"]
    assert (of["K"], of["open"], rec["advice_charge"]["cap"]["ell_a"], of["advice"]) == (
        K, ["free", "circuit"], [2 ** 23], {"is": OR.ADVICE_LINK})
    assert rec["advice_charge"]["charged"]["bits_per_phase"] == [9088]
    assert rec["bound"]["K"] == K and of["tight"] == OR.TIGHT and "periods_tight" not in rec
    assert not {2 ** 23 + 1, K + 2 ** 23 + 1} & set(numbers(rec)) and rec["periods"]["run"]["bits"] == K
    assert all(x in OR.ADVICE_LINK for x in ("charges the advice through the entries", "scanOf refuses any other",
                                             "no ℓ_a, no ℓ_a + 1 and no |a_j|"))
    assert all(x in OR.TIGHT for x in ("in place of the figure of record and never beside it",
                                       "proving every proof block of its scan", "`OVERCOUNT`", "`tied-count`",
                                       "states no figure of any"))
    assert "No bits are added for what is open" in OR.OPEN and "open premise or not" in OR.CAP
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["K"], line["open"], "K_tight" in line) == (K, ["free", "circuit"], False)
    assert K + 2 ** 23 + 1 not in numbers(line)


@pytest.mark.usefixtures("verified")
@pytest.mark.parametrize("held, open_", [(["free", "circuit"], []), (["circuit", "free"], []), (["free"], ["circuit"]),
                                         (["circuit", "x"], ["free"])])
def test_a_tied_charge_is_proved_only_where_the_verdict_held_both_checks_and_else_the_same_K_open(tmp_path, held, open_):
    """A phase charged its tied audits rests on both checks of `CHECKS`: with one the verdict didn't hold, the figure of
    record is the same K, open on it by name, and so is its run's cell, which reads proved only with both held."""
    rec = record(accepted(tmp_path, dict(PUBLIC, held=held)))
    of, run = rec["bound"]["of_record"], rec["periods"]["run"]
    assert (of["held"], of["open"], of["K"]) == ([c for c in OR.CHECKS if c not in open_], open_, K)
    assert (run["bits"], run["how"], run.get("open")) == (K, "open" if open_ else "proved", open_ or None)
    assert of["tight"] == OR.TIGHT and "periods_tight" not in rec


def held_both(out: Path) -> dict:
    """The figure of record of ``out``'s served audit, both checks held."""
    return record(accepted(out, dict(PUBLIC, held=["free", "circuit"])))


@pytest.mark.usefixtures("verified")
def test_with_no_discharge_in_the_lock_every_premise_is_open_by_name_though_both_checks_are_held(
        tmp_path, lock, monkeypatch):
    """A lock listing the guarantees the figure rests on and none of the theorems that discharge its premises, and
    the premises with no discharge named, as the tree's are: every premise of a tied charge is open, by name, so the
    figure of record is the check's K, open, with nothing added, and no cell of it reads proved: the run's and its
    figure over one phase are open, naming each, and a scaled one names them beside its rule. certifier-key alone
    closes, by PeriodLeakRegistered's pin taking the auditor's record and the run's check of it, though
    PeriodLeakRegistered's other premises stay open: the network certifier's verdict is accepted, and the figure and
    its cells are conditional on nothing."""
    for name, premise in UNNAMED.items():
        monkeypatch.setitem(OR.PREMISES, name, premise)
    lock.write_text(json.dumps(lock_doc({g: [] for g in OR.RESTS_ON})))
    rec = held_both(tmp_path)
    of = rec["bound"]["of_record"]
    want = [n for n in OR.PREMISES if n != OR.CERTIFIER_KEY and n not in OR.ENTRIES]
    assert UNPINNED == [] and OR.CERTIFIER_KEY not in UNNAMED and set(OR.PREMISES) == {*UNNAMED, OR.CERTIFIER_KEY}
    assert (of["held"], of["open"], of["K"]) == (["free", "circuit"], want, K)
    assert {"sampled-leg-bridge", "reg-opens-at", "tied-count", "commitment-binds", "list-within"} <= set(of["open"])
    assert all(p["missing"] == [] and p["discharged_by"] == [] for n, p in of["premises"]["premises"].items()
               if n in UNNAMED)
    run, week = rec["periods"]["run"], rec["periods"]["week"]["node"]
    assert (run["bits"], run["how"], run["open"]) == (K, "open", want)
    assert (run["per_phase"]["how"], run["per_phase"]["open"]) == ("open", want)
    assert (week["how"], week["rule"], week["open"]) == ("scaled", "run-to-period", want)
    assert "conditional_on" not in of and "conditional_on" not in rec["periods"] and of["tight"] == OR.TIGHT
    assert rec["verdicts"]["the network certifier"]["verdict"] == "accept"
    assert rec["checked"][OR.AUDITOR_RECORD]["holds"] is True
    assert of["premises"]["counts"] == OR.COUNTS and not set(OR.COUNTS) & set(of["open"])
    assert "periods_tight" not in rec
    assert all(f in OR.PREMISES["sampled-leg-bridge"]["is"]
               for f in ("SampledCodeRuns", "SessionKS,", "SessionRuns", "`list-within`", "`commitment-binds`"))
    assert "Python" in OR.DISCHARGE
    assert "no opening reaches the verifier" in OR.PREMISES["session-wires"]["is"]
    tags = OR.PREMISES["session-tags"]["is"]
    assert "circuitOf at Tags.circuit" in tags and "Tags.circuit.withCoins c" in tags
    padding = OR.PREMISES["tied-padding-reads"]["is"]
    assert all(x in padding for x in ("scan's circuit files whole", "to its header only", "never assumed",
                                      "every padding column", "never a difftest", "CircuitCountsAt"))
    assert "tied-padding-reads" not in json.dumps(OR.STANDING)
    assert all(x in OR.PREMISES["padding-out-of-free"]["is"] for x in ("refuses a tied audit whose units read into",
                                                                       "a column no unit reads"))
    wrong = OR.PREMISES["wrong-bounds"]["is"]
    assert all(x in wrong for x in ("RegMatches", "`sampled-leg-bridge`", "v.law is g.rule.lawOver n",
                                    "Draw.sameLaw", "jeq a (toJson n) → nat? a = some n"))
    assert all(x in OR.PREMISES[OR.CERTIFIER_KEY]["is"] for x in ("certifier-key/v1", "beside trusted-host",
                                                                 "`NOT_ESTABLISHED`", "`UNCHARGED`", "hA, hr and hkey"))
    assert all(x in OR.CHECKED[OR.AUDITOR_RECORD] for x in ("never the prover's node or its deployment.json",
                                                            "an obligation on the auditor's configuration"))
    cut_ = OR.PREMISES["entries-cut"]["is"]
    assert all(x in cut_ for x in ("proved and none sampled", "scanOf's refusal", "the state chain",
                                   "RowsKBoundsCount, in the lock, is its arithmetic alone", "Not yet stated"))
    assert all(x in OR.PREMISES["certifiers-open"]["is"] for x in ("to leave the statement rather than be discharged",
                                                                   "refused by the period check"))
    binds = OR.PREMISES["commitment-binds"]["is"]
    assert "Not what the guarantee says, and nothing it assumes of the prover" in binds and "binds_deployOf" in binds
    assert "What the guarantee says" in OR.COUNTS["client-decodes"] and "no prover's code" in OR.COUNTS["client-decodes"]


@pytest.mark.usefixtures("verified")
@pytest.mark.parametrize("given, open_", [
    ({}, []),
    ({HELD: ["NetTiming.Assumptions.SampledCodeRuns"]}, ["held-performed"]),
    ({HELD: ["Verity.Assumptions.TrustedHost"]}, []),
    ({COUNTED: None}, ["free-count"]),
    ({COVERED: None}, ["schedules-covered"]),
    ({COVERED: ["Verity.Assumptions.TrustedHost"]}, []),
    ({ACCEPTED: ["NetTiming.Assumptions.SampledCodeRuns"]}, ["NetTiming.Assumptions.SampledCodeRuns"]),
    ({ACCEPTED: ["Verity.Assumptions.TrustedHost"]}, []),
    ({ACCEPTED: ["Accounting.Covered"]}, []),
    ({ACCEPTED: ["NetTiming.TapOpens"]}, []),
    ({ACCEPTED: None}, [ACCEPTED])])
def test_the_figure_is_proved_only_where_every_premise_is_discharged_and_none_but_a_standing_assumption_taken(
        tmp_path, lock, given, open_):
    """Both checks held: a premise whose discharge the lock doesn't list, or lists taking an assumption that doesn't
    stand (SampledCodeRuns, a premise about our own Python), is open, and so is an assumption of a guarantee the
    figure rests on that is neither standing nor named in what it takes, by its name, and that guarantee where the lock
    doesn't list it. trusted-host and certifier-coverage stand, and a premise its takes names is that premise's: so
    schedules-covered is open by its name until the Lean audit's Property is in the lock, and discharged once it is,
    resting on trusted-host alone. The figure is the check's K either way, and its run's cell proved only with none
    open. With one check open too, both are open under the figure of record."""
    write_lock(lock, given)
    rec = held_both(tmp_path / "held")
    of = rec["bound"]["of_record"]
    assert (of["open"], of["K"], rec["periods"]["run"]["how"]) == (open_, K, "open" if open_ else "proved")
    assert of["premises"]["open"] == open_
    p = of["premises"]["premises"]
    assert p["held-performed"]["assumes"] == (["NetTiming.Assumptions.SampledCodeRuns"] if "held-performed" in open_
                                              else [])
    assert p["free-count"]["missing"] == ([COUNTED] if "free-count" in open_ else [])
    assert p["schedules-covered"]["missing"] == ([COVERED] if "schedules-covered" in open_ else [])
    rec = record(accepted(tmp_path / "free", dict(PUBLIC, held=["free"])))
    assert rec["bound"]["of_record"]["open"] == ["circuit", *open_] and "periods_tight" not in rec


#: COINS lines' sampled-proofs parts that name no live challenger, as served_verdicts.sh says them
DEVELOPMENT = {
    "served": (f"a local development challenger session on {NODE}, the served node, not an auditor's (PP_CHALLENGER "
               "local)"),
    "deferred": ("a local development challenger session on the follow-on's node, where step 6 defers V3c's audit, not "
                 "an auditor's (PP_CHALLENGER local)"),
    "follow-on": ("a local development challenger session on cpu-6, the follow-on's node, where it runs V3c's audit, "
                  "not an auditor's (PP_CHALLENGER local)"),
    "no peer": "the challenger gate:research@toy-gate (peer ?)",
    "not a gate": "the challenger vy-challenger-1 (peer toy-peer)",
}


def with_coins(tmp_path: Path, said: list[str]) -> dict:
    """The record, written by `main`, of a run with both checks held whose verdicts.txt opens with the lines
    ``said``."""
    out = accepted(tmp_path, dict(PUBLIC, held=["free", "circuit"]), coins=None)
    (out / "verdicts.txt").write_text("".join(x + "\n" for x in said) + (out / "verdicts.txt").read_text())
    assert main([str(out)]) == 0
    return json.loads((out / "record" / "onsite.json").read_text())


@pytest.mark.parametrize("said", [
    *([coins_line(s)] for s in DEVELOPMENT.values()),
    [f"COINS sampled proofs: {DEVELOPMENT['follow-on']}; the other protocols' as the served node's COINS line "
     "names them"],
    [coins_line(GATE), coins_line(DEVELOPMENT["follow-on"])],
    ["COINS the bandwidth audit's: vy-auditor-2's bw_party, inside PoUS's piece"],
    [],
])
@pytest.mark.usefixtures("verified")
def test_coins_no_line_names_live_leave_the_figure_of_record_and_each_cell_conditional_on_them(tmp_path, capsys, said):
    """A run whose COINS lines name a local development challenger session (the served node's, a deferral's or its
    follow-on's), a gate at no named peer, another challenger or no sampled proofs' coins, or that published none, has
    its coins open, live-coins by name: the sampled-proofs verdict, the run's being of record, the figure of record and
    each of its cells are conditional on them, and the figure of record is the check's K, open on them, its run's cell
    never proved, though both checks are held and every premise discharged."""
    rec, cond = with_coins(tmp_path, said), [OR.COINS_OPEN]
    parts = [x.partition("sampled proofs: ")[2].partition(";")[0] or None for x in said]
    assert rec["coins"] == {"open": True, "sampled_proofs": parts, "from": "verdicts.txt COINS", "is": OR.COINS,
                            "open_draws": ["sampled proofs' draw"], "draws": ALL_DRAWN} | (
        {} if said else {"missing": OR.NO_COINS})
    of = rec["bound"]["of_record"]
    assert rec["verdicts"]["sampled proofs"]["conditional_on"] == rec["of_record"]["conditional_on"] == cond
    assert rec["composite_error"]["conditional_on"] == cond
    assert (of["held"], of["open"], of["conditional_on"], of["K"]) == (["free", "circuit"], cond, cond, K)
    run = rec["periods"]["run"]
    assert rec["periods"]["conditional_on"] == cond and (run["bits"], run["how"], run["open"]) == (K, "open", cond)
    assert of["tight"] == OR.TIGHT and "periods_tight" not in rec
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["conditional_on"], line["K"], line["open"]) == (cond, K, cond)
    assert all(x in OR.COINS for x in ("sampled-proofs verdict", "E_w", "never as a live draw's", "`CHAIN_DRAWS`"))


@pytest.mark.usefixtures("verified")
@pytest.mark.parametrize("lead", ["", "Pearl+'s epoch coins are in-process and the record claims no Pearl+ verdict; "])
def test_coins_a_line_names_live_add_no_condition_and_the_figure_of_record_is_proved(tmp_path, capsys, lead):
    """A COINS line naming the live challenger (PP_CHALLENGER gate:USER@HOST at a named peer), whatever leads it, leaves
    the coins closed: nothing is conditional on them, and with both checks held and every premise discharged the
    figure of record is proved. A record written before the record read the coins is conditional on them in its line."""
    rec = with_coins(tmp_path, [coins_line(GATE, lead)])
    assert rec["coins"] == {"open": False, "sampled_proofs": [GATE], "from": "verdicts.txt COINS", "is": OR.COINS,
                            "open_draws": [], "draws": ALL_DRAWN}
    of = rec["bound"]["of_record"]
    assert not any("conditional_on" in x for x in (rec["verdicts"]["sampled proofs"], rec["of_record"], of,
                                                   rec["periods"], rec["composite_error"]))
    assert (of["K"], of["open"], rec["periods"]["run"]["how"]) == (K, [], "proved")
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["conditional_on"], line["K"], line["open"]) == ([], K, [])
    assert OR.summary({k: v for k, v in rec.items() if k != "coins"})["conditional_on"] == [OR.COINS_OPEN]


#: the run's ADVICE-LINE as served_verdicts.py says it on a serve-openai --advice run, declaring B for day {}
DECLARED = ("ADVICE-LINE: B (compiler digest 0123456789abcdef…, its advice toy, ℓ_a 32 bits) registered by the server "
            "once, before its first window, the registration log's builder and the compiler prove_path.py recomputes "
            "from serve.json, and declared day {}'s one program; every closed window's advice committed: PASS")
NO_B = ("ADVICE-LINE: FAIL: no advice_registration.json: the server registered no B (serve-openai without "
        "--advice)")


@pytest.mark.parametrize("days", [["0"], ["3", "3"], [None], []])
def test_one_program_a_day_is_never_claimed_but_open_by_name_beside_the_day_each_advice_line_says(tmp_path, capsys,
                                                                                                    days):
    """Whatever the run's ADVICE-LINEs say (a day's declaration of B, two of them, a line naming no day) or where it
    published none (missing), the record doesn't claim one program a day: it is open by name in the record and its
    line, on both counts (R2's day isn't the challenger's signed time, and the audit reads the node's copy of the
    registration log, not the challenger's signed one), saying the day's registration and its refusals still run. The
    figures don't read it: with the coins live and every premise discharged the figure of record is proved."""
    rec = with_coins(tmp_path, [coins_line(GATE), *(NO_B if d is None else DECLARED.format(d) for d in days)])
    assert rec["one_program_a_day"] == {"open": True, "advice_line_day": days, "from": "verdicts.txt ADVICE-LINE",
                                        "is": OR.DAY} | ({} if days else {"missing": OR.NO_ADVICE_LINE})
    assert (rec["bound"]["of_record"]["K"], rec["bound"]["of_record"]["open"]) == (K, [])
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["not_claimed"], line["conditional_on"], line["K"]) == ([OR.DAY_OPEN, OR.POUW_OPEN], [], K)
    assert all(x in OR.DAY for x in ("doesn't claim", "R2's day isn't yet the challenger's signed time",
                                     "doesn't yet read the challenger's signed registration log",
                                     "(duplicate-declaration)", "(R2-declared)", "which no verdict here does"))


def test_a_discharge_the_properties_lock_lists_counts_as_one_the_guarantees_lock_does(tmp_path, lock, monkeypatch):
    """The Properties' lock holds the theorems that discharge premises at the deployed code and PeriodLeakRegistered,
    which states certifier-key, the guarantees' lock the other guarantees the figure rests on: the record reads both,
    and with the Properties' lock left out each premise is open, and so is PeriodLeakRegistered."""
    props_lists = [t for p in OR.PREMISES.values() for t in p["discharged_by"]] + [OR.PERIOD_REGISTERED]
    write_lock(lock, dict.fromkeys(props_lists))
    props = tmp_path / "properties-lean-audit.json"
    props.write_text(json.dumps(lock_doc({t: [] for t in props_lists})))
    monkeypatch.setattr(OR, "LOCKS", (lock, props))
    assert held_both(tmp_path / "both")["bound"]["of_record"]["open"] == []
    monkeypatch.setattr(OR, "LOCKS", (lock,))
    assert held_both(tmp_path / "one")["bound"]["of_record"]["open"] == [
        *(n for n in OR.PREMISES if n not in OR.ENTRIES), OR.PERIOD_REGISTERED]


@pytest.mark.parametrize("text, error", [(None, "FileNotFoundError"), ("{", "JSONDecodeError"), ("{}", "KeyError")])
def test_a_lock_that_cant_be_read_lists_nothing_and_is_named_so_every_premise_it_would_discharge_is_open(
        tmp_path, lock, monkeypatch, text, error):
    """The Properties' lock missing, not JSON, or with no guarantees: the record states it under ``unread``, by its path
    in the tree and with why, and reads the other lock, so only the premise whose discharge that lock alone would list
    is open under the figure of record, never a crash."""
    write_lock(lock, {COUNTED: None})
    props = tmp_path / "tree" / "Security" / "Properties" / "lean-audit.json"
    props.parent.mkdir(parents=True)
    if text is not None:
        props.write_text(text)
    monkeypatch.setattr(OR, "LOCKS", (lock, props))
    of = held_both(tmp_path / "out")["bound"]["of_record"]
    assert (of["open"], of["K"]) == (["free-count"], K)
    (why,) = of["premises"]["unread"]
    assert why.startswith(f"{props}: {error}: ")
    assert of["state_charge"]["unread"] == [f"{props}: {error}"]
    (why,) = OR.premises((lock, props), root=tmp_path / "tree")["unread"]
    assert why.startswith(f"Security/Properties/lean-audit.json: {error}: ") and str(tmp_path) not in why


def test_an_assumption_a_premise_stands_for_is_open_only_while_that_premise_is(tmp_path, lock, monkeypatch):
    """A guarantee the figure rests on takes the wires premise as a named hypothesis, which the lock lists: named in
    what it takes, by the wires premise, nothing is open while that is discharged; with one of its discharges missing,
    the premise is open, by its own name alone; not named there, the assumption is open by its own name."""
    takes = dict(OR.RESTS_ON[ACCEPTED]["takes"], **{"NetTiming.WiresFit": "wires-fit"})
    monkeypatch.setitem(OR.RESTS_ON, ACCEPTED, dict(OR.RESTS_ON[ACCEPTED], takes=takes))
    write_lock(lock, {ACCEPTED: ["NetTiming.WiresFit"]})
    assert held_both(tmp_path / "a")["bound"]["of_record"]["open"] == []
    write_lock(lock, {ACCEPTED: ["NetTiming.WiresFit"], WIRED: None})
    assert held_both(tmp_path / "b")["bound"]["of_record"]["open"] == ["wires-fit"]
    monkeypatch.setitem(OR.RESTS_ON, ACCEPTED, RESTS_ON[ACCEPTED])
    write_lock(lock, {ACCEPTED: ["NetTiming.WiresFit"]})
    assert held_both(tmp_path / "c")["bound"]["of_record"]["open"] == ["NetTiming.WiresFit"]


def test_a_discharge_counts_only_at_the_statement_its_pin_names(tmp_path, lock, monkeypatch):
    """The lock records no antecedent, and a Property may take a deployment's premise inline (`ScanScopeAcc` takes
    `StmtFlattens` and `ScopeMatches`): a discharge the lock lists but `DISCHARGES` doesn't pin, or pins at another
    statement, leaves its premise open with the lock's digest to read it at, and pinned there it discharges it."""
    stated = "ab" * 32
    monkeypatch.delitem(OR.DISCHARGES, COUNTED)
    write_lock(lock, stated={COUNTED: stated})
    of = held_both(tmp_path / "unpinned")["bound"]["of_record"]
    assert (of["open"], of["K"], of["premises"]["premises"]["free-count"]["unpinned"]) == (
        ["free-count"], K, {COUNTED: stated})
    monkeypatch.setitem(OR.DISCHARGES, COUNTED, {"statement": "ff" * 32, "takes": {}})
    assert held_both(tmp_path / "restated")["bound"]["of_record"]["open"] == ["free-count"]
    monkeypatch.setitem(OR.DISCHARGES, COUNTED, {"statement": stated, "takes": {}})
    assert held_both(tmp_path / "pinned")["bound"]["of_record"]["open"] == []


def test_a_discharge_that_takes_a_premise_discharges_only_once_that_premise_is_and_two_never_each_other(
        tmp_path, lock, monkeypatch):
    """The wires premise's discharge takes the deployment's scope check, a named assumption of its, by the sampled
    leg's bridge: nothing is open while that premise is discharged, both are while it isn't, and two premises whose
    discharges take each other stay open, each discharging nothing."""
    wired, bridge = WIRED, TOY["sampled-leg-bridge"]
    scope = "Properties.NetworkScan.ScopeMatches"
    monkeypatch.setitem(OR.DISCHARGES, wired, dict(OR.DISCHARGES[wired], takes={scope: "sampled-leg-bridge"}))
    write_lock(lock, {wired: [scope]})
    of = held_both(tmp_path / "a")["bound"]["of_record"]
    assert (of["open"], of["premises"]["premises"]["wires-fit"]["takes"]) == ([], ["sampled-leg-bridge"])
    write_lock(lock, {wired: [scope], bridge: None})
    assert held_both(tmp_path / "b")["bound"]["of_record"]["open"] == ["wires-fit", "sampled-leg-bridge"]
    monkeypatch.setitem(OR.DISCHARGES, bridge, dict(OR.DISCHARGES[bridge], takes={"hw": "wires-fit"}))
    write_lock(lock, {wired: [scope]})
    of = held_both(tmp_path / "c")["bound"]["of_record"]
    assert (of["open"], of["K"]) == (["wires-fit", "sampled-leg-bridge"], K)
    assert of["premises"]["premises"]["sampled-leg-bridge"]["assumes"] == []


@pytest.mark.parametrize("stated", ["ff" * 32, None])
def test_a_guarantee_the_figure_rests_on_is_open_by_name_where_the_lock_states_it_otherwise(tmp_path, lock, stated):
    """The lock records a guarantee's named assumptions, never its antecedents, so a statement other than the one whose
    hypotheses and antecedents `RESTS_ON` names (another digest among the lock's reads, or none) may take one the
    record doesn't name: that guarantee is open, by its name, with why, until it is read again and pinned."""
    write_lock(lock, stated={ACCEPTED: stated})
    rec = held_both(tmp_path)
    of = rec["bound"]["of_record"]
    assert (of["open"], of["K"]) == ([ACCEPTED], K)
    assert of["premises"]["rests_on"][ACCEPTED] == {"assumptions": [], "statement": stated, "restated": OR.RESTATED,
                                                    "takes": OR.RESTS_ON[ACCEPTED]["takes"]}
    assert of["tight"] == OR.TIGHT and "conditional_on" not in of


def test_a_guarantee_the_figure_rests_on_with_no_statement_pinned_is_open_even_where_the_lock_lists_it(
        tmp_path, lock, monkeypatch):
    """A guarantee named before what it takes was read at its statement: open by its name, whatever the lock states it
    as."""
    monkeypatch.setitem(OR.RESTS_ON, ACCEPTED, dict(OR.RESTS_ON[ACCEPTED], statement=None))
    write_lock(lock, stated={ACCEPTED: "toy"})
    of = held_both(tmp_path)["bound"]["of_record"]
    assert of["open"] == [ACCEPTED] and of["premises"]["rests_on"][ACCEPTED]["restated"] == OR.RESTATED


@pytest.mark.usefixtures("verified")
def test_no_guarantee_counts_a_phases_tied_audits_so_that_count_is_a_premise_of_a_tied_charge_alone(
        tmp_path, monkeypatch):
    """The record rests on no guarantee that counts a phase's tied audits with its scan: none is in `RESTS_ON`, and
    what one would give is the premise tied-count, with no discharge named. A phase charged its tied audits is open
    on it, by name, with every check held and every other premise discharged. A phase charged its scan's slot entries
    once doesn't rest on it, or on any premise or check of the tied charge: it rests on entries-cut, open by name till
    its statement is in the lock, and on the premises of both charges."""
    for name in ("tied-count", "entries-cut"):
        monkeypatch.setitem(OR.PREMISES, name, UNNAMED[name])
    assert not any("TiedCount" in g for g in (*OR.RESTS_ON, *OR.DISCHARGES))
    assert UNNAMED["tied-count"]["discharged_by"] == UNNAMED["entries-cut"]["discharged_by"] == ()
    assert ("tied-count" in OR.TIE, "entries-cut" in OR.ENTRIES) == (True, True)
    of = held_both(tmp_path / "tied")["bound"]["of_record"]
    assert (of["tie_charged"], of["open"], of["K"]) == ([0], ["tied-count"], K)
    assert "entries-cut" not in of["premises"]["premises"] and set(OR.TIE) <= set(of["premises"]["premises"])
    rec = record(accepted(tmp_path / "once", ONCE))
    of = rec["bound"]["of_record"]
    assert (of["tie_charged"], of["held"], of["open"], of["K"]) == ([], [], ["entries-cut"], K)
    assert not set(OR.TIE) & set(of["premises"]["premises"]) and "entries-cut" in of["premises"]["premises"]
    assert set(of["premises"]["premises"]) == set(OR.PREMISES) - set(OR.TIE)
    assert (rec["periods"]["run"]["how"], rec["periods"]["run"]["open"]) == ("open", ["entries-cut"])


def test_the_record_names_no_guarantee_or_discharge_the_trees_locks_dont_list(monkeypatch, tmp_path):
    """Each guarantee the figure rests on (`RESTS_ON`), each theorem a premise names as its discharge or as stating it
    (`PREMISES`), each pinned discharge (`DISCHARGES`) and RowsStateHeld (`STATE_HELD`) is a guarantee one of the
    tree's locks lists, so the Lean check replays it: a name of a theorem in no Lean file, or of one stated and not
    yet a guarantee, fails here with what to do about it. A premise whose discharge isn't in the lock names none and
    is open."""
    monkeypatch.undo()
    got = OR.unlisted(OR.LOCKS)
    assert not got, "\n".join(f"onsite_record.{where} names {g}: {OR.UNLISTED}" for g, where in got.items())
    assert all(x in OR.UNLISTED for x in ("take the name out", "leaving its premise open", "check.py --update"))
    absent = "Properties.NetworkPeriod.phaseStateHeld"
    monkeypatch.setitem(OR.PREMISES, "rows-state-held", dict(OR.PREMISES["rows-state-held"], discharged_by=(absent,)))
    monkeypatch.setitem(OR.RESTS_ON, "NetTiming.SecurityProofs.PeriodTiedCount", {"statement": None, "takes": {}})
    monkeypatch.setitem(OR.DISCHARGES, "Properties.NetworkCertifier.circuitCounts", {"statement": None, "takes": {}})
    assert OR.unlisted(OR.LOCKS) == {"NetTiming.SecurityProofs.PeriodTiedCount": "RESTS_ON",
                                     "Properties.NetworkCertifier.circuitCounts": "DISCHARGES",
                                     absent: "PREMISES['rows-state-held']'s discharged_by"}
    empty = tmp_path / "lean-audit.json"
    empty.write_text(json.dumps({"guarantees": {}}))
    assert {OR.PERIOD_REGISTERED, ACCEPTED, OR.STATE_HELD} <= set(OR.unlisted((empty,)))
    assert OR.unlisted((empty,))[OR.PERIOD_REGISTERED] == f"PREMISES[{OR.CERTIFIER_KEY!r}]'s stated_by"


def test_each_guarantee_the_figure_rests_on_is_named_by_what_it_takes_at_its_statement_in_security_lock(monkeypatch):
    """Security's lock records no guarantee's antecedents, so `RESTS_ON` names each hypothesis and antecedent of each
    statement by hand, read at the statement's digest among the lock's reads: each guarantee of `RESTS_ON` the tree's
    lock lists is there at that digest, with no named assumption it doesn't name or that doesn't stand, and each Prop it
    takes is a premise's or standing. A hypothesis or antecedent a statement gains changes its digest, and fails here.
    So does a discharge `DISCHARGES` pins, in either lock, and it pins only theorems `PREMISES` names."""
    monkeypatch.undo()
    locks = {f.relative_to(OR.ROOT): json.loads(f.read_text()) for f in OR.LOCKS}
    assert ACCEPTED in locks[Path("Security/lean-audit.json")]["guarantees"]
    assert set(OR.DISCHARGES) <= {t for p in OR.PREMISES.values() for t in p["discharged_by"]}
    for g, r in (OR.RESTS_ON | OR.DISCHARGES).items():
        pins = "RESTS_ON" if g in OR.RESTS_ON else "DISCHARGES"
        assert set(r["takes"].values()) <= set(OR.PREMISES) | set(OR.STANDING) | set(OR.COUNTS) | set(OR.CHECKED), g
        for name, lock in locks.items():
            if g not in lock["guarantees"]:
                continue
            got = OR.statement(lock, g)
            assert got is not None and got == r["statement"], (
                f"{g}'s statement in {name} is {got}, not {r['statement']}, the one onsite_record.{pins} names what it "
                f"takes at: read its hypotheses, antecedents and the conditions its good event reads again, name each in "
                f"its takes with the premise (PREMISES), standing assumption (STANDING), term of what K counts (COUNTS) "
                f"or obligation the run checks (CHECKED) that covers it, and pin {got}")
            assert set(lock["guarantees"][g]["assumptions"]) <= set(r["takes"]) | set(OR.STANDING), g


def test_at_the_trees_lock_certifier_key_closes_where_the_runs_check_of_the_auditors_record_holds_and_only_there(
        monkeypatch):
    """The tree's Properties lock holds PeriodLeakRegistered at the statement `RESTS_ON` pins, which takes hA, hr and
    hkey, the auditor's record, as auditor-record: certifier-key closes where the run's check of it holds, and stays
    open, with why, where it doesn't or no run's inputs were checked. sampled-leg-bridge and reg-opens-at, which that
    statement takes too, stay open on their own."""
    monkeypatch.undo()
    r = OR.RESTS_ON[OR.PERIOD_REGISTERED]
    assert {k for k, v in r["takes"].items() if v == OR.AUDITOR_RECORD} == {"hA", "hr", "hkey"}
    ok = OR.premises(OR.LOCKS, checked={OR.AUDITOR_RECORD: {"holds": True}})
    assert ok["unread"] == [] and ok["rests_on"][OR.PERIOD_REGISTERED]["statement"] == r["statement"]
    assert not {"missing", "restated"} & set(ok["rests_on"][OR.PERIOD_REGISTERED])
    assert OR.CERTIFIER_KEY not in ok["open"] and {"sampled-leg-bridge", "reg-opens-at"} <= set(ok["open"])
    assert "unchecked" not in ok["premises"][OR.CERTIFIER_KEY]
    for checked, why in (({OR.AUDITOR_RECORD: {"holds": False, "why": "toy"}}, "toy"), (None, OR.NOT_CHECKED)):
        got = OR.premises(OR.LOCKS, checked=checked)
        assert OR.CERTIFIER_KEY in got["open"] and got["premises"][OR.CERTIFIER_KEY]["unchecked"] == why


def test_each_audit_is_stated_the_runs_cap_and_no_phase_is_charged_it_whichever_audits_its_scan_names(tmp_path):
    """Two audits, each stated the run's cap: whether the public names no scan's tied audits or each phase names its
    own, the figure of record is the check's K, no phase charged a cap, and the tied charge's audits are the ones the
    scans name, every audit where they name none."""
    two = dict(PUBLIC, K=PUBLIC["K"] + 9088 + 63, eps="1/524288", windows=2, scans=[SCAN, SCAN])
    rec = record(accepted(tmp_path / "all", two, windows=2, audits=(NO_ADVICE, NO_ADVICE)))
    of = rec["bound"]["of_record"]
    assert rec["advice_charge"]["cap"] == {"ell_a": [2 ** 23, 2 ** 23], "is": OR.CAP}
    assert (of["K"], of["advice"], of["tie_charged"], of["state_charge"]["tied"]) == (
        two["K"], {"is": OR.ADVICE_LINK}, [0, 1], [0, 1])
    tied = dict(two, scans=[dict(SCAN, tied=[1]), dict(SCAN, tied=[1])])
    of = record(accepted(tmp_path / "tied", tied, windows=2, audits=(NO_ADVICE, NO_ADVICE)))["bound"]["of_record"]
    assert (of["K"], of["tie_charged"], of["state_charge"]["tied"]) == (two["K"], [0, 1], [1])


@pytest.mark.parametrize("binds", [{"advice": {"bits": 2 ** 23, "root": "ef" * 64}},
                                   {"compiler": {"digest": "01" * 64, "name": "B"}}])
def test_an_audit_whose_registration_binds_an_advice_or_a_compiler_leaves_no_figure_of_record(tmp_path, binds):
    """An audit's own advice is in no term of K and isn't the run's cap: the period takes only a circuit the verifier
    registered (`records_audit.py period` refuses the rest), so such an audit is the figure of record's error."""
    bound = {**NO_ADVICE, "window": {**NO_ADVICE["window"], **binds}}
    rec = record(accepted(tmp_path, audits=(NO_ADVICE, bound)))
    assert rec["bound"]["of_record"] == rec["advice_charge"]["cap"] == {
        "error": f"{OR.CASE} commitment: ValueError: the registrations of audits [1] bind an advice or a compiler, whose "
                 "bits no term of K charges (records_audit.py period refuses them)"}
    assert rec["periods"]["missing"].startswith("no figure of record") and rec["bound"]["K"] == K


#: stand-ins for the digests `STATE_PINS` pins: RowsStateHeld's statement and the definitions it reads
STATE_TOY = {g: f"toy-{g}" for g in OR.STATE_PINS}


def with_state_held(lock: Path, digests: dict[str, str | None], reader: str = OR.STATE_HELD) -> None:
    """``lock`` listing RowsStateHeld too (`OR.STATE_HELD`), its statement at ``digests``' and the definitions
    ``reader`` reads at theirs, each unlisted where None."""
    doc = json.loads(lock.read_text())
    doc["guarantees"][OR.STATE_HELD] = {"assumptions": [], "signature": [f"{OR.STATE_HELD} : Toy.RowsStateHeld"]}
    doc["reads"]["Toy.Stated"] = {"definitions": {"Toy.RowsStateHeld": digests[OR.STATE_HELD]}}
    doc["reads"]["Toy.Period"] = {"guarantees": [reader], "definitions": {
        d: h for d, h in digests.items() if d != OR.STATE_HELD and h is not None}}
    lock.write_text(json.dumps(doc))


@pytest.mark.parametrize("how, unpinned", [
    ("its proof stands over the activations entry", {}),
    ("not in the lock", {g: None for g in STATE_TOY}),
    ("restated", {OR.STATE_HELD: "ab" * 32}),
    ("over the registered entries alone", {"NetworkCheck.Period.auditStateEntries": None}),
    ("over another stateHeldIn", {"NetworkCheck.Period.stateHeldIn": "cd" * 32}),
    ("over another stateEntries", {"NetworkCheck.Period.stateEntries": "ef" * 32}),
    ("its definitions another guarantee's", {g: None for g in STATE_TOY if g != OR.STATE_HELD}),
])
def test_the_figure_says_its_charge_was_state_free_only_where_rows_state_held_stands_over_the_activations_entry(
        tmp_path, lock, monkeypatch, how, unpinned):
    """The rows form charges a tied audit its stateFree where its reads of the scan's state are the scan's own, over the
    entries auditStateEntries reads, its window's activations entry among them. The figure of record says that charge
    applied only where the lock holds RowsStateHeld at its pinned statement reading stateHeldIn, auditStateEntries and
    the stateEntries both call at theirs, and free otherwise, naming each digest the lock gives that isn't pinned: not
    in the lock, restated, read over the registered entries alone (no auditStateEntries), another stateHeldIn or another
    stateEntries, or those definitions read by another guarantee. An audit with no circuit record has stateFree its
    free, and the figure stands either way."""
    monkeypatch.setattr(OR, "STATE_PINS", STATE_TOY)
    if how != "not in the lock":
        with_state_held(lock, STATE_TOY | unpinned, reader="Toy.other" if how.endswith("another guarantee's") else
                        OR.STATE_HELD)
    of = record(accepted(tmp_path))["bound"]["of_record"]
    assert of["state_charge"] == {"charge": "free" if unpinned else "stateFree", "is": OR.STATE_CHARGE, "tied": [0],
                                  "with_record": []} | ({"unpinned": unpinned} if unpinned else {})
    assert of["K"] == K


#: the period's one audit with its circuit record, the form whose stateFree may be less than its free
RECORDED = {"audits": [{"circuit": "c0" * 64, "order": {}, "pins": {}, "program": {}, "steps": 0}]}


@pytest.mark.usefixtures("verified")
@pytest.mark.parametrize("stands", [True, False])
def test_a_tied_audit_with_a_circuit_record_has_a_figure_of_record_only_where_its_state_free_charge_stands(
        tmp_path, lock, monkeypatch, stands):
    """A tied audit whose parameters state its circuit record may be charged its stateFree, less than its free. Where
    RowsStateHeld stands over the activations entry, that charge is the figure's; where it doesn't, the public doesn't
    separate the two, so there is no figure of record and no period cells, the bound's K still stated. A phase charged
    its scan's slot entries once charges no tied audit either way, and has its figure."""
    monkeypatch.setattr(OR, "STATE_PINS", STATE_TOY)
    if stands:
        with_state_held(lock, STATE_TOY)
    rec = record(accepted(tmp_path, params=RECORDED))
    of = rec["bound"]["of_record"]
    charge = {"charge": "stateFree" if stands else "free", "is": OR.STATE_CHARGE, "tied": [0], "with_record": [0]} | (
        {} if stands else {"unpinned": {g: None for g in STATE_TOY}, "missing": OR.STATE_UNCOVERED})
    assert of["state_charge"] == charge and rec["bound"]["K"] == K
    if stands:
        assert of["K"] == K and "missing" not in rec["periods"]
        return
    assert of == {"missing": OR.STATE_UNCOVERED, "state_charge": charge}
    assert rec["periods"] == {"missing": f"no figure of record: {OR.STATE_UNCOVERED}"}
    once = record(accepted(tmp_path / "once", ONCE, params=RECORDED))["bound"]["of_record"]
    assert (once["K"], once["state_charge"]) == (K, {"charge": "entries", "is": OR.STATE_CHARGE, "tied": [],
                                                     "with_record": []})


def test_at_the_trees_locks_the_charge_is_state_free(monkeypatch):
    """`STATE_PINS` are the digests the tree's Properties lock records for RowsStateHeld's statement and the stateHeldIn,
    auditStateEntries and stateEntries it reads: a change to any of them fails here, rather than every figure's charge
    turning free unremarked."""
    monkeypatch.undo()
    got = OR.state_charge({"params": {}}, {"scans": [{"tied": []}]}, 0, OR.LOCKS)
    assert got["charge"] == "stateFree", f"the tree's locks give {got.get('unpinned')}, {got.get('unread')}"
    assert OR.state_charge({"params": {}}, {"scans": []}, 0, OR.LOCKS)["charge"] == "entries"


def test_the_tied_audits_the_charge_reads_are_the_ones_each_scan_names(tmp_path):
    """With scans[j].tied, the charge reads the audits each phase names, and only those with a circuit record can
    differ."""
    two = dict(PUBLIC, K=PUBLIC["K"] + 9088 + 63, eps="1/524288", windows=2,
               scans=[dict(SCAN, tied=[1]), dict(SCAN, tied=[1])])
    params = {"audits": [RECORDED["audits"][0], {"program": {}, "steps": 0}]}
    of = record(accepted(tmp_path, two, windows=2, audits=(NO_ADVICE, NO_ADVICE), params=params))["bound"]["of_record"]
    assert (of["state_charge"]["tied"], of["state_charge"]["with_record"], "missing" in of) == ([1], [], False)


@pytest.mark.usefixtures("verified")
def test_a_run_that_published_no_advice_registration_or_one_unpadded_states_no_cap_and_keeps_its_figure(tmp_path):
    """ℓ_a is the cap of the run's published advice registration (`REGISTRATION`), never 0 for want of one: with none
    the cap is missing, and one whose advice isn't padded to its compiler's cap is its error. No figure charges the
    cap, so the figure of record and its periods stand either way."""
    rec = record(accepted(tmp_path / "none", registration=None))
    assert rec["advice_charge"]["cap"] == {"missing": OR.NO_REGISTRATION}
    assert rec["bound"]["of_record"]["K"] == rec["periods"]["run"]["bits"] == rec["bound"]["K"] == K
    unpadded = {**PUBLISHED, "binding": {"advice": {"root": "ab" * 64, "bits": 392}}}
    rec = record(accepted(tmp_path / "unpadded", registration=unpadded))
    assert rec["advice_charge"]["cap"] == {
        "error": f"{OR.REGISTRATION}: ValueError: its advice's length 392 isn't padded to its compiler's cap {2 ** 23}"}
    assert rec["bound"]["of_record"]["K"] == rec["periods"]["run"]["bits"] == K


@pytest.mark.usefixtures("verified")
@pytest.mark.parametrize("registration, cap", [
    ({k: v for k, v in PUBLISHED.items() if k != "compiler"}, None), ({**PUBLISHED, "compiler": {"name": "c"}}, None),
    ({**PUBLISHED, "compiler": {"cap_bits": 0}}, 0), ({**PUBLISHED, "compiler": {"cap_bits": -8}}, -8),
    ({**PUBLISHED, "compiler": {"cap_bits": True}}, True), ({**PUBLISHED, "compiler": {"cap_bits": "8388608"}}, "8388608"),
    ({**PUBLISHED, "compiler": {"cap_bits": 8.0}}, 8.0), ([PUBLISHED], None)])
def test_a_published_advice_registration_stating_no_cap_of_at_least_1_bit_is_the_caps_error_by_name(
        tmp_path, registration, cap):
    """No compiler, or a compiler whose cap_bits is missing, 0, negative, or not a whole number, states no ℓ_a: the
    record gives the cap's error by name, never an ℓ_a of 0. No figure charges it, so the figure of record stands."""
    rec = record(accepted(tmp_path, registration=registration))
    why = (f"{OR.REGISTRATION}: ValueError: its compiler states no cap_bits of at least 1 bit ({cap!r}), the ℓ_a the "
           "record states beside the bound, so it states none rather than 0")
    assert rec["advice_charge"]["cap"] == {"error": why}
    assert rec["bound"]["of_record"]["K"] == rec["periods"]["run"]["bits"] == rec["bound"]["K"] == K


def test_a_commitment_the_record_cant_read_leaves_no_figure_of_record_and_no_bound(tmp_path):
    """The case's commitment holds the period's audits. Where the record can't read it, the figure of record is its
    error (the audits it would charge), and scan-state-pinned can't say whether an audit's window commits the scan's state, so it
    doesn't hold: the network certifier's verdict is not established and there is no bound."""
    out = accepted(tmp_path)
    case = dict(json.loads((out / OR.CASE).read_text()), commitment="00")
    (out / OR.CASE).write_text(json.dumps(case))
    fig, cap = OR.figure(out, case, PUBLIC, K, {"open": []})
    assert fig == cap and fig["error"].startswith(f"{OR.CASE} commitment: JSONDecodeError")
    rec = record(out)
    assert rec["checked"][OR.SCAN_STATE_PINNED] == {
        "is": OR.CHECKED[OR.SCAN_STATE_PINNED], "holds": False,
        "why": f"{OR.CASE}'s commitment doesn't list the period's audits: JSONDecodeError"}
    assert rec["verdicts"]["the network certifier"]["open"] == [OR.SCAN_STATE_PINNED]
    assert "missing" in rec["bound"] and rec["periods"] == {"missing": "no bound"} and "periods_tight" not in rec


@pytest.mark.usefixtures("verified")
def test_w63k4_at_a_4_s_tick_charges_800_bits_a_phase_beside_its_advice_whatever_the_traffic(tmp_path):
    """Three execution phases of w63k4's schedule at a 4 s tick: its 4 lanes over T = 32 buckets carry at most 128
    connections a phase, each capped at 5 egress cells, so k_sched and k_fail charge 128 · ⌈log2 6⌉ = 384 bits a phase
    each, the period's check's terms however few connections were served; with the jitter T_j log2 d_jit = 32 at
    d_jit = 2, a phase's terms but ℓ_{a,j} and k_{skip,j} are 800 bits."""
    sys.path.insert(0, str(HERE.parent.parent / "network_traces"))
    import scan_zk as SZ
    schedule = dataclasses.asdict(SZ.W63K4.params) | {"bucket_ms": 4000}
    scan = dict(SCAN, jitter=schedule["window"])
    public = dict(PUBLIC, K=3 * (768 + 9088 + 32), cells=3 * 384, schedules=3 * 384, windows=3, scans=[scan] * 3,
                  eps="3/1048576")
    rec = record(accepted(tmp_path, public, windows=3, schedule=schedule))
    b = rec["bound"]
    assert [t["bits"] for t in b["terms"] if t.get("phase") is None] == [1152, 1152]
    jitter = [t["bits"] for t in b["terms"] if t["symbol"] == "T_j \\log_2 d_{jit}"]
    assert jitter == [32] * 3 and (1152 + 1152) // 3 + jitter[0] == 800
    assert [t.get("times") for t in b["delta_net"]["terms"]] == [None, 3, 3, 3]
    assert rec["periods"]["run"]["delta_net"] == OR.delta_net(Fraction(3, 2 ** 20), 3)
    assert b["connections"] == {"count": 3 * 128, "from": OR.CONNECTIONS}
    assert (rec["periods"]["phase_s"], rec["periods"]["run"]["seconds"], b["period_s"]) == ("128", "384", "384")
    assert rec["periods"]["week"]["node"]["parameters"]["connections"] == str(128 * math.ceil(7 * 86400 / 128))
    assert rec["periods"]["egress"]["capacity_bits_per_phase"] == 4 * 4 * 32 * 512 * 8


@pytest.mark.usefixtures("verified")
@pytest.mark.parametrize("once", [True, False])
def test_w63k4s_slot_entries_are_charged_once_under_its_scan_proved_whole_and_else_in_k_skip(tmp_path, once):
    """w63k4's schedule with a toy tied audit: w63k4 at a 4 s tick over three execution phases, each phase's scan with
    every proof block drawn, its tied audit one of 16 proof blocks (V3c's registration has millions) drawn subset:1.
    The period check charges each phase its scan's E_w = 896 slot-entry bits once, as ℓ_{a,j}'s ``entries``, with
    k_{skip,j} 0 and no audit read at δ_i: 1,696 bits a phase with the schedule's and the jitter's 800, the figure of
    record, of the schedule's egress capacity of 4 lanes · 4 slots · 32 buckets of 512-byte cells a phase; eps is each
    phase's scan's 2^-64. It rests on no check of the tied charge, so it is proved with every premise discharged
    whatever the public holds. A period check that charges none once states the same 896 bits in k_{skip,j}, the tied
    audit charged, its draw read at δ_i = 2^-64 with a bound of 15 beside its answer's 8 at 1/2: the same 1,696 a
    phase, on the tied charge's checks and premises, open on a check the verdict didn't hold. Neither adds the run's
    advice cap."""
    sys.path.insert(0, str(HERE.parent.parent / "network_traces"))
    import scan_zk as SZ
    schedule = dataclasses.asdict(SZ.W63K4.params) | {"bucket_ms": 4000}
    scan = dict(SCAN, delta=f"1/{2 ** 64}", free=0, jitter=32) | ({"entries": 896} if once else {"faulty_entries": 896})
    audit = {"delta": "1/2", "drawn": 1, "units": 16, "wrong_units": 8}
    charged = [] if once else [dict(audit, delta=f"1/{2 ** 64}", wrong_units=15)] * 3
    public = dict(PUBLIC, K=3 * 1696, cells=3 * 384, schedules=3 * 384, windows=3,
                  scans=[dict(scan, tied=[j]) for j in range(3)], audits=[audit] * 3, charged=charged,
                  eps=str(Fraction(3 + len(charged), 2 ** 64)), held=["free", "circuit"])
    rec = record(accepted(tmp_path, public, windows=3, schedule=schedule, audits=(NO_ADVICE,) * 3))
    b, run = rec["bound"], rec["periods"]["run"]
    in_skip = 0 if once else 896
    assert b["skip"] == {"is": OR.SKIP, "bits_per_phase": [in_skip] * 3, "entries_per_phase": [in_skip] * 3}
    assert rec["advice_charge"]["charged"]["bits_per_phase"] == b["entries_per_phase"] == [896 - in_skip] * 3
    assert [t["lean"] for t in b["terms"] if t["symbol"] == "\\ell_{a,j}"] == [
        [f"scans[{j}].free"] + [f"scans[{j}].entries"] * once for j in range(3)]
    assert b["delta"]["charged"] == str(Fraction(len(charged), 2 ** 64)) and b["delta"]["charged_audits"] == charged
    of = b["of_record"]
    assert (of["K"], of["open"], of["tie_charged"]) == (3 * 1696, [], [] if once else [0, 1, 2])
    assert (of["state_charge"]["charge"], of["state_charge"]["tied"]) == (
        ("entries", []) if once else ("free", [0, 1, 2]))
    assert ("entries-cut" in of["premises"]["premises"], "tied-count" in of["premises"]["premises"]) == (once, not once)
    capacity = 4 * 4 * 32 * 512 * 8
    assert (run["per_phase"]["bits"], run["how"]) == (1696, "proved") and run["share_of_capacity"] == 1696 / capacity
    assert rec["periods"]["year"]["datacenter"]["share_of_capacity"] == 1696 / capacity
    assert rec["periods"]["egress"]["capacity_bits_per_phase"] == capacity == 2 ** 21
    opened = record(accepted(tmp_path / "open", dict(public, held=["free"]), windows=3, schedule=schedule,
                             audits=(NO_ADVICE,) * 3))
    assert opened["bound"]["of_record"]["open"] == ([] if once else ["circuit"])
    assert (opened["periods"]["run"]["per_phase"]["bits"], opened["periods"]["run"]["how"]) == (
        1696, "proved" if once else "open")
    assert not {2 ** 23 + 1, 1696 + 2 ** 23 + 1} & set(numbers(opened["bound"]) + numbers(opened["periods"]))


@pytest.mark.usefixtures("verified")
@pytest.mark.parametrize("drawn", [1, 16])
def test_eps_is_pooled_over_each_monitoring_period_at_its_windows_with_the_tied_audits_the_period_reads(
        tmp_path, capsys, drawn):
    """w63k4 over three execution phases, each phase's scan drawn whole at 2^-64 and its tied audit drawn subset:1, read
    at δ_i = 2^-64: three tied audits read, so eps a window is the scan's δ and one δ_i, and over each monitoring
    period's cell its windows (its phases times its links) times that, the cell's own eps, with the tied audits read
    over those windows. Drawn whole, none is read and eps is the scans' δ alone. The ONSITE line states the count and
    each period's pooled eps beside the per-window term, and the record and its line leave the node as written."""
    sys.path.insert(0, str(HERE.parent.parent / "network_traces"))
    import scan_zk as SZ
    schedule = dataclasses.asdict(SZ.W63K4.params) | {"bucket_ms": 4000}
    entries = 896 if drawn < 16 else 0
    scan = dict(SCAN, delta=f"1/{2 ** 64}", free=0, faulty_entries=entries, jitter=32)
    audit = {"delta": "1/2", "drawn": drawn, "units": 16, "wrong_units": 8 if drawn < 16 else 0}
    charged = [dict(audit, delta=f"1/{2 ** 64}", wrong_units=15)] * 3 if drawn < 16 else []
    public = dict(PUBLIC, K=3 * (800 + entries), cells=3 * 384, schedules=3 * 384, windows=3, scans=[scan] * 3,
                  audits=[audit] * 3, charged=charged, eps=str(Fraction(3 + len(charged), 2 ** 64)),
                  held=["free", "circuit"])
    out = accepted(tmp_path, public, windows=3, schedule=schedule)
    assert main([str(out)]) == 0
    text, said = (out / "record" / "onsite.json").read_text(), capsys.readouterr().out.strip()
    assert SV.public_doc(text) == text and SV.public_line(said) == said
    p, line = json.loads(text)["periods"], json.loads(said[len("ONSITE "):])["eps_pooled"]
    read, d = len(charged), Fraction(1, 2 ** 64)

    def at(w: int) -> dict:
        return {"phases": w, "eps": str(w * d * (3 + read) / 3), "scan": str(w * d),
                "charged": str(w * d * read / 3), "tied_audits_read": str(Fraction(read * w, 3))}

    e = p["eps"]
    assert e["tied_audits"] == {"read": read, "delta_i": [f"1/{2 ** 64}"] * bool(read), "is": OR.TIED_READ}
    assert (e["is"], e["per_window"], e["run"]) == (OR.POOLED_EPS, at(1), at(3))
    assert e["run"]["eps"] == public["eps"] == p["run"]["delta_net"][0]["value"]
    assert (line["tied_audits_read"], line["per_window"]) == (read, at(1)["eps"])
    for name in LB.PERIODS:
        for where in ("node", "datacenter", "world"):
            c = p[name][where]
            w = c["parameters"]["phases"] * c["parameters"]["links"]
            assert e[name][where] == at(w) and e[name][where]["eps"] == c["delta_net"][0]["value"] == line[name][where]
    assert e["week"]["node"]["tied_audits_read"] == str(read * math.ceil(7 * 86400 / 128) // 3)


@pytest.mark.parametrize("caps", [[], [4, 4], [5] * (4 * 63)])
def test_a_case_whose_caps_arent_its_schedules_is_the_connections_error(tmp_path, caps):
    """The period's check refuses caps that aren't the schedule's (`Schedule.caps`), an empty list included, so a
    case's other caps beside an accepted answer leave the connections and the periods unfilled."""
    rec = record(accepted(tmp_path, caps=caps))
    why = (f"{OR.CASE}: ValueError: the case's {len(caps)} caps aren't its schedule's (Schedule.caps: egress_cells = 4 "
           "for each of its lanes · m · T = 252 connections), which the period's check refuses")
    assert rec["bound"]["K"] == K and rec["bound"]["connections"] == {"error": why} and rec["periods"] == {"error": why}


def census_line(name: str, *fields: str) -> dict:
    d = next(d for d in json.loads((LB.CENSUS / name).read_text()) if d["role"] == "reference")
    digest = hashlib.sha512(json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    return {"file": f"benchmarks/census/{name}", "id": d["id"], "read": {f: d[f] for f in fields},
            "sha512": digest.hexdigest()}


@pytest.mark.usefixtures("verified")
def test_the_runs_period_is_proved_and_every_monitoring_period_scaled_beside_one_phase(tmp_path):
    """Both checks held, every premise discharged and the coins live, so nothing is open under the figure of record:
    its run's cell reads proved, and each scaled one names its rule alone."""
    p = held_both(tmp_path)["periods"]
    phase = Fraction(63 * 20, 1000)
    one = {"bits": K, "share": K / LB.WEIGHTS.bits, "share_served": K / LB.SERVED.bits,
           "share_of_capacity": K / CAPACITY, "vacuous": False, "how": "proved",
           "delta_net": [{"symbol": "eps", "value": "1/1048576"}, {"symbol": "ε_ks", "times": 1},
                         {"symbol": "ε_bind", "times": 1}, {"symbol": "ε_tap", "times": 1}],
           "eps_per_phase_below": "1", "statement": LB.STATEMENTS["per_period"]}
    assert p["phase_s"] == str(phase) and p["run"] == one | {
        "seconds": str(phase), "phases": 1, "connections": 252, "per_phase": one}
    links = {"node": 1, "datacenter": LB.datacenter_links(), "world": LB.world_links()}
    rules = {"node": "run-to-period", "datacenter": "datacenter", "world": "world"}
    for name, seconds in LB.PERIODS.items():
        w = math.ceil(Fraction(seconds) / phase)
        assert (p[name]["seconds"], p[name]["phases"]) == (seconds, w)
        for where, n in links.items():
            c = p[name][where]
            assert (c["bits"], c["delta_net"], c["how"], c["rule"]) == (
                K * w * n, OR.delta_net(Fraction(1, 2 ** 20) * w * n, w * n), "scaled", rules[where])
            assert c["delta_net"][0]["value"] == str(Fraction(w * n, 2 ** 20))
            assert c["vacuous"] == (w * n >= 2 ** 20) and c["share"] == K * w * n / LB.WEIGHTS.bits
            assert c["share_served"] == K * w * n / LB.SERVED.bits and c["share_of_capacity"] == K / CAPACITY
            assert c["eps_per_phase_below"] == str(Fraction(1, w * n))
            assert c["parameters"] == {"phases": w, "phase_s": str(phase), "connections": str(252 * w * n), "links": n}
            assert c["per_phase"] == (one if where == "node" else {
                "bits": K * n, "share": K * n / LB.WEIGHTS.bits, "share_served": K * n / LB.SERVED.bits,
                "share_of_capacity": K / CAPACITY,
                "delta_net": OR.delta_net(Fraction(n, 2 ** 20), n),
                "vacuous": n >= 2 ** 20, "eps_per_phase_below": str(Fraction(1, n)), "how": "scaled", "rule": where,
                "statement": OR.STATEMENTS[where]})
    assert p["week"]["datacenter"]["statement"] == LB.STATEMENTS["datacenter"]
    assert {k: r["sources"] for k, r in p["rules"].items()} == {
        "per-phase": [OR.FROM_BOUND], "run-to-period": [OR.FROM_BOUND, OR.FROM_SCHEDULE],
        "datacenter": [census_line("deployments.json", "machines")],
        "world": [census_line("adversary.json", "total_flop", "flop_per_s_per_h100", "monitoring_period_s"),
                  census_line("deployments.json", "gpus_per_machine")]}
    assert p["rules"]["datacenter"]["sources"][0]["id"] == "dc-12500-machines-600s"
    assert {k: r["rule"] for k, r in p["rules"].items()} == OR.RULES
    world = census_line("adversary.json", "model", "total_params", "weights_bytes", "weights_datatype")
    assert p["weights"] == {
        "share": {"of": LB.WEIGHTS.label, "is": OR.WEIGHTS, "bits": 8 * world["read"]["weights_bytes"], "source": world},
        "share_served": {"of": "the demo's served model (Llama-3.2-1B) in BF16", "bits": LB.SERVED.bits}}
    assert world["read"]["model"] == "DeepSeek-V4-Pro" and world["id"] == "world-inference-2027"


@pytest.mark.usefixtures("verified")
def test_eps_is_split_into_the_scans_and_the_charged_audits_delta_beside_each_cells_threshold(tmp_path):
    """The served defaults: the scan drawn whole at 2^-20, and a tied audit drawn subset:1 at δ = 1/2, whose δ the
    period's eps carries. The bound says which δ is which; a cell whose eps a phase is past its threshold is vacuous."""
    public = dict(PUBLIC, eps="524289/1048576", audits=[{"delta": "1/2", "drawn": 1, "units": 16, "wrong_units": 8}])
    rec = record(accepted(tmp_path, public))
    assert rec["bound"]["delta"] == {"scan": "1/1048576", "charged": "1/2", "eps_per_phase": "524289/1048576",
                                     "is": OR.DELTA, "charged_audits": {"missing": OR.NO_READINGS}}
    assert (rec["periods"]["eps"]["tied_audits"], rec["periods"]["eps"]["per_window"]) == (
        {"missing": OR.NO_READINGS}, {"phases": 1, "eps": "524289/1048576", "scan": "1/1048576", "charged": "1/2"})
    run, year = rec["periods"]["run"], rec["periods"]["year"]["datacenter"]
    assert (run["vacuous"], run["eps_per_phase_below"]) == (False, "1")
    assert year["vacuous"] and Fraction(year["eps_per_phase_below"]) < Fraction(public["eps"])
    assert Fraction(year["eps_per_phase_below"]) == Fraction(1, year["parameters"]["phases"] * LB.datacenter_links())


def test_scans_at_two_deltas_are_the_bounds_error(tmp_path):
    two = dict(PUBLIC, K=PUBLIC["K"] + 9088 + 63, eps="3/1048576", windows=2,
               scans=[SCAN, dict(SCAN, delta="1/524288")])
    rec = record(accepted(tmp_path, two, windows=2))
    assert rec["bound"]["error"].startswith("eps 3/1048576 isn't m · δ_scan plus the charged audits' δ")


def test_an_eps_that_isnt_the_scans_delta_plus_each_charged_readings_delta_i_is_the_bounds_error(tmp_path):
    """Where the public lists each charged audit's reading at δ_i, eps is m · δ_scan plus those δ_i exactly: one that
    counts the audit's answer's δ = 1/2 in their place is the bound's error, so the record never reports it."""
    audit = {"delta": "1/2", "drawn": 1, "units": 16, "wrong_units": 8}
    public = dict(PUBLIC, eps="524289/1048576", audits=[audit],
                  charged=[dict(audit, delta=f"1/{2 ** 64}", wrong_units=15)])
    rec = record(accepted(tmp_path, public))
    assert rec["bound"]["error"] == ("eps 524289/1048576 isn't m · δ_scan plus each charged reading's δ_i: m · δ_scan "
                                     f"is 1/1048576, and the readings' δ_i are ['1/{2 ** 64}']")
    assert "of_record" not in rec["bound"]


def with_deltas(out: Path, **deltas: object) -> Path:
    """``out`` with its window report's rows (by key) stating each given δ."""
    f = out / "record" / "verdicts.json"
    doc = json.loads(f.read_text())
    for key, d in deltas.items():
        doc["protocols"][key]["delta"] = d
    f.write_text(json.dumps(doc))
    return out


@pytest.mark.parametrize("eps, stated, vacuous", [("1/1048576", "262145/524288", False),
                                                  ("524289/1048576", "524289/524288", True)])
def test_the_composites_error_sums_each_subprotocols_delta_with_its_source(tmp_path, eps, stated, vacuous):
    """Sampled proofs at its served δ = 1/2 and PoUW at 2^-20 from their rows, and the network certifier's eps from its
    bound: one tied audit's draw read at sampled proofs' 1/2 and again inside the network's eps is two terms. Their sum is
    stated as a lower bound on the error, inference-only's δ missing and the rest unvalued."""
    public = dict(PUBLIC, eps=eps, audits=[{"delta": str(Fraction(eps) - Fraction(1, 2 ** 20)), "drawn": 1,
                                            "units": 16, "wrong_units": 8}])
    rec = record(with_deltas(accepted(tmp_path, public), integrity="1/2", work=2.0 ** -20))
    ce = rec["composite_error"]
    assert [(t["subprotocol"], t.get("delta")) for t in ce["terms"]] == [
        ("sampled proofs", "1/2"), ("PoUW", "1/1048576"), ("inference-only", None), ("the network certifier", eps)]
    assert ce["terms"][0]["from"] == "record/verdicts.json protocols.integrity.delta: its answer's integrity-profile δ"
    assert ce["terms"][2] == {"subprotocol": "inference-only", "missing": OR.NO_NCI_DELTA}
    assert ce["terms"][3]["from"].startswith(OR.EPS["source"])
    assert (ce["stated"], ce["vacuous"], ce["is"], ce["unvalued"]) == (stated, vacuous, OR.COMPOSITE, OR.COMPOSITE_UNVALUED)
    assert "counts in both" in OR.COMPOSITE and "lower bound" in OR.COMPOSITE_UNVALUED
    assert OR.summary(rec)["composite_error"] == {"stated": stated, "vacuous": vacuous}


def test_a_composite_term_the_run_doesnt_state_is_missing_and_one_that_isnt_a_number_its_error(tmp_path):
    rec = record(with_deltas(accepted(tmp_path), integrity="half"))
    sampled, pouw, _nci, net = rec["composite_error"]["terms"]
    assert sampled["error"] == "its row's δ 'half' isn't a number" and pouw["missing"] == "its row states no δ"
    assert rec["composite_error"]["stated"] == net["delta"] == PUBLIC["eps"]


def test_a_composite_term_stated_as_a_float_is_the_ratio_it_was_given_as(tmp_path):
    """2^-64 as a float is 1/2^64 and 0.01 is 1/100, never a nearby decimal or ratio."""
    sampled, pouw, _nci, _net = record(with_deltas(accepted(tmp_path), integrity=0.01, work=2.0 ** -64))[
        "composite_error"]["terms"]
    assert (sampled["delta"], pouw["delta"]) == ("1/100", f"1/{2 ** 64}")


@pytest.mark.usefixtures("verified")
def test_a_node_cell_is_proved_where_the_runs_period_covers_its_monitoring_period_and_nothing_is_open(tmp_path):
    """An execution phase of 63 buckets of 10,000 s covers a week but not a month: the node's week is the run's K,
    proved where nothing is open under the figure of record, and its datacenter scaled from that by its machines; its
    month is scaled from the run. With a check of the tied charge open, the node's week is the same K, open, naming
    it, never proved, and each scaled cell names it beside its rule."""
    slow = dict(SCHEDULE, bucket_ms=10_000_000)
    p = record(accepted(tmp_path, dict(PUBLIC, held=["free", "circuit"]), schedule=slow))["periods"]
    week, month = p["week"], p["month"]
    assert (week["node"]["bits"], week["node"]["how"], "rule" not in week["node"]) == (K, "proved", True)
    assert week["node"]["parameters"] == {"phases": 1, "phase_s": "630000", "links": 1, "connections": "252"}
    n = LB.datacenter_links()
    assert (week["datacenter"]["bits"], week["datacenter"]["rule"]) == (K * n, "datacenter")
    assert "open" not in week["node"] and "open" not in week["datacenter"]
    w = math.ceil(Fraction(LB.PERIODS["month"], 630000))
    assert (month["node"]["bits"], month["node"]["rule"], month["phases"]) == (K * w, "run-to-period", w)
    p = record(accepted(tmp_path / "open", dict(PUBLIC, held=["free"]), schedule=slow))["periods"]
    week = p["week"]
    assert (week["node"]["bits"], week["node"]["how"], week["node"]["open"]) == (K, "open", ["circuit"])
    assert (week["datacenter"]["how"], week["datacenter"]["rule"], week["datacenter"]["open"]) == (
        "scaled", "datacenter", ["circuit"])
    assert not [c for c in (p["run"], p["run"]["per_phase"], *(p[name][where] for name in LB.PERIODS
                                                                for where in ("node", "datacenter", "world")))
                if c["how"] == "proved" or c["open"] != ["circuit"]]


@pytest.mark.usefixtures("verified")
def test_a_scaled_cell_over_several_phases_rounds_up(tmp_path):
    two = dict(PUBLIC, K=PUBLIC["K"] + 9088 + 63 + 1, eps="1/524288", windows=2,
               scans=[SCAN, dict(SCAN, faulty_sets=1)])
    p = record(accepted(tmp_path, two, windows=2))["periods"]
    kr = two["K"]
    w = math.ceil(Fraction(LB.PERIODS["month"]) / Fraction(63, 50))
    assert w % 2 and p["run"]["bits"] == kr and p["run"]["per_phase"]["rule"] == "per-phase"
    assert p["month"]["node"]["bits"] == math.ceil(Fraction(kr, 2) * w) != Fraction(kr, 2) * w
    assert p["month"]["node"]["parameters"]["connections"] == str(252 * w)
    assert p["run"]["per_phase"]["bits"] == math.ceil(Fraction(kr, 2))


def test_a_case_of_another_period_leaves_the_periods_unfilled(tmp_path):
    out = accepted(tmp_path, windows=2)
    why = f"{OR.CASE}: ValueError: the case's period has 2 execution phases, and its answer 1"
    rec = record(out)
    assert rec["periods"] == {"error": why} and rec["bound"]["connections"] == {"error": why}


def test_the_record_lists_the_re_proof_inputs_and_which_of_outs_the_run_wrote(tmp_path):
    """What a later re-proof of the sampled units needs is listed by name, whatever the verdicts: the serve dir's, which
    the record can't see, by their paths there, naming no window, and OUT's with whether the run wrote each; how each must
    be kept is stated beside them."""
    out = accepted(tmp_path)
    got = record(out)["reproof"]
    assert got["kept"] == OR.REPROOF_KEPT and list(got["inputs"]) == list(OR.REPROOF)
    serve = {k for k, v in got["inputs"].items() if v["in"] == "serve dir"}
    assert {"values", "openings", "draw", "coins", "release", "advice", "weights"} <= serve
    assert not any("present" in got["inputs"][k] for k in serve) and not SV.NAMED.search(json.dumps(got))
    written = {OR.REGISTRATION, str(OR.CASE), "record/verdicts.json", str(OR.AUDIT)}
    assert {v["path"]: v["present"] for v in got["inputs"].values() if v["in"] == "OUT"} == {
        v[1]: v[1] in written for v in OR.REPROOF.values() if v[0] == "OUT"}
    for at, path, _ in OR.REPROOF.values():
        if at == "OUT":
            (out / path).parent.mkdir(parents=True, exist_ok=True)
            (out / path).touch()
    assert all(v["present"] for v in record(out)["reproof"]["inputs"].values() if v["in"] == "OUT")
    assert record(served_out(tmp_path / "rehearsal"))["reproof"]["kept"] == OR.REPROOF_KEPT
    assert got["copy"] == {"missing": OR.NO_COPY}


def _reproof_files(path: str, tie: bool = False) -> list[str]:
    """A file of window 7 at class c64 for each a `REPROOF` path names (one under each directory), H1-tied where ``tie``."""
    out = []
    for part in path.split(", "):
        m = re.fullmatch(r"(.*)\{([^}]*)\}(.*)", part)
        for p in [m[1] + x + m[3] for x in m[2].split(",")] if m else [part]:
            p = p.replace("<K>", "7").replace("<class>", "c64").replace("<tie>", "-tie" if tie else "")
            out.append(p + "m0-private.bin" if p.endswith("/") else p)
    return out


def test_the_serve_dirs_re_proof_inputs_are_checked_against_their_copys_listing_by_path_alone(tmp_path):
    """The serve dir's inputs, witness, are copied off the node: the record names the copy's node and path and says of
    each input whether the listing holds each file its path names and at least one under each directory it names, and
    nothing of the listing's window names; one file short is that input not copied, and a copy that isn't NODE:/PATH
    or a listing it can't read is the copy's error."""
    files = _reproof_files
    out = accepted(tmp_path)
    serve = [p for at, p, _ in OR.REPROOF.values() if at == "serve dir"]
    listing = tmp_path / "private" / "listing.txt"
    listing.parent.mkdir()
    listing.write_text("\n".join(f for p in serve for f in files(p)) + "\n")
    got = record(out, copy="vy-nebius-cpu-3:/home/research/keep/reproof", listing=listing)["reproof"]
    assert got["copy"] == {"node": "vy-nebius-cpu-3", "path": "/home/research/keep/reproof", "not_copied": []}
    assert all(v["copied"] for v in got["inputs"].values() if v["in"] == "serve dir")
    assert not any("copied" in v for v in got["inputs"].values() if v["in"] == "OUT")
    assert not SV.NAMED.search(json.dumps(got)) and "c64" not in json.dumps(got)
    listing.write_text("\n".join(f for p in serve for f in files(p) if not f.endswith("/receipt.json")) + "\n")
    got = record(out, copy="vy-nebius-cpu-3:/home/research/keep/reproof", listing=listing)["reproof"]
    assert got["copy"]["not_copied"] == ["draw"] and got["inputs"]["values"]["copied"]
    assert record(out, copy="vy-nebius-cpu-3", listing=listing)["reproof"]["copy"] == {
        "error": "--reproof-copy 'vy-nebius-cpu-3' isn't NODE:/PATH"}
    assert record(out, copy="n:/p", listing=tmp_path / "none.txt")["reproof"]["copy"] == {
        "error": "the copy's listing (--reproof-listing) can't be read: FileNotFoundError"}


def test_under_h1s_tie_the_re_proofs_circuit_is_the_record_reading_h1s_tables(tmp_path):
    """Under H1_TIE the window commit's circuit is held to the verifier's record reading H1's tables (served_verdicts.sh's
    `v3c_record`, window-circuit-w<K>-<class>-tie.json), so a tied run's copy holding only the untied record at the
    class hasn't copied the circuit; `--h1-tie` says the run is tied, and the record names the circuit it needs."""
    out, copy = accepted(tmp_path), "vy-nebius-cpu-3:/home/research/keep/reproof"
    serve = [p for at, p, _ in OR.REPROOF.values() if at == "serve dir"]
    listing = tmp_path / "private" / "listing.txt"
    listing.parent.mkdir()
    for tie in (False, True):
        listing.write_text("\n".join(f for p in serve for f in _reproof_files(p, tie)) + "\n")
        got = record(out, copy=copy, listing=listing, tie=tie)["reproof"]
        assert got["copy"]["not_copied"] == []
        assert got["inputs"]["circuit"]["path"] == f"window-circuit-w<K>-<class>{'-tie' * tie}.json"
        assert main([str(out), "--reproof-copy", copy, "--reproof-listing", str(listing), *(["--h1-tie"] * tie)]) == 0
        written = json.loads((out / "record" / "onsite.json").read_text())["reproof"]
        assert written["copy"]["not_copied"] == [] and written["inputs"]["circuit"] == got["inputs"]["circuit"]
        assert "<tie>" not in json.dumps(written)
    untied = [f for p in serve for f in _reproof_files(p)]
    listing.write_text("\n".join(untied) + "\n")
    assert record(out, copy=copy, listing=listing, tie=True)["reproof"]["copy"]["not_copied"] == ["circuit"]
    assert main([str(out), "--reproof-copy", copy, "--reproof-listing", str(listing), "--h1-tie"]) == 0
    assert json.loads((out / "record" / "onsite.json").read_text())["reproof"]["copy"]["not_copied"] == ["circuit"]
    [tied] = OR.named("window-circuit-w<K>-<class><tie>.json", tie=True)
    assert tied.fullmatch("window-circuit-w7-c64-tie.json") and not tied.fullmatch("window-circuit-w7-c64.json")


def test_the_record_leaves_the_node_as_written(tmp_path):
    out = accepted(tmp_path)
    f = tmp_path / "advice-charge.json"
    f.write_text(json.dumps(MEASURED))
    assert main([str(out), "--advice-charge", str(f)]) == 0
    text = (out / "record" / "onsite.json").read_text()
    assert SV.public_doc(text) == text


def test_the_published_form_which_drops_windows_gives_the_same_record(tmp_path):
    full = record(accepted(tmp_path / "full"))
    out = accepted(tmp_path / "published", {k: v for k, v in PUBLIC.items() if k != "windows"})
    case = json.loads((out / OR.CASE).read_text())
    case["params"].pop("windows")
    (out / OR.CASE).write_text(json.dumps(case))
    assert record(with_onsite(out)) == full


@pytest.mark.parametrize("public, why", [
    (dict(PUBLIC, K=K + 1), f"K is {K + 1}, and its terms sum to {K}"),
    ({k: v for k, v in PUBLIC.items() if k != "scans"}, "not NetworkCheck.Period's rows form"),
    (dict(PUBLIC, scans=[{k: v for k, v in SCAN.items() if k != "faulty_entries"}]),
     "the rows form's public lacks a term: KeyError: 'faulty_entries'"),
    (dict(PUBLIC, windows=2), "1 scans for the period's 2 execution phases"),
    (dict(PUBLIC, K=756 + 756, scans=[], windows=0), "the rows form's public has no scan"),
])
def test_a_public_that_isnt_the_rows_forms_sum_is_the_bounds_error(tmp_path, public, why):
    rec = record(accepted(tmp_path, public))
    assert rec["bound"]["error"].startswith(why) and rec["periods"] == {"missing": "no bound"}


def test_a_test_runs_measured_advice_charge_is_beside_the_bound_as_that_runs(tmp_path, capsys):
    out = accepted(tmp_path)
    f = tmp_path / "advice-charge.json"
    f.write_text(json.dumps(MEASURED))
    assert main([str(out), "--advice-charge", str(f)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    assert rec["advice_charge"]["measured"] == {
        "label": MEASURED["label"], "run": "toy-test", "from": str(f), "bits_per_commit_period": MEASURED["bits"],
        "bits_per_phase": MEASURED["phase_bits"], "phases": OR.TEST_PHASES, "commit_periods_measured": 2,
        "finishes": MEASURED["finishes"],
        "prompt_len": MEASURED["prompt_len"], "output_len": MEASURED["output_len"]}
    assert (rec["advice_charge"]["served"], rec["advice_charge"]["charged"]["bits_per_phase"]) == (OR.ADVICE, [9088])
    assert json.loads(capsys.readouterr().out.strip()[7:])["advice_measured_on"] == "toy-test"
    f.write_text(json.dumps(MEASURED | {"phase_bits": None, "by_phase": {"missing": "no clock"}}))
    assert main([str(out), "--advice-charge", str(f)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    assert rec["advice_charge"]["measured"]["bits_per_phase"] == {"missing": "no clock"}
    capsys.readouterr()
    f.write_text(json.dumps(MEASURED | {"format": "verity/other/v1"}))
    assert main([str(out), "--advice-charge", str(f)]) == 1
    assert "not verity/advice-charge/v2" in capsys.readouterr().err


def test_the_run_is_of_record_only_with_the_onsite_composite_accepting(tmp_path):
    """With no composite of record, nothing states the parameters the period's check read, so auditor-record doesn't
    hold and the network certifier's accepted verdict is not established, nor is sampled proofs', its leg there, which
    credits no member; onsite's composite, accepting, makes the run of record, and its rejection leaves it not."""
    out = accepted(tmp_path, onsite=False)
    rec = record(out)
    assert {k: v["verdict"] for k, v in rec["verdicts"].items()} == {
        "sampled proofs": OR.NOT_ESTABLISHED, "PoUW": "accept", "inference-only": "accept", "remote": "accept",
        "the network certifier": OR.NOT_ESTABLISHED, "onsite": "missing", **dict.fromkeys(OR.MEMORY, "missing")}
    assert rec["verdicts"]["onsite"]["why"] == "the run has no composite of verity/remote-network/v0"
    assert rec["verdicts"]["sampled proofs"]["credit"] == UNESTABLISHED
    assert rec["checked"][OR.AUDITOR_RECORD]["why"] == (
        f"the composite of record has no {OR.NETWORK} case stating the SHA-512 of the parameters the period's check read")
    assert rec["of_record"] == {"holds": False, "missing": ["onsite"], "not_accepting": [],
                                "not_established": ["sampled proofs", "the network certifier"], "not_credited": [],
                                "not_claimed": ["PoUW"], "conditional_on": [OR.CERTIFIER_KEY],
                                "uncredited": OR.UNESTABLISHED_CREDIT, **ONSITE_OF_RECORD}
    doc = json.loads((with_onsite(out) / "composite" / "remote-network" / "composite.json").read_text())
    assert record(out)["of_record"] == {"holds": True, "missing": [], "not_accepting": [], "not_established": [],
                                           "not_credited": [], "not_claimed": ["PoUW"], **ONSITE_OF_RECORD}
    (out / "composite" / "remote-network" / "composite.json").write_text(json.dumps(
        doc | {"missing": [OR.NETWORK], "verdict": {"id": OR.ONSITE, "verdict": "reject", "refusal": "components"}}))
    rec = record(out)
    assert rec["verdicts"]["onsite"] == {"protocol": OR.ONSITE, "verdict": "reject", "refusal": "components",
                                         "from": "composite/remote-network/composite.json",
                                         "not_established": [OR.NETWORK]}
    assert rec["of_record"] == {"holds": False, "missing": [], "not_accepting": ["onsite"], "not_established": [],
                                "not_credited": [], "not_claimed": ["PoUW"], **ONSITE_OF_RECORD}


@pytest.mark.parametrize("leg, row, verdict", [
    ("accept", "reject", "accept"),
    ("reject", "accept", "reject"),
    ("no case", "accept", OR.NOT_ESTABLISHED),
    (None, "accept", OR.NOT_ESTABLISHED),
    (None, None, "missing"),
])
def test_sampled_proofs_verdict_is_its_leg_in_the_composite_of_record_never_the_records_audits_v1_row(
        tmp_path, leg, row, verdict):
    """Sampled proofs' verdict is the outcome of remote's sampled proofs case in the composite of record
    (VStar.Recursive.Sampled on OneStageZk's case), whatever the window report's integrity row, the records audit's
    one-stage v1 answer, says: a v1 verdict is no verdict. Where the composite of record has no such case, or the run
    no composite of record, it is not established, the row's answer beside it, crediting no member and stating no
    bound; missing where the row is missing too."""
    out = accepted(tmp_path, onsite=False)
    f = out / "record" / "verdicts.json"
    report = json.loads(f.read_text())
    report["protocols"]["integrity"] = ({"owner": "compute-accounting", "verdict": "missing", "why": "toy"}
                                        if row is None else report["protocols"]["integrity"] | {"verdict": row})
    f.write_text(json.dumps(report))
    if leg == "no case":
        with_composite(out, "remote-network", OR.ONSITE, (OR.NETWORK, OR.INFERENCE_ONLY, OR.POUW_ID))
    elif leg is not None:
        with_onsite(out, sampled=leg)
    rec = record(out)
    v = rec["verdicts"]["sampled proofs"]
    src = "composite/remote-network/composite.json"
    unread = {"no case": f"{OR.ONSITE}'s composite has no {SAMPLED_V1} case",
              None: f"its case is read from {OR.ONSITE}'s composite, and the run has none"}
    expect = {
        "accept": {"protocol": SAMPLED_V1, "verdict": "accept", "from": f"{src} cases", "credit": CREDITED},
        "reject": {"protocol": SAMPLED_V1, "verdict": "reject", "from": f"{src} cases", "credit": CREDITED},
        OR.NOT_ESTABLISHED: {"protocol": SAMPLED_V1, "verdict": OR.NOT_ESTABLISHED, "answer": "accept",
                             "from": "record/verdicts.json protocols.integrity",
                             "why": f"{OR.V1_UNREAD}, and {unread.get(leg)}", "credit": UNESTABLISHED},
        "missing": {"verdict": "missing", "why": unread[None], "credit": CREDITED}}[verdict]
    assert v == expect
    assert ("per_window" in rec["sampled_proofs"]) is (verdict == "accept")
    assert ("sampled proofs" in rec["of_record"]["not_established"]) is (verdict == OR.NOT_ESTABLISHED)
    assert OR.summary(rec)["verdicts"]["sampled proofs"] == verdict


def test_under_the_network_certifiers_composite_without_pouw_the_run_is_of_record_over_its_subprotocols_alone(
        tmp_path, capsys):
    """A run with the network certifier's composite without PoUW, under its own id: it is the composite of record, named
    by that id, though remote's and remote-network's, which compose PoUW's case, reject naming it missing. The run is of
    record over its verdict and its subprotocols', inference-only's from its case there, reading neither PoUW's nor
    remote's; its error sums no PoUW term; and the record and its line credit no verified compute, the line saying so in
    words at the composite's id. Its rejection leaves the run not of record; with none, remote-network is the composite
    of record again, PoUW's term back in its error, since naming PoUW's case missing it composes PoUW; and two such
    composites refuse the record, naming both."""
    out = accepted(tmp_path)
    with_composite(out, "remote", OR.REMOTE, REMOTE_IDS, "reject", missing=(OR.POUW_ID,), inference_only="reject")
    with_composite(out, "remote-network", OR.ONSITE, (OR.NETWORK, *REMOTE_IDS), "reject", missing=(OR.POUW_ID,))
    with_composite(out, "remote-network-no-pouw", WITHOUT, WITHOUT_IDS)
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    v = rec["verdicts"]
    assert v["onsite"] == {"protocol": WITHOUT, "verdict": "accept",
                           "from": "composite/remote-network-no-pouw/composite.json"}
    assert v["inference-only"] == {"protocol": OR.INFERENCE_ONLY, "verdict": "accept",
                                   "from": "composite/remote-network-no-pouw/composite.json cases",
                                   "alpha_gamma": {"missing": OR.ALPHA_GAMMA_UNSTATED}}
    assert (v["remote"]["verdict"], v["remote"]["not_claimed"], v["PoUW"]["not_claimed"]) == (
        "reject", OR.POUW_OPEN, OR.POUW_OPEN)
    assert rec["of_record"] == {"holds": True, "composite": WITHOUT, "missing": [], "not_accepting": ["remote"],
                                "not_established": [], "not_credited": [], "not_claimed": ["PoUW", "remote"],
                                "verified_compute": "none"}
    ce = rec["composite_error"]
    assert (ce["composite"], [t["subprotocol"] for t in ce["terms"]]) == (
        WITHOUT, ["sampled proofs", "inference-only", "the network certifier"])
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["of_record"], line["composite"], line["verified_compute"], line["compute"]) == (
        True, WITHOUT, "none", f"no verified compute is credited: {WITHOUT}'s verdict is no compute claim")
    with_composite(out, "remote-network-no-pouw", WITHOUT, WITHOUT_IDS, "reject")
    of = record(out)["of_record"]
    assert (of["holds"], of["not_accepting"]) == (False, ["remote", "onsite"])
    (out / "composite" / "remote-network-no-pouw" / "composite.json").unlink()
    rec = record(out)
    assert (rec["of_record"]["composite"], rec["verdicts"]["onsite"]["protocol"], rec["of_record"]["not_claimed"]) == (
        OR.ONSITE, OR.ONSITE, ["PoUW"])
    assert [t["subprotocol"] for t in rec["composite_error"]["terms"]] == [
        "sampled proofs", "PoUW", "inference-only", "the network certifier"]
    with_composite(out, "a", WITHOUT, WITHOUT_IDS)
    with_composite(out, "b", f"{WITHOUT}-b", WITHOUT_IDS)
    with pytest.raises(ValueError, match=re.escape(f"2 composites of the network certifier without PoUW ({WITHOUT}, "
                                                   f"{WITHOUT}-b)")):
        record(out)


#: α and Γ in the inference-only case's private policy.json, numbers no other part of a toy record holds
ALPHA, GAMMA = [192837465, 564738291], 918273645


@pytest.mark.parametrize("policy, said", [
    ({"alpha": ALPHA, "gamma": GAMMA, "read": SC.FROM_DIVISION},
     {"read": "the division", "is": OR.ALPHA_GAMMA_FROM_DIVISION}),
    (None, {"read": "given", "is": OR.ALPHA_GAMMA_GIVEN}),
    ({"alpha": ALPHA, "gamma": GAMMA, "read": "set by hand"},
     {"error": "the inference-only case's policy.json doesn't say served_cases.py read it off the division"}),
    ([ALPHA, GAMMA], {"error": "the inference-only case's policy.json isn't a JSON object: AttributeError"}),
])
def test_the_inference_only_verdict_says_where_alpha_and_gamma_came_from_never_a_registered_policy(
        tmp_path, capsys, policy, said):
    """Under the composite without PoUW, the inference-only verdict and the ONSITE line say where the case's α and Γ
    came from by its private policy.json's read alone: off the run's own division, so the verdict confirms the division
    against itself, where served_cases.py --from-division wrote it; the run's ALPHA and GAMMA where the case has none;
    an error for any other file. Neither says α or Γ, and none changes the verdict or the run's being of record."""
    out = with_composite(accepted(tmp_path), "remote-network-no-pouw", WITHOUT, WITHOUT_IDS)
    f = tmp_path / "serve" / "private" / "inference-only" / "policy.json"
    if policy is not None:
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps(policy))
    assert main([str(out), "--inference-only-case-policy", str(f)]) == 0
    written = (out / "record" / "onsite.json").read_text()
    line = capsys.readouterr().out.strip()
    rec = json.loads(written)
    assert (rec["verdicts"]["inference-only"]["verdict"], rec["verdicts"]["inference-only"]["alpha_gamma"],
            rec["of_record"]["holds"]) == ("accept", said, True)
    assert json.loads(line[len("ONSITE "):])["inference_only_alpha_gamma"] == said
    assert not any(str(x) in text for x in (*ALPHA, GAMMA) for text in (written, line))


@pytest.mark.usefixtures("verified")
@pytest.mark.parametrize("how", ["unlisted", "restated", "another key", "no registration", "a case rewritten"])
def test_while_certifier_key_is_open_an_accepted_network_verdict_is_not_established_and_its_bound_conditional_on_it(
        tmp_path, lock, capsys, how):
    """certifier-key open, every other premise discharged, both checks held and the coins live: the network
    certifier's verdict is not established, its Lean answer's accept beside it, never cited as an accept with an open
    condition, so the run isn't of record; its bound stands, the figure of record charging ℓ_a + 1 for the tied audit,
    and it, its cells, the composite's error and the run's being of record are conditional on certifier-key by name.
    It is open where the lock doesn't hold PeriodLeakRegistered at its pinned statement (unlisted or restated, each
    open too), and where the run's check of the auditor's record fails: a record of another key than the period's, no
    registration of the auditor's, or a case whose key was rewritten after the composite of record composed it, the
    auditor's record and registration stating the rewritten key. A network verdict that didn't accept stays what it
    was."""
    out = accepted(tmp_path, dict(PUBLIC, held=["free", "circuit"]))
    if how == "unlisted":
        write_lock(lock, {OR.PERIOD_REGISTERED: None})
    elif how == "restated":
        write_lock(lock, stated={OR.PERIOD_REGISTERED: "ab" * 32})
    elif how == "another key":
        auditor_files(out, key=key_record(dict(CERT_KEY, public=ANOTHER)))
    elif how == "no registration":
        auditor_args(out)["site_registration"].unlink()
    else:
        case = json.loads((out / OR.CASE).read_text())
        (out / OR.CASE).write_text(json.dumps(case | {"params": case["params"] | {"key": ANOTHER}}))
        auditor_files(out, key=key_record(dict(CERT_KEY, public=ANOTHER)), site=site_registration(ANOTHER))
    assert main([str(out)]) == 0
    rec, cond = json.loads((out / "record" / "onsite.json").read_text()), [OR.CERTIFIER_KEY]
    by_lock = how in ("unlisted", "restated")
    assert rec["checked"][OR.AUDITOR_RECORD]["holds"] is by_lock
    if how == "a case rewritten":
        assert rec["checked"][OR.AUDITOR_RECORD]["why"] == (
            f"the period's parameters ({OR.CASE}'s params) aren't, by SHA-512, those its check read (the composite of "
            f"record's {OR.NETWORK} case's params_sha512)")
    assert ("unchecked" in rec["bound"]["of_record"]["premises"]["premises"][OR.CERTIFIER_KEY]) is not by_lock
    nv = rec["verdicts"]["the network certifier"]
    assert {k: nv[k] for k in ("protocol", "verdict", "answer", "open", "why")} == {
        "protocol": OR.NETWORK, "verdict": OR.NOT_ESTABLISHED, "answer": "accept", "open": cond, "why": OR.KEY_OPEN}
    assert rec["of_record"] == {"holds": False, "missing": [], "not_accepting": [],
                                "not_established": ["the network certifier"], "not_credited": [],
                                "not_claimed": ["PoUW"], "conditional_on": cond, **ONSITE_OF_RECORD}
    of = rec["bound"]["of_record"]
    assert (rec["bound"]["K"], of["open"], of["conditional_on"], of["K"]) == (
        K, cond + [OR.PERIOD_REGISTERED] * by_lock, cond, K)
    assert rec["periods"]["conditional_on"] == rec["composite_error"]["conditional_on"] == cond
    assert "conditional_on" not in rec["verdicts"]["sampled proofs"] and rec["coins"]["open"] is False
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["verdicts"]["the network certifier"], line["of_record"], line["conditional_on"]) == (
        OR.NOT_ESTABLISHED, False, cond)
    assert record(served_out(tmp_path / "rehearsal"))["verdicts"]["the network certifier"]["verdict"] == "missing"


ANOTHER = "d4" * 64
#: each way the auditor's inputs fail auditor-record, as (its certifier-key record, its registration, the period's
#: params, why): None where the run gives no such input, a str where the file holds those bytes
REFUSED_INPUTS = {
    "no record": (None, site_registration(), CERTIFIER, "the run gave no --certifier-key, the auditor's input it reads"),
    "no registration": (key_record(), None, CERTIFIER,
                        "the run gave no --site-registration, the auditor's input it reads"),
    "unreadable": ("{", site_registration(), CERTIFIER, "--certifier-key can't be read: JSONDecodeError"),
    "another schema": (key_record() | {"schema": "verity/other/v1"}, site_registration(), CERTIFIER,
                       f"--certifier-key isn't a {WC.RECORD} record of a key and a session"),
    "no session": ({k: v for k, v in key_record().items() if k != "session"}, site_registration(), CERTIFIER,
                   f"--certifier-key isn't a {WC.RECORD} record of a key and a session"),
    "another key": (key_record(dict(CERT_KEY, public=ANOTHER)), site_registration(ANOTHER), CERTIFIER,
                    "the period's key isn't the auditor's record's"),
    "another session": (key_record(session="6f" * 32), site_registration(), CERTIFIER,
                        "the period's session isn't the auditor's record's"),
    "another LMS": (key_record(dict(CERT_KEY, params=[64, 8, 15])), site_registration(), CERTIFIER,
                    "the period's lms isn't the auditor's record's"),
    "a toy LMS": (key_record(), site_registration(), dict(CERTIFIER, lms=dict(CERTIFIER["lms"], toy=True)),
                  "the period's lms isn't the auditor's record's"),
    "another key and session": (key_record(dict(CERT_KEY, public=ANOTHER), "6f" * 32), site_registration(ANOTHER),
                                CERTIFIER, "the period's key and session aren't the auditor's record's"),
    "no key in the period": (key_record(), site_registration(), {k: v for k, v in CERTIFIER.items() if k != "key"},
                             f"the period's parameters ({OR.CASE}'s params) state no LMS parameters, key and session"),
    "no period": (key_record(), site_registration(), None,
                  f"the period's parameters ({OR.CASE}'s params) state no LMS parameters, key and session"),
    "a log its check refuses": (key_record(), site_registration(placed=ANOTHER), CERTIFIER,
                                "the auditor's registration log doesn't check: entry 0: owner"),
    "another composite": (key_record(), site_registration() | {"params": site_registration(composite="onsite")["params"]},
                          CERTIFIER, "the auditor's registration log doesn't check: composite: the log is "
                                     "remote-network's, not onsite's"),
    "another form": (key_record(), {"auditor": site_registration()["params"]["auditor"], "log": site_registration()["log"]},
                     CERTIFIER, "--site-registration isn't a registration's params and log: Malformed: not {params, log}"),
    "no certifier holding the key": (key_record(), site_registration(ANOTHER), CERTIFIER,
                                     "the auditor's registration holds the record's key for no network certifier its "
                                     "log places"),
}


def test_auditor_record_holds_where_the_auditors_record_and_registration_hold_the_periods_key_lms_and_session(tmp_path):
    """auditor-record on the run (`OR.auditor_record`): the period's parameters are, by SHA-512, those the composite of
    record composed, the auditor's certifier-key record states their key, LMS parameters and session, and its
    registration, which checks, holds that key for a network certifier its log places; it names that certifier and
    each input it read by SHA-512."""
    a = auditor_files(tmp_path / "out")
    assert OR.auditor_record(CERTIFIER, sha512_of(CERTIFIER), a["certifier_key"], a["site_registration"]) == {
        "is": OR.CHECKED[OR.AUDITOR_RECORD], "holds": True, "certifier": "cert0",
        "from": {flag: hashlib.sha512(a[f].read_bytes()).hexdigest()
                 for flag, f in (("--certifier-key", "certifier_key"), ("--site-registration", "site_registration"))}}


@pytest.mark.parametrize("how", list(REFUSED_INPUTS))
def test_auditor_record_doesnt_hold_where_an_input_isnt_given_or_isnt_the_periods_and_says_why(tmp_path, how):
    """auditor-record doesn't hold, saying why, where the run gives no record or registration of the auditor's or one
    can't be read, where the record isn't a certifier-key/v1 record or its key, LMS parameters or session aren't the
    period's (a toy LMS included), where the period states none, where the registration log is one its check refuses or
    of another composite than its params', or isn't a registration's params and log, and where its auditor holds the
    record's key for no network certifier its log places."""
    key, site, params, why = REFUSED_INPUTS[how]
    files = []
    for name, doc in (("certifier-key.json", key), ("site-registration.json", site)):
        if doc is not None:
            (tmp_path / name).write_text(doc if isinstance(doc, str) else json.dumps(doc))
        files.append(None if doc is None else tmp_path / name)
    got = OR.auditor_record(params, sha512_of(params), *files)
    assert (got["is"], got["holds"], got["why"]) == (OR.CHECKED[OR.AUDITOR_RECORD], False, why)
    assert "certifier" not in got


@pytest.mark.parametrize("composed, why", [
    (None, f"the composite of record has no {OR.NETWORK} case stating the SHA-512 of the parameters the period's "
           "check read"),
    (sha512_of(dict(CERTIFIER, key=ANOTHER)), f"the period's parameters ({OR.CASE}'s params) aren't, by SHA-512, those "
                                               f"its check read (the composite of record's {OR.NETWORK} case's "
                                               "params_sha512)"),
    (sha512_of({**CERTIFIER, "windows": 1}), f"the period's parameters ({OR.CASE}'s params) aren't, by SHA-512, those "
                                              f"its check read (the composite of record's {OR.NETWORK} case's "
                                              "params_sha512)"),
])
def test_auditor_record_doesnt_hold_where_the_periods_params_arent_by_sha512_those_the_composite_of_record_composed(
        tmp_path, composed, why):
    """The period's parameters are the ones its check read only where their SHA-512 over canonical JSON is the one the
    composite of record states for the network certifier's case: where it states none, or that of other parameters,
    whether another key or any field more, auditor-record doesn't hold, though the auditor's record and registration
    state those parameters' key, LMS parameters and session."""
    a = auditor_files(tmp_path / "out")
    got = OR.auditor_record(CERTIFIER, composed, a["certifier_key"], a["site_registration"])
    assert (got["holds"], got["why"]) == (False, why) and "certifier" not in got


#: the circuit files the period's one audit lists, as `records_audit` pins them
FILES = [{"sha512": "e1" * 64, "template": "toy-a"}, {"sha512": "e2" * 64, "template": "toy-b"}]


@pytest.mark.parametrize("how, why", [
    ("the files verified", None),
    ("another file", "audit 0's listed circuit files aren't, by SHA-512, those its audit's sessions were verified "
                     f"against ({OR.PERIOD}'s circuits)"),
    ("one file fewer", "audit 0's listed circuit files aren't, by SHA-512, those its audit's sessions were verified "
                       f"against ({OR.PERIOD}'s circuits)"),
    ("another audit's period", f"audit 0 lists circuit files, and the run's {OR.PERIOD} is another audit's"),
    ("no period", f"the period lists circuit files, and {OR.PERIOD} or its case's commitment can't be read: "
                  "FileNotFoundError"),
    ("the scan's files", "the period lists the scan's circuit files (scan_circuits), and the run publishes no record "
                         "of the files the scan's sessions were verified against"),
])
def test_an_accepted_network_verdict_is_established_only_where_each_listed_circuit_file_is_the_one_verified(
        tmp_path, capsys, how, why):
    """auditor-listing on the run: each circuit file the period's audit lists is, by SHA-512, the one its audit's
    sessions were verified against, V3c's published period (`OR.PERIOD`), whose registration is the one the case's
    commitment holds for that audit. Where it holds, the network certifier's accepted verdict stands, the run of
    record. Where it doesn't, whatever else holds, the verdict is not established, its Lean answer's accept beside it,
    with no bound and nothing conditional on the listing: another file, a file fewer, the period of another audit, no
    published period, or the scan's files listed, of which the run publishes no record."""
    files = FILES if how != "one file fewer" else FILES[:1]
    listed = {"audits": [{"circuits": files}]} | ({"scan_circuits": FILES} if how == "the scan's files" else {})
    out = accepted(tmp_path, dict(PUBLIC, held=["free", "circuit"]), params=listed)
    verified = [FILES[0], dict(FILES[1], sha512="e3" * 64)] if how == "another file" else FILES
    registration = NO_ADVICE if how != "another audit's period" else {"format": NO_ADVICE["format"],
                                                                      "window": {"circuit": "ef" * 64}}
    if how != "no period":
        (out / OR.PERIOD).parent.mkdir(parents=True, exist_ok=True)
        (out / OR.PERIOD).write_text(json.dumps({"registration": registration, "circuits": verified}))
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    got, nv = rec["checked"][OR.AUDITOR_LISTING], rec["verdicts"]["the network certifier"]
    assert (got["is"], got["holds"], got.get("why")) == (OR.CHECKED[OR.AUDITOR_LISTING], why is None, why)
    assert rec["checked"][OR.AUDITOR_RECORD]["holds"] is True
    line = json.loads(capsys.readouterr().out.strip()[7:])
    if why is None:
        assert (got["listed"], nv["verdict"], rec["of_record"]["holds"], "conditional_on" in rec["of_record"]) == (
            2, "accept", True, False)
        assert rec["bound"]["of_record"]["K"] == K and line["verdicts"]["the network certifier"] == "accept"
        return
    assert {k: nv[k] for k in ("protocol", "verdict", "answer", "open", "why")} == {
        "protocol": OR.NETWORK, "verdict": OR.NOT_ESTABLISHED, "answer": "accept", "open": [OR.AUDITOR_LISTING],
        "why": OR.LISTING_UNMET}
    assert rec["bound"] == {"missing": f"the network certifier's verdict is {OR.NOT_ESTABLISHED} ({OR.LISTING_UNMET}): "
                                       "only its accepted answer certifies K"}
    assert rec["periods"] == {"missing": "no bound"}
    assert rec["of_record"] == {"holds": False, "missing": [], "not_accepting": [],
                                "not_established": ["the network certifier"], "not_credited": [],
                                "not_claimed": ["PoUW"], **ONSITE_OF_RECORD}
    assert (line["verdicts"]["the network certifier"], line["of_record"], line["K"]) == (OR.NOT_ESTABLISHED, False, None)


@pytest.mark.parametrize("text, unread", [(None, f"the run has no {OR.CASE}"),
                                          ("{", f"{OR.CASE} can't be read: JSONDecodeError"),
                                          ("[]", f"{OR.CASE} isn't an object")])
def test_a_period_listing_no_circuit_files_holds_auditor_listing_and_one_with_no_case_doesnt(tmp_path, text, unread):
    """A period whose audits list no circuit files holds auditor-listing, listing none, with no published period read;
    with no certifier case, or one that can't be read or isn't an object, the run can't say what the period lists, so
    it doesn't hold, saying why, the certifier's verdict is not established, both auditor-record and auditor-listing
    open, and there is no bound."""
    out = accepted(tmp_path)
    files_held = {"holds": True, "read_by": [OR.ONSITE], "compared": ["circuit", "public"]}
    assert record(out)["checked"][OR.AUDITOR_LISTING] == {"is": OR.CHECKED[OR.AUDITOR_LISTING], "holds": True,
                                                          "listed": 0, "registered_files": files_held}
    if text is None:
        (out / OR.CASE).unlink()
    else:
        (out / OR.CASE).write_text(text)
    rec = record(out)
    assert rec["checked"][OR.AUDITOR_LISTING] == {
        "is": OR.CHECKED[OR.AUDITOR_LISTING], "holds": False, "why": f"no case lists the period's audits: {unread}",
        "registered_files": files_held}
    assert rec["checked"][OR.SCAN_STATE_PINNED] == {
        "is": OR.CHECKED[OR.SCAN_STATE_PINNED], "holds": False, "why": f"no case lists the period's audits: {unread}"}
    nv = rec["verdicts"]["the network certifier"]
    assert (nv["verdict"], nv["open"], nv["why"]) == (
        OR.NOT_ESTABLISHED, [OR.CERTIFIER_KEY, OR.AUDITOR_LISTING, OR.SCAN_STATE_PINNED],
        f"{OR.KEY_OPEN}; {OR.LISTING_UNMET}; {OR.STATE_UNPINNED}")
    assert "missing" in rec["bound"] and rec["periods"] == {"missing": "no bound"}


#: the scan's state as V3c's audit's window commits it in its activations, the entry the period holds to the scan's
#: state (`NetworkCheck.Period.auditStateEntries`), and V3c's audit's registration whose window commits it
STATE_ENTRY = {"value": OR.SCAN_STATE, "root": "5a" * 64, "domain": "d5" * 64, "leaves": 64, "words": 8}
STATED = {**NO_ADVICE, "window": {**NO_ADVICE["window"],
                                  "activations": {"program": "ab" * 64, "values": {OR.SCAN_STATE: STATE_ENTRY}}}}


def scan_state_run(tmp_path: Path, registered: dict | None = None, registration: dict = STATED,
                   accepted_: bool = True, audit: dict = AUDITED, audits: tuple[dict, ...] = (STATED,)) -> Path:
    """An accepted run, both checks held, whose case's commitment holds ``audits`` and whose V3c audit published its
    period of ``registration`` (`OR.PERIOD`), its record ``audit`` (`OR.AUDIT`) and its summary (`OR.AUDIT_SUMMARY`),
    accepting where ``accepted_``, its registered reads opening the window's own values at ``registered``'s entries
    (none where None)."""
    out = accepted(tmp_path, dict(PUBLIC, held=["free", "circuit"]), audits=audits)
    (out / OR.PERIOD).parent.mkdir(parents=True, exist_ok=True)
    (out / OR.PERIOD).write_text(json.dumps({"registration": registration}))
    (out / OR.AUDIT).write_text(json.dumps(audit))
    summary = {"audit": {"accepted": accepted_, "outcome": "accept" if accepted_ else "reject", "why": None}}
    (out / OR.AUDIT_SUMMARY).write_text(json.dumps(
        summary | ({} if registered is None else {"registered": {"own": registered}})))
    return out


NO_OWN = (f"audit 0's window commits {OR.SCAN_STATE} in its activations, and the run's {OR.AUDIT_SUMMARY} states no "
          "entry its registered reads opened it at (registered.own)")


@pytest.mark.parametrize("how, why", [
    ("opened at the entry", None),
    ("another root", f"the entry audit 0's registered reads opened {OR.SCAN_STATE} at isn't, by its root, its window's "
                     "activations entry"),
    ("another domain", f"the entry audit 0's registered reads opened {OR.SCAN_STATE} at isn't, by its domain, its "
                       "window's activations entry"),
    ("another value, root and domain", f"the entry audit 0's registered reads opened {OR.SCAN_STATE} at isn't, by its "
                                       "value and root and domain, its window's activations entry"),
    ("no reads of it", NO_OWN),
    ("no registered reads", NO_OWN),
    ("no summary", f"audit 0's window commits {OR.SCAN_STATE} in its activations, and the run's {OR.AUDIT_SUMMARY} "
                   "can't be read: FileNotFoundError"),
    ("no period", f"audit 0's window commits {OR.SCAN_STATE} in its activations, and the run's {OR.PERIOD} can't be "
                  "read: FileNotFoundError"),
    ("another audit's period", f"audit 0's window commits {OR.SCAN_STATE} in its activations, and the run's "
                               f"{OR.PERIOD} is another audit's"),
    ("not accepted", f"V3c's audit didn't accept ({OR.AUDIT_SUMMARY}'s audit.accepted)"),
    ("reads not reported", "V3c's audit record doesn't report exactly each sampled member's registered reads "
                           f"({OR.AUDIT}'s registered_reads)"),
])
def test_the_network_and_composite_verdicts_are_established_only_where_v3cs_reads_of_the_scans_state_opened_at_its_entry(
        tmp_path, capsys, how, why):
    """scan-state-pinned on the run: V3c's audit's window commits the scan's state in its activations, the entry the
    period holds to the scan's state, and one-stage's registration checks read nothing of the window, so the audit's
    driver pins its registered reads there. Where the entry its registered reads opened the state at (its published
    summary's registered.own) is that activations entry by value, root and domain, the audit accepted and its record
    reported each sampled member's registered reads, the network certifier's verdict and the composite of record's stand
    and the run is of record. Where any differs or its evidence is missing (no entry stated, no summary, no published
    period or another audit's, the audit not accepting, the reads unreported), both are not established, each its Lean
    accept beside it, and there is no bound."""
    pin = {"another root": dict(STATE_ENTRY, root="5b" * 64), "another domain": dict(STATE_ENTRY, domain="d6" * 64),
           "another value, root and domain": {"value": "weights/w@8", "root": "ef" * 64, "domain": "fe" * 64},
           "no reads of it": None}.get(how, STATE_ENTRY)
    out = scan_state_run(tmp_path, None if how == "no registered reads" else {} if pin is None else {OR.SCAN_STATE: pin},
                         registration=NO_ADVICE if how == "another audit's period" else STATED,
                         accepted_=how != "not accepted",
                         audit=audit_record(reported={MEMBER: False}) if how == "reads not reported" else AUDITED)
    if how in ("no summary", "no period"):
        (out / (OR.AUDIT_SUMMARY if how == "no summary" else OR.PERIOD)).unlink()
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    got, v = rec["checked"][OR.SCAN_STATE_PINNED], rec["verdicts"]
    assert (got["is"], got["holds"], got.get("why"), got["audits"]) == (
        OR.CHECKED[OR.SCAN_STATE_PINNED], why is None, why, [0])
    assert rec["checked"][OR.AUDITOR_LISTING]["holds"] is rec["checked"][OR.AUDITOR_RECORD]["holds"] is True
    line = json.loads(capsys.readouterr().out.strip()[7:])
    if why is None:
        assert (v["the network certifier"]["verdict"], v["onsite"]["verdict"], rec["of_record"]["holds"]) == (
            "accept", "accept", True)
        assert rec["bound"]["of_record"]["K"] == K and line["verdicts"]["the network certifier"] == "accept"
        return
    assert {k: v["the network certifier"][k] for k in ("protocol", "verdict", "answer", "open", "why")} == {
        "protocol": OR.NETWORK, "verdict": OR.NOT_ESTABLISHED, "answer": "accept", "open": [OR.SCAN_STATE_PINNED],
        "why": OR.STATE_UNPINNED}
    assert {k: v["onsite"][k] for k in ("protocol", "verdict", "answer", "open", "why")} == {
        "protocol": OR.ONSITE, "verdict": OR.NOT_ESTABLISHED, "answer": "accept", "open": [OR.SCAN_STATE_PINNED],
        "why": OR.STATE_UNPINNED}
    assert rec["bound"] == {"missing": f"the network certifier's verdict is {OR.NOT_ESTABLISHED} ({OR.STATE_UNPINNED}): "
                                       "only its accepted answer certifies K"}
    assert rec["periods"] == {"missing": "no bound"} and v["remote"]["verdict"] == "accept"
    assert rec["of_record"]["holds"] is False
    assert rec["of_record"]["not_established"] == ["the network certifier", "onsite"]
    assert ({k: line["verdicts"][k] for k in ("the network certifier", "onsite")}, line["K"]) == (
        {"the network certifier": OR.NOT_ESTABLISHED, "onsite": OR.NOT_ESTABLISHED}, None)


@pytest.mark.parametrize("values", [{}, {"weights/w@8": dict(STATE_ENTRY, value="weights/w@8")},
                                    {OR.SCAN_STATE: dict(STATE_ENTRY, value="weights/w@8")}])
def test_scan_state_pinned_holds_where_no_audits_window_commits_the_scans_state_in_its_activations(tmp_path, values):
    """No audit's window commits the scan's state in its activations as the period reads it (an entry at that value
    naming it, `auditStateEntries`): none at all, another value's, or an entry at that value naming another, which the
    period doesn't read as the audit's state (and refuses where the audit then has none, `stateTied`). Then the check
    holds, naming no audit, with no summary or period read, and the verdicts stand."""
    audit = {**NO_ADVICE, "window": {**NO_ADVICE["window"], "activations": {"program": "ab" * 64, "values": values}}}
    rec = record(accepted(tmp_path, dict(PUBLIC, held=["free", "circuit"]), audits=(audit,)))
    assert rec["checked"][OR.SCAN_STATE_PINNED] == {"is": OR.CHECKED[OR.SCAN_STATE_PINNED], "holds": True, "audits": []}
    assert rec["verdicts"]["the network certifier"]["verdict"] == rec["verdicts"]["onsite"]["verdict"] == "accept"


def test_two_audits_committing_the_scans_state_need_two_published_periods_and_the_run_publishes_one(tmp_path):
    """The run publishes one V3c audit: where a second audit of the commitment commits the scan's state in its
    activations, its period isn't the one published, and the check doesn't hold, naming both."""
    other = {**STATED, "window": {**STATED["window"], "circuit": "ef" * 64}}
    rec = record(scan_state_run(tmp_path, {OR.SCAN_STATE: STATE_ENTRY}, audits=(STATED, other)))
    got = rec["checked"][OR.SCAN_STATE_PINNED]
    assert (got["holds"], got["audits"], got["why"]) == (
        False, [0, 1], f"audit 1's window commits {OR.SCAN_STATE} in its activations, and the run's {OR.PERIOD} is "
                       "another audit's")


def test_a_composite_of_record_withheld_on_two_counts_names_both(tmp_path):
    """The composite of record not established for a registered file that isn't θ's is withheld again where V3c's
    reads of the scan's state didn't open at its entry: both are open, each with why."""
    out = scan_state_run(tmp_path, {OR.SCAN_STATE: dict(STATE_ENTRY, root="5b" * 64)})
    Path(json.loads(auditor_args(out)["sampled_case"].read_text())["blobs"][0]).write_bytes(b"another circuit\n")
    v = record(out)["verdicts"]
    assert (v["onsite"]["verdict"], v["onsite"]["open"], v["onsite"]["why"]) == (
        OR.NOT_ESTABLISHED, [OR.REGISTERED_FILES, OR.SCAN_STATE_PINNED], f"{OR.FILES_UNMET}; {OR.STATE_UNPINNED}")
    assert v["the network certifier"]["open"] == [OR.AUDITOR_LISTING, OR.SCAN_STATE_PINNED]


NOT_THETAS = "the {} file the case registers (its blob {}) isn't, by SHA-512, θ's proofs.{}"
#: the public file changed after the draw, its header kept
CHANGED = b"toy public header\nanother public body"
#: the public file as a5 completed it once the tie held, a member's not re-laid: its header completed, its body the staged one
COMPLETED = b"toy completed header\n" + FILES_REGISTERED[1].partition(b"\n")[2]
NOT_PINNED = ("the public file the case registers isn't, by SHA-512, one --public-pin pinned at staging before the draw or "
              "completed once the tie held")


@pytest.mark.parametrize("how, why", [
    ("both bound", None),
    ("public unbound", None),
    ("another circuit file", NOT_THETAS.format("circuit", 0, "circuit")),
    ("another public file", NOT_THETAS.format("public", 1, "public")),
    ("public unbound, another circuit file", NOT_THETAS.format("circuit", 0, "circuit")),
    ("another θ", f"--sampled-proofs-case's params aren't, by SHA-512, those each composite that reads {SAMPLED_V1}'s "
                  "case states (its params_sha512)"),
    ("no case", f"the run's composites read {SAMPLED_V1}'s case, and the run gave no --sampled-proofs-case"),
    ("a file gone", "the case's circuit file (its blob 0) can't be read: FileNotFoundError"),
    ("a public file changed after the draw", NOT_PINNED),
    ("a public file completed once the tie held, the pin's completed entry with it", None),
    ("a public file completed once the tie held, the pin's completed entry another file's", NOT_PINNED),
    ("a public file completed once the tie held, its body not the staged body",
     "--public-pin's completed file for m0 isn't completed from its staged one: its body isn't the staged body"),
    ("a public file completed once the tie held, on a member with no staged entry",
     "--public-pin's m1 has a completed file and no staged one"),
    ("a public file changed after the draw, the pin with it",
     "--public-pin's bodies aren't, by their leaf layer, those the registration the case commits to holds (R4)"),
    ("no pin", "the run gave no --public-pin, the public file's SHA-512 pinned at staging before the draw"),
    ("another commitment", "the case's commitment isn't the one each composite that reads the case states"),
    ("θ naming another layout", "θ's layout (proofs.members) isn't the one --public-pin fixed before the draw: its member "
                                "0 is ['m0', 2], the pin's ['m0', 1]"),
    ("θ naming no layout", "θ names no layout (proofs.members), so the drawn member isn't held to the one --public-pin "
                           "fixed"),
])
def test_the_registered_files_condition_establishes_the_verdicts_it_bears_on_only_where_each_file_is_the_one_theta_names(
        tmp_path, how, why):
    """auditor-listing's registered-files condition, the remote theorem's on its sampled leg: each file remote's sampled
    proofs case registers is, by SHA-512, the one its params θ name, θ the one the composite reading the case states.
    Where θ names both files and both are its, every verdict stands and the run is of record. Where θ names no public
    file, the public file isn't bound: sampled proofs' verdict is not established, its accept beside it, crediting no
    member, so the run isn't of record, the circuit file still compared, and the evidence's digest of the public file
    never stands in for θ's. Where θ names the public file, it is also one V3c's audit pinned (``--public-pin``): as
    staged before the draw, or as a5 completed it once the tie held, its completed entry, its body the member's staged
    body (`onsite_record.completed_from`; an instance-major member's re-laid,
    `test_a_completed_public_file_is_held_to_its_members_staged_body`). A public file changed after the draw, θ naming
    the changed one, isn't the pin's, nor is one beside a completed entry of another file, a completed entry whose body
    isn't its member's staged body or one on a member with no staged entry, and a pin rewritten to it isn't, by its
    bodies' leaf layer, the registration's the case commits to, which the coins committed to. θ's layout is the one the
    pin fixed before the draw (`onsite_record.layout_pinned`): θ naming another or none doesn't hold. Where a file isn't
    θ's (the circuit file or the public file), θ isn't the one the composite states, the run
    gave no case or a file it registers is gone, the public file isn't the pinned one, or the run gave no pin or its
    case commits to a registration the composite doesn't state, auditor-listing doesn't hold: the network certifier's
    verdict, the composite of record's and sampled proofs', its leg there, are not established, each its Lean accept
    beside it, with no bound and no member credited, and the run isn't of record."""
    out = accepted(tmp_path)
    circuit, public = (sha512_hex(b) for b in FILES_REGISTERED)
    unbound = how.startswith("public unbound")
    changed = how.startswith("a public file changed after the draw")
    completed = how.startswith("a public file completed once the tie held")
    if unbound:
        sampled_case(out, {"circuit": circuit, "registered": None}, evidence={"proofs": {"public": public}})
    elif changed or completed:
        kept = COMPLETED if completed and not how.endswith("not the staged body") else CHANGED
        sampled_case(out, {"circuit": circuit, "public": sha512_hex(kept), "registered": None, "members": LAYOUT})
    elif how.startswith("θ naming"):
        sampled_case(out, {"circuit": circuit, "public": public, "registered": None}
                     | ({"members": [["m0", 2]]} if how.endswith("another layout") else {}))
    with_composite(out, "remote-network-no-pouw", WITHOUT, WITHOUT_IDS)
    with_onsite(out)
    case = auditor_args(out)["sampled_case"]
    blobs = json.loads(case.read_text())["blobs"]
    if how.endswith("another circuit file"):
        Path(blobs[0]).write_bytes(b"another circuit\n")
    elif how == "another public file":
        Path(blobs[1]).write_bytes(b"another public file")
    elif how == "another θ":
        sampled_case(out, {"circuit": circuit, "public": public, "registered": {"reads": []}})
    elif how == "a file gone":
        Path(blobs[0]).unlink()
    elif changed:
        Path(blobs[1]).write_bytes(CHANGED)
        if how.endswith("the pin with it"):
            auditor_args(out)["public_pin"].write_text(json.dumps(pin_of(CHANGED)))
    elif completed:
        Path(blobs[1]).write_bytes(kept)
        entry = b"toy completed header\nanother file's body" if how.endswith("another file's") else kept
        pin = pin_of(FILES_REGISTERED[1])
        pin["members"]["m0"]["relaid"] = False
        done = {"file": sha512_hex(entry), "body": sha512_hex(entry.partition(b"\n")[2])}
        if how.endswith("no staged entry"):
            pin["members"]["m1"] = {"k": 1, "completed": done}
        else:
            pin["members"]["m0"]["completed"] = done
        auditor_args(out)["public_pin"].write_text(json.dumps(pin))
    elif how == "another commitment":
        case.write_text(json.dumps(json.loads(case.read_text()) | {"commitment": registration_of(pin_of(CHANGED))}))
    rec = record(out, **({"sampled_case": None} if how == "no case" else {"public_pin": None} if how == "no pin"
                         else {}))
    v, listing = rec["verdicts"], rec["checked"][OR.AUDITOR_LISTING]
    files, line = listing["registered_files"], OR.summary(rec)
    assert (files["holds"], files["read_by"], files.get("why"), files.get("unbound")) == (
        why is None, [WITHOUT, OR.ONSITE], why, OR.PUBLIC_UNBOUND if unbound else None)
    assert (listing["holds"], listing.get("why"), listing["listed"]) == (why is None, why, 0)
    if why is None or how in ("another public file", "a file gone", "no pin", "another commitment") or changed or completed:
        assert files["compared"] == (["circuit"] if unbound else ["circuit", "public"])
    withheld = (["sampled proofs"] if unbound or why else []) + ([] if why is None else ["the network certifier",
                                                                                        "onsite"])
    assert rec["of_record"]["not_established"] == withheld and rec["of_record"]["holds"] is (not withheld)
    bears = ("sampled proofs", "the network certifier", "onsite")
    assert {k: line["verdicts"][k] for k in bears} == {k: OR.NOT_ESTABLISHED if k in withheld else "accept" for k in bears}
    assert line["of_record"] is (not withheld) and v["remote"]["verdict"] == "accept"
    assert v["sampled proofs"]["credit"] == (UNESTABLISHED if withheld else CREDITED)
    if withheld:
        assert v["sampled proofs"] == {
            "protocol": SAMPLED_V1, "verdict": OR.NOT_ESTABLISHED, "answer": "accept",
            "from": "composite/remote-network-no-pouw/composite.json cases", "open": [OR.REGISTERED_FILES],
            "why": OR.PUBLIC_UNBOUND if unbound else OR.FILES_UNMET, "credit": UNESTABLISHED}
        assert rec["sampled_proofs"]["missing"] == (
            f"sampled proofs' verdict is {OR.NOT_ESTABLISHED}: only its accepted answer states a bound")
    if why is None:
        assert "K" in rec["bound"] and line["K"] is not None
        return
    assert {k: v["the network certifier"][k] for k in ("verdict", "answer", "open", "why")} == {
        "verdict": OR.NOT_ESTABLISHED, "answer": "accept", "open": [OR.AUDITOR_LISTING], "why": OR.FILES_UNMET}
    assert v["onsite"] == {"protocol": WITHOUT, "verdict": OR.NOT_ESTABLISHED, "answer": "accept",
                           "from": "composite/remote-network-no-pouw/composite.json", "open": [OR.REGISTERED_FILES],
                           "why": OR.FILES_UNMET}
    assert rec["bound"] == {"missing": f"the network certifier's verdict is {OR.NOT_ESTABLISHED} ({OR.FILES_UNMET}): "
                                       "only its accepted answer certifies K"}
    assert rec["periods"] == {"missing": "no bound"} and line["K"] is None


RELAY_NOT = ("--public-pin's completed file for m0 isn't completed from its staged one: its body isn't the staged body "
             "re-laid port-major")


@pytest.mark.parametrize("how, relaid, done, why", [
    ("no draw, its staged file", True, None, None),
    ("completed, its staged body", False, "staged", None),
    ("completed, re-laid", True, "re-laid", None),
    ("completed, another body re-laid", True, "another re-laid", RELAY_NOT),
    ("completed, re-laid, a ref not the identity's", True, "a ref off", RELAY_NOT),
    ("completed, its staged body, the member re-laid", True, "staged", RELAY_NOT),
    ("completed, re-laid, its header naming no instances", True, "re-laid",
     "--public-pin's completed file for m0 isn't completed from its staged one: its header names no instances: KeyError"),
    ("completed, re-laid, the member not re-laid", False, "re-laid",
     "--public-pin's completed file for m0 isn't completed from its staged one: its body isn't the staged body"),
    ("completed, the pin not saying whether it re-lays", None, "re-laid",
     "--public-pin doesn't say whether completing m0 re-lays its body (its relaid), so its completed file isn't held to "
     "its staged one"),
    ("completed, no staged entry", True, "re-laid", "--public-pin's m0 has a completed file and no staged one"),
])
def test_a_completed_public_file_is_held_to_its_members_staged_body(tmp_path, how, relaid, done, why):
    """A drawn member's completed file (a5's ``completed`` entry, pinned once the tie held) is accepted only completed
    from the same member's staged body, which R4 holds to the registration: a member not re-laid keeps the staged body
    under its completed header, and an instance-major member's (the staged entry's ``relaid``) is the staged body re-laid
    port-major under identity refs (`flock_inputs.relay`), its instances read off the completed header. A member with no
    draw is held to its staged pair alone. Any other completed entry is refused, naming the member."""
    from verity.ml.windows import flock_inputs as FI

    n, ports = 4, 2
    body = bytes(i % 251 for i in range(n * ports * FI.BC_BYTES))
    laid = FI.relay(body, n, ports)
    bodies = {"staged": body, "re-laid": laid, "another re-laid": FI.relay(body[::-1], n, ports),
              "a ref off": laid[:-4] + (n - 2).to_bytes(4, "little")}
    staged = b'{"instances": 4, "staged": true}\n' + body
    entry = {"k": 0, "file": sha512_hex(staged), "body": sha512_hex(body)} | ({} if relaid is None else {"relaid": relaid})
    public = tmp_path / "pub-4.bin"
    public.write_bytes(staged)
    if done is not None:
        head = b'{"completed": true}\n' if how.endswith("naming no instances") else b'{"instances": 4, "completed": true}\n'
        public.write_bytes(head + bodies[done])
        entry["completed"] = {"file": sha512_hex(public.read_bytes()), "body": sha512_hex(bodies[done])}
    if how.endswith("no staged entry"):
        entry = {"k": 0, "completed": entry["completed"]}
    pin = {"format": "verity/public-pin/v1", "leaf_layer": sha512_hex(canonical_json({"m0": sha512_hex(body)})),
           "members": {"m0": entry}}
    (tmp_path / "public-pin.json").write_text(json.dumps(pin))
    reg = registration_of(pin)
    assert OR.pinned(tmp_path / "public-pin.json", reg, {reg}, public) == why


MOVED = "θ's layout (proofs.members) isn't the one --public-pin fixed before the draw: its member {}"


@pytest.mark.parametrize("members, why", [
    ([["T/c0", 2], ["T/c1", 3]], None),
    ([["T/c0", 3], ["T/c1", 2]], MOVED.format("0 is ['T/c0', 3], the pin's ['T/c0', 2]")),
    ([["T/c0", 1], ["T/c1", 4]], MOVED.format("0 is ['T/c0', 1], the pin's ['T/c0', 2]")),
    ([["T/c1", 3], ["T/c0", 2]], MOVED.format("0 is ['T/c1', 3], the pin's ['T/c0', 2]")),
    ([["T/c0", 2]], MOVED.format("1 is none, the pin's ['T/c1', 3]")),
    ([["T/c0", 2], ["T/c1", 3], ["U/c0", 1]], MOVED.format("2 is ['U/c0', 1], the pin's none")),
    (None, "θ names no layout (proofs.members), so the drawn member isn't held to the one --public-pin fixed"),
])
def test_thetas_layout_is_held_to_the_one_a5_pinned_before_the_draw(tmp_path, members, why):
    """θ's layout (proofs.members), by which v2 finds the drawn member, is the one a5 pinned at staging: each member's
    key and ``n`` in unit order (`a5.pin_layout`, by ``k``). A boundary moved by one unit either way, two members
    swapped, a member left out or one added, or no layout, is refused, naming the first member that differs."""
    pin = tmp_path / "public-pin.json"
    pin.write_text(json.dumps({"members": {"T/c1": {"k": 1, "n": 3}, "T/c0": {"k": 0, "n": 2}}}))
    assert OR.layout_pinned(pin, members) == why


@pytest.mark.parametrize("entries, why", [
    ({"T/c0": {"k": 0}}, "its member T/c0 states no int k and positive int n"),
    ({"T/c0": {"k": 0, "n": 0}}, "its member T/c0 states no int k and positive int n"),
    ({"T/c0": {"k": 1, "n": 2}}, "its members' k aren't 0 to the member count"),
])
def test_a_pin_stating_no_layout_holds_no_theta(tmp_path, entries, why):
    """A pin from before a5 pinned each member's ``n``, or whose members' ``k`` aren't 0 to the member count, states no
    layout, so no θ's is held to it."""
    pin = tmp_path / "public-pin.json"
    pin.write_text(json.dumps({"members": entries}))
    assert OR.layout_pinned(pin, [["T/c0", 2]]) == (
        "--public-pin states no layout of the window at staging (each member's k and n), which θ's proofs.members is "
        f"held to: ValueError: {why}")


@pytest.mark.parametrize("audit, credited, not_credited, missing", [
    (AUDITED, [MEMBER], {}, None),
    (audit_record((MEMBER, "ToyNorm"), {MEMBER: True, "ToyNorm": False}), [MEMBER], {"ToyNorm": OR.UNREPORTED}, None),
    (audit_record((MEMBER, "ToyNorm"), {MEMBER: True}), [MEMBER], {"ToyNorm": OR.UNREPORTED}, None),
    (audit_record((MEMBER,), {MEMBER: True, "ToyNorm": True}), [MEMBER], {"ToyNorm": OR.UNLISTED}, None),
    (audit_record((), {}, ("ToyNorm",)), [], {"ToyNorm": OR.UNCOMPLETED}, None),
    ({"accepted": True, "registered_reads": {"ok": True, "chunks": {MEMBER: {"ok": True}}}}, [], None,
     OR.NO_COMPLETION),
    (None, [], None, f"the run published no audit record ({OR.AUDIT})"),
])
def test_sampled_proofs_credits_only_members_whose_registered_file_was_completed_and_names_each_other(
        tmp_path, capsys, audit, credited, not_credited, missing):
    """Sampled proofs credits a drawn member only where its audit completed the member's --registered file at the
    window's committed roots and the member's Lean verdict of record accepted it reporting exactly its reads: a member
    completed but not so reported, one reported whose file the audit's record lists no completion of, and one the audit
    couldn't complete are each listed as not credited, by name, with why, and an audit record stating no completion,
    or none published, credits no member. The run is of record only where every drawn member is credited."""
    out = accepted(tmp_path, dict(PUBLIC, held=["free", "circuit"]))
    if audit is None:
        (out / OR.AUDIT).unlink()
    else:
        (out / OR.AUDIT).write_text(json.dumps(audit))
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    c = rec["verdicts"]["sampled proofs"]["credit"]
    assert (c["credited"], c.get("not_credited"), c.get("missing"), c["is"]) == (credited, not_credited, missing,
                                                                                 OR.CREDIT)
    of = rec["of_record"]
    assert (of["not_credited"], of.get("uncredited"), of["holds"]) == (
        sorted(not_credited or ()), missing, not not_credited and missing is None)
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["not_credited"], line["of_record"]) == (of["not_credited"], of["holds"])
    assert all(x in OR.CREDIT for x in ("every port", "committed roots", "before any session", "checkComplete",
                                        "not credited, by name", "a visible failure"))


#: the PoUW part of a COINS line where the run served PoUW H1 (served_verdicts.sh's H1_COINS under H1_GATE)
H1_SERVED = ("PoUW H1: each window's coins from a session of its own on vy-challenger-1's gate (peer toy-peer, T 63, "
             "S 16), a development run under the lane's development key: it credits nothing")


def test_pouw_credits_no_freed_compute_whatever_its_row_states_and_is_not_claimed_by_name(tmp_path, capsys):
    """A PoUW row stating freed compute: the PoUW verdict carries none of its figures, so no freed compute is in any of
    the record's totals; the record credits none and doesn't claim it, by name in the record and its line, beside the
    PoUW piece each COINS line says the run served."""
    work = {"protocol": "verity/pouw/period/v1", "owner": "compute-accounting", "verdict": "accept", "reason": None,
            "quantity": {"freed_flops": 123456789}, "answer": {"public": {"freed": 987654321}}}
    out = served_out(tmp_path, work=work, coins=None)
    (out / "verdicts.txt").write_text(f"COINS sampled proofs: {GATE}; {H1_SERVED}; the bandwidth audit's: "
                                      "vy-auditor-2's bw_party, inside PoUS's piece\n" + coins_line(GATE) + "\n")
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    assert rec["verdicts"]["PoUW"] == {"protocol": "verity/pouw/period/v1", "verdict": "accept",
                                       "from": "record/verdicts.json protocols.work", "not_claimed": OR.POUW_OPEN}
    assert rec["pouw_freed_compute"] == {"open": True, "credited": None, "served": [H1_SERVED, "PoUW H1: not served "
                                                                                              "(H1_GATE empty)"],
                                         "from": "verdicts.txt COINS", "is": OR.POUW}
    assert not {123456789, 987654321} & set(numbers(rec))
    assert json.loads(capsys.readouterr().out.strip()[7:])["not_claimed"] == [OR.DAY_OPEN, OR.POUW_OPEN]
    assert all(x in OR.POUW for x in ("OnsiteFreed", "noFoldRound", "eps1-sm120-h1fx1", "credits nothing", "Pearl+"))


@pytest.mark.parametrize("work, verdict", [
    ({"verdict": "missing", "why": "development run, credits nothing: toy", "development": {"verdict": "accept"}},
     {"verdict": "missing", "why": "development run, credits nothing: toy", "development": True}),
    ({"protocol": "verity/pouw/period/v1", "verdict": "reject", "reason": "toy"},
     {"protocol": "verity/pouw/period/v1", "verdict": "reject", "reason": "toy"}),
])
def test_whether_the_run_is_of_record_reads_no_pouw_verdict_which_stays_listed_as_it_is(tmp_path, capsys, work, verdict):
    """An otherwise accepted run, every member credited, whose PoUW row is a development piece's (missing, crediting
    nothing) or a rejection: the run is of record, since the record claims nothing of PoUW (`NOT_CLAIMED`), and PoUW's
    verdict is listed as its row states it, missing and development or rejected, and not claimed."""
    out = accepted(tmp_path)
    report = json.loads((out / "record" / "verdicts.json").read_text())
    report["protocols"]["work"] = {"owner": "compute-accounting", **work}
    (out / "record" / "verdicts.json").write_text(json.dumps(report))
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    assert rec["verdicts"]["PoUW"] == verdict | {"from": "record/verdicts.json protocols.work",
                                                 "not_claimed": OR.POUW_OPEN}
    gone = ["PoUW"] * (verdict["verdict"] == "missing")
    assert rec["of_record"] == {"holds": True, "missing": gone, "not_accepting": ["PoUW"] * (not gone),
                                "not_established": [], "not_credited": [], "not_claimed": ["PoUW"], **ONSITE_OF_RECORD}
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["of_record"], line["verdicts"]["PoUW"]) == (True, verdict["verdict"])
    assert record(with_onsite(out, sampled="reject"))["of_record"]["holds"] is False


def memory_row(key: str, verdict: str) -> dict:
    """Memory's row ``key`` of the window report answering ``verdict``, as `served_verdicts.row_of` writes it."""
    return {"protocol": f"toy/{key}/v0", "owner": "memory-accounting", "verdict": verdict,
            "reason": None if verdict == "accept" else f"toy: {key} refused", "commitment": "ab" * 64,
            "quantity": {"bytes": 123456789} if verdict == "accept" else None, "delta": None, "epsilon": None, "runs": []}


def with_memory(out: Path, rows: dict[str, dict | None]) -> Path:
    """``out`` with its window report's memory rows ``rows``, by row name: each the row given, about `NODE` where it
    answers (`served`) and its source real, none where None."""
    f = out / "record" / "verdicts.json"
    report = json.loads(f.read_text())
    for key, row in rows.items():
        report["protocols"].pop(key, None)
        report["servers"].pop(key, None)
        report["sources"].pop(key, None)
        if row is not None:
            report["protocols"][key], report["servers"][key], report["sources"][key] = row, served(row), "real"
    f.write_text(json.dumps(report))
    return out


#: memory's rows of the window report, by the verdict that reads each
MEMORY_ROWS = {"PoUS": "space", "the bandwidth audit": "bandwidth", "the wipe": "wipe"}
#: a run's memory rows: none, each accepting, each refusing, each as a product without them drops it, and one of each
NO_MEMORY = dict.fromkeys(MEMORY_ROWS.values())
MEMORY_ACCEPTING = {key: memory_row(key, "accept") for key in MEMORY_ROWS.values()}
MEMORY_REFUSING = {key: memory_row(key, "reject") for key in MEMORY_ROWS.values()}
MEMORY_DROPPED = {key: {"verdict": "dropped", "owner": "memory-accounting", "why": f"not run: toy drops {key}"}
                  for key in MEMORY_ROWS.values()}
MEMORY_MIXED = {"space": MEMORY_ACCEPTING["space"], "bandwidth": MEMORY_REFUSING["bandwidth"], "wipe": None}


def test_memorys_three_verdicts_are_among_the_records_and_read_the_rows_memory_names():
    assert OR.MEMORY == tuple(MEMORY_ROWS) and set(OR.MEMORY) < set(OR.VERDICTS)
    assert {k: OR.ROWS[k] for k in OR.MEMORY} == MEMORY_ROWS
    assert {k: SV.NAMES[key] for k, key in MEMORY_ROWS.items()} == {k: k for k in MEMORY_ROWS}


@pytest.mark.parametrize("name, key", list(MEMORY_ROWS.items()))
@pytest.mark.parametrize("row, verdict", [
    ("accept", {"verdict": "accept"}),
    ("reject", {"verdict": "reject"}),
    ("error", {"verdict": "error"}),
    ("dropped", {"verdict": "missing"}),
    (None, {"verdict": "missing"}),
])
def test_each_memory_verdict_is_listed_as_its_row_states_it_and_missing_where_the_run_has_none(
        tmp_path, capsys, name, key, row, verdict):
    """Each of memory's three verdicts reads its own row of the window report: an accepting row's accept and a refusing
    row's refusal with its reason, under the row's protocol; missing with the row's why where the run's product drops
    it, and missing, naming the row, where the report has none. Each says it is listed, the two other rows' verdicts
    stay missing, no figure of the row reaches the record, and the run's line prints the verdict beside the others."""
    given = None if row is None else MEMORY_DROPPED[key] if row == "dropped" else memory_row(key, row)
    out = with_memory(served_out(tmp_path), NO_MEMORY | {key: given})
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    src = {"from": f"record/verdicts.json protocols.{key}", "listed": OR.LISTED}
    expect = {"accept": {"protocol": f"toy/{key}/v0", "verdict": "accept"},
              "reject": {"protocol": f"toy/{key}/v0", "verdict": "reject", "reason": f"toy: {key} refused"},
              "error": {"protocol": f"toy/{key}/v0", "verdict": "error", "reason": f"toy: {key} refused"},
              "dropped": {"verdict": "missing", "why": f"not run: toy drops {key}"},
              None: {"verdict": "missing", "why": f"the window report has no {key} row"}}[row]
    assert expect["verdict"] == verdict["verdict"] and rec["verdicts"][name] == expect | src
    others = {k: rec["verdicts"][k] for k in OR.MEMORY if k != name}
    assert others == {k: {"verdict": "missing", "why": f"the window report has no {MEMORY_ROWS[k]} row",
                          "from": f"record/verdicts.json protocols.{MEMORY_ROWS[k]}", "listed": OR.LISTED}
                      for k in others}
    assert 123456789 not in numbers(rec)
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert list(line["verdicts"]) == sorted(OR.VERDICTS)
    assert {k: line["verdicts"][k] for k in OR.MEMORY} == dict.fromkeys(OR.MEMORY, "missing") | {name: verdict["verdict"]}


def test_a_development_memory_row_is_missing_and_says_so(tmp_path):
    row = {"verdict": "missing", "owner": "memory-accounting", "why": "development run, credits nothing: toy",
           "development": {"verdict": "accept"}}
    rec = record(with_memory(served_out(tmp_path), NO_MEMORY | {"space": row}))
    assert rec["verdicts"]["PoUS"] == {"verdict": "missing", "why": row["why"], "development": True,
                                       "from": "record/verdicts.json protocols.space", "listed": OR.LISTED}


def of_record_today(tmp_path: Path, run: str) -> tuple[Path, dict]:
    """A toy run and the ``of_record`` the record states of it with no memory verdict among its verdicts: one of record
    under onsite's composite, one whose composite of record rejects, one whose sampled proofs leg does, one with no
    composite of record, one of record under the composite without PoUW, and a rehearsal's."""
    no = {"missing": [], "not_accepting": [], "not_established": [], "not_credited": [], "not_claimed": ["PoUW"]}
    if run == "rehearsal":
        return served_out(tmp_path), {
            "holds": False, "missing": ["PoUW", "inference-only", "remote", "the network certifier", "onsite"],
            "not_accepting": [], "not_established": ["sampled proofs"], "not_credited": [], "not_claimed": ["PoUW"],
            "uncredited": OR.UNESTABLISHED_CREDIT, **ONSITE_OF_RECORD}
    if run == "no composite of record":
        return accepted(tmp_path, onsite=False), {
            "holds": False, **no, "missing": ["onsite"],
            "not_established": ["sampled proofs", "the network certifier"], "conditional_on": [OR.CERTIFIER_KEY],
            "uncredited": OR.UNESTABLISHED_CREDIT, **ONSITE_OF_RECORD}
    out = accepted(tmp_path)
    if run == "of record":
        return out, {"holds": True, **no, **ONSITE_OF_RECORD}
    if run == "sampled proofs rejecting":
        return with_onsite(out, sampled="reject"), {"holds": False, **no, "not_accepting": ["sampled proofs"],
                                                    **ONSITE_OF_RECORD}
    if run == "onsite rejecting":
        f = out / "composite" / "remote-network" / "composite.json"
        f.write_text(json.dumps(json.loads(f.read_text()) | {
            "missing": [OR.NETWORK], "verdict": {"id": OR.ONSITE, "verdict": "reject", "refusal": "components"}}))
        return out, {"holds": False, **no, "not_accepting": ["onsite"], **ONSITE_OF_RECORD}
    assert run == "of record without PoUW"
    with_composite(out, "remote", OR.REMOTE, REMOTE_IDS, "reject", missing=(OR.POUW_ID,), inference_only="reject")
    with_composite(out, "remote-network", OR.ONSITE, (OR.NETWORK, *REMOTE_IDS), "reject", missing=(OR.POUW_ID,))
    with_composite(out, "remote-network-no-pouw", WITHOUT, WITHOUT_IDS)
    return out, {"holds": True, **no, "composite": WITHOUT, "not_accepting": ["remote"],
                 "not_claimed": ["PoUW", "remote"], "verified_compute": "none"}


@pytest.mark.parametrize("run", ["of record", "of record without PoUW", "sampled proofs rejecting", "onsite rejecting",
                                 "no composite of record", "rehearsal"])
@pytest.mark.parametrize("memory", [NO_MEMORY, MEMORY_REFUSING, MEMORY_DROPPED, MEMORY_MIXED, MEMORY_ACCEPTING],
                         ids=["absent", "refusing", "dropped", "mixed", "accepting"])
def test_whether_the_run_is_of_record_reads_none_of_memorys_three_verdicts(tmp_path, capsys, run, memory):
    """A run is of record exactly when it is with no memory verdict among the record's: its three rows absent, refusing,
    dropped, accepting or one of each, ``of_record`` is, field for field, what it states of that run without them, a
    run of record staying one though each refuses and a run not of record staying none though each accepts; it names
    none of the three in any of its lists, and the run's line says the same of the run."""
    out, today = of_record_today(tmp_path, run)
    assert record(out)["of_record"] == today
    assert main([str(with_memory(out, memory))]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    assert rec["of_record"] == today
    assert not set(OR.MEMORY) & {k for x in today.values() if isinstance(x, list) for k in x}
    states = {k: ("missing" if (memory[key] or {"verdict": "dropped"})["verdict"] == "dropped"
                  else memory[key]["verdict"]) for k, key in MEMORY_ROWS.items()}
    assert {k: rec["verdicts"][k]["verdict"] for k in OR.MEMORY} == states
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["of_record"], {k: line["verdicts"][k] for k in OR.MEMORY}) == (today["holds"], states)


#: a server that isn't `NODE`, and what a window report can name for a row
OTHER = "toy-other-server"
WHERE = {"the node": NODE, "another server": OTHER, "no server": None}
#: the verdicts that are a row of the window report's, by their row; sampled proofs' is its composite leg
ROW_READ = {name: key for name, key in OR.ROWS.items() if name != "sampled proofs"}
#: the verdicts read from a composite, which the window report names no server for
COMPOSITE_READ = ("inference-only", "remote", "onsite")


def about(out: Path, servers: dict[str, str | None]) -> Path:
    """``out`` with its window report naming ``servers``' server for each of those rows, none where None."""
    f = out / "record" / "verdicts.json"
    report = json.loads(f.read_text())
    report["servers"] |= servers
    f.write_text(json.dumps(report))
    return out


def answering(out: Path, key: str, verdict: str) -> Path:
    """``out`` with its window report's row ``key`` answering ``verdict``, with a reason."""
    f = out / "record" / "verdicts.json"
    report = json.loads(f.read_text())
    report["protocols"][key] |= {"verdict": verdict, "reason": f"toy: {key} answered {verdict}"}
    f.write_text(json.dumps(report))
    return out


def every_row_accepting(tmp_path: Path, **kw) -> Path:
    """A run of record (`accepted`) whose window report has all six rows, each accepting about `NODE`."""
    return with_memory(accepted(tmp_path, **kw), MEMORY_ACCEPTING)


def elsewhere(key: str, verdict: str, src: str | None = None) -> str:
    """Why the row ``key`` (the window report's unless ``src``), answering ``verdict`` about `OTHER`, is missing."""
    src = src or f"record/verdicts.json protocols.{key}"
    return OR.ELSEWHERE.format(key=key, src=src, verdict=verdict, server=OTHER)


def unserved(key: str, src: str | None = None) -> str:
    """Why the accepting row ``key`` (the window report's unless ``src``) no server is named for is missing."""
    return OR.UNSERVED.format(key=key, src=src or f"record/verdicts.json protocols.{key}")


def off_node(base: dict, name: str, why: str) -> dict:
    """The verdict ``name`` reads where its row is no answer about the node: missing with ``why``, from its row, with
    what the record says of it whatever it reads (``listed``, ``not_claimed``), as ``base``, its read about the node."""
    return {"verdict": "missing", "why": why, "from": f"record/verdicts.json protocols.{OR.ROWS[name]}",
            **{k: base[name][k] for k in ("listed", "not_claimed") if k in base[name]}}


def test_the_nine_verdicts_are_the_six_read_from_the_window_reports_rows_and_the_three_read_from_composites():
    assert set(OR.VERDICTS) == set(ROW_READ) | {"sampled proofs"} | set(COMPOSITE_READ)
    assert set(OR.ROWS.values()) == set(SV.NAMES) and {SV.NAMES[key]: key for key in SV.NAMES} == OR.ROWS


@pytest.mark.parametrize("name, key", list(ROW_READ.items()))
@pytest.mark.parametrize("where", list(WHERE))
def test_a_verdict_read_from_a_row_is_accept_only_where_the_report_says_its_answer_is_about_its_node(
        tmp_path, capsys, name, key, where):
    """Each verdict read from a row of the window report, its row accepting: accept where the report's ``servers``
    names its node for the row; missing where it names another server, naming it, and where it names none, each said in
    its why; every other verdict reads as it did, and the run's line says the same."""
    out = every_row_accepting(tmp_path)
    base = record(out)["verdicts"]
    assert {k: base[k]["verdict"] for k in ROW_READ} == dict.fromkeys(ROW_READ, "accept")
    assert main([str(about(out, {key: WHERE[where]}))]) == 0
    v = json.loads((out / "record" / "onsite.json").read_text())["verdicts"]
    expect = {"the node": base[name],
              "another server": off_node(base, name, elsewhere(key, "accept")),
              "no server": off_node(base, name, unserved(key))}[where]
    assert v == base | {name: expect}
    assert OTHER in elsewhere(key, "accept")
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert line["verdicts"][name] == ("accept" if where == "the node" else "missing")


@pytest.mark.parametrize("name, key", list(ROW_READ.items()))
@pytest.mark.parametrize("answer", ["reject", "error"])
@pytest.mark.parametrize("where", ["another server", "no server"])
def test_a_row_that_doesnt_accept_is_missing_about_another_server_and_reads_as_it_states_naming_none(
        tmp_path, name, key, answer, where):
    """A row that rejects or errs: about another server it is no answer about this node, so its verdict is missing,
    naming the server and the answer; naming no server it reads as it states, as the window report composes it."""
    out = answering(every_row_accepting(tmp_path), key, answer)
    here = record(out)["verdicts"]
    assert (here[name]["verdict"], here[name]["reason"]) == (answer, f"toy: {key} answered {answer}")
    v = record(about(out, {key: WHERE[where]}))["verdicts"]
    assert v[name] == (here[name] if where == "no server" else off_node(here, name, elsewhere(key, answer)))
    assert {k: v[k] for k in ROW_READ if k != name} == {k: here[k] for k in ROW_READ if k != name}


@pytest.mark.parametrize("where", list(WHERE))
@pytest.mark.parametrize("leg", ["accept", None])
def test_sampled_proofs_verdict_is_its_composite_leg_whatever_server_the_integrity_row_is_about(tmp_path, leg, where):
    """Sampled proofs' verdict is its leg in the composite of record, which the window report names no server for: with
    that leg accepting it reads the same whatever server the report names for the integrity row. Where the composite of
    record has no such leg, the row's answer is beside a verdict not established only where it is about the node, and
    the verdict is missing where the row is about another server or none."""
    out = every_row_accepting(tmp_path, onsite=leg is not None)
    base = record(out)["verdicts"]["sampled proofs"]
    assert base["verdict"] == ("accept" if leg else OR.NOT_ESTABLISHED) and (leg or base["answer"] == "accept")
    v = record(about(out, {"integrity": WHERE[where]}))["verdicts"]["sampled proofs"]
    if leg or where == "the node":
        assert v == base
    else:
        none = f"its case is read from {OR.ONSITE}'s composite, and the run has none"
        assert (v["verdict"], v["why"]) == ("missing", none)
        assert "answer" not in v


@pytest.mark.parametrize("name", COMPOSITE_READ)
@pytest.mark.parametrize("where", list(WHERE))
def test_a_verdict_read_from_a_composite_reads_the_same_whatever_server_the_report_names_for_its_rows(
        tmp_path, name, where):
    """Inference-only's, remote's and onsite's verdicts are read from the run's composites, never from a row of the
    window report, whose ``servers`` has no entry for them: each reads the same with every row about the node, about
    another server and about none."""
    out = every_row_accepting(tmp_path)
    base = record(out)["verdicts"][name]
    assert base["verdict"] == "accept" and "record/verdicts.json" not in base["from"]
    report = json.loads((out / "record" / "verdicts.json").read_text())
    assert not set(report["servers"]) & set(COMPOSITE_READ) and set(report["servers"]) == set(OR.ROWS.values())
    assert record(about(out, dict.fromkeys(report["servers"], WHERE[where])))["verdicts"][name] == base


@pytest.mark.parametrize("servers", [None, [], "toy"])
def test_a_report_stating_no_servers_reads_accept_from_none_of_its_rows(tmp_path, servers):
    out = every_row_accepting(tmp_path)
    base = record(out)["verdicts"]
    f = out / "record" / "verdicts.json"
    report = json.loads(f.read_text())
    report.pop("servers")
    f.write_text(json.dumps(report | ({} if servers is None else {"servers": servers})))
    v = record(out)["verdicts"]
    assert {k: v[k] for k in ROW_READ} == {k: off_node(base, k, unserved(key)) for k, key in ROW_READ.items()}


#: the record's figures read from a row of the window report beyond its verdict, by the row each reads
FIGURES = {"sampled proofs' δ": "integrity", "PoUW's δ": "work", "the placement": "communication",
           "sampled proofs' guarantee": "integrity", "the bound": "communication"}


def figures(rec: dict) -> dict:
    """Each of `FIGURES` as ``rec`` states it; a δ is its term of the composite's error."""
    terms = {t["subprotocol"]: t for t in rec["composite_error"]["terms"]}
    return {"sampled proofs' δ": terms["sampled proofs"], "PoUW's δ": terms["PoUW"], "the placement": rec["placement"],
            "sampled proofs' guarantee": rec["sampled_proofs"], "the bound": rec["bound"]}


def as_missing(out: Path, key: str, why: str) -> Path:
    """``out`` with its window report's row ``key`` a missing row saying ``why``, about no server."""
    f = out / "record" / "verdicts.json"
    report = json.loads(f.read_text())
    report["protocols"][key], report["servers"][key] = {"verdict": "missing", "why": why}, None
    f.write_text(json.dumps(report))
    return out


@pytest.mark.parametrize("figure, key", list(FIGURES.items()))
@pytest.mark.parametrize("where", list(WHERE))
def test_a_row_that_is_no_answer_about_the_node_gives_the_record_no_figure(tmp_path, figure, key, where):
    """Each figure the record reads from a row, its row accepting: sampled proofs' and PoUW's δ in the composite's
    error, the network certifier's placement and bound, and sampled proofs' guarantee. About the node the figure is the
    row's; about another server, or naming none, it is what the record states where that row is missing: no δ, which
    the composite's stated sum then leaves out, no placement, no bound and no guarantee. The window report's writer
    always names its node for the integrity row, so that row's cases here are a hand-made report."""
    out = with_deltas(every_row_accepting(tmp_path), integrity="1/2", work=2.0 ** -20)
    base = record(out)
    stated = {"integrity": Fraction(1, 2), "work": Fraction(1, 2 ** 20), "communication": Fraction(PUBLIC["eps"])}
    assert base["composite_error"]["stated"] == str(sum(stated.values()))
    assert all("delta" in figures(base)[x] for x in ("sampled proofs' δ", "PoUW's δ"))
    assert figures(base)["the placement"] and "per_window" in base["sampled_proofs"] and base["bound"]["K"] == K
    rec = record(about(out, {key: WHERE[where]}))
    if where == "the node":
        assert rec == base
        return
    got = figures(rec)[figure]
    why = unserved(key) if where == "no server" else elsewhere(key, "accept")
    assert got == figures(record(as_missing(out, key, why)))[figure] != figures(base)[figure]
    if figure.endswith("δ"):
        assert got == {"subprotocol": SV.NAMES[key], "from": figures(base)[figure]["from"],
                       "missing": "its row states no δ"}
        assert rec["composite_error"]["stated"] == str(sum(stated.values()) - stated[key])
    elif figure == "the placement":
        assert got is None
    elif figure == "the bound":
        assert set(got) == {"missing"} and why in got["missing"]
        assert rec["composite_error"]["stated"] == str(sum(stated.values()) - stated[key])
    else:
        assert "per_window" not in got and "pooled" not in got


@pytest.mark.parametrize("key", list(OR.ROWS.values()))
@pytest.mark.parametrize("where", ["another server", "no server"])
def test_the_record_of_a_row_that_is_no_answer_about_the_node_is_the_record_of_that_row_missing(tmp_path, key, where):
    """One place decides it: with any of the six rows accepting about another server, or naming none, the whole record,
    every verdict and every figure, is the record of the same run with that row missing and saying why. The integrity
    row's cases are a hand-made report, the writer always naming its node for that row."""
    out = with_deltas(every_row_accepting(tmp_path), integrity="1/2", work=2.0 ** -20)
    base = record(out)
    rec = record(about(out, {key: WHERE[where]}))
    why = unserved(key) if where == "no server" else elsewhere(key, "accept")
    assert rec == record(as_missing(out, key, why)) != base


def brought(key: str, src: str | None = None, verdict: str = "accept") -> str:
    """Why the row ``key`` (the window report's unless ``src``), answering ``verdict`` about `NODE`, whose source is a
    fixture's, is missing."""
    src = src or f"record/verdicts.json protocols.{key}"
    return OR.BROUGHT.format(key=key, src=src, verdict=verdict, source="fixture")


def sourced(out: Path, sources: dict[str, str | None] | None) -> Path:
    """``out`` with its window report stating ``sources``' source for each of those rows, no entry where None; with no
    ``sources`` at all where ``sources`` is None, as a report written before the writer stated them."""
    f = out / "record" / "verdicts.json"
    report = json.loads(f.read_text())
    if sources is None:
        report.pop("sources")
    else:
        report["sources"] = {k: x for k, x in (report["sources"] | sources).items() if x is not None}
    f.write_text(json.dumps(report))
    return out


@pytest.mark.parametrize("name, key", list(ROW_READ.items()))
@pytest.mark.parametrize("source", ["real", "fixture", "unstated", "no sources"])
def test_an_accept_about_the_node_is_the_runs_own_only_where_its_source_is_real(tmp_path, name, key, source):
    """Each verdict read from a row of the window report, its row accepting about the node: where the report's
    ``sources`` says the answer is a fixture's, brought from a recorded run, the verdict is missing and says so, and the
    whole record is that of the row missing; where it says real, states no source for the row or has no ``sources`` at
    all, the writer's default being real, the record is the same record."""
    out = with_deltas(every_row_accepting(tmp_path), integrity="1/2", work=2.0 ** -20)
    base = record(out)
    stated = {"real": {key: "real"}, "fixture": {key: "fixture"}, "unstated": {key: None}, "no sources": None}[source]
    rec = record(sourced(out, stated))
    if source != "fixture":
        assert rec == base
        return
    assert rec["verdicts"] == base["verdicts"] | {name: off_node(base["verdicts"], name, brought(key))}
    assert rec == record(as_missing(out, key, brought(key))) != base


@pytest.mark.parametrize("name, key", list(ROW_READ.items()))
@pytest.mark.parametrize("verdict", ["reject", "error"])
def test_an_answer_about_the_node_brought_from_a_recorded_run_is_missing_whatever_it_says(tmp_path, name, key, verdict):
    """A reject or an error about the node whose source the report says is a fixture's is no answer of this run's
    either: the verdict is missing, naming the answer, where a real one reads as it states, and the whole record is that
    of the row missing, so the record states no reject or error the run didn't produce."""
    out = answering(every_row_accepting(tmp_path), key, verdict)
    here = record(out)
    rec = record(sourced(out, {key: "fixture"}))
    why = brought(key, verdict=verdict)
    assert here["verdicts"][name]["verdict"] == verdict and verdict in why and "accept" not in why
    assert rec["verdicts"] == here["verdicts"] | {name: off_node(here["verdicts"], name, why)}
    assert rec == record(as_missing(out, key, why)) != here


@pytest.mark.parametrize("name, key", list(ROW_READ.items()))
@pytest.mark.parametrize("verdict", ["accept", "reject", "error"])
def test_a_row_naming_no_server_reads_as_it_does_whatever_its_source(tmp_path, name, key, verdict):
    """A row the report names no server for is read as the report composes it, whatever source it states: an accept is
    missing because no server is named (`OR.UNSERVED`), not because it was brought, and a reject or an error reads as it
    states."""
    out = about(answering(every_row_accepting(tmp_path), key, verdict), {key: None})
    here = record(out)
    v = here["verdicts"][name]
    assert (v["verdict"], v.get("why")) == (("missing", unserved(key)) if verdict == "accept" else (verdict, None))
    assert record(sourced(out, {key: "fixture"})) == here


@pytest.mark.parametrize("leg", ["accept", None])
def test_sampled_proofs_verdict_is_its_composite_leg_whatever_source_the_report_states_for_the_integrity_row(
        tmp_path, leg):
    """The writer always states the integrity row's answer real, so this is a hand-made report: with its source a
    fixture's, sampled proofs' verdict is still its leg in the composite of record, and where the composite of record
    has no such leg the row's answer is not beside a verdict not established: the verdict is missing."""
    out = every_row_accepting(tmp_path, onsite=leg is not None)
    base = record(out)["verdicts"]["sampled proofs"]
    v = record(sourced(out, {"integrity": "fixture"}))["verdicts"]["sampled proofs"]
    assert base["verdict"] == ("accept" if leg else OR.NOT_ESTABLISHED)
    assert (v == base) if leg else (v["verdict"] == "missing" and "answer" not in v)


@pytest.mark.parametrize("verdict", ["accept", "reject", "error"])
@pytest.mark.parametrize("source", ["real", "fixture"])
def test_the_period_checks_answer_a_remote_run_takes_is_held_to_the_source_its_piece_states(tmp_path, verdict, source):
    """The row the record takes from the period check's answer where remote's report drops the network certifier, its
    piece naming the report's node: an answer its piece's `SERVED` record says is a fixture's is missing whatever it
    says, naming the answer, with no bound and the run not of record; one it says is real is the run's own and reads as
    it answers (an answer that isn't a core answer is the row's error)."""
    rec = record(remote_dropped(tmp_path, verdict, PIECE_NAMES["the node"], source))
    v, src = rec["verdicts"]["the network certifier"], "pieces/certifier/answer.json"
    reads = verdict if source == "real" else "missing"
    assert v["verdict"] == reads and (reads != "missing" or v["why"] == brought("communication", src, verdict))
    assert rec["of_record"]["holds"] is (reads == "accept") and ("K" in rec["bound"]) is (reads == "accept")


@pytest.mark.parametrize("run", ["of record", "of record without PoUW"])
@pytest.mark.parametrize("key", list(OR.ROWS.values()))
def test_a_run_of_record_whose_network_certifiers_accept_is_a_fixtures_is_not_of_record(tmp_path, run, key):
    """``of_record`` follows from the verdicts: with the network certifier's accept a fixture's, a run of record is not
    of record, that verdict among its missing ones; PoUW's, which it doesn't read, is named missing and the run stays of
    record; memory's three and the integrity row, whose answer is no verdict, change nothing."""
    out, today = of_record_today(tmp_path, run)
    assert today["holds"] and record(with_memory(out, MEMORY_ACCEPTING))["of_record"] == today
    rec = record(sourced(out, {key: "fixture"}))
    assert rec["of_record"] == today | {"communication": {"holds": False, "missing": ["the network certifier"]},
                                        "work": {"missing": ["PoUW"]}}.get(key, {})


@pytest.mark.parametrize("run", ["of record", "of record without PoUW"])
@pytest.mark.parametrize("key", list(OR.ROWS.values()))
@pytest.mark.parametrize("where", ["another server", "no server"])
def test_a_run_of_record_with_a_needed_verdicts_row_off_its_node_is_not_of_record(tmp_path, capsys, run, key, where):
    """``of_record`` is computed from the verdicts: a run of record whose network certifier's row, the one row a verdict
    it reads is read from, is about another server or names none is not of record, that verdict among its missing ones.
    Moving any other row leaves it of record: PoUW's verdict, which it doesn't read, is named missing and nothing else
    of it changes; memory's three it neither reads nor names; and sampled proofs' verdict is its leg in the composite
    of record, never the integrity row."""
    out, today = of_record_today(tmp_path, run)
    assert today["holds"] and record(with_memory(out, MEMORY_ACCEPTING))["of_record"] == today
    assert main([str(about(out, {key: WHERE[where]}))]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    needed = key == OR.ROWS["the network certifier"]
    assert rec["of_record"] == today | {"communication": {"holds": False, "missing": ["the network certifier"]},
                                        "work": {"missing": ["PoUW"]}}.get(key, {})
    assert ("missing" in rec["bound"]) is needed
    assert json.loads(capsys.readouterr().out.strip()[7:])["of_record"] is not needed


def test_sampled_proofs_guarantee_is_its_windows_epsilon_at_delta_beside_its_pooled_count_over_each_period(
        tmp_path, capsys):
    """subset:1 over a window's 16 proof blocks at δ = 2^-40: the window's bound is 15, ε = 15/16, and pooled at most
    ⌊16 · ln 2^40⌋ = 443 incorrect proof blocks over any number of windows (IntegrityHorizon). Each monitoring period
    states that count for the node, and each link's at δ / links summed for a datacenter and the world, with its share of
    the period's proof blocks at a window an execution phase of 63/50 s, scaled by its rule. The line states the
    window's ε beside each pooled cell, never alone."""
    out = accepted(tmp_path)
    assert main([str(out)]) == 0
    g = json.loads((out / "record" / "onsite.json").read_text())["sampled_proofs"]
    assert (g["is"], g["refused"], g["from"], "conditional_on" in g) == (
        OR.SAMPLED, OR.MULTI, "record/verdicts.json protocols.integrity.answer", False)
    assert g["per_window"] == {"units": N, "law": {"subset": 1}, "delta": str(DELTA), "bound": N - 1, "epsilon": "15/16",
                           "statement": "Flock.Guarantees.EndToEndHidden"}
    p, phase = g["pooled"], Fraction(63 * 20, 1000)
    assert (p["statement"], p["is"], p["q"], p["incorrect_at_most"]) == (
        "Flock.Guarantees.IntegrityHorizon", OR.POOLED, "1/16", 443)
    assert (p["error"]["delta"], p["gates"], p["window_s"]) == (
        str(DELTA), {"missing": OR.NO_GATES}, {"seconds": str(phase), "is": OR.WINDOW_S})
    links = {"node": 1, "datacenter": LB.datacenter_links(), "world": LB.world_links()}
    for name, seconds in LB.PERIODS.items():
        w = math.ceil(Fraction(seconds) / phase)
        assert (p[name]["seconds"], p[name]["sampling_blocks"]) == (seconds, w)
        for where, n in links.items():
            f = n * math.floor(N * math.log(n * 2 ** 40))
            assert p[name][where] == {"links": n, "sampling_blocks": w * n, "proof_blocks": N * w * n, "incorrect_at_most": f,
                                      "share": f / (N * w * n), "how": "scaled",
                                      "rule": "window-to-period" if where == "node" else where}
    assert {k: r["sources"] for k, r in p["rules"].items()} == {
        "window-to-period": [OR.SAMPLED_FROM, OR.FROM_SCHEDULE],
        "datacenter": [census_line("deployments.json", "machines")],
        "world": [census_line("adversary.json", "total_flop", "flop_per_s_per_h100", "monitoring_period_s"),
                  census_line("deployments.json", "gpus_per_machine")]}
    text, line = (out / "record" / "onsite.json").read_text(), capsys.readouterr().out.strip()
    assert SV.public_doc(text) == text and SV.public_line(line) == line
    assert (g["member"], json.loads(line[7:])["open_in_lean"]) == (OR.MEMBER_OPEN, [OR.MEMBER_OPEN])
    line = json.loads(line[7:])["sampled_proofs"]
    assert line["per_window"] == {"epsilon": "15/16", "delta": str(DELTA), "units": N, "law": {"subset": 1}}
    assert line["pooled"] == {"incorrect_at_most": 443, **{
        name: {where: {k: p[name][where][k] for k in ("incorrect_at_most", "share")} for where in links}
        for name in LB.PERIODS}}


def test_the_pooled_count_is_the_floor_of_ln_links_over_delta_over_q_past_a_floats_precision():
    """⌊ln(links/δ)/q⌋: at q = 1 and δ = 2^-40, ⌊40 ln 2⌋ = 27; at q = 1/16 over 12,500 links, ⌊16 ln(12,500 · 2^40)⌋ =
    594; and at δ a part in 10^20 above e^-k, where ln(1/δ) is k less a part in 10^20 that a float's log can't see, k − 1,
    and below it, k."""
    assert (OR.horizon(Fraction(1), DELTA), OR.horizon(Fraction(1, 16), DELTA, 12_500)) == (27, 594)
    with decimal.localcontext() as c:
        c.prec = 60
        for k in (1, 7, 30):
            e = decimal.Decimal(-k).exp()
            above, below = (Fraction(e * (1 + s * decimal.Decimal(10) ** -20)) for s in (1, -1))
            assert (OR.horizon(Fraction(1), above), OR.horizon(Fraction(1), below)) == (k - 1, k)


def test_a_run_with_no_certifier_case_states_the_pooled_count_and_no_period_cells(tmp_path):
    g = record(with_onsite(served_out(tmp_path)))["sampled_proofs"]
    assert (g["per_window"]["epsilon"], g["pooled"]["incorrect_at_most"]) == ("15/16", 443)
    assert g["pooled"]["periods"] == {"missing": f"no {OR.CASE}: a window's length is an execution phase's, its params' "
                                                 "schedule (`WINDOW_S`)"}
    assert not set(LB.PERIODS) & set(g["pooled"])
    assert OR.guarantee(g)["pooled"] == {"incorrect_at_most": 443, "periods": g["pooled"]["periods"]}


def test_a_draw_of_a_member_of_each_stratum_has_no_pooled_figure_and_is_named_refused(tmp_path):
    """stratified:1, H1_TIE's default, draws one member of each template's stratum: its window's bound is stated, but
    IntegrityHorizon pools only a uniform k-subset, so it has no pooled figure, and the record names such a draw as one
    a run of record doesn't make, which writes no sampled-proofs case and which Lean's per-member form refuses."""
    g = record(with_onsite(served_out(tmp_path, draw=STRATIFIED1, bound=14)))["sampled_proofs"]
    assert g["per_window"]["law"] == {"law": "stratified", "strata": [{"name": "A", "size": 8, "k": 1},
                                                                  {"name": "B", "size": 8, "k": 1}]}
    assert (g["per_window"]["epsilon"], g["pooled"]) == ("7/8", {"missing": OR.NOT_POOLED})
    assert all(s in OR.MULTI for s in ("stratified:1", "`a5.v2_proof`", "MULTI_REFUSED", "subset:1 under --zk"))
    assert OR.guarantee(g)["pooled"] == {"missing": OR.NOT_POOLED}


@pytest.mark.parametrize("coins", [None, "a local development challenger session"])
def test_sampled_proofs_guarantee_rests_on_its_coins_and_is_missing_but_for_an_accepted_answer(tmp_path, coins):
    """The guarantee reads the records audit's answer's profile and bound only once sampled proofs' verdict, its leg in
    the composite of record, accepted: rejected there, or not established, it states no bound."""
    out = with_onsite(served_out(tmp_path, coins=coins))
    g = record(out)["sampled_proofs"]
    assert (g["conditional_on"], g["per_window"]["epsilon"]) == ([OR.COINS_OPEN], "15/16")
    f = out / "record" / "verdicts.json"
    report = json.loads(f.read_text())
    report["protocols"]["integrity"]["answer"] = {}
    f.write_text(json.dumps(report))
    assert record(out)["sampled_proofs"]["error"] == ("its answer's profile or public lacks a field: KeyError: "
                                                         "'profile'")
    g = record(with_onsite(out, sampled="reject"))["sampled_proofs"]
    assert g["missing"] == "sampled proofs' verdict is reject: only its accepted answer states a bound"
    assert "per_window" not in g and OR.guarantee(g) == {"missing": g["missing"]}
    (out / "composite" / "remote-network" / "composite.json").unlink()
    assert record(out)["sampled_proofs"]["missing"] == (
        f"sampled proofs' verdict is {OR.NOT_ESTABLISHED}: only its accepted answer states a bound")


@pytest.mark.usefixtures("verified")
@pytest.mark.parametrize("n2, scan, named", [("subset:8", "all", ["the period draw"]),
                                             ("all", "stratified:4", ["the scan's draw"]),
                                             (None, None, ["the period draw", "the scan's draw"])])
def test_a_chain_draw_that_doesnt_take_everything_is_open_under_live_coins_by_its_name(tmp_path, capsys, n2, scan,
                                                                                         named):
    """The network certifier's period draw or its scan's draw taking K of what it draws from (subset:K, stratified:k),
    or a run that published no chain report: each such draw is open under live-coins by its name, though sampled
    proofs' COINS line names the live challenger, so the network certifier's figure of record is open on them though
    both checks are held and every premise discharged, and it, its cells, the composite's error and
    the run's being of record are conditional on the coins; sampled proofs' verdict, on live coins, isn't."""
    out = accepted(tmp_path, dict(PUBLIC, held=["free", "circuit"]), draws=None if n2 is None else chain_report(n2, scan))
    assert main([str(out)]) == 0
    rec, cond = json.loads((out / "record" / "onsite.json").read_text()), [OR.COINS_OPEN]
    assert (rec["coins"]["open"], rec["coins"]["open_draws"], rec["coins"]["sampled_proofs"]) == (True, named, [GATE])
    assert rec["coins"]["draws"].get("missing") == (None if n2 else OR.NO_CHAIN_REPORT)
    of = rec["bound"]["of_record"]
    assert (of["open"], of["conditional_on"], of["K"]) == (cond, cond, K)
    assert rec["periods"]["conditional_on"] == rec["composite_error"]["conditional_on"] == cond
    assert rec["of_record"]["conditional_on"] == cond and "conditional_on" not in rec["verdicts"]["sampled proofs"]
    assert json.loads(capsys.readouterr().out.strip()[7:])["conditional_on"] == cond


@pytest.mark.usefixtures("verified")
def test_one_premise_open_leaves_the_checks_K_of_record_open_on_it_by_name_with_nothing_added(
        tmp_path, monkeypatch, capsys):
    """Both checks held, the coins live and every premise discharged but padding-out-of-free: the figure of record is
    the check's K, open by that name alone and conditional on nothing, with no ℓ_a + 1 added for the tied audit, and
    its run's cell reads open, naming it, in the record and its line."""
    monkeypatch.setitem(OR.PREMISES, "padding-out-of-free", UNNAMED["padding-out-of-free"])
    out = accepted(tmp_path, dict(PUBLIC, held=["free", "circuit"]))
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    of, run = rec["bound"]["of_record"], rec["periods"]["run"]
    assert (of["open"], of["K"], of["tight"], "conditional_on" in of) == (["padding-out-of-free"], K, OR.TIGHT, False)
    assert (run["bits"], run["how"], run["open"]) == (K, "open", ["padding-out-of-free"])
    assert "periods_tight" not in rec and "conditional_on" not in rec["periods"]
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["K"], line["open"], "K_tight" in line) == (K, ["padding-out-of-free"], False)


def test_a_remote_run_whose_report_drops_the_certifier_takes_its_period_checks_answer_and_else_the_drop(tmp_path):
    """Remote's report drops the network certifier whatever its pieces hold; with the split certifier behind the front
    its period check's answer, beside its case under OUT/pieces/certifier, is the verdict and the bound, as onsite's
    report composes it, where its piece names the report's node as the server it is about."""
    out = accepted(tmp_path)
    report = json.loads((out / "record" / "verdicts.json").read_text())
    report["protocols"]["communication"] = SV.dropped_row("communication", "remote")
    report["servers"]["communication"] = None
    (out / "record" / "verdicts.json").write_text(json.dumps(report))
    rec = record(out)
    assert rec["verdicts"]["the network certifier"] == {
        "verdict": "missing", "why": report["protocols"]["communication"]["why"],
        "from": "record/verdicts.json protocols.communication"}
    assert "missing" in rec["bound"]
    (out / OR.CASE.parent / "answer.json").write_text(json.dumps(answer(PUBLIC)))
    (out / OR.CASE.parent / SV.SERVED).write_text(json.dumps({"server": NODE}))
    rec = record(out)
    assert rec["verdicts"]["the network certifier"] == {"protocol": OR.NETWORK, "verdict": "accept",
                                                        "from": "pieces/certifier/answer.json"}
    assert rec["bound"]["K"] == K and rec["placement"] == {"trusted-host": "the certifier on the gate's host"}
    assert rec == record(accepted(tmp_path / "onsite")) | {"verdicts": rec["verdicts"]}


#: the composite each run of record reads remote's verdict from and the one it reads as its composite of record: its
#: directory under OUT/composite and the product it states
READ_FROM = {"of record": {"remote": ("remote", OR.REMOTE), "of record": ("remote-network", OR.ONSITE)},
             "of record without PoUW": {"remote": ("remote", OR.REMOTE),
                                        "of record": ("remote-network-no-pouw", WITHOUT)}}


def unread_composite(out: Path, name: str, how: str) -> list[str]:
    """``out`` with its composite NAME one the record doesn't read, and the paths of the files that make it so, in the
    order the record names them: a second composite.json stating its product, in composite/NAME-again (``two of one
    product``), or its own stating none (``no product``)."""
    f = out / "composite" / name / "composite.json"
    if how == "no product":
        f.write_text(json.dumps({k: x for k, x in json.loads(f.read_text()).items() if k != "product"}))
        return [f"composite/{name}/composite.json"]
    (out / "composite" / f"{name}-again").mkdir()
    (out / "composite" / f"{name}-again" / "composite.json").write_text(f.read_text())
    return [f"composite/{name}/composite.json", f"composite/{name}-again/composite.json"]


def saying(x, instead: dict[str, str]):
    """The JSON value ``x`` with each of ``instead``'s texts, wherever a string of it says one, replaced by its
    other."""
    if isinstance(x, str):
        for old, new in instead.items():
            x = x.replace(old, new)
        return x
    if isinstance(x, dict):
        return {k: saying(v, instead) for k, v in x.items()}
    return [saying(v, instead) for v in x] if isinstance(x, list) else x


def test_the_composites_the_record_reads_are_one_a_product_each_stating_its_product(tmp_path):
    """`OR.composites`: one composite a product is read, by the product it states, and none is refused; of a product
    two state, neither is read; one that states no product, or isn't an object, is read for none; each refusal names
    its files."""
    out, _ = of_record_today(tmp_path, "of record")
    found, refused = OR.composites(out)
    assert {p: path for p, (path, _) in found.items()} == {
        OR.REMOTE: "composite/remote/composite.json", OR.ONSITE: "composite/remote-network/composite.json"}
    assert refused == {}
    two = unread_composite(out, "remote", "two of one product")
    found, refused = OR.composites(out)
    assert set(found) == {OR.ONSITE} and set(refused) == {OR.REMOTE} and len(refused[OR.REMOTE][1]) == 2
    assert refused[OR.REMOTE][0] == OR.TWO_OF_ONE.format(n=2, product=OR.REMOTE, paths=", ".join(two))
    none = unread_composite(out, "remote-network", "no product")
    found, refused = OR.composites(out)
    assert found == {} and set(refused) == {OR.REMOTE, None}
    assert refused[None][0] == OR.NO_PRODUCT.format(paths=none[0], s="s", these="it")
    (out / "composite" / "x").mkdir()
    (out / "composite" / "x" / "composite.json").write_text("[]")
    assert OR.composites(out)[1][None][0] == OR.NO_PRODUCT.format(
        paths=f"{none[0]}, composite/x/composite.json", s="", these="them")


@pytest.mark.parametrize("run", list(READ_FROM))
@pytest.mark.parametrize("which", ["remote", "of record"])
@pytest.mark.parametrize("how", ["two of one product", "no product"])
def test_a_composite_the_record_doesnt_read_gives_it_no_verdict_and_no_figure(tmp_path, run, which, how):
    """A run of record whose remote composite, or whose composite of record, the record doesn't read (`OR.composites`):
    the run has two composites stating that product, or its one states no product. The record is still written, and it
    is the record of the same run without those files, each verdict that finds no composite of the product saying why
    and naming the files: so no verdict and no figure is read from them. Only remote's under the composite without
    PoUW, which that run doesn't read, leaves the run of record.

    Two cases aren't the record of the run without the files, since which composite is the composite of record is read
    from the composites. Two composites of the product without PoUW: that product stays the composite of record and
    the record reads none of it. A composite without PoUW stating no product: the run has no composite stating that
    product, so its composite of record is onsite's, as for a run with no composite without PoUW; no verdict is
    missing, so none names the file."""
    out, today = of_record_today(tmp_path, run)
    base = record(out)
    assert base["of_record"] == today and today["holds"]
    name, product = READ_FROM[run][which]
    paths = unread_composite(out, name, how)
    rec = record(out)
    two = how == "two of one product"
    why = (OR.TWO_OF_ONE.format(n=2, product=product, paths=", ".join(paths)) if two
           else OR.NO_PRODUCT.format(paths=paths[0], s="s", these="it"))
    changed = {k: v["verdict"] for k, v in rec["verdicts"].items() if v != base["verdicts"][k]}
    if (run, which) == ("of record without PoUW", "of record") and not two:
        assert rec["of_record"]["composite"] == OR.ONSITE and why not in json.dumps(rec)
        assert changed == {"sampled proofs": "accept", "inference-only": "reject", "remote": "reject",
                           "onsite": "reject"}
        assert not rec["of_record"]["holds"] and all(product not in str(v) for v in rec["verdicts"].values())
        return
    not_read = {"remote": ["remote"] + ["inference-only"] * (run == "of record"),
                "of record": ["onsite"] + ["inference-only"] * (run != "of record")}[which]
    beside = dict.fromkeys(("sampled proofs", "the network certifier"), OR.NOT_ESTABLISHED) if which != "remote" else {}
    assert changed == dict.fromkeys(not_read, "missing") | beside
    for k in not_read:
        assert rec["verdicts"][k]["why"] == (why if two else f"{base_none(k, product)}; {why}")
    assert all(p in rec["verdicts"][k]["why"] for p in paths for k in not_read)
    assert which == "remote" or why in rec["verdicts"]["sampled proofs"]["why"]
    assert rec["of_record"]["holds"] is ((run, which) == ("of record without PoUW", "remote"))
    assert rec["of_record"].get("composite") == today.get("composite")
    if (run, which) == ("of record without PoUW", "of record"):
        return
    for p in paths:
        (out / p).unlink()
    nones = {base_none(k, product) for k in (*not_read, "inference-only")}
    assert rec == saying(record(out), {none: why if two else f"{none}; {why}" for none in nones})


def base_none(name: str, product: str) -> str:
    """What the verdict ``name``, read from ``product``'s composite, says where the run has no such composite."""
    return {"remote": "remote's composite didn't run: the run has no composite/remote/composite.json",
            "onsite": f"the run has no composite of {product}",
            "inference-only": f"its case is read from {product}'s composite, and the run has none"}[name]


#: what the period check's piece under OUT/pieces/certifier states of the server its answer is about, as
#: `served_verdicts.piece_server` reads it: its `SERVED` record's server, else the node `pieces` ran it on (`RAN`)
PIECE_NAMES = {
    "the node": {SV.SERVED: NODE},
    "the node, run on another": {SV.SERVED: NODE, SV.RAN: OTHER},
    "the node it ran on": {SV.RAN: NODE},
    "another server": {SV.SERVED: OTHER, SV.RAN: NODE},
    "another server it ran on": {SV.RAN: OTHER},
    "no server": {},
}


REFUSAL = "toy: the period check refused"


def remote_dropped(tmp_path: Path, verdict: str, names: dict[str, str], source: str | None = None) -> Path:
    """A run of record under remote's product: its window report drops the network certifier, naming no server for it,
    and its period check's answer under OUT/pieces/certifier is ``verdict`` (``error``: one that isn't a core answer,
    which `served_verdicts.piece_row` reads as the row's error), its piece's records naming ``names`` and, where given,
    its `SERVED` record the answer's ``source``."""
    out = accepted(tmp_path)
    f = out / "record" / "verdicts.json"
    report = json.loads(f.read_text())
    report["protocols"]["communication"] = SV.dropped_row("communication", "remote")
    report["servers"]["communication"] = None
    f.write_text(json.dumps(report))
    refusal = {"protocol": OR.NETWORK, "verdict": "reject", "reason": REFUSAL, "commitment": "ab" * 64, "public": None,
               "profile": None}
    said = {"accept": answer(PUBLIC), "reject": refusal, "error": {"protocol": OR.NETWORK, "verdict": "accept"}}
    (out / OR.CASE.parent / "answer.json").write_text(json.dumps(said[verdict]))
    for name, server in names.items():
        stated = {"source": source} if source and name == SV.SERVED else {}
        (out / OR.CASE.parent / name).write_text(json.dumps({"server": server} | stated))
    return out


@pytest.mark.parametrize("names", list(PIECE_NAMES))
@pytest.mark.parametrize("verdict", ["accept", "reject"])
def test_the_period_checks_answer_a_remote_run_takes_is_held_to_the_server_its_piece_names(
        tmp_path, capsys, verdict, names):
    """The row the record takes from the period check's answer where remote's report drops the network certifier is
    held to the same rule as a row of the report, by the server its piece names (`served_verdicts.piece_server`: its
    `SERVED` record's, else the node it ran on): about the report's node it reads as it answers, and an accept makes the
    run of record with its bound; about another server it is missing, naming the server and the answer; an accept naming
    no server is missing, and a reject naming none reads as it states. Only the accept about the node is of record."""
    out = remote_dropped(tmp_path, verdict, PIECE_NAMES[names])
    assert main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    src = "pieces/certifier/answer.json"
    here = {"protocol": OR.NETWORK, "verdict": verdict, "from": src}
    here |= {"reason": REFUSAL} if verdict == "reject" else {}
    off = {"verdict": "missing", "from": src}
    expect = (here if names.startswith("the node") or (names, verdict) == ("no server", "reject") else
              off | {"why": unserved("communication", src) if names == "no server" else
                     elsewhere("communication", verdict, src)})
    assert rec["verdicts"]["the network certifier"] == expect
    holds = expect == here and verdict == "accept"
    assert rec["of_record"]["holds"] is holds and ("K" in rec["bound"]) is holds
    assert rec["of_record"]["missing"] == ["the network certifier"] * (expect["verdict"] == "missing")
    assert rec["of_record"]["not_accepting"] == ["the network certifier"] * (expect["verdict"] == "reject")
    assert json.loads(capsys.readouterr().out.strip()[7:])["of_record"] is holds


def test_a_run_with_no_window_report_writes_no_record(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
    assert "record/verdicts.json is missing" in capsys.readouterr().err and not (tmp_path / "record").exists()


def published(cap: int) -> dict:
    """The run's advice registration as published, its advice padded to its compiler's cap ℓ_a = ``cap``."""
    return SV.public_registration({"row_bits": 256, "rows": -(-cap // 256),
                                   "binding": {"advice": {"root": "ab" * 64, "bits": cap}},
                                   "compiler": {"name": "c", "digest": "d", "cap_bits": cap}})


#: two runs' serving settings, (S, MAX_TOKENS), each with the ℓ_a its registration states
SETTINGS = ((1, 1, 1328), (16, 4, 7432))


def test_the_callers_serving_settings_label_the_record_and_its_line_and_are_missing_where_it_states_none(
        tmp_path, capsys):
    out = accepted(tmp_path)
    assert main([str(out), "--window-steps", "16", "--max-tokens", "4"]) == 0
    text = (out / "record" / "onsite.json").read_text()
    assert json.loads(text)["serving"] == {"window_steps": 16, "max_tokens": 4, "is": OR.SERVING}
    line = capsys.readouterr().out.strip()
    assert json.loads(line[7:])["serving"] == {"window_steps": 16, "max_tokens": 4}
    assert SV.public_doc(text) == text and SV.public_line(line) == line
    assert main([str(out)]) == 0
    assert json.loads((out / "record" / "onsite.json").read_text())["serving"] == {"missing": OR.NO_SERVING}
    assert json.loads(capsys.readouterr().out.strip()[7:])["serving"] is None


@pytest.mark.parametrize("args", [["--window-steps", "16"], ["--max-tokens", "4"],
                                  ["--window-steps", "0", "--max-tokens", "4"],
                                  ["--window-steps", "x", "--max-tokens", "4"]])
def test_serving_settings_are_given_both_or_neither_each_a_whole_number_of_at_least_1(tmp_path, args):
    out = accepted(tmp_path)
    with pytest.raises(SystemExit) as e:
        main([str(out), *args])
    assert e.value.code == 2 and not (out / "record" / "onsite.json").exists()


@pytest.mark.usefixtures("verified")
def test_runs_at_two_serving_settings_go_beside_each_other_each_figure_labelled_with_its_own(tmp_path, capsys):
    paths = []
    for s, m, cap in SETTINGS:
        out = accepted(tmp_path / f"s{s}", registration=published(cap))
        assert main([str(out), "--window-steps", str(s), "--max-tokens", str(m)]) == 0
        paths.append(out / "record" / "onsite.json")
    capsys.readouterr()
    f = tmp_path / "beside" / "onsite-beside.json"
    assert main(["--beside", *map(str, paths), "--to", str(f)]) == 0
    doc = json.loads(f.read_text())
    assert (doc["format"], doc["is"]) == (OR.BESIDE, OR.BESIDE_IS)
    runs = doc["runs"]
    assert [(r["record"], r["run"], r["tree"], r["serving"], r["ell_a"]) for r in runs] == [
        (str(p), "toy-run", TREE, {"window_steps": s, "max_tokens": m}, [cap]) for p, (s, m, cap) in zip(paths, SETTINGS)]
    assert [(r["of_record"], r["conditional_on"], r["not_claimed"], r["not_credited"], r["K"], r["answers"], r["open"])
            for r in runs] == [(True, [], [OR.DAY_OPEN, OR.POUW_OPEN], [], K, OR.VERIFIED, ["free", "circuit"])
                               ] * len(SETTINGS)
    for p, r in zip(paths, runs):
        rec = json.loads(p.read_text())
        run = rec["periods"]["run"]
        assert r["periods"]["run"] == {"bits": run["bits"], "share": run["share"], "phases": 1,
                                       "per_phase": run["per_phase"]["bits"]}
        assert all(r["periods"][n][w] == {k: rec["periods"][n][w][k] for k in ("bits", "share", "how")}
                   for n in LB.PERIODS for w in ("node", "datacenter", "world"))
        assert r["sampled_proofs"] == OR.guarantee(rec["sampled_proofs"]) and "year" in r["sampled_proofs"]["pooled"]
        assert "conditional_on" not in r["periods"] and not {"K_tight", "periods_tight"} & set(r)
        assert r["periods"]["answers"] == OR.VERIFIED
    assert runs[1]["periods"]["run"]["bits"] == runs[0]["periods"]["run"]["bits"] == K
    assert runs[0]["ell_a"] != runs[1]["ell_a"] and {r["periods"]["year"]["node"]["how"] for r in runs} == {"scaled"}
    said = (OR.ONSITE, "none", f"no verified compute is credited: {OR.ONSITE}'s verdict is no compute claim")
    assert [(r["composite"], r["verified_compute"], r["compute"]) for r in runs] == [said] * len(SETTINGS)
    line = capsys.readouterr().out.strip()
    assert line.startswith("ONSITE-BESIDE ") and [
        (x["serving"], x["K"], x["answers"], x["composite"], x["verified_compute"], x["compute"])
        for x in json.loads(line[14:])] == [(r["serving"], r["K"], OR.VERIFIED, *said) for r in runs]


def test_beside_states_the_answers_beside_each_runs_K_and_prints_a_records_periods_as_it_wrote_them(
        tmp_path, capsys, monkeypatch):
    """Each run's K has beside it, in the document and in the line, what its own record is read as, and a record's
    periods are printed as it wrote them: the missing with its why, and the cells of a record written with them, under
    that record's label."""
    paths = []
    for i, (s_, m, cap) in enumerate(SETTINGS):
        with monkeypatch.context() as mp:
            mp.setattr(OR, "ANSWERS_VERIFIED", i == 0)    # the first record says its answers were verified
            out = accepted(tmp_path / f"s{s_}", registration=published(cap))
            assert main([str(out), "--window-steps", str(s_), "--max-tokens", str(m)]) == 0
        paths.append(out / "record" / "onsite.json")
    capsys.readouterr()
    f = tmp_path / "beside" / "onsite-beside.json"
    assert main(["--beside", *map(str, paths), "--to", str(f)]) == 0
    with_cells, without = json.loads(f.read_text())["runs"]
    wrote = json.loads(paths[0].read_text())["periods"]
    assert "answers" not in wrote and with_cells["periods"]["answers"] == OR.VERIFIED
    assert with_cells["periods"]["run"]["bits"] == wrote["run"]["bits"] == K
    assert all(with_cells["periods"][n][w] == {k: wrote[n][w][k] for k in ("bits", "share", "how")}
               for n in LB.PERIODS for w in ("node", "datacenter", "world"))
    assert without["periods"] == {"missing": OR.NOT_SCALED}
    labels = (OR.VERIFIED, OR.ANSWERS)
    assert [(r["K"], r["answers"]) for r in (with_cells, without)] == [
        (K, label) for label in labels]
    line = capsys.readouterr().out.strip()
    assert [(x["K"], x["answers"]) for x in json.loads(line[14:])] == [
        (r["K"], label) for r, label in zip((with_cells, without), labels)]


def test_beside_refuses_by_path_a_record_of_another_format_or_one_stating_no_serving_settings(tmp_path, capsys):
    out = accepted(tmp_path)
    assert main([str(out)]) == 0
    rec, other, f = out / "record" / "onsite.json", tmp_path / "other.json", tmp_path / "beside.json"
    other.write_text(json.dumps({"format": OR.BESIDE}))
    capsys.readouterr()
    assert main(["--beside", str(rec), "--to", str(f)]) == 1
    assert capsys.readouterr().err.strip() == f"ONSITE-BESIDE: not written: {rec} {OR.UNLABELLED}"
    assert main(["--beside", str(other), "--to", str(f)]) == 1
    assert f"{other} is {OR.BESIDE!r}, not {OR.FORMAT}" in capsys.readouterr().err
    assert not f.exists()


@pytest.mark.parametrize("args", [[], ["--beside", "r.json"], ["out", "--beside", "r.json", "--to", "b.json"],
                                  ["--beside", "r.json", "--to", "b.json", "--window-steps", "1", "--max-tokens", "1"]])
def test_beside_takes_the_records_and_to_alone_and_otherwise_the_record_takes_an_out(args):
    with pytest.raises(SystemExit) as e:
        main(args)
    assert e.value.code == 2
