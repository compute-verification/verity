"""verity_vllm.program.predict.weights_tie at TINY: the verifier's tie of the pinned checkpoint X is X's public bytes (each
shard's header state and SHA-512, each other width's field SHA-512, the embedding table's after its tag) and reads every row
of every rooted weight parameter once, the table's as row-seg rows, and making it commits no weight; the honest committer's
registration is `weights_registration`'s, its units are their Definitions' and its last units' digests are the verifier's,
so the audit's pins hold and the verifier pins the roots; a committer serving another checkpoint's words as X's ends at
another digest, and forging the last units' digests makes their outputs not their Definition's; the verifier pins nothing
on an audit that didn't accept the verifier's own tie; and a shard the tie can't read is refused by name."""
from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from tests.check.test_weights_of_record import write_safetensors
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import GRAIN, KEY, _checkpoint
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline import cli
from verity_vllm.program.predict import weights_tie as WT
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.protocols.pouw.h1 import registered as H1R

from verity.core.protocols.sampled_proofs.one_stage import draw as D
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.commitments.gates import sha512_stream as SS
from verity.ml.commitments.gates.sha512 import IV
from verity.ml.windows import interface as IF
from verity.ml.windows import rows as CS

LAW, CAP = D.parse_law("work:1"), 16
EMBED = "model.embed_tokens.weight"


def _ends_of(win: IF.Window) -> list[str]:
    return [descriptor_id(fn) for fn, _ in win.circuit.templates if descriptor_id(fn).startswith("Sha512StreamLast")]


def _ends(win: IF.Window) -> dict[str, list[int]]:
    """Each last unit's recorded digest row (its ``o0`` words), by the supplied value its read names."""
    ends = _ends_of(win)
    rows = {t: np.concatenate([b.words[-1] for b in win._recorded[t]]) for t in ends}
    return {r.param: rows[r.reader][r.at].tolist() for r in win.circuit.reads if r.reader in ends and r.port == "o0"}


def _pins(want: dict) -> dict[str, dict]:
    """The entries of the audit's ``--registered`` file, by value."""
    return {p: e for doc in want["registered"] for p, e in doc["values"].items()}


def _unopened(win: IF.Window, c: CS.Committed, d: Path, pins: dict[str, dict]) -> set[tuple[str, str]]:
    """Lean's registered reads of the written window ``d`` on the host (`Flock.Registered.checkPort`), from its records
    and the verifier's ``pins`` alone: each member's ports, ``(member, port)``, whose table rows' ``b || c`` don't open
    the pinned root at their stated positions."""
    mine, doc = IF.of_records(d, LAW, IF.resolve)
    files = IF.registered_reads(doc, mine, pins, registered=G)
    assert set(files) == {m.key for m in mine.members}
    written, bad = json.loads((d / "window.json").read_text())["members"], set()
    for k, m in enumerate(mine.members):
        for p, port in files[m.key]["ports"].items():
            dom, entry, bc = win.registered[port["value"]].domain, written[k]["registered"][p], c.bc[m.port(p)]
            assert dom.domain_id.hex() == port["domain"]
            if not all(G.verify_read(dom, bytes.fromhex(port["root"]), pos, bc[r * CS.BC_BYTES:(r + 1) * CS.BC_BYTES],
                                     bytes.fromhex(path), port["row_schema"])
                       for r, (pos, path) in enumerate(zip(entry["positions"], entry["paths"]))):
                bad.add((m.key, p))
    return bad


def _summary(t: WT.Tie, registration: dict, record: dict, roots: dict[str, str], **kw) -> dict:
    """An accepting records audit's summary of ``t`` as `checked_pins` reads it, the window's roots ``roots``."""
    want = WT.audit_inputs(registration, record, t, LAW, CAP)
    return {"ok": True, "circuit": want["circuit"], "run_root": want["run"], "outputs": {"units": t.units(), "proved": True},
            "registered": {"pins": {p: e["root"] for p, e in _pins(want).items()}, "roots": roots}, **kw}


@pytest.fixture
def x(tmp_path):
    return _checkpoint(tmp_path / "x")


def test_the_tie_is_the_checkpoints_public_bytes_and_commits_no_weight(x, monkeypatch):
    """One chain for the shard, from the verifier's own state after its header (its pending bytes the header's length mod
    128) through every tensor's rows in file order to its SHA-512, one for the embedding's other width (TINY's tied LM
    head reads it @512 and ``x0`` @64: @512 is the shard's, sorted first), from the IV over the tensor's bytes to their
    SHA-512, and one for its table @64, from the IV over the tag and the tensor's bytes; each rooted parameter's every row
    read once, the rotary table (no root) none. Its templates: the shard's rows at its one phase (each row whole blocks)
    and its last row, the field's rows and last row, and the table's rows (one byte pending) and last row, whose ``data``
    alone are row-seg ports. The verifier makes the tie, its circuit's digest and its audit inputs without committing a
    weight (`weight_value` and the H1 committer refuse here), and its shape entries are the weights record's, which names
    no root."""
    reg, (d, man) = S._registered(), x
    registration = WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN)
    record = WW.weights_record(reg, d, man, grain=GRAIN)

    def no(*a, **k):
        raise AssertionError("the verifier commits no weight")

    monkeypatch.setattr(WW, "weight_value", no)
    monkeypatch.setattr(H1R, "commit", no)
    t = WT.tie(reg, d, man, grain=GRAIN)
    shard = (Path(d) / "model.safetensors").read_bytes()
    base = 8 + struct.unpack("<Q", shard[:8])[0]
    embed = WR.CheckpointIndex(d, verify_shards=False).array(EMBED).tobytes()
    cv, pend = SS.state(shard[:base])
    tagged = WT.TABLE_TAG + embed
    assert [(c.name, c.cv, c.pending, c.length, c.digest) for c in t.chains] == [
        ("tie/model.safetensors", tuple(cv), pend, len(shard), hashlib.sha512(shard).digest()),
        (f"tie/weights/{EMBED}@64", tuple(IV), b"", len(embed), hashlib.sha512(embed).digest()),
        (f"tie/weights/{EMBED}@64/table", tuple(IV), WT.TABLE_TAG, len(tagged), hashlib.sha512(tagged).digest())]
    assert t.entries == record["registered_entries"] and not [e for e in t.entries.values() if "root" in e]
    read = Counter(r for c in t.chains for r in c.rows)
    assert read == Counter((p, j) for p, e in t.entries.items() for j in range(e["leaves"]))
    assert list(t.unrooted) == list(registration["unrooted"]) and not set(t.unrooted) & set(t.entries)
    h = base % 128
    tag = len(WT.TABLE_TAG)
    assert [(descriptor_id(fn), n) for fn, n in t.circuit().templates] == [
        (descriptor_id(fn), n) for fn, n in ((SS.row(h, 512), 176), (SS.row(h, 64), 4), (SS.last(h, 64, len(shard)), 1),
                                             (SS.row(0, 64), 255), (SS.last(0, 64, len(embed)), 1),
                                             (SS.row(tag, 64), 255), (SS.last(tag, 64, len(tagged)), 1))]
    assert IF.seg_ports(t.circuit().registered) == {
        descriptor_id(fn): {"data"} for fn in (SS.row(tag, 64), SS.last(tag, 64, len(tagged)))}
    got = WT.audit_inputs(registration, record, t, LAW, CAP)
    assert got["outputs"] == [[0, 693]] and got["units"] == 693 and got["run"] == t.run
    assert got["registered"][0] == {"program": registration["program"], "values": registration["values"]}
    assert set(got["registered"][1]["values"]) == {n for c in t.chains for n in c.supplied()}
    assert got["circuit"] == t.digest(LAW, CAP) != WT.tie(reg, d, man, grain=GRAIN, run_id="other").digest(LAW, CAP)


def test_the_honest_registration_is_accepted(x, tmp_path):
    """The committer's registration of X is `weights_registration`'s (one key, one checkpoint, one registration), every
    unit of its tie window is its Definition's on its inputs, and each last unit's digest is the verifier's, so its read
    opens the supplied root. Written, its registration records each weight's registered root, the verifier's window from
    its records alone is the tie's (one circuit digest), and the verifier pins the roots on an accepting audit of it."""
    reg, (d, man) = S._registered(), x
    registration, t, win = WT.commit(reg, KEY, d, man, law=LAW, cap=CAP, grain=GRAIN)
    assert registration == WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN)
    assert win.mismatches(kernel="auto") == {}
    assert _ends(win) == {f"{c.name}/sha512": IF.digest_words(c.digest) for c in t.chains}
    c = win.close()
    win.write(c, tmp_path / "w", {"tie": WT.TIE_TAG}, registrar=R)
    rec = json.loads((tmp_path / "w" / "registration.json").read_text())
    roots = rec["window"]["registered_roots"]
    assert {p: roots[p] for p in registration["values"]} == WW.WeightsRegistration.of(registration, "r").roots
    record = WW.weights_record(reg, d, man, grain=GRAIN)
    want = WT.audit_inputs(registration, record, t, LAW, CAP)
    mine, _doc = IF.of_records(tmp_path / "w", LAW, IF.resolve)
    assert rec["window"]["circuit"] == want["circuit"] and mine.run == want["run"]
    pins = _pins(want)
    assert set(roots) == set(pins) and {p: roots[p] for p in pins} == {p: e["root"] for p, e in pins.items()}
    assert _unopened(win, c, tmp_path / "w", pins) == set()
    assert WT.checked_pins(registration, record, t, _summary(t, registration, record, roots), LAW, CAP) == \
        WW.registration_pins(registration, record)


def test_another_checkpoints_words_are_refused(x, tmp_path):
    """A committer serving checkpoint Y's words as X's registers their roots (each another), and its honest units hash
    them from X's header to another digest than X's: each last unit's digest read opens no supplied root, and every other
    read opens. Forging each last unit's digest to X's opens every read and makes those units' outputs not their
    Definitions'. The
    verifier pins nothing on an audit that refused (nor on a forged arm's), nor on one of another run, nor one that left a
    unit unproved or held another root."""
    reg, (d, man) = S._registered(), x
    honest = WW.weights_registration(reg, KEY, d, checkpoints_manifest=man, grain=GRAIN)
    served = WW.record_words(reg, *_checkpoint(tmp_path / "y", seed=4))
    registration, t, win = WT.commit(reg, KEY, d, man, law=LAW, cap=CAP, grain=GRAIN, served=served)
    assert set(registration["values"]) == set(honest["values"])
    assert not [p for p, e in registration["values"].items() if e["root"] == honest["values"][p]["root"]]
    assert win.mismatches(kernel="auto") == {}
    digests = {f"{c.name}/sha512": IF.digest_words(c.digest) for c in t.chains}
    assert not [v for v, row in _ends(win).items() if row == digests[v]]
    record = WW.weights_record(reg, d, man, grain=GRAIN)
    pins = _pins(WT.audit_inputs(registration, record, t, LAW, CAP))
    ends = {(m.key, "o0") for m in win.window.members if m.template in _ends_of(win)}
    c = win.close()
    win.write(c, tmp_path / "w", {"tie": WT.TIE_TAG}, registrar=R)
    assert len(ends) == len(t.chains) and _unopened(win, c, tmp_path / "w", pins) == ends
    _r, _t, forged = WT.commit(reg, KEY, d, man, law=LAW, cap=CAP, grain=GRAIN, served=served, forge_digest=True)
    assert _ends(forged) == digests
    assert forged.mismatches(kernel="auto") == {t: [0] for t in _ends_of(forged)}
    fc = forged.close()
    forged.write(fc, tmp_path / "f", {"tie": WT.TIE_TAG}, registrar=R)
    assert _unopened(forged, fc, tmp_path / "f", pins) == set()
    roots = WW.WeightsRegistration.of(registration, "r").roots
    for bad, why in [({"ok": False}, "didn't accept"), ({"forged": {"ok": True}}, "didn't accept"),
                     ({"run_root": "00" * 64}, "another circuit or run"),
                     ({"outputs": {"units": t.units(), "proved": False}}, "not every one"),
                     ({"registered": {}}, "held no registered pins")]:
        with pytest.raises(ValueError, match=why):
            WT.checked_pins(registration, record, t, _summary(t, registration, record, roots, **bad), LAW, CAP)
    other = {**roots, next(iter(roots)): "ab" * 64}
    with pytest.raises(ValueError, match="roots for .* aren't the registration's"):
        WT.checked_pins(registration, record, t, _summary(t, registration, record, other), LAW, CAP)
    assert WT.checked_pins(registration, record, t, _summary(t, registration, record, roots), LAW, CAP) == roots


def test_the_command_pins_only_on_an_accepting_audit(x, tmp_path, capsys):
    """`verity-vllm weights-tie` as `benchmarks/prove_path/run.sh` runs it: ``commit`` writes the registration (its key
    drawn 0600) and the tie window, ``verify`` the verifier's tie of the same circuit, run and units, and ``pins`` writes
    the roots on an accepting audit's summary and nothing (exit 1) on one that refused; a forged digest without served
    words is a usage error."""
    reg, (d, man) = S._registered(), x
    (tmp_path / "phi.json").write_text(json.dumps(reg.phi.to_json()))
    common = ["--phi", str(tmp_path / "phi.json"), "--checkpoint-dir", d, "--manifest", man,
              "--grain", f"{GRAIN.gemm_words},{GRAIN.rows},{GRAIN.select}", "--law", "work:1", "--cap", str(CAP)]
    path, key = tmp_path / "registration.json", tmp_path / "key"
    with pytest.raises(SystemExit) as e:
        cli.main(["weights-tie", "commit", *common, "--key-file", str(key), "--registration-out", str(path),
                  "--out", str(tmp_path / "w"), "--forge-digest"])
    assert e.value.code == 2 and not key.exists()
    assert cli.main(["weights-tie", "commit", *common, "--key-file", str(key), "--registration-out", str(path),
                     "--out", str(tmp_path / "w")]) == 0
    committed = json.loads(capsys.readouterr().out)
    assert key.stat().st_mode & 0o777 == 0o600 and committed["mismatched"] == {}
    assert cli.main(["weights-tie", "verify", *common, "--registration", str(path), "--out", str(tmp_path / "v")]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown == json.loads((tmp_path / "v" / "tie.json").read_text()) == {k: committed[k] for k in ("circuit", "run", "units")}
    registration, record = json.loads(path.read_text()), WW.weights_record(reg, d, man, grain=GRAIN)
    t = WT.tie(reg, d, man, grain=GRAIN)
    accepting = _summary(t, registration, record, committed["roots"])
    pins = tmp_path / "pins.json"
    for summary, rc in (({**accepting, "ok": False}, 1), (accepting, 0)):
        (tmp_path / "summary.json").write_text(json.dumps(summary))
        assert cli.main(["weights-tie", "pins", *common, "--registration", str(path), "--summary", str(tmp_path / "summary.json"),
                         "--out", str(pins)]) == rc
        assert json.loads(capsys.readouterr().out)["pinned"] is (rc == 0) and pins.exists() is (rc == 0)
    assert json.loads(pins.read_text()) == WW.registration_pins(registration, record)


def _rewrite(d: str, man: str, tensors: dict, *, gap: str | None = None, tail: bytes = b"") -> None:
    """``d``'s shard rewritten as ``tensors`` (name -> (dtype, array)), ``gap``'s data 128 bytes after the tensor before it
    and ``tail`` after the last, and the manifest re-pinned to it."""
    path = Path(d) / "model.safetensors"
    write_safetensors(str(path), tensors)
    if gap is not None:
        n = struct.unpack("<Q", path.read_bytes()[:8])[0]
        hdr, body = json.loads(path.read_bytes()[8:8 + n]), path.read_bytes()[8 + n:]
        lo = hdr[gap]["data_offsets"][0]
        for k, v in hdr.items():
            if k != "__metadata__" and v["data_offsets"][0] >= lo:
                v["data_offsets"] = [x + 128 for x in v["data_offsets"]]
        hj = json.dumps(hdr, separators=(",", ":")).encode()
        path.write_bytes(struct.pack("<Q", len(hj)) + hj + body[:lo] + bytes(128) + body[lo:])
    path.write_bytes(path.read_bytes() + tail)
    doc = json.loads(Path(man).read_text())
    doc["checkpoints"][0]["files"]["model.safetensors"]["sha256"] = WR.sha256_file(str(path))
    Path(man).write_text(json.dumps(doc))


@pytest.mark.parametrize("change, why", [
    ("extra", "tensor model.extra.weight is no registered weight parameter's rows"),
    ("gap", "bytes [{at}, {lo}) between its tensors"),
    ("tail", "bytes [{at}, {end}) after its last tensor"),
    ("cast", "its field is '1:1' (cast 'F32->bf16 (RNE)'"),
    ("unpinned", "the checkpoint is not of record"),
    ("narrow", "{p}: its rows are 64 bytes; C-Flock proves a unit only with a row of at least 112 bytes"),
])
def test_a_shard_the_tie_cant_read_is_refused_by_name(x, change, why):
    """A tensor no registered parameter holds, bytes between two tensors or after the last, a tensor the served model
    casts (a norm stored F32, served bf16: its rows are other bytes than the shard's), a shard the manifest doesn't pin and
    a parameter of rows of 32 words (no row a unit C-Flock proves) are each refused by name."""
    reg, (d, man) = S._registered(), x
    ck = WR.CheckpointIndex(d, verify_shards=False)
    tensors = {n: (ck.dtype(n), ck.array(n)) for n in sorted(ck.tensors, key=lambda n: ck.tensors[n][3])}
    norm, kw = "model.norm.weight", {}
    if change == "narrow":
        kw["record"] = rec = WW.weights_record(reg, d, man, grain=GRAIN)
        p, e = min((p, e) for p, e in rec["registered_entries"].items() if e["words"] == 512)
        e["words"], e["leaves"] = 32, e["leaves"] * 16
        why = why.format(p=p)
    elif change == "extra":
        _rewrite(d, man, {**tensors, "model.extra.weight": ("BF16", np.zeros(64, np.uint16))})
    elif change == "gap":
        _rewrite(d, man, tensors, gap=norm)
        hdr_end = WR.CheckpointIndex(d, verify_shards=False).tensors[norm][3]
        why = why.format(at=hdr_end - 128, lo=hdr_end)
    elif change == "tail":
        _rewrite(d, man, tensors, tail=b"\0" * 5)
        end = (Path(d) / "model.safetensors").stat().st_size
        why = why.format(at=end - 5, end=end)
    elif change == "cast":
        f32 = (tensors[norm][1].astype(np.uint32) << 16).view(np.float32)
        _rewrite(d, man, {**tensors, norm: ("F32", f32)})
    elif change == "unpinned":
        p = Path(d) / "model.safetensors"
        p.write_bytes(p.read_bytes()[:-1] + bytes([p.read_bytes()[-1] ^ 1]))
    with pytest.raises(ValueError, match=re.escape(why)):
        WT.tie(reg, d, man, grain=GRAIN, **kw)
