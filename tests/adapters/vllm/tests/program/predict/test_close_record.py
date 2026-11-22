"""A served window's record made at the serve's close (`close_record.Record.make`), on the processor."""
from __future__ import annotations

import json
import re
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verity.commitments import docket_digest as DD

from tests.program.predict.test_window_rows import (
    CLASS,
    GRAIN,
    LANES,
    RUN,
    _located,
    _read,
    _registered,
    _served,
    _window,
)
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import advice_windows as AW
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_plan as WP
from verity_vllm.program.predict import window_rows as WR
from verity_vllm.program.predict import window_strings as WS
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.close_record import Record, development_of, served_tables, weight_elements


def _committer(held):
    """A committer as the close reads it: each step's members in one uint8 block, each at its offset (``dev_off``)."""
    layouts, blocks = {}, {}
    for (step, name), a in sorted(held.items()):
        b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        at = sum(x.size for x in blocks.get(step, []))
        layouts.setdefault(step, []).append(SimpleNamespace(name=name, ordinal=0, dev_off=at, stream_off=-1))
        blocks.setdefault(step, []).append(b)
    return SimpleNamespace(_layouts=layouts, _gpu_blocks={s: (torch.from_numpy(np.concatenate(bs)),) for s, bs in blocks.items()})


def _close(tmp_path, monkeypatch):
    """TINY's window at class 2,4,2 as the serve's close holds it: its plan, served values, a committer holding each member
    at its offset in its step's block, and the close's plan and binding; the checkpoint's words, the weights registration
    and the advice opening stood in."""
    low, fields, _values, _weights, slots = _window()
    served, held = [], {}
    for k, s in enumerate(slots):
        rows, store = _served(low, s, T=max(WR.positions(low.programs[k].fn)) + 1 - s.row[1])
        held.update({(step, f"s{k}:{op}/{m}"): a for (step, op, m), a in store.items()})
        served.append(WR.Served(rows, *_read(store), locate=_located(store, f"s{k}:")))
    units = WP.Plan(low, served)
    ids = [f"q{k}" for k in range(len(LANES))]
    bodies = {i: {"prompt": p[:lp], "max_tokens": m} for i, (_o, lp, m, p) in zip(ids, LANES)}
    bound = {"compiler": {"name": "c", "digest": "cd" * 64}, "advice": {"root": "ef" * 64, "bits": 0}}
    weight_values = WW.weight_values
    monkeypatch.setattr(ADV, "bodies", lambda out, window: bodies)
    monkeypatch.setattr(AW, "opened", lambda compiler, doc, codec=None: (None, bound))
    monkeypatch.setattr(WW, "record_words", lambda *a, **k: fields)
    monkeypatch.setattr(WW, "weight_values", lambda r, s, _reg, keep_rows=False: weight_values(r, s, None, GRAIN, keep_rows=keep_rows))
    (tmp_path / "advice" / "w0" / "private").mkdir(parents=True)
    (tmp_path / "advice" / "w0" / "private" / "opening.json").write_text("{}")
    plan = SimpleNamespace(cls=CLASS, out=tmp_path, compiler=None, _registered=_registered)
    binding = SimpleNamespace(rc=SimpleNamespace(run_root=bytes.fromhex(RUN)), com=_committer(held))
    compiled = SimpleNamespace(requests=[SimpleNamespace(id=i) for i in ids])
    return low, units, served, plan, binding, compiled


def test_the_records_weights_read_a_fields_flat_elements_in_its_checkpoint_shape():
    """The checkpoint's words come in each field's shape (an embedding table V x H); a weights read is a flat element range of
    it, so a table row at token t is elements [t*H, (t+1)*H). The close's stand-in fields are flat, where a slice of rows and
    one of elements agree."""
    table = np.arange(6 * 4, dtype=np.uint16).reshape(6, 4)
    read = weight_elements({"embed": table})
    assert read("embed", 5 * 4, 6 * 4).tolist() == table[5].tolist()
    assert read("embed", 3, 7).tolist() == [3, 4, 5, 6]
    with pytest.raises(ValueError, match="weight other: the checkpoint of record composes no such field"):
        read("other", 0, 1)


def test_the_records_words_hold_the_rotary_table_the_serve_served(tmp_path, monkeypatch):
    """The checkpoint's shards hold no rotary table: the weights record holds it to the analytic table, and the record reads
    the words the serve served (``private/cos_sin_cache.npy``), refused outside the table's bound. The close's stand-in
    fields hold a rotary table of their own, so the record never asked the serve for one."""
    table = np.arange(8, dtype=np.uint16).reshape(2, 4)
    rot = {"field": "rot", "pinned": "analytic", "analytic": {"table": "cos_sin"}}
    monkeypatch.setattr(WW, "_fields", lambda registered: (None, []))
    monkeypatch.setattr(WW, "_composed", lambda *a: (a[-1]("w", [1, 2]), {"fields": [{"field": "w"}, rot]})[1])
    bound = {"pass": {"status": "pass"}}
    monkeypatch.setattr(WW.WR, "_check_analytic", lambda f, words: bound["pass"])
    assert served_tables(tmp_path) is None
    (tmp_path / "private").mkdir()
    np.save(tmp_path / "private" / "cos_sin_cache.npy", table)
    got = Record("rate:1/4", "ck", tmp_path / "salt").words(None, tmp_path)
    assert sorted(got) == ["rot", "w"] and got["rot"].tolist() == table.tolist()
    assert sorted(WW.record_words(None, "ck")) == ["w"], "no served table: the field is left out"
    bound["pass"] = {"status": "fail", "why": "past the bound"}
    with pytest.raises(ValueError, match="outside the analytic table's bound"):
        WW.record_words(None, "ck", tables=served_tables(tmp_path))


def test_a_development_design_says_its_draw():
    """A development design's record says DEVELOPMENT and its draw; a stratified design is refused."""
    d = development_of({"law": "subset", "k": 4}, 100)
    assert d == {"says": d["says"], "draws": 4, "of": 100} and d["says"].startswith("DEVELOPMENT")
    with pytest.raises(ValueError, match="uniform draw"):
        development_of({"law": "stratified", "k": 4}, 100)


@pytest.mark.slow
def test_the_serves_close_records_its_window_with_strings_from_where_its_rows_are_committed(tmp_path, monkeypatch):
    """`Record.make` over TINY's window at class 2,4,2 (`_close`): the window recorded, closed and written under
    ``windows/w0/record/``, its strings' cost and the twin's comparison on the sampled ports returned. What no cheaper test
    reaches: the serve's own path from its close's plan to a written record. Slow (minutes) for the close."""
    low, units, served, plan, binding, compiled = _close(tmp_path, monkeypatch)
    got = Record("rate:1/4", "unread", tmp_path / "salt_stream", sample=2, workers=2).make(
        plan, 0, compiled, low, units, served, binding)
    to = tmp_path / "windows" / "w0" / "record"
    assert got["out"] == str(to) and got["run"] == RUN and got["population"] > 0
    s = got["strings"]
    assert s["where"] == "HostTwin" and s["acquired"] > 0 and s["acquired"] + s["host"] > s["acquired"]
    assert s["twin"]["equal"] and s["twin"]["ports"] == min(2, s["twin"]["of"]) > 0
    reg = json.loads((to / "registration.json").read_text())
    assert SS.SCAN_STATE not in [e["value"] for e in reg.get("registered", [])] and DD.digest(reg)
    assert SS.SCAN_STATE not in reg["window"].get("registered_roots", {})
    assert {"rows", "record", "close", "write"} <= set(got["seconds"])


@pytest.mark.slow
def test_the_cards_part_runs_first_and_the_record_is_made_from_its_handoff_alone(tmp_path, monkeypatch):
    """The card's part of the close (`Record.handoff`) leaves `window_strings.HANDOFF`; `Record.make` from it, with no
    committer, closes the window with the card's strings, the rest from the host, every handed port equal to the host
    reference's; a card string altered fails by its port's name, and a handoff without its key by the file's. What no
    cheaper test reaches: the record after the lease on a real window's leaf rows. Slow (minutes) for the close."""
    low, units, served, plan, binding, compiled = _close(tmp_path, monkeypatch)
    rec = Record("rate:1/4", "unread", tmp_path / "salt_stream", sample=64, workers=2)
    h = rec.handoff(plan, 0, low, units, binding)
    to = tmp_path / "windows" / "w0" / "handoff"
    assert h["out"] == str(to) and h["files"] == list(WS.HANDOFF) and h["units"] > 0
    assert all((to / f).is_file() for f in WS.HANDOFF)
    after = SimpleNamespace(rc=binding.rc, com=None)
    got = rec.make(plan, 0, compiled, low, units, served, after, handoff=to)
    s = got["strings"]
    assert s["where"] == "card, handed" and s["acquired"] > 0 and s["twin"]["equal"] and s["twin"]["ports"] == s["twin"]["of"] > 0
    key, card, meta = WS.load_card(to)
    k = sorted(card)[0]
    WS.save_card(to, {**card, k: (*card[k][:3], card[k][3] ^ np.uint8(1))}, key, meta)
    with pytest.raises(ValueError, match=re.escape(f"port {k}")):
        rec.make(plan, 0, compiled, low, units, served, after, handoff=to)
    (to / "key.bin").unlink()
    with pytest.raises(ValueError, match="no key.bin"):
        rec.make(plan, 0, compiled, low, units, served, after, handoff=to)


def test_the_launchers_strings_line_prints_the_cards_handed_strings(tmp_path, capsys):
    """The record's strings are the card's handed ones (`window_strings.handed_strings`), which carry no
    ``acquired_s``, or the host's (`window_strings.strings_of`): the launcher's WINDOW-STRINGS line prints either, so a
    run whose record took the card's strings doesn't end in a KeyError after its record."""
    text = (Path(__file__).parents[6] / "adapters/vllm/tools/served_run/h100_smoke.sh").read_text()
    line = text[text.index("card_s = "):]
    line = line[:line.index('"))\n') + 3]
    twin = {"ports": 8, "of": 27, "units": 874496, "equal": True, "unequal": []}
    handed = {"ports": 139, "where": "card, handed", "acquired": 3602952, "host": 163500, "host_s": 4.4, "handed_unleafed": 0}
    (tmp_path / "line.py").write_text(line)
    runpy.run_path(str(tmp_path / "line.py"), {"s": handed, "t": twin, "k": 0})
    runpy.run_path(str(tmp_path / "line.py"), {"s": {**handed, "where": "HostTwin", "acquired_s": 0.0}, "t": twin, "k": 0})
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("WINDOW-STRINGS window 0: 139 ports, card (card, handed) 3602952 units, host 163500 units")
    assert "EQUAL" in out[0] and "3602952 units in 0.0 s" in out[1]


def test_the_launcher_runs_the_cards_parts_under_the_lease_and_the_record_after_it():
    """``h100_smoke.sh``: under the lease (its ``card`` phase) the serve alone, whose close leaves the handoff; after the
    lease, the record from it (verity-vllm close-record), before the advice is written again and window-commit runs; the
    record runs at the close only under ``RECORD_IN_LEASE=1``."""
    text = (Path(__file__).parents[6] / "adapters/vllm/tools/served_run/h100_smoke.sh").read_text()
    card = text[text.index('if [ "${1:-}" = card ]'):]
    card = card[:card.index("\nfi\n")]
    assert '"${SERVE[@]}"' in card and "FIN" not in card and "WC[@]" not in card
    at = [text.index(x) for x in ('"${LEASE[@]}" ||', '"${FIN[@]}" --serve-run "$D" --window "$K"', 'cli advice "$D"',
                                  'timeout -k 1m "$WC_TIMEOUT"')]
    assert at == sorted(at)
    assert '[ "$RECORD_IN_LEASE" != 1 ] || REC_ARGS+=(--record-in-lease)' in text and "no proof is made" in text


def test_a_deployment_registers_once_and_each_epochs_close_reads_its_registration(tmp_path, monkeypatch):
    """`register` writes the deployment's one registration (its record with the program at the top level, no protocol, the
    embedding's table among its entries, the salts beside it) and refuses a second, naming the first's digest; an epoch's
    close (`Record.roots`, `window_weights.weight_values`) reads it and writes nothing, refusing a deployment with none, so
    two epochs commit the registered roots under the kept salts and the record's digest stays the one written."""
    import functools

    from verity.commitments import domains as DM
    from verity.commitments import salt_file as SF

    from tests.program.predict.test_window_circuit import GRAIN, S, _checkpoint
    from verity_vllm.program.predict.window_weights import DEPLOYMENT_RECORD, REGISTRATION, register
    from verity_vllm.records.weights_registration import OF_RECORD

    monkeypatch.setattr(WW, "weights_registration", functools.partial(WW.weights_registration, grain=GRAIN))
    reg, (d, man) = S._registered(), _checkpoint(tmp_path / "ck")
    out = tmp_path / "deployment"
    out.mkdir()
    (out / "serve.json").write_text(json.dumps({"model": "org/m", "revision": "r"}))
    close = Record("rate:1/4", d, tmp_path / "salt", checkpoints=man)
    with pytest.raises(ValueError, match="holds no registration: a deployment registers once, before its first epoch"):
        close.roots(reg, out)
    digest = register(reg, out, d, man)
    def held() -> dict[str, bytes]:
        return {str(f.relative_to(out)): f.read_bytes() for f in sorted(out.rglob("*")) if f.is_file()}

    files = held()
    assert {REGISTRATION, DEPLOYMENT_RECORD, OF_RECORD, "serve.json"} < set(files)
    assert all(f.startswith(SF.beside(out / REGISTRATION).name + "/") for f in set(files) - {REGISTRATION, DEPLOYMENT_RECORD, OF_RECORD, "serve.json"})
    doc, rec = json.loads(files[REGISTRATION]), json.loads(files[DEPLOYMENT_RECORD])
    assert DD.digest(rec) == digest and rec == WW.deployment_record(doc) and "protocol" not in rec
    assert (rec["format"], rec["program"], rec["previous"]) == (DM.REGISTRATION, doc["program"], None)
    assert "weights/model.embed_tokens.weight@64/table" in [e["value"] for e in rec["registered"]]
    with pytest.raises(ValueError, match=re.escape(f"holds the deployment's registration already ({digest})")):
        register(reg, out, d, man)
    words = WW.record_words(reg, d, man)
    for _epoch in range(2):
        assert close.roots(reg, out) == ADV.registered_weights(doc)
        served = WW.weight_values(reg, words, out / REGISTRATION, GRAIN)
        assert {p: v.root.hex() for p, v in served.items()} == {e["value"]: e["root"] for e in rec["registered"]}
        assert held() == files
