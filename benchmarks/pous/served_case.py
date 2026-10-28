"""PoUS's and the wipe's pieces for the composite (`benchmarks/one_stage/composite.py`), built from a served window's
kept outputs: each piece's case, ``case.json``, the ``{"commitment", "evidence", "params"}`` its Lean component's
``serve`` reads (the commitment and evidence hex, no blobs), and that component's own answer on it, ``answer.json``,
whose commitment is the component's digest of the case's commitment: its Lean's with ``--exe``; else PoUS's Python's,
which its Lean is difftested against, and the wipe's leg's own. Both go to the piece's private directory (0700, its files 0600), which is then the
directory composite.py takes as ``--piece``.

    served_case.py --window DIR [--private P] [--exe LEAN] [--hf-home H] [--p2-lib LIB] [--device cuda]

``DIR`` is a piece's public directory as the served run keeps it, and its ``answer.json`` names the protocol:

* **PoUS** (`verity/pous/p2/v4/run`, `served_piece.py`): ``DIR/answer.json``, under its hiding commitment, and
  ``DIR/pous.json``, with the commitment's binding; and the private directory (``--private``, by default the one
  ``pous.json`` names): ``run.json`` (setup's published outputs, the signed session with its coin openings, the session
  id, the trusted challenger key and the run protocol's own answer), ``opening.json`` (the hiding commitment's opening)
  and ``digests.bin`` (the memory challenger's table of ``C*``). The case's evidence holds setup's own inputs too: the
  commitment to ``W``, the salt and ``C*``. ``C*`` is encoded again from the served snapshot at the kept salt
  (`reencode`), on ``--device`` as the piece encoded it (the vLLM adapter; ``libp2v1.so`` from ``--p2-lib`` on a GPU),
  and is refused unless it gives the kept commitment, root and table.
* **The wipe** (`verity/wipe/v0`, `streamed_wipe/served_wipe.sh`): ``DIR/answer.json`` and the case its leg wrote,
  ``wipe.case.json``, in the private directory or else in ``DIR``. The private directory is by default
  ``SERVE_DIR/private/wipe``, ``SERVE_DIR`` the one ``DIR/served.json`` names.

Before anything is written, the case is held to what the run published. For PoUS, the private opening opens the public
answer's commitment, under the window's binding, to the digest of the case's commitment. For the wipe, the answer's
commitment is that digest. For both, the component's verifier (`period.answer`, `wipe_period.answer`, the Python its
Lean port is held to) answers the case, decoded from its own bytes, as the kept answer does. A piece that fails any of
these is refused (`Refused`) and nothing is written.

The case and its answer stay on the prover's side, in that 0700 directory, and so does composite.py's ``--out``, where it
writes every piece's evidence again (``evidence.json``). The auditor's side reads what it read before: each piece's
public ``answer.json`` (PoUS's under its hiding commitment) and the composite's answer. This prints one ``CASE`` line,
naming the protocol, the verdict and where the piece is. `served_case.sh` runs it on a served run's pieces.

Exit 0 when both files are written, 1 on a refusal, 2 when ``DIR/answer.json`` is neither protocol's or on a usage
error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period
from verity.core.protocols.pous import run as R
from verity.core.protocols.pous import wipe_party, wipe_period
from verity.ml.commitments import advice
from verity.ml.commitments.identity import canonical_json_bytes

HERE = Path(__file__).resolve().parent
CASE, ANSWER = "case.json", "answer.json"
WIPE_CASE = "wipe.case.json"
KEYS = ("commitment", "evidence", "params")


class Refused(Exception):
    """The kept outputs don't make a case the run's answer is about; the message says why."""


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise Refused(reason)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: {type(e).__name__}: {e}") from None


def _canonical(b: bytes, what: str) -> Any:
    """``b`` parsed, refused unless it is canonical JSON, the only form the Lean component decodes."""
    try:
        x = json.loads(b)
    except ValueError as e:
        raise Refused(f"the case's {what} is not JSON: {e}") from None
    _need(canonical_json_bytes(x) == b, f"the case's {what} is not canonical JSON")
    return x


def _shape(case: Mapping[str, Any]) -> tuple[Any, Any]:
    """The case's commitment and evidence decoded, refused unless the case is composite.py's form."""
    _need(sorted(case) == sorted(KEYS), f"a case holds exactly {', '.join(KEYS)}, not {sorted(case)}")
    try:
        c, e = bytes.fromhex(case["commitment"]), bytes.fromhex(case["evidence"])
    except (TypeError, ValueError):
        raise Refused("the case's commitment and evidence are hex") from None
    return _canonical(c, "commitment"), _canonical(e, "evidence")


def digest(case: Mapping[str, Any]) -> str:
    """The component's digest of the case's commitment: SHA-512 of its bytes, for PoUS and the wipe alike."""
    return hashlib.sha512(bytes.fromhex(case["commitment"])).hexdigest()


def pous_answer(case: Mapping[str, Any], *, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
                protocol: str = period.PROTOCOL) -> dict[str, Any]:
    """`PousCheck.RunProtocol`'s answer on ``case`` by its Python (`period.answer`), from the case's bytes alone."""
    commitment, ev = _shape(case)
    p = case["params"]
    try:
        published, session = R.Setup.from_json(commitment), R.ChallengePhase.from_json(ev["session"])
        point, sid = period.point_of(p["point"]), bytes.fromhex(p["session_id"])
        trusted, digests = [bytes.fromhex(t) for t in p["trusted"]], bytes.fromhex(ev["digests"])
        inputs = period.setup_inputs(ev["setup"])
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"the PoUS case doesn't decode: {type(e).__name__}: {e}") from None
    return period.answer(point, schemes, trusted, published, digests, session, sid, protocol, setup_inputs=inputs)


def wipe_answer(case: Mapping[str, Any], *, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
                protocol: str = wipe_period.PROTOCOL) -> dict[str, Any]:
    """`MemoryChallenger.WipeProtocol`'s answer on ``case`` by its Python (`wipe_period.answer`), from the case's bytes
    alone; refused unless the case is exactly what `wipe_period.case` makes of what it decodes to."""
    commitment, ev = _shape(case)
    p = case["params"]
    try:
        c = wipe_party.commitment_of(commitment)
        signed = None if ev["signed"] is None else wipe_party.signed_of(ev["signed"])
        server, session = bytes.fromhex(p["server"]), bytes.fromhex(p["session"])
        trusted = [bytes.fromhex(t) for t in p["trusted"]]
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"the wipe's case doesn't decode: {type(e).__name__}: {e}") from None
    _need(wipe_period.case(server, session, trusted, c, signed) == dict(case),
          "the wipe's case is not the one its decoded commitment, session and params make")
    return wipe_period.answer(server, session, trusted, c, signed, schemes, protocol)


def judged(case: Mapping[str, Any], answer: dict[str, Any], exe: str | os.PathLike | None) -> dict[str, Any]:
    """``answer``, or with ``exe`` (the component's Lean, `pous-run-protocol` or `wipe-protocol`) the Lean's answer on
    ``case`` (`period.serve`), refused unless it agrees with ``answer`` but for the reason."""
    if exe is None:
        return answer
    try:
        lean = period.serve(exe, [case])[0]
    except (OSError, IndexError, ValueError, subprocess.CalledProcessError) as e:
        raise Refused(f"{exe} doesn't answer the case: {type(e).__name__}: {e}"[:1000]) from None
    _need({k: v for k, v in lean.items() if k != "reason"} == {k: v for k, v in answer.items() if k != "reason"},
          f"{exe} answers the case {lean.get('verdict')} ({lean.get('reason')}), its Python {answer['verdict']} "
          f"({answer['reason']})")
    return lean


#: the longest slice of a string `write` writes at once: a served PoUS case's evidence is gigabytes of hex
CHUNK = 1 << 26


def _dump(f: Any, doc: Mapping[str, Any]) -> None:
    """``doc`` as ``json.dumps(doc, sort_keys=True, separators=(",", ":"))`` would write it, each top-level string
    longer than `CHUNK` written raw in slices. Those are a case's commitment and evidence, hex once `_shape` has read
    them, which JSON escapes nothing of."""
    for i, k in enumerate(sorted(doc)):
        v = doc[k]
        f.write(("," if i else "{") + json.dumps(k) + ":")
        if isinstance(v, str) and len(v) > CHUNK:
            f.write('"')
            for j in range(0, len(v), CHUNK):
                f.write(v[j:j + CHUNK])
            f.write('"')
        else:
            f.write(json.dumps(v, sort_keys=True, separators=(",", ":")))
    f.write(("}" if doc else "{}") + "\n")


def write(private: Path, case: Mapping[str, Any], answer: Mapping[str, Any]) -> Path:
    """``case.json`` and ``answer.json`` into ``private`` (0700, each file 0600): the piece composite.py reads."""
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(private, 0o700)
    for name, doc in ((CASE, case), (ANSWER, answer)):
        path = private / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            _dump(f, doc)
        os.chmod(path, 0o600)
    return private


def reencode(repo: str, rev: str, published: R.Setup, point: R.Point, digests: bytes, *, hf_home: Path,
             device: str = "cuda", p2_lib: str | None = None, pins: Path | None = None) -> period.SetupInputs:
    """Setup's own inputs encoded again as the piece encoded them (`served_piece.py`): the linears of ``repo``'s snapshot
    at ``rev`` held to the pin, committed (the vLLM adapter's ``W``), and ``C*`` encoded at the kept salt; refused unless
    they give the kept commitment to ``W``, point, root and table."""
    import torch

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import served_piece as SP

    snapshot = hf_home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    try:
        ck, wrong = SP.checkpoint(snapshot, SP._pin(repo, rev, SP.PINS if pins is None else pins))
    except SP.Refusal as e:
        raise Refused(str(e)) from None
    _need(not wrong, "the snapshot is not the pinned checkpoint's: " + "; ".join(wrong))
    model, _ = SP.linears(ck, torch.device(device))
    del ck
    pous, types = SP.pous, SP.pous.linear_types
    pous.linear_types = SP._linear_types
    try:
        _, _, tensors = pous._plaintext(SP.ENTRY, model)
        layout, commitment = pous._commit(pous.DEFAULT_SCHEME, tensors)
        _need(commitment == published.commitment, "the snapshot's W is not the one setup committed to")
        store = pous._encode(layout, tensors, commitment, published.salt, "the kept salt", p2_lib)
        try:
            _need((store.point.params, store.point.limit_ns) == (point.params, point.limit_ns),
                  f"C* is at {store.point}, not at the run's point {point} (the challenge count aside)")
            _need(store.root == published.root, "C* encoded at the kept salt has another root than setup published")
            _need(store.digests == digests, "C* encoded at the kept salt is not the memory challenger's table")
            return commitment, published.salt, store.segments()
        finally:
            store.free()
    finally:
        pous.linear_types = types


def pous_piece(window: Path, private: Path | None = None, *, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
               inputs: period.SetupInputs | None = None, exe: str | os.PathLike | None = None,
               **encode: Any) -> tuple[Path, dict[str, Any]]:
    """PoUS's piece from the public ``window`` and its private directory (module docstring); setup's own inputs are
    ``inputs`` when given, else `reencode`'s, with ``encode`` its keywords, and the answer is the Lean's with ``exe``
    (`judged`). Returns the private directory and the answer written there."""
    public, summary = _json(window / ANSWER), _json(window / "pous.json")
    if private is None:
        _need(bool(summary.get("private")), f"{window}/pous.json names no private directory: the run's session didn't "
                                            f"complete ({public.get('verdict')}: {public.get('reason')}), so it has no case")
        private = Path(summary["private"])
    run, opening = _json(private / "run.json"), _json(private / "opening.json")
    try:
        digests = (private / "digests.bin").read_bytes()
        inner, published, point = run["answer"], R.Setup.from_json(run["published"]), period.point_of(run["point"])
        session, sid = R.ChallengePhase.from_json(run["session"]), bytes.fromhex(run["session_id"])
        trusted = [bytes.fromhex(t) for t in run["trusted"]]
        op = advice.Opening(opening["index"], bytes.fromhex(opening["row"]), bytes.fromhex(opening["salt"]),
                            tuple(bytes.fromhex(p) for p in opening["path"]))
        bind, root = bytes.fromhex(summary["binding"]), bytes.fromhex(public["commitment"])
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise Refused(f"the kept PoUS record doesn't read: {type(e).__name__}: {e}") from None
    row = hashlib.sha512(canonical_json_bytes(published.to_json())).digest()
    _need(advice.verify(root, 1, 8 * len(row), bind, op),
          "the private opening doesn't open the public answer's commitment under the window's binding")
    _need(op.row == row, "the opened row is not the digest of setup's published outputs")
    _need(inner["commitment"] == row.hex(), "the run's own answer is not about setup's published outputs")
    _need({k: public.get(k) for k in ("protocol", "public", "reason", "verdict")} ==
          {k: inner.get(k) for k in ("protocol", "public", "reason", "verdict")},
          "the public answer is not the run's own answer under a hiding commitment")
    if inputs is None:
        inputs = reencode(str(summary.get("model")), str(summary.get("revision")), published, point, digests, **encode)
    _need(tuple(inputs[:2]) == (published.commitment, published.salt),
          "setup's own commitment and salt are not the ones it published")
    case = period.case(point, sid, trusted, published, digests, session, period.setup_case(point.params, *inputs))
    del inputs
    got = pous_answer(case, schemes=schemes, protocol=public["protocol"])
    _need(got == inner, f"the run protocol answers the case {got['verdict']} ({got['reason']}), not as the run did "
                        f"({inner['verdict']}: {inner['reason']})")
    got = judged(case, got, exe)
    return write(private, case, got), got


#: the answer fields the wipe's verifier and its Lean agree on, as `wipe_period.leg` holds them (the reasons may differ)
WIPE_FIELDS = ("commitment", "protocol", "public", "verdict")


def wipe_piece(window: Path, private: Path | None = None, *, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
               exe: str | os.PathLike | None = None) -> tuple[Path, dict[str, Any]]:
    """The wipe's piece from its public ``window`` and its kept case (module docstring). The answer written is the
    leg's own, the public one without the placement the run added to its profile (the Lean's when the leg ran its
    judge), or with ``exe`` the Lean's now (`judged`). Returns the private directory and that answer."""
    public = _json(window / ANSWER)
    served = _json(window / "served.json") if (window / "served.json").is_file() else {}
    if private is None:
        _need(bool(served.get("serve_dir")), f"{window} has no served.json naming the serve dir, so no private directory")
        private = Path(served["serve_dir"]) / "private" / "wipe"
    kept = private / WIPE_CASE if (private / WIPE_CASE).is_file() else window / WIPE_CASE
    case = _json(kept)
    got = wipe_answer(case, schemes=schemes, protocol=public.get("protocol", wipe_period.PROTOCOL))
    _need(public.get("commitment") == digest(case), "the answer's commitment is not the digest of the case's commitment")
    if served.get("server") is not None:
        _need(bytes.fromhex(case["params"]["server"]) == str(served["server"]).encode(),
              f"the case is a wipe of another server than {served['server']}")
    prof = None if public.get("profile") is None else {k: v for k, v in public["profile"].items() if k != "placement"}
    own = {k: public.get(k) for k in (*WIPE_FIELDS, "reason")} | {"profile": prof}
    _need({k: v for k, v in own.items() if k != "reason"} == {k: v for k, v in got.items() if k != "reason"},
          f"the wipe protocol answers the case {got['verdict']} ({got['reason']}), not as the run did "
          f"({public.get('verdict')}: {public.get('reason')})")
    own = judged(case, own, exe)
    return write(private, case, own), own


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", type=Path, required=True, help="the piece's public directory (its answer.json)")
    ap.add_argument("--private", type=Path, help="its private directory (default: the one the run kept)")
    ap.add_argument("--hf-home", type=Path, help="PoUS (required): the HF cache the serve read its snapshot from")
    ap.add_argument("--p2-lib", help="PoUS: libp2v1.so's directory (benchmarks/torch_gpu/p2_stage/build.sh's lib/)")
    ap.add_argument("--device", default="cuda", help="PoUS: where C* is encoded again")
    ap.add_argument("--exe", help="the component's Lean (pous-run-protocol, wipe-protocol), whose answer is written")
    a = ap.parse_args(argv)
    try:
        protocol = _json(a.window / ANSWER).get("protocol")
    except (Refused, AttributeError) as e:
        print(f"CASE: no answer in {a.window}: {e}", file=sys.stderr)
        return 2
    try:
        if protocol == period.PROTOCOL:
            if a.hf_home is None:
                ap.error("a PoUS piece needs --hf-home, the HF cache the serve read its snapshot from")
            private, answer = pous_piece(a.window, a.private, exe=a.exe, hf_home=a.hf_home, p2_lib=a.p2_lib,
                                         device=a.device)
        elif protocol == wipe_period.PROTOCOL:
            private, answer = wipe_piece(a.window, a.private, exe=a.exe)
        else:
            print(f"CASE: {a.window}/answer.json is no PoUS or wipe answer ({protocol})", file=sys.stderr)
            return 2
    except Refused as e:
        print(f"CASE: refused: {e}", flush=True)
        return 1
    print("CASE " + json.dumps({"protocol": answer["protocol"], "verdict": answer["verdict"], "piece": str(private)},
                               sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
