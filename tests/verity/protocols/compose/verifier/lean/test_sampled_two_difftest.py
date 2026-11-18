"""SPOT's Lean at two levels (`vstar-recursive sampled serve`, `VStar.Recursive.Sampled.protocol`, which hands two
levels to `VStar.Recursive.SampledTwo`) against its Python twin (`reference.spot`, two levels' parse `sampled_two.py`)
on parameters each refused at one step of the parse: the levels' and the form's own steps, whose reasons must agree
word for word, and the decoders it hands its parts to, whose outcomes must agree but for the reason. The parameters are
the one-stage difftest's two-stage run's (`_two_stage_case`) at two levels, both registrations naming one challenger
key, each case one change to them. The twin decides no case past the interior's template circuits, so the verdict on
an audit that passes every step is the Lean's, not this test's."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import strategies as st

import reference as R
import sampled_two as T
import verity_difftest as D
from verity_config.verity_settings import REGISTRY as VERITY_SETTINGS
from verity_difftest import canonical

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

OS = R.ONE_STAGE
PORT = D.ProtocolPort("vstar-recursive", R.spot, st.nothing(), serve=("sampled", "serve"),
                      package=R.ROOT / "verity/protocols/proofs/spot_flock_recursion/verifier/lean")
server = D.server_fixture(PORT)
DROP = object()


@functools.cache
def _key(seed: int) -> dict:
    """The challenger key of `seed`, at the ledger's parameters."""
    return sys.modules["test_challenger_difftest"]._ledger_session(random.Random(f"two-stage-form/{seed}"), seed=seed).key.to_json()


def _challenged(p: dict, label: str, seed: int = 0) -> dict:
    return {**p, "challenger": {"key": _key(seed), "label": label}}


@functools.cache
def _base_text() -> str:
    c = OS._two_stage_case(0)
    it = c["interior"]
    return canonical({
        "levels": 2, "boundary": _challenged(c["params"], "draw/0"), "challenger": c["challenger"],
        "delta": c["delta"],
        "interior": {"members": {}, "params": _challenged(it["params"], "draw/second/0"), "replay": it["replay"],
                     "values": it["values"]},
        "outputs": [], "public": c["public"], "rule": c["rule"], "values": c["values"]})


def base() -> dict:
    """Two levels' parameters on the two-stage run, which the twin parses through the template circuits."""
    return json.loads(_base_text())


def _set(p: dict, path: tuple, v) -> dict:
    p = copy.deepcopy(p)
    *up, last = path
    at = p
    for k in up:
        at = at[k]
    if v is DROP:
        del at[last]
    else:
        at[last] = v
    return p


def _two_strata() -> dict:
    """A boundary over a setup whose proof blocks under its query are more than one stratum."""
    s = next(s for s in OS.SETUPS if len(OS.DR.strata_sizes(s.program, s.query)) > 1)
    b = base()["boundary"]
    return {**b, "law": {"k": 1, "law": "subset"}, "ports": s.ports, "program": s.desc, "query": s.query}


#: Each case's change to the base parameters.
CHANGES = {
    "not-an-object": lambda p: [],
    "no-levels": lambda p: _set(p, ("levels",), DROP),
    "levels-one": lambda p: _set(p, ("levels",), 1),
    "levels-zero": lambda p: _set(p, ("levels",), 0),
    "levels-three": lambda p: _set(p, ("levels",), 3),
    "levels-a-string": lambda p: _set(p, ("levels",), "2"),
    "shape-no-interior": lambda p: _set(p, ("interior",), DROP),
    "levels-one-shape-one": lambda p: {k: v for k, v in _set(p, ("levels",), 1).items()
                                       if k not in ("boundary", "interior", "rule")},
    "a-key-missing": lambda p: _set(p, ("delta",), DROP),
    "a-key-more": lambda p: _set(p, ("more",), 1),
    "outputs-not-a-list": lambda p: _set(p, ("outputs",), "credited"),
    "outputs": lambda p: _set(p, ("outputs",), ["credited"]),
    "rule-all": lambda p: _set(p, ("rule",), {"rule": "all"}),
    "rule-sampling-design": lambda p: _set(p, ("rule",), {"law": "rate:1/2", "rule": "law"}),
    **{f"rule-odd-{n}": (lambda r: lambda p: _set(p, ("rule",), r))(r) for n, r in enumerate(OS.ODD_RULES)},
    "challenger-not-a-string": lambda p: _set(p, ("challenger",), 3),
    "public-not-a-list": lambda p: _set(p, ("public",), "x"),
    "public-item": lambda p: _set(p, ("public",), [1]),
    "boundary-empty": lambda p: _set(p, ("boundary",), {}),
    "boundary-no-window": lambda p: _set(p, ("boundary", "window_root"), DROP),
    "boundary-two-strata": lambda p: _set(p, ("boundary",), _two_strata()),
    "interior-not-an-object": lambda p: _set(p, ("interior",), "x"),
    "interior-a-key-missing": lambda p: _set(p, ("interior", "replay"), DROP),
    "boundary-no-challenger": lambda p: _set(p, ("boundary", "challenger"), DROP),
    "interior-no-challenger": lambda p: _set(p, ("interior", "params", "challenger"), DROP),
    "interior-other-key": lambda p: _set(p, ("interior", "params", "challenger", "key"), _key(1)),
    "interior-params-empty": lambda p: _set(p, ("interior", "params"), {"challenger": p["interior"]["params"]["challenger"]}),
    "interior-no-window": lambda p: _set(p, ("interior", "params", "window_root"), DROP),
    "interior-proofs": lambda p: _set(p, ("interior", "params", "proofs"), {"templates": "x"}),
}


def case(params) -> dict:
    return {"commitment": "7b7d", "evidence": "7b7d", "params": params}


def refusal(params) -> Exception:
    """What the twin's parse of the parameters raises: its own refusal, a delegated decoder's (`sampled_two.Delegated`),
    or the `ValueError` at the mirror's end."""
    try:
        R._form_params(params)
    except (D.Reject, ValueError) as e:
        return e
    raise AssertionError("the parse returned: every case's parameters are refused before the evidence")


def test_base_parses_through_the_mirror() -> None:
    with pytest.raises(ValueError, match="outside the mirror"):
        R._form_params(base())


def test_the_own_refusals_are_reached() -> None:
    """Every refusal the levels and the form make themselves is some case's: a case list that missed one would hold
    nothing of it."""
    refusals = [refusal(change(base())) for change in CHANGES.values()]
    assert all(isinstance(e, D.Reject) for e in refusals), refusals
    reasons = {str(e) for e in refusals if not isinstance(e, T.Delegated)}
    for own in (T.ONE_DEFINITION, T.ONE_KEY, T.OUTPUTS_REFUSED, "the rule is not two-stage",
                f"the parameters name no proofs: {R.V1_REFUSED}"):
        assert own in reasons
    for code in ("levels-params-mismatch", "levels-not-positive", "levels-not-built", "levels-not-integer",
                 "levels-shape"):
        assert any(r.startswith(code) for r in reasons), code


@pytest.mark.parametrize("name", list(CHANGES))
def test_refusal(server, name: str) -> None:
    c = case(CHANGES[name](base()))
    lean, py = PORT.lean(server, c), PORT.python(c)
    assert py[0] == "reject", py
    if not isinstance(refusal(c["params"]), T.Delegated):
        assert lean == py
    else:
        assert PORT.key(lean) == PORT.key(py)


def test_two_definitions_are_refused_by_name(server) -> None:
    """A boundary population of replay blocks under two Definitions is refused by `ONE_DEFINITION`, in the Lean and the
    twin alike: the theorem counts replay blocks, a share by weight only where they are one Definition's."""
    c = case(CHANGES["boundary-two-strata"](base()))
    lean, py = PORT.lean(server, c), PORT.python(c)
    assert lean[0] == py[0] == "reject", (lean, py)
    assert json.loads(lean[1])["reason"] == json.loads(py[1])["reason"], (lean, py)
    assert T.ONE_DEFINITION in json.loads(lean[1])["reason"], lean


def test_the_base_is_refused(server) -> None:
    """Past the mirror the Lean still refuses: the members are none."""
    assert PORT.lean(server, case(base()))[0] == "reject"


#: One honest two-stage play's case, as ``sampled_two_small.sh`` with no arguments keeps it (its ``$OUT/form``:
#: params.json, commitment.json, evidence.json and blobs/, the play's circuit, public file and session's files). Its
#: files are past what git holds, so `artifacts.json` registers the directory and `suites.py` fetches it.
CASE = Path(__file__).resolve().parent / "fixtures" / "sampled-two-case"
#: What mends a kept case SPOT no longer accepts because an honest play changed: a play made afresh, which the script
#: accepts before it keeps it.
AFRESH = ("if an honest play changed, keep one made afresh: `bash benchmarks/one_stage/sampled_two_small.sh` on a "
          "node, its $OUT/form's params.json, commitment.json, evidence.json and blobs/ as a directory, `research "
          "data put --kind fixture/v1 --tree DIR --preserve`, and its id at this path in artifacts.json")


def _case_dir() -> Path:
    d = Path(VERITY_SETTINGS.VERITY_SAMPLED_TWO_CASE or CASE)
    assert (d / "params.json").is_file(), (
        f"{d} is not fetched: `uv run research data fetch-fixtures "
        "tests/verity/protocols/compose/verifier/lean/fixtures/sampled-two-case`")
    return d


def _at_two_levels(j: dict) -> dict:
    """A kept case's params or evidence at two levels: a case kept before ``levels`` named the two-stage form by
    ``form``, which the levels replace."""
    return {**{k: v for k, v in j.items() if k != "form"}, "levels": 2}


def _check(tmp_path: Path, params: dict | None = None, evidence: dict | None = None) -> subprocess.CompletedProcess:
    """`vstar-recursive sampled check` on the kept play, at two levels, its params or evidence replaced where given."""
    d = _case_dir()
    files = []
    for name, given in (("params.json", params), ("evidence.json", evidence)):
        f = tmp_path / name
        f.write_text(canonical(_at_two_levels(given if given is not None else json.loads((d / name).read_text()))))
        files.append(f)
    argv = [str(files[0]), str(d / "commitment.json"), str(files[1]), *map(str, sorted((d / "blobs").iterdir()))]
    return subprocess.run([str(D.build(PORT.package, PORT.exe)), "sampled", "check", *argv],
                          capture_output=True, text=True, check=False)


def _profile(j) -> dict | None:
    if isinstance(j, dict):
        if isinstance(j.get("profile"), dict):
            return j["profile"]
        return next((p for v in j.values() if (p := _profile(v)) is not None), None)
    return None


@pytest.mark.slow
def test_two_levels_accept_one_honest_play_and_profile_both(tmp_path) -> None:
    """SPOT accepts some play at two levels: one honest two-stage run's case (`CASE`, or the directory
    ``VERITY_SAMPLED_TWO_CASE`` names, a run's own ``$OUT/form``) passes `vstar-recursive sampled check`, and the
    accept's integrity profile lists both levels, coarse to fine: the replay blocks, the boundary registration's
    population, then the interior's, each with its sampling design and its drawn count, and no credited block. It
    catches a change after which two levels accept no play, which every refusal case above passes, or profile one."""
    on = _check(tmp_path)
    assert on.returncode == 0, (on.stdout + on.stderr)[-4000:] + "\n" + AFRESH
    prof = _profile(json.loads(on.stdout))
    assert prof is not None, on.stdout[-4000:]
    population = json.loads((_case_dir() / "commitment.json").read_text())["population"]
    levels = prof["levels"]
    assert [lv["name"] for lv in levels][:1] == ["replay"] and len(levels) == 2, levels
    assert levels[0]["size"] == population, (levels, population)
    assert all(lv["draw"] is not None and lv["size"] >= 1 for lv in levels), levels
    assert sorted(prof["drawn"]) == sorted(lv["name"] for lv in levels), prof["drawn"]
    assert "credited" not in prof, prof


#: how the Lean's refusal of an interior audit on the boundary registration's window begins (`SampledTwo.ONE_WINDOW`)
ONE_WINDOW = "the interior audit's window is the boundary registration's"


@pytest.mark.slow
def test_two_levels_refuse_the_honest_play_on_one_window(tmp_path) -> None:
    """Two levels read an audit on two windows only: the kept honest play (`CASE`) with its interior audit's window
    root set to the boundary registration's is refused at its parameters and by name (`ONE_WINDOW`). The parameters
    still parse to the end, where the refusal is, which no refusal case above reaches: their interior names no
    members."""
    params = json.loads((_case_dir() / "params.json").read_text())
    assert params["interior"]["params"]["window_root"] != params["boundary"]["window_root"]
    params["interior"]["params"]["window_root"] = params["boundary"]["window_root"]
    on = _check(tmp_path, params=params)
    assert on.returncode == 1 and ONE_WINDOW in on.stdout, (on.stdout + on.stderr)[-4000:]


def _pinned(tmp_path: Path, sha512: str, retired: list[str], params: dict | None = None) -> subprocess.CompletedProcess:
    """`_check` under `vstar-recursive sampled --challenger-key-sha512 sha512 --challenger-retired FILE`, FILE a key file
    retiring the public keys `retired`."""
    d, keyfile = _case_dir(), tmp_path / "challenger-key.json"
    keyfile.write_text(json.dumps({"retired": [{"pubkey": k} for k in retired]}))
    files = []
    for name, given in (("params.json", params), ("evidence.json", None)):
        f = tmp_path / name
        f.write_text(canonical(_at_two_levels(given if given is not None else json.loads((d / name).read_text()))))
        files.append(f)
    argv = [str(files[0]), str(d / "commitment.json"), str(files[1]), *map(str, sorted((d / "blobs").iterdir()))]
    return subprocess.run([str(D.build(PORT.package, PORT.exe)), "sampled", "--challenger-key-sha512", sha512,
                           "--challenger-retired", str(keyfile), "check", *argv],
                          capture_output=True, text=True, check=False)


UNPINNED = "challenger's key not pinned: the case names it"


def _key() -> str:
    params = json.loads((_case_dir() / "params.json").read_text())
    (key,) = {params["boundary"]["challenger"]["key"]["public"],
              params["interior"]["params"]["challenger"]["key"]["public"]}
    return key


@pytest.mark.slow
@pytest.mark.parametrize("pin, retired, verdict", [
    ("right", False, None), ("other", False, "challenger-key-not-pinned"), ("other", True, "challenger-key-retired"),
    ("right", True, "challenger-key-retired")])
def test_the_honest_play_under_a_pin(tmp_path, pin, retired, verdict) -> None:
    """The kept honest play (`CASE`) under the pin of its challenger's key is accepted, with the unpinned check's line;
    under another pin it is refused at its parameters by `challenger-key-not-pinned`, and with its key in the verifier's
    retired list by `challenger-key-retired`, the pinned key included. Unpinned, the check says so first on stderr."""
    key = _key()
    sha = hashlib.sha512(bytes.fromhex(key)).hexdigest() if pin == "right" else hashlib.sha512(b"forger").hexdigest()
    on = _pinned(tmp_path, sha, [key] if retired else [])
    if verdict is None:
        off = _check(tmp_path)
        assert on.returncode == 0 and on.stdout == off.stdout, (on.stdout + on.stderr)[-4000:]
        assert off.stderr.splitlines()[0] == UNPINNED and UNPINNED not in on.stderr, off.stderr[-4000:]
    else:
        assert on.returncode == 1 and json.loads(on.stdout)["reason"].startswith(f"params: {verdict}: "), \
            (on.stdout + on.stderr)[-4000:]


@pytest.mark.slow
def test_a_forged_window_root_under_the_pinned_key_is_refused_by_the_ledgers_open(tmp_path) -> None:
    """The kept honest play (`CASE`) under its challenger's pin, its interior audit's window root a root the
    challenger's ledger opens no session on: the pin passes and the ledger's open refuses it (`auditedOpen`)."""
    params = json.loads((_case_dir() / "params.json").read_text())
    params["interior"]["params"]["window_root"] = hashlib.sha512(b"forged window").hexdigest()
    on = _pinned(tmp_path, hashlib.sha512(bytes.fromhex(_key())).hexdigest(), [], params)
    assert on.returncode == 1 and "challenger-key" not in on.stdout, (on.stdout + on.stderr)[-4000:]
    assert "opens no challenger session for this window" in on.stdout, (on.stdout + on.stderr)[-4000:]


@pytest.mark.slow
@pytest.mark.parametrize("pin", ["forger", None])
def test_remote_v1_under_the_pin_refuses_a_spot_member_whose_challenger_is_the_forgers(tmp_path, pin) -> None:
    """Remote's composite (`remote-v1`) under the auditor's pin, its SPOT member the kept play (`CASE`), whose
    challenger is not the pinned key: refused at its parameters by `challenger-key-not-pinned` before any member is
    read; without the pin the same parameters are refused for another reason, since they hold no other member."""
    spot = _at_two_levels(json.loads((_case_dir() / "params.json").read_text()))
    params = tmp_path / "params.json"
    params.write_text(canonical({"components": [{"id": "verity/sampled-proofs/v1", "params": spot}]}))
    (tmp_path / "commitment.bin").write_bytes(b"")
    (tmp_path / "evidence.json").write_text("{}")
    keyfile = tmp_path / "challenger-key.json"
    keyfile.write_text(json.dumps({"retired": []}))
    exe = D.build(R.ROOT / "verity/protocols/compose/verifier/remote", "remote-v1")
    flags = ["--challenger-key-sha512", hashlib.sha512(b"auditor").hexdigest(), "--challenger-retired", str(keyfile)]
    on = subprocess.run([str(exe), *(flags if pin else []), "check", str(params), str(tmp_path / "commitment.bin"),
                         str(tmp_path / "evidence.json")], capture_output=True, text=True, check=False)
    reason = json.loads(on.stdout)["reason"]
    assert on.returncode == 1, (on.stdout + on.stderr)[-4000:]
    assert reason.startswith("params: challenger-key-not-pinned: ") == bool(pin), reason


def _evidence() -> dict:
    """The kept honest play's evidence, with its second-stage statements fetched."""
    return json.loads((_case_dir() / "evidence.json").read_text())


def _refuses(tmp_path: Path, evidence: dict, reason: str) -> None:
    """The kept play with its evidence replaced by `evidence` is refused, by `reason`."""
    on = _check(tmp_path, evidence=evidence)
    assert on.returncode == 1 and reason in on.stdout, (on.stdout + on.stderr)[-4000:]


@pytest.mark.slow
def test_two_levels_refuse_a_session_count_mismatch(tmp_path) -> None:
    """Two levels read one statement a session: the kept honest play (`CASE`) with one second-stage statement dropped,
    so the statements are fewer than the sessions, is refused by `STMT_COUNT`, the reason the twin's reference gives.
    The counts agree only at the honest play, which the accept test passes; no refusal case above reaches this check,
    whose play's interior names no members."""
    ev = _evidence()
    own = ev["statements"]
    assert len(own) >= 1, ev
    short = own[:-1]
    reason = T.stmts_sessions(own, short)
    assert reason == T.STMT_COUNT
    _refuses(tmp_path, {**ev, "statements": short}, reason)


@pytest.mark.slow
def test_two_levels_refuse_a_statement_that_is_not_its_session_s(tmp_path) -> None:
    """Two levels read each statement as its session's: the kept honest play (`CASE`) with its first second-stage
    statement replaced by another, so the statement at a position is not the one its session makes, is refused by
    `STMT_SESSION 0`, the reason the twin's reference gives."""
    ev = _evidence()
    own = ev["statements"]
    assert len(own) >= 1, ev
    swapped = [{"template": "not-the-session-s"}, *own[1:]]
    assert canonical(swapped[0]) != canonical(own[0])
    reason = T.stmts_sessions(own, swapped)
    assert reason == T.STMT_SESSION(0)
    _refuses(tmp_path, {**ev, "statements": swapped}, reason)
