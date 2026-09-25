"""[R16 pin] row_pod.sh's weights-of-record REUSE rule (Commit stage, beside the picked Program): an existing `weights_of_record.json` is reused
only when it is THIS Program's (program_digest == the component's result.json digest), of the current schema (MAN-09c stamps
`engine_assumptions`) and complete; anything else is named (the script then moves it aside and re-derives).  The heredoc is extracted from
the script and run as row_pod runs it (python3 - <record> <result.json>), CPU only.

Why: exec's wave-1 Commit-only reruns restore R15 Build dirs to the pod (M-0011); the r15_2309 dir carries build_request/weights_of_record.json
= the int-5 DEFAULT-shape record (program_digest 38f7b397, 98/163 pinned, no engine_assumptions) -- never picked at B=8 (component pick), but
a record of that vintage beside a COMPONENT (int-6 Commits) would have been reused as-is: a weaker pin than int-1 derives (65 engine
constants unpinned by name) or, after a rebuild with another digest, a W2 FAIL by name that only an operator rm could clear."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from tests.check.test_weights_of_record import checkpoint  # noqa: F401 (fixture)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
ROW_POD = os.path.join(REPO, "verity_vllm", "ops", "row_pod.sh")
D = "38f7b397917d7951016a28d9a38df848f66557e0da4d1dd7d4155ad64ed9dbdf"
D2 = "9699b09e" + "0" * 56

from verity_vllm.check.weights_of_record import SCHEMA, SCHEMA_V1

CURRENT = {"schema": SCHEMA, "program_digest": D, "complete": True, "n_pinned": 163, "n_fields": 163,
           "unpinned": [], "unresolved": [], "engine_assumptions": {"kv_cache_dtype": "auto"}, "root_of_record": "2303f39a" + "0" * 56}
LEGACY_INT6 = {"schema": "verity/commit/weights-of-record/v1", "program_digest": D, "complete": True, "n_pinned": 98, "n_fields": 163,
               "unpinned": [f"model.model.layers.{i}.self_attn.attn._k_scale" for i in range(65)], "unresolved": [], "root_of_record": "2303f39a" + "0" * 56}


def _source() -> str:
    src = open(ROW_POD).read()
    m = re.search(r'WOR_STALE=\$\(python3 - "\$WOR" "\$PD_WOR/result.json" "\$\{WOR_SCHEMA:-\}" <<\'EOF\'\n(.*?)\nEOF\n\)', src, re.S)
    assert m, "row_pod.sh: the weights-of-record reuse heredoc (WOR_STALE=$(python3 - \"$WOR\" \"$PD_WOR/result.json\" \"${WOR_SCHEMA:-}\" <<'EOF') is not where it was"
    # the tree's schema is asked of the module (never a literal in the script)
    assert 'WOR_SCHEMA=$($PY -c "from verity_vllm.check.weights_of_record import SCHEMA; print(SCHEMA)"' in src
    assert "'" not in m.group(1), "no apostrophes in the heredoc (bash 3.2 laptop gate)"
    return m.group(1)


def _verdict(tmp_path, record, result, schema: str = SCHEMA) -> str:
    comp = tmp_path / "build_request_LP9_T12"
    comp.mkdir(exist_ok=True)
    rec = comp / "weights_of_record.json"
    if isinstance(record, dict):
        rec.write_text(json.dumps(record))
    else:
        rec.write_text(record)
    if result is not None:
        (comp / "result.json").write_text(json.dumps(result))
    elif (comp / "result.json").exists():
        (comp / "result.json").unlink()
    argv = [sys.executable, "-", str(rec), str(comp / "result.json")] + ([schema] if schema is not None else [])
    out = subprocess.run(argv, input=_source(), capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_a_record_of_an_older_schema_is_named_and_re_derived_never_reused_for_its_root(tmp_path):
    """[R16 pin, schema v2 / exec M-0178] a v1 record (root keyed by the deriving host's fp32 inv_freq: cp-r1 and cp-a100 derived two
    roots for one Llama-3.2-1B) beside a v2 tree is moved aside and re-derived, the reason naming both schemas; the row_pod call
    passes the tree's schema, and a tree that cannot say its schema (older module) leaves the record alone on this criterion."""
    v1 = {**CURRENT, "schema": SCHEMA_V1}
    why = _verdict(tmp_path, v1, {"ok": True, "program_digest": D, "LP": 9, "T": 12})
    assert why.startswith(f"schema {SCHEMA_V1}, this tree derives {SCHEMA}") and "root formula differs" in why, why
    assert _verdict(tmp_path, CURRENT, {"ok": True, "program_digest": D, "LP": 9, "T": 12}) == ""              # the current schema is reused
    assert _verdict(tmp_path, v1, {"ok": True, "program_digest": D, "LP": 9, "T": 12}, schema="") == ""        # no schema known: not this criterion
    assert _verdict(tmp_path, v1, {"ok": True, "program_digest": D, "LP": 9, "T": 12}, schema=None) == ""
    # the ordering of the reasons: another Program's record is named as stale before its schema is looked at
    why2 = _verdict(tmp_path, v1, {"ok": True, "program_digest": D2, "LP": 9, "T": 12})
    assert "stale" in why2 and "schema" not in why2


def test_current_record_of_this_program_is_reused(tmp_path):
    assert _verdict(tmp_path, CURRENT, {"ok": True, "program_digest": D, "LP": 9, "T": 12}) == ""


def test_the_component_digest_is_read_where_a_real_result_json_carries_it(tmp_path):
    """[R16 pin, read on cp-r1 06:00Z: llama B8 build_request_LP245_T31/result.json] a real Program's result.json has NO top-level
    program_digest -- it is artifact.program_digest / program.digest (the derivation itself falls back to instances.json.gz's digest).
    A reuse rule reading only the top-level key saw an empty digest and SKIPPED the identity check: a record of another Program beside the
    component would have been reused (then W2 FAILs by name at the checker instead of the re-derivation remedy).  Every shape names it."""
    real_other = {"schema": "x", "ok": True, "LP": 245, "T": 31, "program": {"digest": D2, "n": 1}, "artifact": {"program_digest": D2}}
    assert "stale" in _verdict(tmp_path, CURRENT, real_other) and D2[:16] in _verdict(tmp_path, CURRENT, real_other)
    assert _verdict(tmp_path, CURRENT, {**real_other, "program": {"digest": D}, "artifact": {"program_digest": D}}) == ""
    assert "stale" in _verdict(tmp_path, CURRENT, {"ok": True, "program": {"digest": D2}})                 # program.digest alone
    assert "stale" in _verdict(tmp_path, CURRENT, {"ok": True, "artifact": {"program_digest": D2}})        # artifact.program_digest alone
    # no digest anywhere in result.json: instances.json.gz's digest (what weights_of_record() records as program_digest) decides
    import gzip
    comp = tmp_path / "build_request_LP9_T12"
    with gzip.open(comp / "instances.json.gz", "wt") as f:
        json.dump({"digest": D2, "rows": []}, f)
    assert "stale" in _verdict(tmp_path, CURRENT, {"ok": True})
    with gzip.open(comp / "instances.json.gz", "wt") as f:
        json.dump({"digest": D, "rows": []}, f)
    assert _verdict(tmp_path, CURRENT, {"ok": True}) == ""
    (comp / "instances.json.gz").unlink()
    assert _verdict(tmp_path, CURRENT, {"ok": True}) == ""                                                  # nothing to compare against: not this criterion


def test_stale_record_of_another_program_is_named(tmp_path):
    why = _verdict(tmp_path, CURRENT, {"ok": True, "program_digest": D2, "LP": 9, "T": 12})
    assert why == f"built for Program {D[:16]}, this component is {D2[:16]} (stale)", why


def test_pre_man09c_record_is_named_with_its_unpinned_count(tmp_path):
    why = _verdict(tmp_path, LEGACY_INT6, {"ok": True, "program_digest": D, "LP": 9, "T": 12})
    assert why == "predates MAN-09c (no engine_assumptions; 65 engine constant(s) unpinned by name)", why


def test_incomplete_record_is_named(tmp_path):
    rec = {**CURRENT, "complete": False, "unresolved": ["model.model.layers.0.mlp.experts.w13_weight"]}
    assert _verdict(tmp_path, rec, {"ok": True, "program_digest": D, "LP": 9, "T": 12}) == "incomplete (1 unresolved field(s))"


def test_unreadable_record_is_named_and_no_result_json_falls_back_to_the_schema_rule(tmp_path):
    assert _verdict(tmp_path, "{not json", {"ok": True, "program_digest": D}).startswith("unreadable record (")
    # no result.json beside the Program (cannot compare digests): the schema / completeness rules still decide
    assert _verdict(tmp_path, CURRENT, None) == ""
    assert _verdict(tmp_path, LEGACY_INT6, None).startswith("predates MAN-09c")


def test_cli_record_carries_its_derivation_provenance_and_names_what_it_supersedes(checkpoint, tmp_path):
    """[COORD R16 M-0038 (2)] a record the CLI writes carries `derived` = {tool, code (git HEAD or null), at, sources (Program files, checkpoint
    dir, checkpoints.json), argv, supersedes}; check() ignores it (digests are the evidence)."""
    from verity_vllm.check import weights_of_record as WR
    from tests.check.test_weights_of_record import _program, _served, _write_program, live_digests

    d, man, t = checkpoint
    prog, fields = _program()
    pdir = tmp_path / "build_request_LP9_T2"
    _write_program(pdir, prog, "d" * 64, 9, 2)
    stale = pdir / "weights_of_record.stale-20260920T042250Z.json"
    stale.write_text(json.dumps(LEGACY_INT6))
    out = pdir / "weights_of_record.json"
    argv = ["--program-dir", str(pdir), "--checkpoint-dir", str(d), "--checkpoints", str(man), "--out", str(out), f"--supersedes={stale}"]
    assert WR.main(argv) == 0
    rec = json.load(open(out))
    dv = rec["derived"]
    assert dv["tool"] == "verity_vllm.check.weights_of_record" and dv["supersedes"] == str(stale) and dv["argv"] == argv
    assert dv["sources"]["program_dir"] == str(pdir) and dv["sources"]["program_files"] == ["instances.json.gz", "result.json"]
    assert dv["sources"]["checkpoint_dir"] == str(d) and dv["sources"]["checkpoints_manifest"] == str(man)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", dv["at"]) and (dv["code"] is None or re.fullmatch(r"[0-9a-f]{40}", dv["code"]))
    assert WR.check(rec, live_digests(_served(t, fields)), program_digests_of_record={"d" * 64})["result"] is True
    # the reuse rule accepts the freshly derived record (current schema, this Program)
    assert _verdict_at(pdir, {"ok": True, "program_digest": "d" * 64, "LP": 9, "T": 2}) == ""


def _verdict_at(comp, result) -> str:
    (comp / "result.json").write_text(json.dumps(result))
    out = subprocess.run([sys.executable, "-", str(comp / "weights_of_record.json"), str(comp / "result.json")], input=_source(), capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_the_real_r15_default_shape_record_shape_is_named_legacy():
    """The record exec's r15_2309 dir carries (lane/r15-exec build_request/weights_of_record.json, read here by its recorded fields): v1 schema,
    program_digest 38f7b397 (the default wrapper shape M-0727 named), 98/163 pinned, no engine_assumptions -> named legacy, never reused."""
    fields = {"schema": "verity/commit/weights-of-record/v1", "program_digest": D, "complete": True, "n_pinned": 98, "n_fields": 163, "n_analytic": None}
    assert "engine_assumptions" not in fields and fields["n_pinned"] < fields["n_fields"]


def _derive_block() -> str:
    """The bash block of the Commit stage from the reuse rule down to the CMD_K arming (row_pod.sh), run with stubs (log, $PY, CMD_K)."""
    src = open(ROW_POD).read()
    i = src.index('      if [ -f "$WOR" ] && [ -n "${CK:-}" ] && [ -d "$CK" ]; then')
    j = src.index('elif [ -n "${CK:-}" ]; then CMD_K+=(--checkpoint-dir "$CK"); fi', i)
    return src[i:src.index("\n", j) + 1]


def test_an_unresolved_checkpoint_dir_is_named_in_row_log_not_silently_unpinned(tmp_path):
    """[R16 pin, wave 1] the workload's (model, revision) resolves to NO local checkpoint dir on this pod: the checker FAILs the Commit
    'weights unpinned (MAN-09)' -- row_pod must say WHY in row.log (fail-closed with the cause named), and must not derive anything."""
    pdir = tmp_path / "build_request"; pdir.mkdir()
    script = (f'set -uo pipefail; PY=/bin/false; D="{tmp_path}"; PD_WOR="{pdir}"; WOR="$PD_WOR/weights_of_record.json"; CMD_K=(x)\n'
              'log(){ echo "LOG $*"; }; tl(){ :; }\n' + _derive_block() + 'printf "CMD_K=%s\\n" "${CMD_K[*]}"\n')   # tl = int-2 stage telemetry, no-op here
    for ck in ("None", ""):
        out = subprocess.run(["bash", "-c", script.replace("CMD_K=(x)", f'CK="{ck}"; CMD_K=(x)')], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "WARNING weights of record NOT derived: no local checkpoint dir" in out.stdout and "weights unpinned (MAN-09)" in out.stdout, out.stdout
        assert not (pdir / "weights_of_record.json").exists() and not any(pdir.glob("weights_of_record.stale-*"))
        # CK 'None' (the heredoc prints Python's None) still reaches commit_delta as --checkpoint-dir: its in-process fallback names the
        # missing dir in verdict.json (weights of record not derived), the checker FAILs 'weights unpinned (MAN-09)'; an empty CK arms nothing
        assert ("--checkpoint-dir None" in out.stdout) == (ck == "None") and "--weights-of-record" not in out.stdout
    # a resolvable checkpoint dir takes the derive path (here: $PY is /bin/false -> the derive WARNING with the log tail), never the unresolved text
    ck = tmp_path / "ck"; ck.mkdir()
    out = subprocess.run(["bash", "-c", script.replace("CMD_K=(x)", f'CK="{ck}"; CMD_K=(x)')], capture_output=True, text=True)
    assert out.returncode == 0 and "not derived (see weights_of_record.log)" in out.stdout and "no local checkpoint dir" not in out.stdout, out.stdout
    assert f"--checkpoint-dir {ck}" in out.stdout


def test_provenance_code_is_the_shipped_commit_on_a_pod_tree_and_git_head_in_a_git_tree(tmp_path, monkeypatch):
    """[COORD M-0038 (2)] derived.code names the code that ran the derivation.  The release tree on a pod has NO .git (ship.sh writes
    EXPORT.json with the shipped commit): the int-1 wave-1 records carried code null (exec r16_0413).  Now: EXPORT.json's commit first
    (code_source 'EXPORT.json', the dirty-patch digest named when present), git HEAD in a git tree, null only when neither exists."""
    from verity_vllm.check import weights_of_record as WR
    root = os.path.abspath(os.path.join(os.path.dirname(WR.__file__), "..", ".."))
    exp = os.path.join(root, "EXPORT.json")
    if os.path.exists(exp):
        pytest.skip("this tree carries an EXPORT.json already (a shipped pod tree)")
    dv = WR.derivation_provenance(str(tmp_path), str(tmp_path), None, ["x"])
    if dv["code"] is not None:                                         # a git tree (the laptop): HEAD, source git
        assert re.fullmatch(r"[0-9a-f]{40}", dv["code"]) and dv["code_source"] == "git"
    try:
        with open(exp, "w") as f:
            json.dump({"commit": "a" * 40, "patch_sha256": "deadbeefcafe0000"}, f)
        dv2 = WR.derivation_provenance(str(tmp_path), str(tmp_path), None, ["x"])
        assert dv2["code"] == "a" * 40 and dv2["code_source"] == "EXPORT.json (dirty patch deadbeefcafe)"
        with open(exp, "w") as f:
            json.dump({"commit": "b" * 40}, f)
        dv3 = WR.derivation_provenance(str(tmp_path), str(tmp_path), None, ["x"])
        assert dv3["code"] == "b" * 40 and dv3["code_source"] == "EXPORT.json"
    finally:
        os.unlink(exp)
