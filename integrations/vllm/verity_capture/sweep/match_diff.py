"""match_diff -- the DIFFERENTIAL HARNESS for the global Match (R17 lane mfast; BRIEF §5 "Acceptance and targets", COORD R17-7.2).

Runs the checker of record (`MATCH_IMPL=record`: `cb_a.global_match` GM-01 with X-09 inside and the alternate within-step criterion
beside the verdict, exactly as `row_pod.sh` invokes it) and the candidate (`MATCH_IMPL=fast`: `cb_a.global_match_fast`) on the SAME
capture artifacts, in separate processes, and compares

  * the VERDICT and the failed-check list;
  * every per-check outcome by name (G1..G8, X-09 inside, the alternate criterion): status, problem count, the problem texts and
    their CLASS (digits/hex normalised);
  * the POPULATION counts: fold instances / attributed / unattributed / engine steps (G2), identity map (G4), query vs fold population,
    unexercised declared VUs, boundary probes (G7), per request: instances, local steps, TokenSelects, prefill chunks, output events,
    generated length, first/finish step, achieved concurrency (G6), X-09 attribution channels;
  * the MISMATCH LOCATIONS: per request `first_divergence` (local step, unmatched fold/component instances and their specs, forward refs),
    divergent step counts, permutation records (G8), dependency-order violations, segmentation disagreements;
  * and, at `--tolerance strict` (the default), the WHOLE result document minus volatile keys (utc, seconds, impl, phases, paths).

For every run it records the UNPROFILED elapsed wall time, CPU (user+sys, self+children), the peak RSS of the whole process tree
(polled -- parallel candidates are covered; PSS beside it on Linux) and the phase timestamps the checker writes (`phases`).

Fixtures (all on disk, both implementations read the same files):
  toy   -- the GM-01 CPU fixtures of `cb_a/tests/test_global_match.py` (real `b1` Definitions, the production fold) built into
           `--fixtures-dir`: positives (B2 unequal shapes, staggered arrivals with a mixed step, EOS-truncated request, UUID request ids,
           LAG=1 async trace, FP8-shaped multi-node weights Record) and the ADVERSARIAL set, each mutation applied to the artifacts:
           omitted request (Program side / record side), duplicated request (Program side / record header), wrong operands (an
           instance rewired to another producer of the same step; to a STALE earlier-step version), cross-request KV read, component of
           the wrong request, missing component, swapped outputs, wrong arrival, wrong EOS set / no EOS declared, LAG=0 header on a
           LAG=1 trace and the reverse, broken interface (a decode row fed a token the request never sampled), manifest edited after the
           Build, undeclared request UUID, consumer-before-producer emission, mis-emitted sampler, and the r14-v1 sequence criterion
           still refusing the co-scheduled-prefill permutation; plus the ACCEPTED dependency-safe within-step permutation (M-ARRIVAL-ORDER,
           DECISION M-0021) which BOTH implementations must PASS under `dag`.
  real  -- `--real NAME=<row attempt dir>` (build_workload/workload_program.json + manifest.json + match/): the positive run and the
           artifact-level mutations that do not need the fold rewritten (swapped outputs, omitted / duplicated request in the Program
           artifact, manifest edit) always; the log / fold rewrites (LAG header flip, fed-token interface break, operand rewire) when the
           files are under `--mutate-max-mb`.  The B8 1024/128 representative case runs this way on cp-m2.
           REAL FAMILY OF RECORD (the fixtures every Match change is gated on; a fixture names its TREE, it is not a row of record):
             llama32-1b L40S B1 256/32 and B8 256/32 (/vol/cp/r17/mfast/mfast-real/{b1,b8}_256_32, S-r17-04);
             OLMoE B2 mixedstop (rev's mutation set, S-r17-30 (1));
             smollm2-135m L40S B2 32/32 mixedstop -- exec's r17_2007 Build+Match (cp-dev2 row
             smollm2-135m__bf16__l40s__tp1__b2__i32__o32__mixedstop__greedy__bi-eager; workload document
             workloads/<row>.json sha256 7e045283686573878943940e5375fbe435a1e4a27637abb76f1f28f25d7be68d, repo 64f326e78):
             the B>=2 MIXED-STOP record -- r0 EOS-stops at 12 of cap 32 with LAG 1 exercised (eos_verified), r1 length-stops -- the
             lag-exercised real-record leg of the fast-as-checker mechanism at B>=2 (rev M-0300 / M-0953; lag_header_flip is a real
             negative here, not a SKIP).  9 GREEN on f7ecee433 (M-0943; out/gen/sweep/lanes/mfast/match_diff_mixedstop_b2_f7ecee433.md).
  tp    -- `--tp NAME=<tp row dir>:build=<build root>`: tp's D90 (TP2) row per rank -- the positive fold of each rank and every case of
           tp's negative set (wrong peer / link kind / producer class / same-rank peer, recorder rows dropped / off by one step / received
           bytes, .pt bytes, provenance, swapped rank logs, dropped collective), EXPECT = the GM-01 document of record tp's tree wrote
           beside each fold; without a TP set the five named cases are listed as SKIP (the gap visible, never silently absent).

Gate: exit 0 iff every non-skipped fixture is GREEN = the record's verdict is the expected one (positives PASS; negatives FAIL with the
named check among the failed ones), the candidate's is the same, and the two result documents agree at the chosen tolerance.  A SKIP is
listed, never counted as green.  Nothing here weakens a check or accepts by hash: the harness only reads what both checkers wrote.

    python -m verity_capture.sweep.match_diff build --fixtures-dir /tmp/mfast-fx
    python -m verity_capture.sweep.match_diff run --fixtures-dir /tmp/mfast-fx --out out/gen/sweep/evidence/mfast/harness/<attempt> \\
        [--real b8_1024_128=/vol/cp/r17/mprof/<row>/<attempt>] [--python /workspace/cp/venv/bin/python] [--tolerance strict|class]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import dataclasses
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

GM_MODULE = "verity_capture.experimental.cb_a.global_match"
IMPLS = ("record", "fast")
REF_IMPL = "record@ref"   # the checker of record imported from a REFERENCE tree (--ref-tree): the record must not drift from it
VOLATILE_KEYS = frozenset({"utc", "seconds", "impl", "phases", "oracle_dir", "log", "recomputed_from"})
_NUM_RE = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-f]{12,}\b|\d+(?:\.\d+)?")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------------------------------------------------- fixtures


@dataclasses.dataclass
class Fixture:
    name: str
    family: str                 # toy | real | tp
    kind: str                   # positive | negative | skip
    expect_verdict: str         # PASS | FAIL | COND_LAG (negative iff the baseline declares async or has an EOS stop) | SKIP
    expect_failed: list         # checks that must appear in failed_checks when FAIL is expected
    args: list                  # global_match CLI args (no --out)
    note: str = ""
    skip_reason: str | None = None
    baseline: str | None = None  # for COND_* expectations: the fixture whose result decides the expectation
    provenance: dict | None = None  # real rows: what of the artifact set is the record's and what was re-derived / reconstructed, by name
    expect_problem_substr: str | None = None  # a named problem text both checkers must carry (rev's EXPECT.problem_substr)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Fixture":
        return Fixture(**d)


def _norm_class(msg: str) -> str:
    """The CLASS of a problem text: every number / hex digest replaced by '#'."""
    return _NUM_RE.sub("#", msg)


def _sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=str)


# ------------------------------------------------------------------------------------------------ record / fold mutations
# Every mutation edits the PERSISTED artifacts (what both checkers read).  They are the reviewer's attack set on the toy fixtures and
# the artifact-level attacks on real captures.


def edit_log(oracle_dir: str, fn) -> int:
    """Rewrite <oracle>/capture/log.jsonl.gz: `fn(tag, dict) -> dict | list[dict] | None` per record (None = unchanged).  Returns edits."""
    p = os.path.join(oracle_dir, "capture", "log.jsonl.gz")
    tag_re = re.compile(r'^\{"\$e":\s*"(\w+)"')
    out, n = [], 0
    with gzip.open(p, "rt") as f:
        for line in f:
            m = tag_re.match(line)
            tag = m.group(1) if m else None
            r = fn(tag, line)
            if r is None:
                out.append(line if line.endswith("\n") else line + "\n")
                continue
            n += 1
            for d in (r if isinstance(r, list) else [r]):
                out.append(json.dumps(d, separators=(",", ":")) + "\n")
    with gzip.open(p, "wt") as f:
        f.writelines(out)
    return n


def _edit_tagged(tag_want: str, edit):
    def fn(tag, line):
        if tag != tag_want:
            return None
        d = json.loads(line)
        return edit(d)
    return fn


def mutate_header_async(oracle_dir: str, value: bool) -> int:
    """LAG header flip: RunHeader.profile.engine_kwargs.async_scheduling (and any top-level engine_kwargs) := value."""
    def ed(d):
        prof = d.get("profile") if isinstance(d.get("profile"), dict) else {}
        if isinstance(prof.get("engine_kwargs"), dict):
            prof["engine_kwargs"]["async_scheduling"] = value
        if isinstance(d.get("engine_kwargs"), dict):
            d["engine_kwargs"]["async_scheduling"] = value
        return d
    n = edit_log(oracle_dir, _edit_tagged("RunHeader", ed))
    vp = os.path.join(oracle_dir, "capture", "versions.json")
    if os.path.exists(vp):   # the record's third attestation location must say the same, else G1 names the disagreement instead
        v = json.load(open(vp))
        if isinstance(v.get("engine_kwargs"), dict) and "async_scheduling" in v["engine_kwargs"]:
            v["engine_kwargs"]["async_scheduling"] = value
            json.dump(v, open(vp, "w"))
    return n


def mutate_header_duplicate_request(oracle_dir: str, index: int = -1) -> int:
    def ed(d):
        reqs = (d.get("workload") or {}).get("requests") or []
        if reqs:
            reqs.append(dict(reqs[index]))
        return d
    return edit_log(oracle_dir, _edit_tagged("RunHeader", ed))


def mutate_fed_token(oracle_dir: str, request_id: str, prompt_len: int) -> dict:
    """Broken interface: the request's FIRST decode row (position == prompt_len) is fed a token it never sampled -- the StepStart's
    input_token_ids entry and the StepLink value for that row are set to another id (consistent with each other, so the failure is the
    output->input association, not the link's self-consistency)."""
    found = {}

    def ed_start(d):
        if found:
            return None
        for k, (r, p) in enumerate(d.get("rows") or []):
            if r == request_id and int(p) == prompt_len:
                toks = d.get("input_token_ids") or []
                if k < len(toks):
                    found.update({"step": int(d["step"]), "row": k, "was": int(toks[k])})
                    toks[k] = int(toks[k]) + 1
                    d["input_token_ids"] = toks
                    return d
        return None
    edit_log(oracle_dir, _edit_tagged("StepStart", ed_start))
    if not found:
        raise ValueError(f"{request_id}: no decode row at position {prompt_len} in the record")

    def ed_link(d):
        if int(d.get("step", -1)) == found["step"] and int(d.get("row", -1)) == found["row"] and d.get("request_id") == request_id:
            d["value"] = found["was"] + 1
            return d
        return None
    edit_log(oracle_dir, _edit_tagged("StepLink", ed_link))
    return found


def _expand_aliases(desc: dict) -> tuple[list, dict]:
    body = desc["definitions"][desc["root"]]["body"]
    nodes = body["nodes"]

    def expand(j):
        if isinstance(j, dict) and "alias" in j:
            k, a = j["alias"]
            return expand(nodes[k]["args"][a]["refs"])
        if isinstance(j, dict) and "concat" in j:
            return {"concat": [expand(x) for x in j["concat"]]}
        return j
    for nd in nodes:
        for a in nd["args"]:
            a["refs"] = expand(a["refs"])
    body["ret"] = expand(body["ret"])
    return nodes, body


def rewire_operand(oracle_dir: str, request_id: str, mode: str = "same_step") -> dict:
    """WRONG OPERANDS on the persisted fold: one instance of `request_id` that reads a node of its own request is rewired to ANOTHER
    producer of the same output type -- of the same step (`same_step`: the wrong producer) or of an EARLIER step of the request
    (`earlier_step`: a stale version).  Spec, form, out type and every other operand are unchanged; the record's digest is recomputed
    so the artifact is self-consistent (the attack is the wiring, not a corrupt file).  Returns the edit (consumer, old, new)."""
    from verity.ir import codec
    pj = os.path.join(oracle_dir, "program.json")
    ij = os.path.join(oracle_dir, "instances.jsonl")
    doc = json.load(open(pj))
    desc = doc["program"]
    n_in = doc["n_input_nodes"]
    fid_of = (lambda nd: desc["callees"][nd["fn"]]) if "callees" in desc else (lambda nd: nd["fn"])
    # instances.jsonl streamed: only (request, step) per node is kept (a B8 256/32 fold is 160k instances / 190 MB -- never a list of dicts)
    meta: dict[int, dict] = {}
    with open(ij) as f:
        head = json.loads(f.readline())
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            meta[n_in + m["node"]] = {"request": m.get("request"), "step": m.get("step")}
    # aliases expanded on a COPY for the search; the edit lands on the record's own node (alias replaced by the concrete reference)
    nodes_orig = desc["definitions"][desc["root"]]["body"]["nodes"]
    nodes, _body = _expand_aliases(copy.deepcopy(desc))

    def is_ts(k):
        return str(fid_of(nodes[k])).startswith(("TokenSelect", "Gumbel"))
    pick = None
    for c, nd in enumerate(nodes):
        mc = meta.get(c)
        if not mc or mc.get("request") != request_id or is_ts(c):
            continue
        for ai, a in enumerate(nd["args"]):
            refs = a["refs"]
            if not (isinstance(refs, dict) and "affine" in refs and refs["affine"][0] == "n"):
                continue
            p = refs["affine"][1]
            mp = meta.get(p)
            if not mp or mp.get("request") != request_id or is_ts(p):
                continue
            want_step = mc.get("step") if mode == "same_step" else None
            for p2 in range(p - 1, -1, -1) if mode == "same_step" else range(0, p):
                m2 = meta.get(p2)
                if not m2 or m2.get("request") != request_id or p2 == p or is_ts(p2) or p2 >= c:
                    continue
                if nodes[p2]["out"] != nodes[p]["out"]:
                    continue
                if mode == "same_step" and m2.get("step") != want_step:
                    continue
                if mode == "earlier_step" and not (m2.get("step") is not None and mc.get("step") is not None and m2["step"] < mc["step"]):
                    continue
                if any(isinstance(b["refs"], dict) and "affine" in b["refs"] and b["refs"]["affine"][:2] == ["n", p2] for b in nd["args"]):
                    continue
                pick = (c, ai, p, p2)
                break
            if pick:
                break
        if pick:
            break
    if pick is None:
        raise ValueError(f"{request_id}: no (consumer, producer, alternative producer of the same type) triple for mode {mode!r}")
    c, ai, p, p2 = pick
    refs = copy.deepcopy(nodes[c]["args"][ai]["refs"])
    refs["affine"][1] = p2
    nodes_orig[c]["args"][ai]["refs"] = refs
    doc["digest"] = codec.program_digest(desc)
    json.dump(doc, open(pj, "w"), separators=(",", ":"), sort_keys=True)
    head["digest"] = doc["digest"]
    # the instance rows are unchanged: new header line + the file's own bytes after its first line
    tmp = ij + ".tmp"
    with open(ij, "rb") as src, open(tmp, "wb") as dst:
        src.readline()
        dst.write((json.dumps(head) + "\n").encode())
        shutil.copyfileobj(src, dst, 1 << 20)
    os.replace(tmp, ij)
    return {"consumer_node": c, "arg": ai, "old_producer": p, "new_producer": p2, "mode": mode,
            "consumer_spec": str(fid_of(nodes[c])), "consumer_step": meta[c].get("step"), "new_producer_step": meta[p2].get("step")}


def swap_outputs(tokens_path: str) -> None:
    tj = json.load(open(tokens_path))
    reqs = tj["requests"]
    if len(reqs) < 2:
        raise ValueError("swap_outputs needs two requests")
    a, b = reqs[0], reqs[1]
    a["generated_token_ids"], b["generated_token_ids"] = b["generated_token_ids"], a["generated_token_ids"]
    json.dump(tj, open(tokens_path, "w"))


# ------------------------------------------------------------------------------------------------------ toy fixtures


def _toy_env():
    """The GM-01 toy (cb_a/tests/test_global_match.py) as a library: real b1 Definitions + the production fold on CPU."""
    from verity_capture.tests import test_fold_synthetic as toy
    if not toy._real_memory_available():   # the tests' `memory` fixture: the stub memory model when the real one cannot import
        from verity_capture import fold as fold_mod
        fold_mod.MemoryModel = toy.StubMemory
        fold_mod.row_view = toy.stub_row_view
        fold_mod.rows_view = toy.stub_rows_view
    from verity_capture.experimental.cb_a.tests import test_global_match as T
    from verity_capture.experimental.cb_a.tests import test_batch_decomp_e2e as e2e
    from verity_capture.experimental.cb_a import global_match as gm
    return T, e2e, gm


def _gp_args(gp: dict, fx_dir: str, oracle: str, within_step: str = "dag") -> list:
    gpp = os.path.join(fx_dir, "global_program.json")
    _dump(gp, gpp)
    return ["--global-program", gpp, "--oracle", oracle, "--within-step", within_step]


def _assemble_lag(T, e2e, gm, reqs_trace, reqs_decl, path, header_async: bool, declared_async: bool, eos=(11,)):
    """A record whose header attests `async_scheduling=header_async` against a Program declaring `declared_async` (stop_rule.eos_lag_rows
    1/0).  `reqs_trace` may carry a placeholder row after the EOS (LAG=1 trace: sampled = [.., EOS, discarded]); tokens.json reports the
    ANSWER only (up to and including the first EOS), finish_step = the EOS step -- the driver's view (vllm_adapter tokens.json)."""
    import pathlib
    from verity_capture.fold import fold
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    wl = T.workload(reqs_decl, path / "workload.json")
    evs = T.batched_log(reqs_trace, wl_sha=T._sha(wl))
    hdr = evs[0]
    prof = copy.deepcopy(hdr.profile)
    prof["engine_kwargs"]["async_scheduling"] = header_async
    evs[0] = dataclasses.replace(hdr, profile=prof)
    r = fold(evs, e2e.profile(), strict=True)
    assert r.report.unsupported == 0 and not r.errors, (r.report.unsupported, r.errors)
    oracle = T.persist(r, evs, path / "batched", reqs_trace)
    tk = T.tokens_json(reqs_trace)
    for q, t in zip(reqs_trace, tk["requests"]):
        s = list(q["sampled"])
        if any(int(x) in eos for x in s):
            k = next(i for i, x in enumerate(s) if int(x) in eos) + 1
            t["generated_token_ids"], t["n_generated"] = s[:k], k
            t["finish_step"], t["finish_reason"] = T._arr(q) + k - 1, "stop"
    json.dump(tk, open(os.path.join(oracle, "capture", "tokens.json"), "w"))
    comps = {}
    for q in reqs_trace:
        sh = (len(q["prompt"]), T._cap(q) - 1)
        if sh not in comps:
            comps[sh] = T.component(q, path / f"comp_LP{sh[0]}_T{sh[1]}")
    # [R17-1, int-2] the mock Program DECLARES its lag (a Program that declares none fails G6 by name on every EOS stop -- 'rebuild the
    # workload Program with the declared async-scheduling specialization'); the fixture's declaration = the header it is built against
    gp = gm.mock_global_program(wl, list(comps.values()), eos_token_ids=list(eos), lag=1 if declared_async else 0)
    gp["applicability"]["engine"]["async_scheduling"] = declared_async
    gp["stop_rule"]["eos_lag_rows"] = 1 if declared_async else 0
    return gp, oracle


def build_toy_fixtures(fx_root: str) -> list:
    T, e2e, gm = _toy_env()
    import pathlib
    root = pathlib.Path(fx_root) / "toy"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    out: list[Fixture] = []

    def add(name, kind, expect, failed, args, note=""):
        out.append(Fixture(name=f"toy:{name}", family="toy", kind=kind, expect_verdict=expect, expect_failed=list(failed), args=args, note=note))

    def mk(name, reqs, **kw):
        d = root / name
        gp, oracle = T.assemble(reqs, d, **kw)
        return d, gp, oracle

    # ---- positives (PASS on both)
    d, gp, o = mk("pos_b2_unequal_shapes", [T.R0, T.R1])
    add("pos_b2_unequal_shapes", "positive", "PASS", [], _gp_args(gp, str(d), o), "B2, LP 3/2, caps 3/2")
    d, gp, o = mk("pos_staggered_mixed_step", [T.R0, T.R1_LATE, T.R2_LATE])
    add("pos_staggered_mixed_step", "positive", "PASS", [], _gp_args(gp, str(d), o), "arrivals 0/1/2, a prefill co-scheduled with decodes")
    d, gp, o = mk("pos_eos_truncated", [T.R0, T.R1_EOS])
    add("pos_eos_truncated", "positive", "PASS", [], _gp_args(gp, str(d), o), "r1 stops on the declared EOS at 2 of 4; compared over its executed prefix")
    d, gp, o = mk("pos_uuid_request_ids", [T.U0, T.U1])
    add("pos_uuid_request_ids", "positive", "PASS", [], _gp_args(gp, str(d), o), "record UUIDs mapped to r<k>")
    d, gp, o = mk("pos_tgrouped_no_displacement", [T.R0, T.RS_LATE], tgrouped=True)
    add("pos_tgrouped_no_displacement", "positive", "PASS", [], _gp_args(gp, str(d), o), "production AttentionScope emission, no displacement")
    d, gp, o = mk("pos_permutation_dependency_safe", [T.R0, T.RX_LATE], tgrouped=True)
    add("pos_permutation_dependency_safe", "positive", "PASS", [], _gp_args(gp, str(d), o),
        "ACCEPTED: M-ARRIVAL-ORDER co-scheduled prefill emitted permuted (same DAG multiset, dependency order kept) -> PASS under dag, layout recorded in G8")
    R1_LAG = {"id": "r1", "prompt": [4, 6], "sampled": [8, 11, 77], "cap": 4}   # EOS 11 at event 1, one placeholder row after it (LAG=1)
    gp, o = _assemble_lag(T, e2e, gm, [T.R0, R1_LAG], [T.R0, R1_LAG], root / "pos_lag1_async_trace", header_async=True, declared_async=True)
    add("pos_lag1_async_trace", "positive", "PASS", [], _gp_args(gp, str(root / "pos_lag1_async_trace"), o),
        "LAG=1: async_scheduling attested + declared; one placeholder row after the EOS, its sample fed nowhere and not in generated_token_ids")
    # FP8-shaped weights Record (one Input node per leaf field): the toy Record + one f32 field
    try:
        from verity_capture import analytic, fold as fold_mod
        from verity.ir.types import Array, Record, Value
        base = analytic.weights_type
        fold_mod.analytic.weights_type = lambda C: Record(*base(C).fields, ("scale32", Array(4, Value(32))))
        try:
            d, gp, o = mk("pos_multi_weight_input_nodes", [T.R0, T.R1])
        finally:
            fold_mod.analytic.weights_type = base
        add("pos_multi_weight_input_nodes", "positive", "PASS", [], _gp_args(gp, str(d), o), "F-fp8-GM1 shape: 5 weight Input nodes, the last unread")
    except Exception as e:  # noqa: BLE001 -- listed, never silently absent
        out.append(Fixture(name="toy:pos_multi_weight_input_nodes", family="toy", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[],
                           skip_reason=f"fixture build failed: {e!r}"[:300]))

    # ---- adversarial set (FAIL on both, by the named check)
    d, gp, o = mk("neg_omit_request_program", [T.R0, T.R1])
    gp["requests"] = [r for r in gp["requests"] if r["request_id"] != "r1"]
    add("neg_omit_request_program", "negative", "FAIL", ["G1", "G2"], _gp_args(gp, str(d), o), "the Program omits r1 the record served")
    # record side: the manifest / Program declare r0 + r1, the record served r0 only (r1's rows are never there to attribute)
    d = root / "neg_omit_request_record"
    d.mkdir()
    wl = T.workload([T.R0, T.R1], d / "workload.json")
    o = T.batched([T.R0], d / "batched", wl_sha=T._sha(wl))
    comps = [T.component(T.R0, d / "comp_LP3_T2"), T.component(T.R1, d / "comp_LP2_T1")]
    gp = gm.mock_global_program(wl, comps, eos_token_ids=[T.EOS], lag=0)   # [R17-1] declared lag 0 = the toy engine's (T.assemble's default)
    add("neg_omit_request_record", "negative", "FAIL", ["G1", "G2"], _gp_args(gp, str(d), o), "the record never scheduled declared r1 (no rows, no instances)")
    d, gp, o = mk("neg_duplicate_request_program", [T.R0, T.R1])
    gp["requests"].append(dict(gp["requests"][1]))
    add("neg_duplicate_request_program", "negative", "FAIL", ["G1"], _gp_args(gp, str(d), o), "the Program lists r1 twice")
    d, gp, o = mk("neg_duplicate_request_record_header", [T.R0, T.R1])
    mutate_header_duplicate_request(o)
    add("neg_duplicate_request_record_header", "negative", "FAIL", ["G1"], _gp_args(gp, str(d), o), "RunHeader.workload lists r1 twice (two labels, one trace)")
    d, gp, o = mk("neg_wrong_operand_same_step", [T.R0, T.R1])
    ed = rewire_operand(o, "r1", "same_step")
    add("neg_wrong_operand_same_step", "negative", "FAIL", ["G3"], _gp_args(gp, str(d), o), f"fold instance rewired to the wrong producer: {ed}")
    d, gp, o = mk("neg_wrong_operand_stale_version", [T.R0, T.R1_LATE])
    ed = rewire_operand(o, "r1", "earlier_step")
    add("neg_wrong_operand_stale_version", "negative", "FAIL", ["G3"], _gp_args(gp, str(d), o), f"fold instance fed an EARLIER step's value (stale version): {ed}")
    d, gp, o = mk("neg_cross_request_kv_read", [T.R0, T.R1], leak=True)
    add("neg_cross_request_kv_read", "negative", "FAIL", ["G5", "G3"], _gp_args(gp, str(d), o), "r1's attention reads r0's KV row (record)")
    d, gp, o = mk("neg_component_of_wrong_request", [T.R0, T.R1])
    gp["requests"][1]["component"] = dict(gp["requests"][0]["component"])
    add("neg_component_of_wrong_request", "negative", "FAIL", ["G3", "G4"], _gp_args(gp, str(d), o), "r1 wired to r0's component (Program)")
    d, gp, o = mk("neg_missing_component", [T.R0, T.R1])
    gp["requests"][1]["component"]["dir"] = None
    add("neg_missing_component", "negative", "FAIL", ["G3"], _gp_args(gp, str(d), o), "no component for r1's shape")
    d, gp, o = mk("neg_swapped_outputs", [T.R0, T.R1])
    swap_outputs(os.path.join(o, "capture", "tokens.json"))
    add("neg_swapped_outputs", "negative", "FAIL", ["G6"], _gp_args(gp, str(d), o), "driver's generated ids of r0/r1 swapped (output association)")
    d, gp, o = mk("neg_wrong_arrival", [T.R0, T.R1_LATE])
    next(r for r in gp["requests"] if r["request_id"] == "r1")["arrive_step"] = 0
    add("neg_wrong_arrival", "negative", "FAIL", ["G6"], _gp_args(gp, str(d), o), "declared arrival 0, executed at 1")
    d, gp, o = mk("neg_wrong_eos_set", [T.R0, T.R1_EOS], eos_token_ids=(99,))
    add("neg_wrong_eos_set", "negative", "FAIL", ["G6"], _gp_args(gp, str(d), o), "early stop on 11, declared EOS set [99]")
    d, gp, o = mk("neg_no_eos_declared", [T.R0, T.R1_EOS], eos_token_ids=None)
    add("neg_no_eos_declared", "negative", "FAIL", ["G6"], _gp_args(gp, str(d), o), "early stop, Program declares no EOS set (fail closed)")
    gp, o = _assemble_lag(T, e2e, gm, [T.R0, R1_LAG], [T.R0, R1_LAG], root / "neg_lag0_header_on_lag1_trace", header_async=False, declared_async=True)
    add("neg_lag0_header_on_lag1_trace", "negative", "FAIL", ["G6"], _gp_args(gp, str(root / "neg_lag0_header_on_lag1_trace"), o),
        "trace has the placeholder row (LAG=1), header attests async_scheduling=False")
    gp, o = _assemble_lag(T, e2e, gm, [T.R0, T.R1_EOS], [T.R0, T.R1_EOS], root / "neg_lag1_header_on_lag0_trace", header_async=True, declared_async=True)
    add("neg_lag1_header_on_lag0_trace", "negative", "FAIL", ["G6"], _gp_args(gp, str(root / "neg_lag1_header_on_lag0_trace"), o),
        "trace has NO placeholder row (LAG=0), header attests async_scheduling=True (a deleted step)")
    d, gp, o = mk("neg_broken_interface_fed_token", [T.R0, T.R1_LATE])
    ed = mutate_fed_token(o, "r1", len(T.R1_LATE["prompt"]))
    add("neg_broken_interface_fed_token", "negative", "FAIL", ["G6"], _gp_args(gp, str(d), o), f"r1's first decode row fed a token it never sampled: {ed}")
    d, gp, o = mk("neg_manifest_edited_after_build", [T.R0, T.R1_LATE])
    wlp = gp["workload"]["path"]
    wl = json.load(open(wlp))
    wl["requests"][1]["max_tokens"] = 5
    json.dump(wl, open(wlp, "w"))
    add("neg_manifest_edited_after_build", "negative", "FAIL", ["G1"], _gp_args(gp, str(d), o), "manifest max_tokens edited after the Build")
    d, gp, o = mk("neg_undeclared_request_uuid", [T.U0, T.U1])
    gp["requests"][1]["workload_request_id"] = "00000000-0000-0000-0000-000000000000"
    add("neg_undeclared_request_uuid", "negative", "FAIL", ["G1"], _gp_args(gp, str(d), o), "the record serves a UUID the Program does not declare")
    d, gp, o = mk("neg_consumer_before_producer", [T.R0, T.R1], tgrouped=True)
    T.move_consumer_before_producer(o, "r1")
    add("neg_consumer_before_producer", "negative", "FAIL", ["G3", "G8"], _gp_args(gp, str(d), o), "same DAG, one consumer emitted before its producer (not a layout)")
    d, gp, o = mk("neg_mis_emitted_sampler", [T.R0, T.R1_LATE], tgrouped=True)
    T.move_consumer_before_producer(o, "r1", sampler_to_head=True)
    add("neg_mis_emitted_sampler", "negative", "FAIL", ["G8"], _gp_args(gp, str(d), o), "r1's sampler re-emitted at the head of its engine step (segmentation disagreement)")
    d, gp, o = mk("neg_sequence_criterion_refuses_permutation", [T.R0, T.RX_LATE], tgrouped=True)
    add("neg_sequence_criterion_refuses_permutation", "negative", "FAIL", ["G3", "G8"], _gp_args(gp, str(d), o, within_step="sequence"),
        "the r14-v1 sequence criterion on the co-scheduled-prefill permutation (alternate leg exercised as the verdict)")

    # ---- TP (D90) -- skipped by name until tp's fold + per-rank compare exist
    for nm, why in (("tp_wrong_peer", "a received collective value from the wrong rank"), ("tp_swapped_rank", "rank logs swapped"),
                    ("tp_wrong_collective_op", "AllReduce where the Program says AllGather"), ("tp_off_by_one_step", "collective at step t+1"),
                    ("tp_wrong_segmentation", "shard segmentation differs from the declared one")):
        out.append(Fixture(name=f"tp:{nm}", family="tp", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[], note=why,
                           skip_reason="pending tp's D90 fold + per-rank compare (ruling R17-4); fixture layout to come from tp"))
    return out


# ------------------------------------------------------------------------------------------------------ real fixtures


def _shadow_copy(src_dir: str, dst_dir: str, rewrite: set[str] | None = None) -> None:
    """A mutable view of a row attempt dir: every file symlinked, except those named in `rewrite` (relative paths) which are copied."""
    rewrite = rewrite or set()
    for base, dirs, files in os.walk(src_dir):
        rel = os.path.relpath(base, src_dir)
        tb = os.path.join(dst_dir, rel) if rel != "." else dst_dir
        os.makedirs(tb, exist_ok=True)
        for f in files:
            s = os.path.join(base, f)
            t = os.path.join(tb, f)
            r = os.path.normpath(os.path.join(rel, f)) if rel != "." else f
            if os.path.lexists(t):
                os.remove(t)
            if r in rewrite:
                shutil.copy2(s, t)
            else:
                os.symlink(os.path.abspath(s), t)


def _relocate_gprog(art_path: str, row_dir: str, out_path: str, workload_path: str | None = None) -> dict:
    """The Build artifact's `inputs.derived` (absolute component paths written on the Build pod) re-rooted onto `row_dir` when the recorded
    path does not exist but `<row_dir>/<basename>` does; `inputs.manifest` (the workload document) -> `workload_path` when given.
    descriptor.json.gz is linked beside the copy so from_gprog's digest verification still runs.  Paths only: no Program content, no digest
    and no request of the artifact is touched (the checker recomputes the digest from the descriptor and refuses a mismatch)."""
    art = json.load(open(art_path))
    moved = {}
    der = (art.get("inputs") or {}).get("derived") or {}
    for k, p in list(der.items()):
        if p and not os.path.exists(p):
            cand = os.path.join(row_dir, os.path.basename(p.rstrip("/")))
            if os.path.isdir(cand):
                der[k] = cand
                moved[k] = cand
    if workload_path:
        art.setdefault("inputs", {})["manifest"] = workload_path
    _dump(art, out_path)
    dsrc = os.path.join(os.path.dirname(art_path), "descriptor.json.gz")
    ddst = os.path.join(os.path.dirname(out_path), "descriptor.json.gz")
    if os.path.exists(dsrc) and not os.path.lexists(ddst):
        os.symlink(os.path.abspath(dsrc), ddst)
    return moved


def _resolve_workload(row_dir: str, art: dict, explicit: str | None) -> str | None:
    """The workload document GM-01 takes as `--workload` (row_pod.sh: workloads/<row_id>.json; the Build artifact records it
    as `inputs.manifest` relative to the repo).  Explicit path > artifact's path (absolute, or relative to the repo) > <repo>/out/gen/sweep/
    workloads/<row_id>.json with row_id from attempt.json or the row dir's parent name."""
    if explicit:
        return os.path.abspath(explicit) if os.path.exists(explicit) else None
    mp = (art.get("inputs") or {}).get("manifest")
    for cand in ([mp, os.path.join(REPO_ROOT, mp)] if mp else []):
        if cand and os.path.exists(cand):
            return os.path.abspath(cand)
    row_id = None
    ap = os.path.join(row_dir, "attempt.json")
    if os.path.exists(ap):
        try:
            row_id = json.load(open(ap)).get("row_id")
        except Exception:  # noqa: BLE001
            row_id = None
    for rid in (row_id, os.path.basename(os.path.dirname(row_dir)), os.path.basename(row_dir)):
        if rid:
            cand = os.path.join(REPO_ROOT, "out", "gen", "sweep", "workloads", f"{rid}.json")
            if os.path.exists(cand):
                return cand
    return None


def build_tp_fixtures(spec: str) -> list:
    """`NAME=<tp row dir>:build=<build root>[:workload=<path>]` -> the per-rank GM-01 fixtures of tp's D90 (TP2) row: the positive fold of
    every rank (`<row>/match/fold_match/rank<r>`) and every case of tp's negative set (`<row>/negs_fold/<case>/fold_match/rank<r>`, the
    fold products of a mutated record: wrong peer / link kind / producer class / same-rank peer link, recorder rows dropped / off by one
    step / received bytes, .pt bytes, provenance of another run, swapped rank logs, dropped collective).  The Program of rank r is
    `<build root>/build_workload_r<r>/workload_program.json` (its components are referenced by absolute path: the tree is mirrored at the
    same path, nothing is rewritten).  EXPECT of every (case, rank) = the GM-01 document tp's tree wrote beside the fold products
    (`global_match.json`: verdict + failed_checks -- the record's verdict of record; the current record must reproduce it, the candidate
    must equal the current record strictly); a rank whose fold failed before GM-01 (no document) is SKIP by name.  No mutation is made
    here: tp's negatives are the TP adversarial set (M-0146), and most are caught at the collective / link level, where per-rank GM-01
    MUST still PASS (a FAIL there would be a false negative of either implementation) -- swapped_rank_logs is the one that reaches GM-01
    (G3+G4 on both ranks)."""
    name, _, rest = spec.partition("=")
    parts = rest.split(":")
    row_dir = os.path.abspath(parts[0])
    build_root, wl_explicit, ws = None, None, "dag"
    for p in parts[1:]:
        if p.startswith("build="):
            build_root = os.path.abspath(p[len("build="):])
        elif p.startswith("workload="):
            wl_explicit = p[len("workload="):]
        elif p:
            ws = p
    out: list[Fixture] = []

    def skip(nm, why):
        out.append(Fixture(name=f"tp:{name}:{nm}", family="tp", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[], skip_reason=why))

    if build_root is None or not os.path.isdir(build_root):
        skip("all", f"tp spec needs :build=<root with build_workload_r<r>/> (got {build_root!r})")
        return out
    ranks = sorted(int(d[4:]) for d in os.listdir(os.path.join(row_dir, "match", "fold_match")) if d.startswith("rank") and d[4:].isdigit()) \
        if os.path.isdir(os.path.join(row_dir, "match", "fold_match")) else []
    if not ranks:
        skip("all", f"no match/fold_match/rank<r> under {row_dir}")
        return out
    gprogs = {r: os.path.join(build_root, f"build_workload_r{r}", "workload_program.json") for r in ranks}
    wl = None
    for r in ranks:
        if os.path.exists(gprogs[r]):
            wl = _resolve_workload(row_dir, json.load(open(gprogs[r])), wl_explicit)
            break
    if not wl:
        skip("all", f"workload document not found for {row_dir} (give :workload=<path>)")
        return out
    prov = {"row_dir": row_dir, "build_root": build_root, "workload": {"path": wl, "sha256": _sha256_file(wl)}, "ranks": ranks,
            "expect_source": "the GM-01 document tp's tree wrote beside each rank's fold products (global_match.json)", "mutations_here": None}
    cases = [("positive", os.path.join(row_dir, "match", "fold_match"), "the row's own record, per rank")]
    negs = os.path.join(row_dir, "negs_fold")
    if os.path.isdir(negs):
        for c in sorted(os.listdir(negs)):
            fm = os.path.join(negs, c, "fold_match")
            if os.path.isdir(fm):
                why = ""
                rp = os.path.join(negs, c, "result.json")
                if os.path.exists(rp):
                    try:
                        why = str(json.load(open(rp)).get("mutation") or "")
                    except Exception:  # noqa: BLE001
                        why = ""
                cases.append((c, fm, why))
    for case, fm, why in cases:
        for r in ranks:
            nm = f"{case}:rank{r}"
            oracle = os.path.join(fm, f"rank{r}")
            gp = gprogs[r]
            missing = [p for p in (gp, os.path.join(oracle, "program.json"), os.path.join(oracle, "instances.jsonl"),
                                   os.path.join(oracle, "capture", "log.jsonl.gz")) if not os.path.exists(p)]
            if missing:
                skip(nm, f"missing {os.path.relpath(missing[0], row_dir)} ({'the fold of this rank did not run / failed before GM-01' if case != 'positive' else 'incomplete row'})")
                continue
            doc_p = os.path.join(oracle, "global_match.json")
            if not os.path.exists(doc_p):
                skip(nm, "no GM-01 document of record beside the fold products (tp's tree did not run GM-01 on this rank)")
                continue
            doc = json.load(open(doc_p))
            expect, failed = doc.get("verdict"), list(doc.get("failed_checks") or [])
            if expect not in ("PASS", "FAIL"):
                skip(nm, f"record document verdict {expect!r}")
                continue
            args = ["--gprog", gp, "--workload", wl, "--oracle", oracle, "--within-step", ws]
            out.append(Fixture(name=f"tp:{name}:{nm}", family="tp", kind="positive" if expect == "PASS" else "negative", expect_verdict=expect,
                               expect_failed=failed, args=args, note=(why or "tp's record") + f" [record doc: {expect} {failed}]",
                               provenance=dict(prov, case=case, rank=r, record_document=doc_p, record_tree=doc.get("tree") or doc.get("revision"))))
    return out


def _component_dirs(art: dict, row_dir: str) -> dict[str, str]:
    """`inputs.derived` ("lp,t" -> component dir) re-rooted onto row_dir when the recorded (pod) path is gone."""
    out = {}
    for k, p in ((art.get("inputs") or {}).get("derived") or {}).items():
        if p and os.path.isdir(p):
            out[k] = p
        else:
            cand = os.path.join(row_dir, os.path.basename(str(p).rstrip("/")))
            if os.path.isdir(cand):
                out[k] = cand
    return out


def _row_provenance(row_dir: str, art: dict) -> dict:
    """What the harness can say about the artifact set BEFORE either checker runs: per component, is descriptor.json.gz the record's
    (sha256 == the Build artifact's component_provenance.descriptor_sha256), is derive-report.json present; was the fold re-derived
    (fold_summary.record.json beside fold_summary.json: digest equality); which files the preserving lane excluded (excluded_sha256.txt)."""
    from verity.ir.codec import canonical_json, program_digest
    prov = {"row_dir": row_dir, "components": {}, "fold": {}, "excluded": []}
    cp = (art.get("inputs") or {}).get("component_provenance") or {}
    for k, d in _component_dirs(art, row_dir).items():
        dp = os.path.join(d, "descriptor.json.gz")
        rec = cp.get(k) or {}
        sha = dig = None
        if os.path.exists(dp):
            with gzip.open(dp, "rb") as f:
                desc = json.loads(f.read())
            sha, dig = hashlib.sha256(canonical_json(desc)).hexdigest(), program_digest(desc)   # the Build artifact's own two fields
        prov["components"][k] = {"dir": d, "descriptor_present": sha is not None,
                                 "descriptor_sha256_matches_artifact": (sha == rec.get("descriptor_sha256")) if (sha and rec.get("descriptor_sha256")) else None,
                                 "program_digest_matches_artifact": (dig == rec.get("descriptor_digest")) if (dig and rec.get("descriptor_digest")) else None,
                                 "derive_report_present": os.path.exists(os.path.join(d, "derive-report.json")),
                                 "instances_present": os.path.exists(os.path.join(d, "instances.json.gz"))}
    m = os.path.join(row_dir, "match")
    fs, fr = os.path.join(m, "fold_summary.json"), os.path.join(m, "fold_summary.record.json")
    if os.path.exists(fs):
        cur = json.load(open(fs))
        prov["fold"] = {"digest": cur.get("digest"), "n_instances": cur.get("n_instances"), "n_steps": cur.get("n_steps"), "re_derived": os.path.exists(fr)}
        if os.path.exists(fr):
            rec = json.load(open(fr))
            prov["fold"]["record_digest"] = rec.get("digest")
            prov["fold"]["digest_equals_record"] = rec.get("digest") == cur.get("digest") and rec.get("n_instances") == cur.get("n_instances")
    ex = os.path.join(row_dir, "excluded_sha256.txt")
    if os.path.exists(ex):
        prov["excluded"] = [ln.split(None, 1)[1].strip() for ln in open(ex) if ln.strip() and len(ln.split(None, 1)) == 2]
    return prov


def _reconstruct_derive_reports(shadow: str, art_shadow_path: str, lp_of: dict[str, int], extra_donors: list[str] | None = None) -> list[dict]:
    """A preserved row may lack a component's derive-report.json (preservation excluded the multi-MB file; excluded_sha256.txt names it).
    `compare.load_derived` reads two things from it: the weights field map (`provenance.weights`: MODEL-level -- identical for every
    request component of the row, verified here against every sibling that has one) and the number of user inputs (`source.example_inputs`,
    one `prompt` of LP tokens for the request wrapper).  For a component without it a report carrying exactly those two things is written
    INTO THE SHADOW (never the source row), from a sibling request component's original, and named in the fixture's provenance.  The
    component Program itself (descriptor.json.gz) is always the record's own file; nothing of it is reconstructed."""
    art = json.load(open(art_shadow_path))
    dirs = _component_dirs(art, shadow)
    donors = {}
    for k, d in list(dirs.items()) + [(f"donor:{p}", p) for p in (extra_donors or [])]:
        p = os.path.join(d, "derive-report.json")
        if os.path.exists(p):
            r = json.load(open(p))
            if r.get("RECONSTRUCTED"):
                continue   # a harness-written report never donates
            prov = r.get("provenance", r)
            src = prov.get("source") if isinstance(prov.get("source"), dict) else {}
            if len(src.get("example_inputs") or src.get("inputs") or [[0]]) == 1 and prov.get("weights"):
                donors[k] = prov
    written = []
    missing = [k for k, d in dirs.items() if not os.path.exists(os.path.join(d, "derive-report.json"))]
    if not missing:
        return written
    if not donors:
        return [{"component": k, "reconstructed": False, "why": "no sibling request component with an original derive-report.json"} for k in missing]
    sig = lambda prov: json.dumps([(w.get("name"), w.get("field"), w.get("shape"), w.get("dtype"), w.get("aliases")) for w in prov["weights"]], sort_keys=True)  # noqa: E731
    sigs = {k: sig(p) for k, p in donors.items()}
    consistent = len(set(sigs.values())) == 1
    donor_k = sorted(donors)[0]
    for k in missing:
        d = dirs[k]
        if not consistent:
            written.append({"component": k, "reconstructed": False, "why": f"sibling weights maps disagree across {sorted(donors)}"})
            continue
        lp = lp_of.get(k)
        rep = {"schema": "verity/match-diff/reconstructed-derive-report/v1",
               "RECONSTRUCTED": f"written by match_diff for the harness: the original derive-report.json was not preserved with this row; "
                                f"weights field map copied from sibling component {donor_k} (model-level, identical across "
                                f"{len(donors)} sibling(s) with an original); example_inputs = one int32 prompt of LP={lp}; nothing else",
               "donor_component": donor_k, "siblings_checked": sorted(donors),
               "provenance": {"weights": donors[donor_k]["weights"], "source": {"example_inputs": [["torch.int32", [lp]]] if lp is not None else [[0]]}}}
        target = os.path.join(d, "derive-report.json")
        if os.path.islink(target):
            os.remove(target)
        _dump(rep, target)
        written.append({"component": k, "reconstructed": True, "donor": donor_k, "siblings_checked": sorted(donors), "path": target})
    return written


def build_real_fixtures(spec: str, fx_root: str, mutate_max_mb: float, mutations: bool = True) -> list:
    """`NAME=<row attempt dir>[:within_step][:workload=<path>]` -> the positive fixture + artifact-level mutations on shadow copies.
    The row attempt dir is what row_pod.sh leaves: build_workload/workload_program.json (+descriptor.json.gz), build_request*/, match/
    {program.json,instances.jsonl,capture/log.jsonl.gz,capture/tokens.json}; the fold products may be re-derived from the capture with
    `run_config --worker fold` (digest must equal the record's fold_summary.json)."""
    name, _, rest = spec.partition("=")
    parts = rest.split(":")
    row_dir = os.path.abspath(parts[0])
    ws, wl_explicit, donors = "dag", None, []
    for p in parts[1:]:
        if p.startswith("workload="):
            wl_explicit = p[len("workload="):]
        elif p.startswith("donor="):
            donors.append(os.path.abspath(p[len("donor="):]))
        elif p:
            ws = p
    art = os.path.join(row_dir, "build_workload", "workload_program.json")
    match = os.path.join(row_dir, "match")
    for p in (art, os.path.join(match, "program.json"), os.path.join(match, "instances.jsonl"), os.path.join(match, "capture", "log.jsonl.gz")):
        if not os.path.exists(p):
            return [Fixture(name=f"real:{name}", family="real", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[],
                            skip_reason=f"missing {os.path.relpath(p, row_dir)} under {row_dir}")]
    art_doc = json.load(open(art))
    wl = _resolve_workload(row_dir, art_doc, wl_explicit)
    if not wl:
        return [Fixture(name=f"real:{name}", family="real", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[],
                        skip_reason=f"workload document not found for {row_dir} (give :workload=<path>)")]
    # a cross-row donor of derive-report.json must be the same model + revision (its row's workload document says which)
    wl_doc = json.load(open(wl))
    ok_donors = []
    for d in donors:
        drow = os.path.dirname(d)
        dwl = _resolve_workload(drow, {}, None)
        dd = json.load(open(dwl)) if dwl else {}
        same = dd.get("model") == wl_doc.get("model") and dd.get("tokenizer_revision") == wl_doc.get("tokenizer_revision")
        ok_donors.append({"dir": d, "row_workload": dwl, "model": dd.get("model"), "revision": dd.get("tokenizer_revision"), "accepted": bool(same)})
    donors = [x["dir"] for x in ok_donors if x["accepted"]]
    lp_of = {f"{r['prompt_len']},{int(r['cap']) - 1}": int(r["prompt_len"]) for r in (art_doc.get("requests") or []) if "prompt_len" in r and "cap" in r}
    root = os.path.join(fx_root, "real", name)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    out: list[Fixture] = []
    provenance = _row_provenance(row_dir, art_doc)
    provenance["workload"] = {"path": wl, "sha256": _sha256_file(wl), "model": wl_doc.get("model"), "tokenizer_revision": wl_doc.get("tokenizer_revision")}
    provenance["donors"] = ok_donors
    provenance["reconstructed_derive_reports"] = None

    def gprog_args(shadow: str, extra_rewrite: set[str] | None = None) -> tuple[list, str]:
        _shadow_copy(row_dir, shadow, rewrite={"build_workload/workload_program.json", "match/capture/tokens.json"} | (extra_rewrite or set()))
        swl = os.path.join(shadow, "workload.json")
        if os.path.lexists(swl):      # a workload document stored INSIDE the row dir was just linked into the shadow: replace the link by the copy
            os.unlink(swl)
        shutil.copy2(wl, swl)          # the workload document, mutable per shadow
        ap = os.path.join(shadow, "build_workload", "workload_program.json")
        _relocate_gprog(os.path.join(row_dir, "build_workload", "workload_program.json"), shadow, ap, os.path.join(shadow, "workload.json"))
        rec = _reconstruct_derive_reports(shadow, ap, lp_of, donors)
        if provenance["reconstructed_derive_reports"] is None:
            provenance["reconstructed_derive_reports"] = rec
        return ["--gprog", ap, "--workload", os.path.join(shadow, "workload.json"), "--oracle", os.path.join(shadow, "match"), "--within-step", ws], ap

    def add(nm, kind, expect, failed, args, note="", baseline=None):
        out.append(Fixture(name=f"real:{name}:{nm}", family="real", kind=kind, expect_verdict=expect, expect_failed=list(failed), args=args,
                           note=note, baseline=baseline, provenance=provenance))

    base_name = f"real:{name}:positive"
    args, _ = gprog_args(os.path.join(root, "positive"))
    add("positive", "positive", "PASS", [], args, f"the row's own record ({row_dir})")
    _dump(provenance, os.path.join(root, "PROVENANCE.json"))
    if not mutations:
        return out
    n_req = len(json.load(open(art)).get("requests") or [])
    sz = lambda rel: os.path.getsize(os.path.join(row_dir, rel)) / 1e6  # noqa: E731
    log_mb = sz("match/capture/log.jsonl.gz")
    fold_mb = sz("match/program.json") + sz("match/instances.jsonl")
    # small-file mutations: always
    if n_req >= 2 and os.path.exists(os.path.join(match, "capture", "tokens.json")):
        args, _ = gprog_args(os.path.join(root, "swapped_outputs"))
        swap_outputs(os.path.join(root, "swapped_outputs", "match", "capture", "tokens.json"))
        add("swapped_outputs", "negative", "FAIL", ["G6"], args, "driver's generated ids of two requests swapped")
    else:
        out.append(Fixture(name=f"real:{name}:swapped_outputs", family="real", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[],
                           skip_reason=f"{n_req} request(s) in the Program: two are needed to swap outputs" if n_req < 2 else "no capture/tokens.json"))
    args, ap = gprog_args(os.path.join(root, "omit_request_program"))
    a = json.load(open(ap))
    a["requests"] = a["requests"][:-1]
    _dump(a, ap)
    add("omit_request_program", "negative", "FAIL", ["G1", "G2"], args, "the Program artifact omits the last request")
    args, ap = gprog_args(os.path.join(root, "duplicate_request_program"))
    a = json.load(open(ap))
    a["requests"].append(dict(a["requests"][-1]))
    _dump(a, ap)
    add("duplicate_request_program", "negative", "FAIL", ["G1"], args, "the Program artifact lists the last request twice")
    args, _ = gprog_args(os.path.join(root, "workload_edited"))
    w = json.load(open(os.path.join(root, "workload_edited", "workload.json")))
    w["requests"][-1]["max_tokens"] = int(w["requests"][-1]["max_tokens"]) + 1
    json.dump(w, open(os.path.join(root, "workload_edited", "workload.json"), "w"))
    add("workload_edited", "negative", "FAIL", ["G1"], args, "workload document's max_tokens edited after the Build (sha + per-request field)")
    # WRONG EOS: the Program's declared stop set names a token a request served mid-answer -> under that declaration the request stopped
    # on an EOS and then ran past it (rows after the EOS beyond the attested lag), and its driver output reports tokens after the stop
    try:
        tj = json.load(open(os.path.join(match, "capture", "tokens.json")))
        cand = None
        for r in tj.get("requests") or []:
            gen = list(r.get("generated_token_ids") or [])
            if len(gen) >= 4:
                cand = (r.get("request_id"), int(gen[len(gen) // 2]), len(gen) // 2)
                break
        if cand:
            args, ap = gprog_args(os.path.join(root, "wrong_eos_set"))
            a = json.load(open(ap))
            a.setdefault("stop_rule", {})["eos_token_ids"] = [cand[1]]
            _dump(a, ap)
            add("wrong_eos_set", "negative", "FAIL", ["G6"], args, f"Program's stop_rule.eos_token_ids := [{cand[1]}], a token {str(cand[0])[:8]} served at event {cand[2]} and ran past")
    except Exception as e:  # noqa: BLE001
        out.append(Fixture(name=f"real:{name}:wrong_eos_set", family="real", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[], skip_reason=repr(e)[:300]))
    # log rewrites: when the log is small enough to rewrite in a reasonable time
    if log_mb <= mutate_max_mb:
        args, _ = gprog_args(os.path.join(root, "lag_header_flip"), {"match/capture/log.jsonl.gz", "match/capture/versions.json"})
        shadow = os.path.join(root, "lag_header_flip")
        try:
            hdr_async = None
            with gzip.open(os.path.join(shadow, "match", "capture", "log.jsonl.gz"), "rt") as f:
                for line in f:
                    if line.startswith('{"$e": "RunHeader"') or line.startswith('{"$e":"RunHeader"'):
                        d = json.loads(line)
                        hdr_async = ((d.get("profile") or {}).get("engine_kwargs") or {}).get("async_scheduling", (d.get("engine_kwargs") or {}).get("async_scheduling"))
                        break
            if isinstance(hdr_async, bool):
                mutate_header_async(os.path.join(shadow, "match"), not hdr_async)
                add("lag_header_flip", "negative", "COND_LAG", ["G6"], args, f"header async_scheduling {hdr_async} -> {not hdr_async}", baseline=base_name)
            else:
                shutil.rmtree(shadow, ignore_errors=True)
                add("lag_header_flip", "skip", "SKIP", [], [], f"the record does not attest async_scheduling ({hdr_async!r}) and its Program declares no "
                    "eos_lag_rows: G6 reads the lag from that attestation only on an EOS stop or a declaration (M-0213 / ruling 21), so a header flip "
                    "changes nothing either checker reads here; LAG=0/1 header attacks run on toy:neg_lag0_header_on_lag1_trace / neg_lag1_header_on_lag0_trace "
                    "and on moe's LAG-declared row when it lands")
                out[-1].skip_reason = out[-1].note
        except Exception as e:  # noqa: BLE001
            out.append(Fixture(name=f"real:{name}:lag_header_flip", family="real", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[], skip_reason=repr(e)[:300]))
        try:
            hdr = json.load(open(os.path.join(row_dir, "build_workload", "workload_program.json")))
            first = (hdr.get("requests") or [{}])[0]
            rid_rec = first.get("workload_request_id") or first.get("request_id")
            args, _ = gprog_args(os.path.join(root, "broken_interface_fed_token"), {"match/capture/log.jsonl.gz"})
            ed = mutate_fed_token(os.path.join(root, "broken_interface_fed_token", "match"), str(rid_rec), int(first["prompt_len"]))
            add("broken_interface_fed_token", "negative", "FAIL", ["G6"], args, f"{rid_rec}: first decode row fed a token it never sampled {ed}")
        except Exception as e:  # noqa: BLE001
            out.append(Fixture(name=f"real:{name}:broken_interface_fed_token", family="real", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[], skip_reason=repr(e)[:300]))
    else:
        for nm in ("lag_header_flip", "broken_interface_fed_token"):
            out.append(Fixture(name=f"real:{name}:{nm}", family="real", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[],
                               skip_reason=f"log.jsonl.gz is {log_mb:.0f} MB > --mutate-max-mb {mutate_max_mb:.0f}"))
    if fold_mb <= mutate_max_mb:
        try:
            hdr = json.load(open(os.path.join(row_dir, "build_workload", "workload_program.json")))
            first = (hdr.get("requests") or [{}])[0]
            rid_rec = first.get("workload_request_id") or first.get("request_id")
            args, _ = gprog_args(os.path.join(root, "wrong_operand_same_step"), {"match/program.json", "match/instances.jsonl"})
            ed = rewire_operand(os.path.join(root, "wrong_operand_same_step", "match"), str(rid_rec), "same_step")
            add("wrong_operand_same_step", "negative", "FAIL", ["G3"], args, f"fold instance rewired to the wrong producer: {ed}")
        except Exception as e:  # noqa: BLE001
            out.append(Fixture(name=f"real:{name}:wrong_operand_same_step", family="real", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[], skip_reason=repr(e)[:300]))
    else:
        out.append(Fixture(name=f"real:{name}:wrong_operand_same_step", family="real", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[],
                           skip_reason=f"program.json + instances.jsonl are {fold_mb:.0f} MB > --mutate-max-mb {mutate_max_mb:.0f}"))
    return out


def build_rev_fixtures(cases_dir: str) -> list:
    """rev's on-disk case set (lane/r17-rev verity_capture/bench/rev_match_fixture_set.py <dir>): `<case>/gprog.json` (the global Program
    dict as check() consumes it), `ORACLE` (relative path of the fold record), `EXPECT.json` {verdict, failed_checks | failed_checks_include,
    problem_substr}.  Imported as family `rev` -- the reviewer's statement of the requirement, run through this harness on both
    implementations, so the two fixture sets and the two comparators (this one, rev_match_diff.py) meet on the same documents."""
    out: list[Fixture] = []
    cases_dir = os.path.abspath(cases_dir)
    idx = os.path.join(cases_dir, "INDEX.json")
    names = json.load(open(idx)).get("cases") if os.path.exists(idx) else sorted(d for d in os.listdir(cases_dir) if os.path.isdir(os.path.join(cases_dir, d)))
    for nm in names:
        d = os.path.join(cases_dir, nm)
        gp, orc, exp = os.path.join(d, "gprog.json"), os.path.join(d, "ORACLE"), os.path.join(d, "EXPECT.json")
        if not (os.path.exists(gp) and os.path.exists(orc) and os.path.exists(exp)):
            out.append(Fixture(name=f"rev:{nm}", family="rev", kind="skip", expect_verdict="SKIP", expect_failed=[], args=[], skip_reason=f"{d}: gprog.json / ORACLE / EXPECT.json incomplete"))
            continue
        oracle = os.path.normpath(os.path.join(d, open(orc).read().strip()))
        e = json.load(open(exp))
        verdict = e.get("verdict")
        failed = list(e.get("failed_checks") or e.get("failed_checks_include") or [])
        out.append(Fixture(name=f"rev:{nm}", family="rev", kind="positive" if verdict == "PASS" else "negative", expect_verdict=verdict, expect_failed=failed,
                           args=["--global-program", gp, "--oracle", oracle, "--within-step", "dag"], note=str(e.get("note") or ""),
                           expect_problem_substr=e.get("problem_substr")))
    return out


def run_rev_diff(script: str, python: str, a_path: str, b_path: str) -> dict:
    """rev's independent comparator (`rev_match_diff.py diff A B`) on the two documents this harness produced: a second opinion recorded
    beside ours (rc 0 = differentially equal by rev's load-bearing keys)."""
    try:
        p = subprocess.run([python, script, "diff", a_path, b_path], capture_output=True, text=True, timeout=600, cwd=REPO_ROOT)
        return {"rc": p.returncode, "tail": (p.stdout + p.stderr)[-800:]}
    except Exception as ex:  # noqa: BLE001
        return {"rc": None, "tail": repr(ex)[:300]}


# ------------------------------------------------------------------------------------------------------ running


def _ps_tree() -> tuple[dict[int, int], dict[int, int]]:
    """{pid: ppid}, {pid: rss_kb} for every process (portable: ps)."""
    try:
        txt = subprocess.run(["ps", "-o", "pid=,ppid=,rss=", "-ax"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return {}, {}
    par, rss = {}, {}
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                pid, ppid, kb = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            par[pid], rss[pid] = ppid, kb
    return par, rss


def _tree_pids(root: int, par: dict[int, int]) -> list[int]:
    kids: dict[int, list[int]] = collections.defaultdict(list)
    for p, pp in par.items():
        kids[pp].append(p)
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(kids.get(p, ()))
    return out


def _pss_kb(pid: int) -> int | None:
    p = f"/proc/{pid}/smaps_rollup"
    try:
        with open(p) as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except Exception:  # noqa: BLE001
        return None
    return None


# ---- harness SELF-TEST: sabotage of the CANDIDATE'S RESULT DOCUMENT, applied by the runner after the checker wrote it (never inside any
# checker module).  Each one is a divergence the harness must turn RED: an accept-everything candidate, a population change, a moved
# mismatch location, a renamed failed check.  `run` never sets it and refuses the gate if any run reports one.
SABOTAGE_ENV = "MATCH_DIFF_SABOTAGE"


def _sabotage(res: dict, mode: str) -> dict:
    before = json.dumps(res, sort_keys=True, default=str)
    changed = False
    if mode == "accept_all":
        res["verdict"], res["failed_checks"], res["reasons"] = "PASS", [], []
        for c in res.get("checks") or []:
            c["status"], c["problems"], c["n_problems"] = "PASS", [], 0
        if res.get("x09"):
            res["x09"]["verdict"], res["x09"]["reasons"] = "PASS", []
        if res.get("alternate_verdict"):
            res["alternate_verdict"].update({"verdict": "PASS", "failed_checks": [], "reasons": []})
    elif mode == "drop_request":
        if res.get("requests"):
            res["requests"].pop()
            changed = True
        if (res.get("x09") or {}).get("requests"):
            res["x09"]["requests"].pop()
    elif mode == "move_mismatch":
        for e in res.get("requests") or []:
            fd = e.get("first_divergence")
            if isinstance(fd, dict) and isinstance(fd.get("step"), int):
                fd["step"] += 1
                changed = True
            elif isinstance(fd, int):
                e["first_divergence"] = fd + 1
                changed = True
        if not changed:
            for c in res.get("checks") or []:
                if c.get("problems"):
                    p = c["problems"][0]
                    m = re.search(r"\d+", p)
                    if m:
                        c["problems"][0] = p[:m.start()] + str(int(m.group()) + 1) + p[m.end():]
                        changed = True
                        break
    elif mode == "rename_failed_check":
        fc = res.get("failed_checks") or []
        if fc:
            old = fc[0]
            ids = [c["id"] for c in res.get("checks") or []]
            new = next((i for i in ids if i not in fc), None)
            if new:
                fc[0] = new
                st = {c["id"]: c for c in res["checks"]}
                st[old]["status"], st[new]["status"] = st[new]["status"], st[old]["status"]
                changed = True
    else:
        raise ValueError(f"{SABOTAGE_ENV}={mode!r}: not one of {SABOTAGES}")
    changed = json.dumps(res, sort_keys=True, default=str) != before   # the document itself decides (e.g. accept_all on a positive whose ALTERNATE verdict was FAIL)
    return {"mode": mode, "changed": changed}


SABOTAGES = ("accept_all", "drop_request", "move_mismatch", "rename_failed_check")


def run_impl(fx: Fixture, impl: str, out_dir: str, python: str, poll_s: float = 0.25, timeout_s: float | None = None,
             sabotage: str | None = None, tree: str | None = None, extra_env: dict | None = None) -> dict:
    """One checker run in a fresh process (`python -m match_diff run-one`), measured from outside: wall, peak tree RSS/PSS (polled),
    plus what the runner reports from inside (rusage, rc, the checker's phases and impl).  `tree`: import the checker from another
    source tree (this file runs as a script there; `verity_capture` resolves to that tree) -- the record of a reference tree."""
    os.makedirs(out_dir, exist_ok=True)
    gm_out = os.path.join(out_dir, "global_match.json")
    run_json = os.path.join(out_dir, "run.json")
    for p in (gm_out, run_json):
        if os.path.exists(p):
            os.remove(p)
    impl_env = "record" if impl == REF_IMPL else impl
    if tree:
        cmd = [python, os.path.abspath(__file__), "run-one", "--impl", impl_env, "--run-json", run_json, "--", *fx.args, "--out", gm_out]
    else:
        cmd = [python, "-m", "verity_capture.sweep.match_diff", "run-one", "--impl", impl_env, "--run-json", run_json, "--", *fx.args, "--out", gm_out]
    env = dict(os.environ)
    env["MATCH_IMPL"] = impl_env
    if tree:
        env["PYTHONPATH"] = os.path.abspath(tree) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop(SABOTAGE_ENV, None)
    if sabotage:
        env[SABOTAGE_ENV] = sabotage
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    t0 = time.time()
    with open(os.path.join(out_dir, "stdout.txt"), "w") as so, open(os.path.join(out_dir, "stderr.txt"), "w") as se:
        proc = subprocess.Popen(cmd, cwd=os.path.abspath(tree) if tree else REPO_ROOT, env=env, stdout=so, stderr=se)
        peak_rss = peak_pss = 0
        n_procs_max = 1
        samples = 0
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            par, rss = _ps_tree()
            if par:
                pids = _tree_pids(proc.pid, par)
                n_procs_max = max(n_procs_max, len(pids))
                tot = sum(rss.get(p, 0) for p in pids)
                peak_rss = max(peak_rss, tot)
                pss = [_pss_kb(p) for p in pids]
                if all(x is not None for x in pss) and pss:
                    peak_pss = max(peak_pss, sum(pss))
                samples += 1
            if timeout_s is not None and time.time() - t0 > timeout_s:
                proc.kill()
                rc = -9
                break
            time.sleep(poll_s)
    wall = time.time() - t0
    inner = json.load(open(run_json)) if os.path.exists(run_json) else {}
    res = json.load(open(gm_out)) if os.path.exists(gm_out) else None
    # peak RSS = the larger of the polled process-tree sum (covers parallel workers; misses a run shorter than the poll) and the
    # runner's own ru_maxrss (exact for the single process, blind to its children's simultaneous footprint)
    peak_rss_mb = max(peak_rss / 1024.0, float(inner.get("maxrss_self_mb") or 0.0))
    return {"impl": impl, "rc": rc, "wall_s": round(wall, 2), "peak_rss_tree_mb": round(peak_rss_mb, 1), "peak_rss_polled_mb": round(peak_rss / 1024, 1),
            "peak_pss_tree_mb": round(peak_pss / 1024, 1) if peak_pss else None,
            "n_procs_max": n_procs_max, "rss_samples": samples, "inner": inner, "result_path": gm_out if res is not None else None,
            "verdict": (res or {}).get("verdict"), "failed_checks": (res or {}).get("failed_checks"), "checker_seconds": (res or {}).get("seconds"),
            "phases": (res or {}).get("phases"), "impl_info": (res or {}).get("impl"), "sabotage": inner.get("sabotage"),
            "stderr_tail": open(os.path.join(out_dir, "stderr.txt")).read()[-1500:] if rc not in (0, 3) else None}


def run_one(argv: list) -> int:
    """The in-process runner: `run-one --impl X --run-json P -- <global_match args>`; reports its own rusage beside the checker's output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=IMPLS)
    ap.add_argument("--run-json", required=True)
    ap.add_argument("gm_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    gm_args = a.gm_args[1:] if a.gm_args[:1] == ["--"] else a.gm_args
    os.environ["MATCH_IMPL"] = a.impl
    import resource
    from verity_capture.experimental.cb_a import global_match as gm
    t0, c0 = time.time(), time.process_time()
    rc = None
    err = None
    try:
        rc = gm.main(gm_args)
    except SystemExit as e:  # argparse errors
        rc = int(e.code) if isinstance(e.code, int) else 2
    except Exception as e:  # noqa: BLE001 -- an exception is a run failure, recorded by name
        import traceback
        err = traceback.format_exc()[-3000:]
        rc = 70
    wall = time.time() - t0
    ru_s, ru_c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    unit = 1024.0 if sys.platform == "darwin" else 1.0   # ru_maxrss: bytes on macOS, KB on Linux
    sab = None
    mode = os.environ.get(SABOTAGE_ENV)
    if mode and a.impl == "fast" and "--out" in gm_args:   # harness self-test only: the candidate's DOCUMENT is altered after the fact
        outp = gm_args[gm_args.index("--out") + 1]
        if os.path.exists(outp):
            doc = json.load(open(outp))
            sab = _sabotage(doc, mode)
            _dump(doc, outp)
    _dump({"impl": a.impl, "rc": rc, "wall_s": round(wall, 2), "cpu_self_s": round(time.process_time() - c0, 2),
           "ru_utime_s": round(ru_s.ru_utime + ru_c.ru_utime, 2), "ru_stime_s": round(ru_s.ru_stime + ru_c.ru_stime, 2),
           "maxrss_self_mb": round(ru_s.ru_maxrss / unit / 1024.0, 1), "maxrss_children_mb": round(ru_c.ru_maxrss / unit / 1024.0, 1),
           "python": sys.version.split()[0], "platform": sys.platform, "error": err, "sabotage": sab}, a.run_json)
    return rc if rc is not None else 70


# ------------------------------------------------------------------------------------------------------ comparison


def _n_requests(fx: Fixture) -> int | None:
    """Requests declared by the fixture's global Program (the sweep runs MATCH_WORKERS = N+1 so at least one worker is idle and the
    round-robin differs from every smaller setting)."""
    for flag in ("--gprog", "--global-program"):
        if flag in fx.args:
            try:
                return len(json.load(open(fx.args[fx.args.index(flag) + 1])).get("requests") or [])
            except Exception:  # noqa: BLE001
                return None
    return None


def workers_sweep_set(fx: Fixture, spec: str) -> list[int]:
    """`spec`: comma-separated ints and/or `N+1` / `N` (N = the fixture's request count); `none` = no sweep."""
    if not spec or spec.strip().lower() in ("none", "0", "off"):
        return []
    n = _n_requests(fx)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.upper().startswith("N"):
            if n is None:
                continue
            out.append(n + (int(tok[1:]) if len(tok) > 1 else 0))
        else:
            out.append(int(tok))
    return sorted({w for w in out if w >= 1})


def store_twin(fx: Fixture, twin_root: str) -> tuple[Fixture, dict] | None:
    """[F2 STORE axis, rev M-0500 (1)] the SAME fold with `instances.jsonl` re-persisted in the OTHER stored form (runs <-> progressions,
    cb_a.instances_form.convert_instances -- lossless by the property test; this is the second proof, on the documents).  The twin oracle
    dir links every other entry of the fixture's oracle dir and holds the converted file; the fixture's args point `--oracle` at it.
    None when the fixture has no oracle dir with an instances.jsonl."""
    if "--oracle" not in fx.args:
        return None
    i = fx.args.index("--oracle")
    oracle = fx.args[i + 1]
    ij = os.path.join(oracle, "instances.jsonl")
    if not os.path.isfile(ij):
        return None
    from verity_capture.experimental.cb_a import instances_form as IF
    have = IF.instances_form_of(ij)
    want = "progressions" if have == "runs" else "runs"
    twin = os.path.join(twin_root, "oracle")
    if os.path.lexists(twin):
        shutil.rmtree(twin)
    os.makedirs(twin)
    for name in os.listdir(oracle):
        if name != "instances.jsonl":
            os.symlink(os.path.abspath(os.path.join(oracle, name)), os.path.join(twin, name))
    t0 = time.time()
    res = IF.convert_instances(ij, os.path.join(twin, "instances.jsonl"), want)
    res["convert_seconds"] = round(time.time() - t0, 2)
    res.update({"from": have, "to": want, "oracle": oracle, "twin": twin})
    args = list(fx.args)
    args[i + 1] = twin
    return Fixture(name=fx.name, family=fx.family, kind=fx.kind, expect_verdict=fx.expect_verdict, expect_failed=fx.expect_failed, args=args,
                   note=f"{fx.note} [store twin: instances.jsonl as {want}]", provenance=fx.provenance, expect_problem_substr=fx.expect_problem_substr), res


def _doc_relocated(path: str, twin: str, oracle: str):
    """The twin run's document with the twin oracle path spelled as the original's (paths are not content; `oracle_dir` is volatile anyway)."""
    return json.loads(open(path).read().replace(json.dumps(twin)[1:-1], json.dumps(oracle)[1:-1]))


def _fold_load_of(impl_info) -> dict:
    """The per-fold-dir {source, instances_schema, operand_refs_form} a document states: `impl.fold_load` (every impl since rev S-r17-30 n1),
    else the fast document's `impl.config.fold_load` (int-2..int-5 fast documents), else {} (a record document before n1)."""
    ii = impl_info or {}
    return ii.get("fold_load") or ((ii.get("config") or {}).get("fold_load")) or {}


def strip_volatile(x):
    if isinstance(x, dict):
        return {k: strip_volatile(v) for k, v in x.items() if k not in VOLATILE_KEYS}
    if isinstance(x, list):
        return [strip_volatile(v) for v in x]
    return x


def deep_diff(a, b, path: str = "$", out: list | None = None, limit: int = 60) -> list:
    out = out if out is not None else []
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                out.append({"path": f"{path}.{k}", "record": "<absent>", "fast": _short(b[k])})
            elif k not in b:
                out.append({"path": f"{path}.{k}", "record": _short(a[k]), "fast": "<absent>"})
            else:
                deep_diff(a[k], b[k], f"{path}.{k}", out, limit)
            if len(out) >= limit:
                break
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": f"{path}.len", "record": len(a), "fast": len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f"{path}[{i}]", out, limit)
            if len(out) >= limit:
                break
        return out
    if a != b:
        out.append({"path": path, "record": _short(a), "fast": _short(b)})
    return out


def _short(x):
    s = json.dumps(x, default=str, sort_keys=True)
    return s if len(s) <= 200 else s[:200] + "..."


_REQ_KEYS = ("request_id", "workload_request_id", "index", "LP", "max_tokens", "n_instances", "local_steps", "token_selects", "prefill_chunks",
             "T_observed", "executed_steps_equal", "n_divergent_steps", "eos_truncated", "unexecuted_steps", "sequence_equal_steps", "operands",
             "within_step", "first_divergence", "first_step", "finish_step", "arrive_step_realized", "output_events", "answer_events", "lag_rows",
             "generated_length", "finish", "eos_verified", "eos_step", "events_fed_forward", "error", "criterion")


def population_view(res: dict) -> dict:
    """The POPULATION COUNTS both checkers must agree on, pulled out by name (rev M-0003 (b)): requests declared / in the record, fold
    instances / attributed / unattributed / engine steps (G2), identities mapped (G4), query vs fold population, unexercised declared
    VUs, boundary probes (G7), interfaces = step links + events fed forward (G6), per request instances x local steps, X-09 attribution."""
    checks = {c.get("id"): c for c in (res.get("checks") or [])}
    g = lambda cid, k: (checks.get(cid) or {}).get(k)  # noqa: E731
    reqs = res.get("requests") or []
    x = res.get("x09") or {}
    att = x.get("attribution") or {}
    conc = res.get("achieved_concurrency") or {}
    return {"requests_declared": g("G1", "n_declared"), "requests_in_record": g("G1", "n_record"),
            "fold_instances": g("G2", "n_instances"), "attributed": g("G2", "attributed"), "unattributed": g("G2", "unattributed"),
            "engine_steps": g("G2", "engine_steps"), "identities_mapped": g("G4", "mapped"),
            "population_query": g("G7", "population_query"), "population_fold": g("G7", "population_fold"),
            "unexercised_declared": g("G7", "unexercised_declared"), "boundary_probes": g("G7", "boundary_probes"),
            "boundary_probe_failures": g("G7", "boundary_probe_failures"), "step_links": g("G6", "step_links"),
            "shared_steps": g("G8", "shared_steps"), "dependency_order_violations": g("G8", "dependency_order_violations"),
            "n_requests": len(reqs),
            "per_request": {str(e.get("request_id")): {"n_instances": e.get("n_instances"), "local_steps": e.get("local_steps"),
                                                        "token_selects": e.get("token_selects"), "prefill_chunks": e.get("prefill_chunks"),
                                                        "events_fed_forward": e.get("events_fed_forward"), "generated_length": e.get("generated_length")}
                            for e in reqs},
            "x09_attributed": att.get("attributed"), "x09_unattributed": att.get("unattributed"), "x09_xreq_total": x.get("xreq_total"),
            "engine_steps_observed": conc.get("engine_steps_observed"), "engine_steps_programmed": conc.get("engine_steps_programmed"),
            "observed_max_concurrency": conc.get("observed_max")}


def classes_view(res: dict) -> dict:
    """The class-level comparison document: verdicts, per-check status + problem count + problem CLASSES, population counts, mismatch
    locations -- what rev's criterion (b) names, independent of message wording that carries run-specific numbers."""
    if not res:
        return {"missing": True}
    checks = {}
    for c in res.get("checks") or []:
        info = {k: v for k, v in c.items() if k not in ("id", "what", "status", "problems", "n_problems")}
        checks[c["id"]] = {"status": c.get("status"), "n_problems": c.get("n_problems"),
                           "problem_classes": sorted(_norm_class(p) for p in (c.get("problems") or [])),
                           "info": strip_volatile(info)}
    x = res.get("x09") or {}
    alt = res.get("alternate_verdict") or {}
    return {"verdict": res.get("verdict"), "failed_checks": res.get("failed_checks"), "within_step_criterion": res.get("within_step_criterion"),
            "reason_classes": sorted(_norm_class(p) for p in (res.get("reasons") or [])),
            "population": population_view(res),
            "checks": checks,
            "requests": [{k: e.get(k) for k in _REQ_KEYS if k in e} for e in (res.get("requests") or [])],
            "x09": {"verdict": x.get("verdict"), "reason_classes": sorted(_norm_class(p) for p in (x.get("reasons") or [])),
                    "attribution": x.get("attribution"), "xreq_total": x.get("xreq_total"),
                    "requests": [{k: e.get(k) for k in ("request_id", "LP", "T", "n_instances", "steps", "canonical_equal", "sequence_equal", "dag_multiset_equal", "criterion", "prefill_chunks")}
                                 for e in (x.get("requests") or [])]},
            "alternate": {k: alt.get(k) for k in ("criterion", "verdict", "failed_checks", "checks", "requests")},
            "achieved_concurrency": res.get("achieved_concurrency"), "eos_truncated_requests": res.get("eos_truncated_requests"),
            "eos_verified": res.get("eos_verified"), "request_ids": res.get("request_ids"), "execution": strip_volatile(res.get("execution")),
            "global_program": strip_volatile(res.get("global_program"))}


def evaluate(fx: Fixture, runs: dict, tolerance: str, baseline_results: dict) -> dict:
    """GREEN / RED / SKIP for one fixture."""
    rec, fast = runs.get("record") or {}, runs.get("fast") or {}
    res_r = json.load(open(rec["result_path"])) if rec.get("result_path") else None
    res_f = json.load(open(fast["result_path"])) if fast.get("result_path") else None
    exp = fx.expect_verdict
    reasons = []
    if exp == "COND_LAG":
        b = baseline_results.get(fx.baseline or "")
        if not b:
            exp = "SKIP"
            reasons.append("baseline result missing: expectation undecidable")
        else:
            declared = ((b.get("execution") or {}).get("declared_engine") or {}).get("async_scheduling") is not None
            eos = bool(b.get("eos_truncated_requests"))
            if declared or eos:
                exp = "FAIL"
                reasons.append(f"lag flip applicable: declared async_scheduling={declared}, eos-truncated requests={eos}")
            else:
                exp = "SKIP"
                reasons.append("lag flip not applicable to this record: the Program declares no async_scheduling and no request stopped early -> the header flip changes nothing either checker may read")
    if exp == "SKIP":
        return {"status": "SKIP", "expected": exp, "reasons": reasons + ([fx.skip_reason] if fx.skip_reason else [])}

    def ok(res, label):
        if res is None:
            reasons.append(f"{label}: no result document")
            return False
        v = res.get("verdict")
        if v != exp:
            reasons.append(f"{label}: verdict {v} != expected {exp}")
            return False
        if exp == "FAIL":
            missing = [c for c in fx.expect_failed if c not in (res.get("failed_checks") or [])]
            if missing:
                reasons.append(f"{label}: expected failed checks {missing} not among {res.get('failed_checks')}")
                return False
        if fx.expect_problem_substr:
            texts = [p for c in (res.get("checks") or []) for p in (c.get("problems") or [])] + list(res.get("reasons") or [])
            if not any(fx.expect_problem_substr in t for t in texts):
                reasons.append(f"{label}: no problem text carries {fx.expect_problem_substr!r}")
                return False
        return True
    ok_r, ok_f = ok(res_r, "record"), ok(res_f, "fast")
    strict = deep_diff(strip_volatile(res_r), strip_volatile(res_f)) if res_r is not None and res_f is not None else [{"path": "$", "record": res_r is not None, "fast": res_f is not None}]
    cls = deep_diff(classes_view(res_r), classes_view(res_f)) if res_r is not None and res_f is not None else strict
    agree = not (strict if tolerance == "strict" else cls)
    if not agree:
        reasons.append(f"record and fast disagree at tolerance {tolerance}: {len(strict if tolerance == 'strict' else cls)} difference(s)")
    if rec.get("rc") not in (0, 3):
        reasons.append(f"record rc={rec.get('rc')}")
    if fast.get("rc") not in (0, 3):
        reasons.append(f"fast rc={fast.get('rc')}")
    ref_ok, ref_diff = True, None
    ref = runs.get(REF_IMPL)
    if ref is not None:   # the record of the reference tree: the current record's document must be the same document, strictly
        res_ref = json.load(open(ref["result_path"])) if ref.get("result_path") else None
        ref_diff = deep_diff(strip_volatile(res_ref), strip_volatile(res_r)) if res_ref is not None and res_r is not None else \
            [{"path": "$", "record@ref": res_ref is not None, "record": res_r is not None}]
        ref_ok = not ref_diff and ref.get("rc") in (0, 3)
        if not ref_ok:
            reasons.append(f"record drifted from the reference tree's record: {len(ref_diff)} difference(s), record@ref rc={ref.get('rc')}")
    status = "GREEN" if (ok_r and ok_f and agree and ref_ok and rec.get("rc") in (0, 3) and fast.get("rc") in (0, 3)) else "RED"
    out = {"status": status, "expected": exp, "record_ok": ok_r, "fast_ok": ok_f, "agree_strict": not strict, "agree_class": not cls,
           "strict_diff": strict[:40], "class_diff": cls[:40], "reasons": reasons}
    if ref is not None:
        out["record_ref_equal"] = ref_ok
        out["record_ref_diff"] = (ref_diff or [])[:40]
    return out


# ------------------------------------------------------------------------------------------------------ report


def _fmt_phases(ph) -> str:
    if not ph:
        return "-"
    return " ".join(f"{p['name']}={p['dt_s']:.1f}" for p in ph)


def render_md(report: dict) -> str:
    L = [f"# match_diff -- differential harness run {report['utc']}", "",
         f"tolerance `{report['tolerance']}` · python `{report['python']}` · host `{report['host']}` · record impl `{report['impl_sources'].get('record')}` · "
         f"fast impl `{report['impl_sources'].get('fast')}` (candidate stage `{report.get('candidate_stage')}`)"
         + (" · **IDENTITY RUN: the candidate module re-exports the checker of record**" if report.get("identity_run") else "")
         + f" · jobs {report.get('jobs', 1)} of {report.get('cpu_count')} cpus" + (" (**concurrent fixtures: wall not of record**)" if (report.get('jobs') or 1) > 1 else "")
         + (f" · drift reference `{report.get('ref_tree')}` (record@ref impl `{report['impl_sources'].get(REF_IMPL)}`): record must equal it strictly" if report.get("ref_tree") else "")
         + (f" · fast re-run at MATCH_WORKERS `{report.get('workers_sweep')}`: same document required (F-r17-07)" if report.get("workers_sweep") else "")
         + (" · STORE axis: every impl re-run on the same fold with instances.jsonl in the other stored form (runs <-> progressions): same document required (F2, M-0500)" if report.get("store_sweep") else ""), "",
         f"**{report['summary']['GREEN']} GREEN / {report['summary']['RED']} RED / {report['summary']['SKIP']} SKIP** — gate {'OPEN' if report['gate'] else 'CLOSED'}", "",
         "| fixture | kind | expected | record | fast | agree strict/class | record==ref | record wall s / peak RSS MB | fast wall s / peak RSS MB | status | notes |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in report["fixtures"]:
        r, f, ev = e.get("runs", {}).get("record") or {}, e.get("runs", {}).get("fast") or {}, e["evaluation"]
        rv = f"{r.get('verdict') or '-'} {','.join(r.get('failed_checks') or []) if r.get('failed_checks') else ''}".strip()
        fv = f"{f.get('verdict') or '-'} {','.join(f.get('failed_checks') or []) if f.get('failed_checks') else ''}".strip()
        L.append(f"| {e['name']} | {e['kind']} | {ev.get('expected')} | {rv} | {fv} | "
                 f"{'y' if ev.get('agree_strict') else ('n' if ev.get('agree_strict') is not None else '-')}/{'y' if ev.get('agree_class') else ('n' if ev.get('agree_class') is not None else '-')} | "
                 f"{'-' if ev.get('record_ref_equal') is None else ('y' if ev.get('record_ref_equal') else 'N')} | "
                 f"{r.get('wall_s', '-')} / {r.get('peak_rss_tree_mb', '-')} | {f.get('wall_s', '-')} / {f.get('peak_rss_tree_mb', '-')} | **{ev['status']}** | "
                 f"{'; '.join(ev.get('reasons') or [])[:220]} |")
    L += ["", "## phases (record | fast), seconds per phase", ""]
    for e in report["fixtures"]:
        r, f = e.get("runs", {}).get("record") or {}, e.get("runs", {}).get("fast") or {}
        if r.get("phases") or f.get("phases"):
            L.append(f"- `{e['name']}`: record {_fmt_phases(r.get('phases'))} | fast {_fmt_phases(f.get('phases'))}")
    reds = [e for e in report["fixtures"] if e["evaluation"]["status"] == "RED"]
    if reds:
        L += ["", "## differences (RED fixtures)", ""]
        for e in reds:
            L.append(f"### {e['name']}")
            for d in (e["evaluation"].get("strict_diff") or [])[:25]:
                L.append(f"- `{d['path']}`: record `{d['record']}` | fast `{d['fast']}`")
            for d in (e["evaluation"].get("record_ref_diff") or [])[:25]:
                L.append(f"- DRIFT `{d['path']}`: record@ref `{d.get('record')}` | record `{d.get('fast')}`")
            for w, sw in (e.get("workers_sweep") or {}).items():
                for d in (sw.get("differences") or [])[:10]:
                    L.append(f"- MATCH_WORKERS={w} `{d['path']}`: default `{d.get('record')}` | W={w} `{d.get('fast')}`")
            for impl_, st in (e.get("store_sweep") or {}).items():
                for d in (st.get("differences") or [])[:10]:
                    L.append(f"- STORE {impl_}@{st['to']} `{d['path']}`: {st['from']} store `{d.get('record')}` | {st['to']} store `{d.get('fast')}`")
    skips = [e for e in report["fixtures"] if e["evaluation"]["status"] == "SKIP"]
    if skips:
        L += ["", "## skipped (listed, never counted green)", ""]
        for e in skips:
            L.append(f"- `{e['name']}`: {'; '.join(e['evaluation'].get('reasons') or [])}")
    L += ["", "## adversarial set coverage", "",
          "| attack (rev M-0003 (c)) | fixtures |", "|---|---|"]
    cov = {"omitted request": ["neg_omit_request_program", "neg_omit_request_record", ":omit_request_program"],
           "duplicated request": ["neg_duplicate_request_program", "neg_duplicate_request_record_header", ":duplicate_request_program"],
           "wrong operands / wrong producer version": ["neg_wrong_operand_same_step", "neg_wrong_operand_stale_version", "neg_cross_request_kv_read", "neg_component_of_wrong_request", ":wrong_operand_same_step"],
           "wrong peers (TP)": ["tp:"],
           "wrong EOS / lag": ["neg_wrong_eos_set", "neg_no_eos_declared", "neg_lag0_header_on_lag1_trace", "neg_lag1_header_on_lag0_trace", ":lag_header_flip", ":wrong_eos_set"],
           "broken interface": ["neg_broken_interface_fed_token", "neg_swapped_outputs", "neg_missing_component", "neg_manifest_edited_after_build", ":broken_interface_fed_token", ":swapped_outputs", ":workload_edited"],
           "dependency-safe permutation ACCEPTED": ["pos_permutation_dependency_safe", "pos_tgrouped_no_displacement", "pos_perm_shared_component_b4", "pos_perm_tgrouped"],
           "shared component across workers (F-r17-07): document independent of MATCH_WORKERS": ["pos_perm_shared_component_b4"],
           "consumer-before-producer / mis-emitted sampler REFUSED": ["neg_consumer_before_producer", "neg_mis_emitted_sampler"]}
    names = [e["name"] for e in report["fixtures"]]
    for k, pats in cov.items():
        hits = [n for n in names if any(p in n for p in pats)]
        st = {e["name"]: e["evaluation"]["status"] for e in report["fixtures"]}
        L.append(f"| {k} | " + ", ".join(f"{n} ({st[n]})" for n in hits) + " |")
    # real rows: what of the artifact set is the record's and what the harness re-derived / reconstructed
    seen = set()
    for e in report["fixtures"]:
        pv = e.get("provenance")
        if not pv or pv.get("row_dir") in seen:
            continue
        seen.add(pv.get("row_dir"))
        if not seen or len(seen) == 1:
            L += ["", "## real artifact sets: provenance", ""]
        fold = pv.get("fold") or {}
        L.append(f"### `{pv.get('row_dir')}`")
        L.append(f"- workload `{(pv.get('workload') or {}).get('path')}` sha256 `{str((pv.get('workload') or {}).get('sha256'))[:16]}` "
                 f"model `{(pv.get('workload') or {}).get('model')}` rev `{str((pv.get('workload') or {}).get('tokenizer_revision'))[:12]}`")
        L.append(f"- fold: digest `{str(fold.get('digest'))[:16]}` instances {fold.get('n_instances')} steps {fold.get('n_steps')}"
                 + (f" — RE-DERIVED from capture/log.jsonl.gz, digest == record's fold_summary: **{fold.get('digest_equals_record')}**" if fold.get("re_derived") else " — the record's own fold products"))
        for k, c in sorted((pv.get("components") or {}).items()):
            L.append(f"- component `{k}` `{os.path.basename(c['dir'])}`: descriptor {'present' if c['descriptor_present'] else 'MISSING'}"
                     f", canonical sha256 == artifact's: {c['descriptor_sha256_matches_artifact']}, program digest == artifact's: {c.get('program_digest_matches_artifact')}"
                     f"; derive-report original: {c['derive_report_present']}")
        for r in pv.get("reconstructed_derive_reports") or []:
            L.append(f"- derive-report.json for `{r['component']}`: {'RECONSTRUCTED (weights field map from ' + str(r.get('donor')) + ', siblings checked ' + str(r.get('siblings_checked')) + ')' if r.get('reconstructed') else 'NOT reconstructed: ' + str(r.get('why'))}")
        if pv.get("excluded"):
            L.append(f"- files the preserving lane excluded (excluded_sha256.txt): {', '.join(pv['excluded'][:12])}{' ...' if len(pv['excluded']) > 12 else ''}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------ main


def load_fixtures(fx_dir: str) -> list:
    p = os.path.join(fx_dir, "fixtures.json")
    return [Fixture.from_json(d) for d in json.load(open(p))] if os.path.exists(p) else []


def _is_tp_placeholder(f: Fixture) -> bool:
    """The five SKIP-by-name tp fixtures the toy builder emits until a real TP set is given (--tp)."""
    return f.family == "tp" and not f.args and f.name.startswith("tp:tp_") and (f.skip_reason or "").startswith("pending tp")


def save_fixtures(fx_dir: str, fixtures: list) -> None:
    _dump([f.to_json() for f in fixtures], os.path.join(fx_dir, "fixtures.json"))


def cmd_build(a) -> int:
    os.makedirs(a.fixtures_dir, exist_ok=True)
    # fixtures.json is cumulative: every family already on disk is kept unless rebuilt here (toy/tp: always unless --real-only;
    # real: per --real spec, or all with --rebuild-real; rev: per --rev-cases)
    fixtures = load_fixtures(a.fixtures_dir)
    if a.rebuild_real:
        fixtures = [f for f in fixtures if f.family != "real"]
    if not a.real_only:
        # the toy builder emits the tp PLACEHOLDERS (SKIP by name); real tp fixtures already on disk (--tp) replace them
        prev_tp = [f for f in fixtures if f.family == "tp" and not _is_tp_placeholder(f)]
        fixtures = [f for f in fixtures if f.family not in ("toy", "tp")] + build_toy_fixtures(a.fixtures_dir)
        if prev_tp:
            fixtures = [f for f in fixtures if not _is_tp_placeholder(f)] + prev_tp
    if not a.toy_only:
        for spec in a.real or []:
            nm = spec.split("=", 1)[0]
            fixtures = [f for f in fixtures if not f.name.startswith(f"real:{nm}:") and f.name != f"real:{nm}"]
            fixtures += build_real_fixtures(spec, a.fixtures_dir, a.mutate_max_mb, mutations=not a.no_real_mutations)
        for spec in getattr(a, "tp", None) or []:
            nm = spec.split("=", 1)[0]
            fixtures = [f for f in fixtures if not (f.name.startswith(f"tp:{nm}:") or _is_tp_placeholder(f))]
            fixtures += build_tp_fixtures(spec)
        if getattr(a, "rev_cases", None):
            fixtures = [f for f in fixtures if f.family != "rev"] + build_rev_fixtures(a.rev_cases)
    save_fixtures(a.fixtures_dir, fixtures)
    print(json.dumps({"fixtures_dir": a.fixtures_dir, "n": len(fixtures), "by_family": dict(collections.Counter(f.family for f in fixtures)),
                      "by_kind": dict(collections.Counter(f.kind for f in fixtures))}, indent=1))
    return 0


def cmd_run(a) -> int:
    import socket
    if a.real or getattr(a, "tp", None) or not load_fixtures(a.fixtures_dir) or a.rebuild:
        cmd_build(a)
    elif getattr(a, "rev_cases", None):
        # rev's case set is cheap to re-import and rev extends it (F-r17-07 added a 19th case): always re-read it from the dir given
        fixtures = [f for f in load_fixtures(a.fixtures_dir) if f.family != "rev"] + build_rev_fixtures(a.rev_cases)
        save_fixtures(a.fixtures_dir, fixtures)
    fixtures = load_fixtures(a.fixtures_dir)
    if a.only:
        fixtures = [f for f in fixtures if any(pat in f.name for pat in a.only)]
    if a.skip_toy:
        fixtures = [f for f in fixtures if f.family != "toy"]
    if a.family:
        fixtures = [f for f in fixtures if f.family in set(a.family)]
    if a.skip_family:
        fixtures = [f for f in fixtures if f.family not in set(a.skip_family)]
    jobs = max(1, int(a.jobs or 1))
    os.makedirs(a.out, exist_ok=True)
    report = {"schema": "verity/match-diff/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tolerance": a.tolerance, "python": a.python,
              "host": socket.gethostname(), "fixtures_dir": os.path.abspath(a.fixtures_dir), "impls": list(a.impls), "fixtures": [], "impl_sources": {},
              "jobs": jobs, "cpu_count": os.cpu_count(), "ref_tree": os.path.abspath(a.ref_tree) if a.ref_tree else None,
              "timing_note": "unprofiled wall + peak RSS per implementation, subprocess per run; jobs > 1 runs FIXTURES concurrently (never the two "
                             "implementations of one fixture at once) -- wall numbers of record come from jobs == 1 on a quiet host" if jobs > 1 else
                             "unprofiled wall + peak RSS per implementation, one subprocess at a time"}
    baseline_results: dict[str, dict] = {}

    def flush_report():
        # written after every fixture: a killed run still leaves the finished fixtures on disk
        report["summary"] = dict(collections.Counter(e["evaluation"]["status"] for e in report["fixtures"]))
        for k in ("GREEN", "RED", "SKIP"):
            report["summary"].setdefault(k, 0)
        sabotaged = any((r or {}).get("sabotage") for e in report["fixtures"] for r in (e.get("runs") or {}).values())
        report["sabotaged"] = sabotaged
        report["gate"] = report["summary"]["RED"] == 0 and report["summary"]["GREEN"] > 0 and not sabotaged
        report["identity_run"] = bool(report["impl_sources"].get("record") and report["impl_sources"].get("record") == report["impl_sources"].get("fast")) \
            or report.get("candidate_stage") == "identity"
        _dump(report, os.path.join(a.out, "match_diff.json"))
        open(os.path.join(a.out, "match_diff.md"), "w").write(render_md(report))
    lock = threading.Lock()
    sweep_spec = a.workers_sweep if "fast" in a.impls else ""
    report["workers_sweep"] = sweep_spec or None
    report["store_sweep"] = bool(getattr(a, "store_sweep", False))

    def run_fixture(fx: Fixture) -> dict:
        """One fixture end to end: every implementation in turn (never two of one fixture at once), then the evaluation."""
        entry = {"name": fx.name, "family": fx.family, "kind": fx.kind, "note": fx.note, "args": fx.args, "runs": {}, "provenance": fx.provenance}
        if fx.kind == "skip" or fx.expect_verdict == "SKIP":
            entry["evaluation"] = {"status": "SKIP", "expected": "SKIP", "reasons": [fx.skip_reason or fx.note]}
            print(f"[match_diff] SKIP {fx.name}: {fx.skip_reason}", flush=True)
            return entry
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", fx.name)
        impls = list(a.impls) + ([REF_IMPL] if a.ref_tree else [])
        for impl in impls:
            if jobs == 1:
                print(f"[match_diff] {fx.name} <- {impl} ...", flush=True, end="")
            r = run_impl(fx, impl, os.path.join(a.out, slug, impl.replace("@", "_")), a.python, timeout_s=a.timeout,
                         tree=a.ref_tree if impl == REF_IMPL else None)
            entry["runs"][impl] = r
            # [F2 / F-r17-12 / rev S-r17-30 n1] EVERY document states what each fold dir was loaded from (the record files) and the STORED
            # form read (instances.jsonl schema + operand_refs_form): shown per run, so a v1 and a v2 store of one fold are told apart in the log
            _fl = _fold_load_of(r.get("impl_info"))
            _fl_note = (" fold_load=" + ";".join(sorted({(v.get("operand_refs_form") or "?") if isinstance(v, dict) else str(v) for v in _fl.values()}))) if _fl else ""
            print(f"{'' if jobs == 1 else '[match_diff] ' + fx.name + ' <- ' + impl + ' ...'} {r['verdict']} rc={r['rc']} wall={r['wall_s']}s rss={r['peak_rss_tree_mb']}MB{_fl_note}", flush=True)
            with lock:
                if r.get("impl_info"):
                    report["impl_sources"].setdefault(impl, r["impl_info"].get("source_sha256"))
                    if impl == "fast":
                        report.setdefault("candidate_stage", r["impl_info"].get("candidate_stage"))
                if r.get("result_path") and fx.kind == "positive" and impl == "record":
                    baseline_results[fx.name] = json.load(open(r["result_path"]))
            # [F2 STORE axis, rev M-0500 (1)] the document must not depend on the STORED form of instances.jsonl: the same fold with the file
            # re-persisted in the other form (runs <-> progressions) -> record == record, fast == fast, strictly (volatile keys aside), and
            # the fast twin's impl.config.fold_load must NAME the other form
            if getattr(a, "store_sweep", False) and impl != REF_IMPL and r.get("result_path") and fx.kind != "skip":
                tw = store_twin(fx, os.path.join(a.out, slug, "store_twin"))
                if tw is not None:
                    tfx, tinfo = tw
                    rt = run_impl(tfx, impl, os.path.join(a.out, slug, f"{impl}_store_{tinfo['to']}"), a.python, timeout_s=a.timeout)
                    st = {"from": tinfo["from"], "to": tinfo["to"], "bytes_from": tinfo["bytes_in"], "bytes_to": tinfo["bytes_out"], "items_from": tinfo["items_in"],
                          "items_to": tinfo["items_out"], "convert_seconds": tinfo["convert_seconds"], "rc": rt["rc"], "verdict": rt["verdict"], "wall_s": rt["wall_s"],
                          "peak_rss_tree_mb": rt["peak_rss_tree_mb"], "wall_s_original": r["wall_s"]}
                    if rt.get("result_path"):
                        dt = deep_diff(strip_volatile(json.load(open(r["result_path"]))), strip_volatile(_doc_relocated(rt["result_path"], tinfo["twin"], tinfo["oracle"])))
                        st["differences"] = dt
                        st["equal"] = not dt
                        # the twin's document must NAME the form it read for the converted fold dir (the per-request component folds
                        # the same run loads are untouched and stay in their own form) -- the fast one (impl.config.fold_load, M-0500 (2))
                        # and, since rev S-r17-30 n1, the RECORD one too (impl.fold_load); a record checker that predates n1 (a ref tree,
                        # a release before the flip) names nothing: recorded as such, not RED (the writer default flips on the tree that names)
                        fl = _fold_load_of(rt.get("impl_info"))
                        named = (fl.get(tinfo["twin"]) or fl.get(os.path.realpath(tinfo["twin"])) or {}).get("operand_refs_form")
                        st["fast_names_form"] = (named == tinfo["to"]) if impl == "fast" else None
                        st["record_names_form"] = (named == tinfo["to"]) if impl == "record" else None
                        if impl == "record" and not fl:
                            st["record_names_form"] = None      # pre-n1 record checker: the document carries no fold_load at all
                        if named != tinfo["to"] and (impl == "fast" or fl):
                            st["equal"] = False
                            st["differences"] = dt + [{"path": f"$.impl{'.config' if impl == 'fast' else ''}.fold_load[{tinfo['twin']}]", "record": tinfo["to"], "fast": named}]
                    else:
                        st["equal"] = False
                        st["differences"] = [{"path": "$", "record": f"{impl} document", "fast": f"no document on the {tinfo['to']} store (rc={rt['rc']})"}]
                    entry.setdefault("store_sweep", {})[impl] = st
                    print(f"[match_diff]   {fx.name} {impl}@store={tinfo['to']}: {'same document' if st['equal'] else 'DIFFERENT DOCUMENT'} wall={rt['wall_s']}s "
                          f"(instances.jsonl {tinfo['bytes_in']} -> {tinfo['bytes_out']} bytes, {tinfo['items_in']} -> {tinfo['items_out']} items)", flush=True)
            # [rev F-r17-07] the candidate's DOCUMENT must be a function of the artifacts alone: the same fast run at MATCH_WORKERS in the
            # sweep set (1, 2 and N+1 by default) must equal the default run strictly (volatile keys aside)
            if impl == "fast" and r.get("result_path") and sweep_spec:
                base_doc = strip_volatile(json.load(open(r["result_path"])))
                entry["workers_sweep"] = {}
                for w in workers_sweep_set(fx, sweep_spec):
                    rw = run_impl(fx, impl, os.path.join(a.out, slug, f"fast_w{w}"), a.python, timeout_s=a.timeout, extra_env={"MATCH_WORKERS": str(w)})
                    sw = {"rc": rw["rc"], "verdict": rw["verdict"], "wall_s": rw["wall_s"], "peak_rss_tree_mb": rw["peak_rss_tree_mb"], "n_procs_max": rw["n_procs_max"]}
                    if rw.get("result_path"):
                        dw = deep_diff(base_doc, strip_volatile(json.load(open(rw["result_path"]))))
                        sw["differences"] = dw
                        sw["equal"] = not dw
                    else:
                        sw["equal"] = False
                        sw["differences"] = [{"path": "$", "record": "fast default document", "fast": f"no document at MATCH_WORKERS={w} (rc={rw['rc']})"}]
                    entry["workers_sweep"][str(w)] = sw
                    print(f"[match_diff]   {fx.name} fast@MATCH_WORKERS={w}: {'same document' if sw['equal'] else 'DIFFERENT DOCUMENT'} wall={rw['wall_s']}s", flush=True)
        entry["evaluation"] = evaluate(fx, entry["runs"], a.tolerance, baseline_results)
        for w, sw in (entry.get("workers_sweep") or {}).items():
            if not sw["equal"]:
                entry["evaluation"]["status"] = "RED"
                entry["evaluation"].setdefault("reasons", []).append(
                    f"fast document depends on MATCH_WORKERS: {len(sw['differences'])} difference(s) at MATCH_WORKERS={w} vs default (first: {sw['differences'][0]['path']})")
        for impl_, st in (entry.get("store_sweep") or {}).items():
            if not st["equal"]:
                entry["evaluation"]["status"] = "RED"
                entry["evaluation"].setdefault("reasons", []).append(
                    f"{impl_} document depends on the STORED form of instances.jsonl: {len(st['differences'])} difference(s) on the {st['to']} store (first: {st['differences'][0]['path']})")
        if a.rev_diff and (entry["runs"].get("record") or {}).get("result_path") and (entry["runs"].get("fast") or {}).get("result_path"):
            rd = run_rev_diff(a.rev_diff, a.python, entry["runs"]["record"]["result_path"], entry["runs"]["fast"]["result_path"])
            entry["evaluation"]["rev_diff"] = rd
            if rd.get("rc") != 0:
                entry["evaluation"]["status"] = "RED"
                entry["evaluation"].setdefault("reasons", []).append(f"rev_match_diff rc={rd.get('rc')}: {rd.get('tail', '')[-200:]}")
        print(f"[match_diff]   -> {entry['evaluation']['status']} {fx.name if jobs > 1 else ''} {'; '.join(entry['evaluation'].get('reasons') or [])}", flush=True)
        return entry

    # positives first (COND expectations read their baseline), then the rest in manifest order; `jobs` fixtures at a time per phase
    positives = [f for f in fixtures if f.kind == "positive"]
    others = [f for f in fixtures if f.kind != "positive"]
    for phase in (positives, others):
        if not phase:
            continue
        if jobs == 1:
            for fx in phase:
                report["fixtures"].append(run_fixture(fx))
                flush_report()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
                futs = {pool.submit(run_fixture, fx): fx for fx in phase}
                for fut in concurrent.futures.as_completed(futs):
                    with lock:
                        report["fixtures"].append(fut.result())
                        flush_report()
            # manifest order in the report regardless of completion order
            pos = {f.name: i for i, f in enumerate(fixtures)}
            report["fixtures"].sort(key=lambda e: pos.get(e["name"], 1 << 30))
    if not report["fixtures"]:
        flush_report()
    else:
        flush_report()
    print(json.dumps({"summary": report.get("summary"), "gate": report.get("gate"), "identity_run": report.get("identity_run"),
                      "sabotaged": report.get("sabotaged"), "out": os.path.abspath(a.out)}, indent=1))
    return 0 if report.get("gate") else 3


SELFTEST_DEFAULT_ONLY = ("toy:pos_b2_unequal_shapes", "toy:pos_permutation_dependency_safe", "toy:neg_wrong_operand_same_step",
                         "toy:neg_omit_request_program", "toy:neg_broken_interface_fed_token", "toy:neg_lag0_header_on_lag1_trace")


def cmd_selftest(a) -> int:
    """Proves the harness DETECTS: the record runs once per fixture; the candidate (identity) runs once per sabotage of its result
    document; every sabotage that changed the document must evaluate RED at `strict`, and every population / check-name sabotage also
    at `class`.  A harness that stays GREEN under any of them cannot gate anything."""
    if not load_fixtures(a.fixtures_dir) or a.rebuild:
        a.real, a.toy_only, a.real_only = [], True, False
        cmd_build(a)
    fixtures = [f for f in load_fixtures(a.fixtures_dir) if f.kind != "skip" and f.expect_verdict not in ("SKIP", "COND_LAG")]
    only = a.only or list(SELFTEST_DEFAULT_ONLY)
    fixtures = [f for f in fixtures if any(p in f.name for p in only)]
    os.makedirs(a.out, exist_ok=True)
    rows, failures = [], []
    for fx in fixtures:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", fx.name)
        rec = run_impl(fx, "record", os.path.join(a.out, slug, "record"), a.python, timeout_s=a.timeout)
        for mode in SABOTAGES:
            fast = run_impl(fx, "fast", os.path.join(a.out, slug, f"fast_{mode}"), a.python, timeout_s=a.timeout, sabotage=mode)
            runs = {"record": rec, "fast": fast}
            ev_s, ev_c = evaluate(fx, runs, "strict", {}), evaluate(fx, runs, "class", {})
            changed = bool((fast.get("sabotage") or {}).get("changed"))
            # what MUST be red: any changed document at strict; population and check-name changes also at class
            must_red_strict = changed
            must_red_class = changed and mode in ("accept_all", "drop_request", "rename_failed_check")
            ok = (not must_red_strict or ev_s["status"] == "RED") and (not must_red_class or ev_c["status"] == "RED") \
                and (changed or (ev_s["status"] == "GREEN" and ev_c["status"] == "GREEN"))
            row = {"fixture": fx.name, "kind": fx.kind, "sabotage": mode, "changed": changed, "strict": ev_s["status"], "class": ev_c["status"],
                   "ok": ok, "strict_reasons": ev_s.get("reasons"), "class_reasons": ev_c.get("reasons"),
                   "strict_diff_paths": [d["path"] for d in (ev_s.get("strict_diff") or [])[:6]]}
            rows.append(row)
            if not ok:
                failures.append(row)
            print(f"[selftest] {fx.name:48s} {mode:20s} changed={changed!s:5s} strict={ev_s['status']:5s} class={ev_c['status']:5s} {'ok' if ok else 'NOT DETECTED'}", flush=True)
    if not rows:
        failures.append({"fixture": None, "sabotage": None, "ok": False, "strict_reasons": [f"no fixture matched {only} in {a.fixtures_dir}"]})
    n_changed = sum(r["changed"] for r in rows)
    if rows and n_changed == 0:
        failures.append({"fixture": None, "sabotage": None, "ok": False, "strict_reasons": ["no sabotage changed any document: nothing was tested"]})
    rep = {"schema": "verity/match-diff-selftest/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows, "failures": failures,
           "ok": not failures, "n_rows": len(rows), "n_changed": n_changed,
           "n_red_strict": sum(r["strict"] == "RED" for r in rows), "n_red_class": sum(r["class"] == "RED" for r in rows)}
    _dump(rep, os.path.join(a.out, "selftest.json"))
    lines = [f"# match_diff self-test {rep['utc']} -- {'OK' if rep['ok'] else 'FAILED'}: {rep['n_changed']} sabotaged documents, "
             f"{rep['n_red_strict']} RED at strict, {rep['n_red_class']} RED at class, {len(failures)} undetected", "",
             "| fixture | kind | sabotage | changed | strict | class | ok |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['fixture']} | {r['kind']} | {r['sabotage']} | {r['changed']} | {r['strict']} | {r['class']} | {'y' if r['ok'] else 'NOT DETECTED'} |" for r in rows]
    lines += ["", "A positive fixture's document is unchanged by `accept_all` / `move_mismatch` / `rename_failed_check` (nothing to accept, no",
              "mismatch, no failed check) and stays GREEN -- an accept-everything candidate is caught by the negatives, which is why they are",
              "part of the gate.  `move_mismatch` alters a problem TEXT's number when a request has no `first_divergence`: caught at strict, not",
              "necessarily at class (class normalises numbers) -- strict is the gate's default tolerance."]
    open(os.path.join(a.out, "selftest.md"), "w").write("\n".join(lines) + "\n")
    print(json.dumps({k: rep[k] for k in ("ok", "n_rows", "n_changed", "n_red_strict", "n_red_class")} | {"out": os.path.abspath(a.out)}, indent=1))
    return 0 if rep["ok"] else 3


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run-one"]:
        return run_one(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--fixtures-dir", required=True)
        p.add_argument("--real", action="append", default=[], help="NAME=<row attempt dir>[:within_step][:workload=<path>][:donor=<component dir>] (repeatable)")
        p.add_argument("--tp", action="append", default=[], help="NAME=<tp row dir>:build=<build root>[:workload=<path>]: tp's per-rank fold products + negative set as family `tp` (repeatable)")
        p.add_argument("--rev-cases", default=None, help="rev's on-disk case set (rev_match_fixture_set.py <dir>) imported as family `rev`")
        p.add_argument("--rev-diff", default=None, help="path to rev_match_diff.py: run its `diff` on every record/fast document pair as a second opinion (rc != 0 -> RED)")
        p.add_argument("--mutate-max-mb", type=float, default=512.0, help="rewrite log / fold files for mutations only under this size")
        p.add_argument("--no-real-mutations", action="store_true")
        p.add_argument("--toy-only", action="store_true")
        p.add_argument("--real-only", action="store_true")
        p.add_argument("--rebuild-real", action="store_true")
    pb = sub.add_parser("build", help="build the fixture set on disk (fixtures.json)")
    common(pb)
    pr = sub.add_parser("run", help="run record + fast on every fixture, compare, report")
    common(pr)
    pr.add_argument("--out", required=True)
    pr.add_argument("--python", default=sys.executable)
    pr.add_argument("--impls", nargs="+", default=list(IMPLS), choices=IMPLS)
    pr.add_argument("--tolerance", choices=("strict", "class"), default="strict")
    pr.add_argument("--only", action="append", default=[], help="substring filter on fixture names (repeatable)")
    pr.add_argument("--skip-toy", action="store_true")
    pr.add_argument("--family", action="append", default=[], choices=("toy", "rev", "real", "tp"), help="run only these families (repeatable)")
    pr.add_argument("--skip-family", action="append", default=[], choices=("toy", "rev", "real", "tp"))
    pr.add_argument("--jobs", type=int, default=1, help="fixtures run concurrently (each fixture's implementations stay sequential); >1 = timings are not of record")
    pr.add_argument("--workers-sweep", default="1,2,N+1", help="[F-r17-07] re-run the fast impl at these MATCH_WORKERS (N = the fixture's request count) and require the "
                                                             "SAME document as the default fast run; `none` to skip (e.g. the real family on a wall-of-record run)")
    pr.add_argument("--store-sweep", action="store_true", help="[F2, rev M-0500 (1)] re-run every impl on the same fold with instances.jsonl converted to the OTHER stored form "
                                                               "(runs <-> progressions; cb_a.instances_form) and require the SAME document; the fast twin must name the form it read")
    pr.add_argument("--ref-tree", default=None, help="a source tree whose checker of record is the DRIFT REFERENCE (e.g. a checkout of lane/r17-base): "
                                                    "its record document must equal this tree's record document strictly, else RED (the record itself may not change)")
    pr.add_argument("--rebuild", action="store_true")
    pr.add_argument("--timeout", type=float, default=None, help="seconds per run; a killed run is RED")
    ps = sub.add_parser("self-test", help="prove the harness turns RED under sabotage of the candidate's result document")
    common(ps)
    ps.add_argument("--out", required=True)
    ps.add_argument("--python", default=sys.executable)
    ps.add_argument("--only", action="append", default=[], help="substring filter on fixture names (default: a positive + negatives of each class)")
    ps.add_argument("--rebuild", action="store_true")
    ps.add_argument("--timeout", type=float, default=None)
    pt = sub.add_parser("ref-tree", help="export the checker's source packages at a git commit into a directory usable as `run --ref-tree` (git archive; no checkout, no branch touched)")
    pt.add_argument("--commit", required=True, help="commit / ref whose checker of record is the drift reference (e.g. lane/r17-base, or the commit before a refactor)")
    pt.add_argument("--out", required=True)
    pt.add_argument("--repo", default=REPO_ROOT)
    pa = sub.add_parser("audit", help="[COORD R17-37 (2)] record audit of a row whose Match of record ran MATCH_IMPL=fast: the checker of record on the same preserved artifacts, strict compare")
    pa.add_argument("--name", required=True, help="row name for the report (e.g. qwen_b8_1024_128)")
    pa.add_argument("--row", required=True, help="the row attempt dir (pulled read-only): build_workload/ + build_request_*/ + match/ (fold products)")
    pa.add_argument("--out", required=True)
    pa.add_argument("--fast-doc", default=None, help="the fast Match of record's global_match.json (the document the audit is against)")
    pa.add_argument("--record-doc", default=None, help="a GM-01 document of record from another tree (verdict-level comparison only, class diff listed)")
    pa.add_argument("--record-tree", default=None, help="run the checker of record from this tree (a `ref-tree` export of the release the fast verdict ran on); default: this tree")
    pa.add_argument("--also-fast", action="store_true", help="also run this tree's MATCH_IMPL=fast on the same artifacts (the differential for a row with no fast document yet)")
    pa.add_argument("--fast-compare", default=None, help="[F-r17-15 (c)] the fast verdict's fast_compare.json (exec's tool: artifacts_sha256 + tree); default: beside --fast-doc")
    pa.add_argument("--allow-tree-mismatch", default=None, metavar="REASON", help="[F-r17-15 (b)] audit across trees anyway; the REASON is written into the report (else RED by name)")
    pa.add_argument("--allow-fast-read-relocated", default=None, metavar="REASON", help="[rev M-0774 n1] accept that the fast checker read other bytes than the row's originals "
                    "(a harness-written fast_compare.json's artifacts_read_sha256 != the row's sha256 beyond workload_program.json: a store twin); the REASON is written into the report (else RED by name)")
    pa.add_argument("--reuse-runs", action="store_true", help="re-emit the report from record/fast runs already under --out/<slug>/ (no re-run)")
    pa.add_argument("--record-result", default=None, metavar="GLOBAL_MATCH_JSON", help="[R17-37 dual path] the checker of record ALREADY ran to completion on the row "
                    "(row_pod's record leg beside the fast one): take its document as the record half instead of re-running -- accepted by CONTENT only "
                    "(impl.name record, impl.source_sha256 == the record tree's global_match.py, oracle_dir = the row), else RED by name")
    pa.add_argument("--workload", default=None)
    pa.add_argument("--donor", action="append", default=[], help="component dir of the same model+revision whose derive-report.json stands in for a missing one (as --real :donor=)")
    pa.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "run": cmd_run, "self-test": cmd_selftest, "ref-tree": cmd_ref_tree, "audit": cmd_audit}[a.cmd](a)


REF_TREE_PATHS = ("integrations/vllm/verity_capture", "integrations/vllm/verity_vllm", "integrations/vllm/pyproject.toml",
                  "packages/verity/src/verity")   # what the checker imports (torch-free): the packages of record, repo-root relative


def _tree_commit(tree: str | None) -> str | None:
    """The commit a source tree stands at: REF_TREE.json (a `ref-tree` export), EXPORT.json (ship.sh), else git."""
    if not tree:
        return None
    for name, key in (("REF_TREE.json", "commit"), ("EXPORT.json", "commit")):
        p = os.path.join(tree, name)
        if os.path.exists(p):
            try:
                v = json.load(open(p)).get(key)
                if v:
                    return str(v)
            except Exception:
                pass
    try:
        return subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def _same_commit(a: str | None, b: str | None) -> bool:
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    return bool(a and b and len(a) >= 7 and len(b) >= 7 and (a.startswith(b) or b.startswith(a)))


FAST_COMPARE_FOUR = ("match/program.json", "match/instances.jsonl", "build_workload/workload_program.json", "match/capture/tokens.json")


def write_fast_compare(fx: "Fixture", doc_path: str, tree: str, row: str | None, run: dict | None = None) -> str:
    """[COORD R17-37 / rev F-r17-15 (c)] fast_compare.json beside a FAST document this harness produced -- exec's schema (fast_compare_pod.sh):
    `tree` = the commit the fast checker ran from (EXPORT.json / REF_TREE.json / git), `artifacts_sha256` = the FULL sha256 of the four files
    of the row the fixture stands on (the row's originals; the shadow's relocated copy of workload_program.json is recorded beside, by name),
    the impl block, env, verdict.  The record audit compares its own four shas to these."""
    doc = json.load(open(doc_path))
    row = os.path.abspath(row) if row else ((fx.provenance or {}).get("row_dir") if fx.provenance else None)
    shadow = os.path.dirname(fx.args[fx.args.index("--oracle") + 1]) if "--oracle" in fx.args else None
    shas, read = {}, {}
    for rel in FAST_COMPARE_FOUR:
        rp = os.path.join(row, rel) if row else None
        sp = os.path.join(shadow, rel) if shadow else None
        if rp and os.path.exists(rp):
            shas[rel] = _sha256_file(rp)
        elif sp and os.path.exists(sp) and rel != "build_workload/workload_program.json":
            shas[rel] = _sha256_file(sp)     # no row dir known: the shadow's file IS the original for everything the shadow did not rewrite
        if sp and os.path.exists(sp):
            read[rel] = _sha256_file(sp)
    fi = doc.get("impl") if isinstance(doc.get("impl"), dict) else {}
    out = {"tool": "mfast match_diff write_fast_compare (COORD R17-37 (2) / rev F-r17-15 (c))", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tree": _tree_commit(tree), "tree_path": os.path.abspath(tree), "row": row, "fixture": fx.name, "shadow": shadow,
           "env": {k: os.environ.get(k) for k in ("MATCH_IMPL", "MATCH_WORKERS", "MATCH_COMPACT") if os.environ.get(k)} | ({"MATCH_WORKERS": str(((fi.get("config") or {}).get("match_workers")))} if (fi.get("config") or {}).get("match_workers") is not None else {}),
           "impl": {k: fi.get(k) for k in ("name", "module", "source_sha256", "candidate_stage")}, "verdict": doc.get("verdict"), "failed_checks": doc.get("failed_checks"),
           "checker_seconds": doc.get("seconds"), "wall_s": (run or {}).get("wall_s"), "peak_rss_tree_mb": (run or {}).get("peak_rss_tree_mb"),
           "artifacts_sha256": shas, "artifacts_read_sha256": read,
           "fast_document": {"path": os.path.abspath(doc_path), "sha256_16": _sha256_file(doc_path)[:16]}}
    p = os.path.join(os.path.dirname(doc_path), "fast_compare.json")
    _dump(out, p)
    return p


def _row_release_commit(row: str) -> tuple[str | None, str | None]:
    """The release commit a row's stages ran on: attempt.json release_commit, commit/versions.json tree.commit, release.json, stages.txt commit=."""
    row = os.path.abspath(row)
    p = os.path.join(row, "attempt.json")
    if os.path.exists(p):
        try:
            v = json.load(open(p)).get("release_commit")
            if v:
                return str(v), "attempt.json release_commit"
        except Exception:
            pass
    p = os.path.join(row, "commit", "versions.json")
    if os.path.exists(p):
        try:
            v = ((json.load(open(p)).get("tree") or {}).get("commit"))
            if v:
                return str(v), "commit/versions.json tree.commit"
        except Exception:
            pass
    p = os.path.join(row, "release.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            v = d.get("commit") or (d.get("tree") or {}).get("commit") if isinstance(d, dict) else None
            if v:
                return str(v), "release.json commit"
        except Exception:
            pass
    p = os.path.join(row, "stages.txt")
    if os.path.exists(p):
        m = re.search(r"commit=([0-9a-f]{7,40})", open(p).read())
        if m:
            return m.group(1), "stages.txt commit="
    return None, None


def cmd_audit(a) -> int:
    """[COORD R17-37 (2)] The RECORD AUDIT of a row whose Match of record ran MATCH_IMPL=fast: the checker of record, run unbounded on the
    SAME preserved artifacts (the row dir pulled read-only: Build components + match/ fold products), from the tree the fast verdict was
    produced on (`--record-tree`, e.g. the release tree) or this one; then the strict compare of the two documents (deep_diff over the
    volatile-stripped text, the class view beside it) and the population view of both.  Optionally (`--also-fast`) this tree's fast
    document on the same artifacts too, so a row with no fast document yet gets its differential here.  GREEN = the record checker's
    document equals the fast Match of record STRICTLY (and, if a document of record from another tree is given with --record-doc, the
    record checker's verdict + failed checks equal its).  Writes audit.json + audit.md; exit 0 iff GREEN."""
    import socket
    os.makedirs(a.out, exist_ok=True)
    spec = f"{a.name}={a.row}" + (f":workload={a.workload}" if a.workload else "") + "".join(f":donor={d}" for d in (a.donor or []))
    fixtures = build_real_fixtures(spec, a.out, 0.0, mutations=False)
    pos = [f for f in fixtures if f.kind == "positive"]
    if not pos:
        print(json.dumps({"status": "RED", "reason": "no positive fixture from the row", "fixtures": [f.to_json() for f in fixtures]}, indent=1))
        return 3
    fx = pos[0]
    report = {"schema": "verity/match-audit/v2", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
              "python": a.python, "row": os.path.abspath(a.row), "name": a.name, "fixture": fx.to_json(),
              "record_tree": os.path.abspath(a.record_tree) if a.record_tree else REPO_ROOT, "this_tree": REPO_ROOT, "this_tree_commit": _tree_commit(REPO_ROOT),
              "fast_doc": os.path.abspath(a.fast_doc) if a.fast_doc else None, "record_doc": os.path.abspath(a.record_doc) if a.record_doc else None,
              "runs": {}, "compare": {}, "reuse_runs": bool(a.reuse_runs)}
    report["record_tree_commit"] = _tree_commit(a.record_tree) if a.record_tree else report["this_tree_commit"]
    reasons = []
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", fx.name)

    def _run(impl: str, tree: str | None) -> dict:
        d = os.path.join(a.out, slug, impl)
        if a.reuse_runs and os.path.exists(os.path.join(d, "global_match.json")) and os.path.exists(os.path.join(d, "run.json")):
            inner = json.load(open(os.path.join(d, "run.json")))
            res = json.load(open(os.path.join(d, "global_match.json")))
            return {"impl": impl, "rc": inner.get("rc"), "wall_s": inner.get("wall_s"), "peak_rss_tree_mb": inner.get("maxrss_self_mb"),
                    "peak_pss_tree_mb": None, "inner": inner, "result_path": os.path.join(d, "global_match.json"), "verdict": res.get("verdict"),
                    "failed_checks": res.get("failed_checks"), "checker_seconds": res.get("seconds"), "phases": res.get("phases"), "impl_info": res.get("impl"),
                    "reused": True}
        return run_impl(fx, impl, d, a.python, tree=tree)

    if a.also_fast:
        print(f"[audit] {fx.name} <- fast (this tree) ...", flush=True, end="")
        rf = _run("fast", None)
        report["runs"]["fast"] = rf
        print(f" {rf.get('verdict')} rc={rf.get('rc')} wall={rf.get('wall_s')}s rss={rf.get('peak_rss_tree_mb')}MB{' (reused)' if rf.get('reused') else ''}", flush=True)
        if not a.fast_doc and rf.get("result_path"):
            report["fast_doc"] = rf["result_path"]
            if not rf.get("reused") or not os.path.exists(os.path.join(os.path.dirname(rf["result_path"]), "fast_compare.json")):
                write_fast_compare(fx, rf["result_path"], REPO_ROOT, a.row, rf)
    if getattr(a, "record_result", None):
        # [COORD R17-37 dual path, fp8 M-0949] the checker of record ALREADY RAN TO COMPLETION on the same artifacts (row_pod's record leg was
        # not killed while the fast one ran beside it): its document is the record half of this audit -- accepted BY CONTENT, never by path:
        # (i) its impl block names the record checker (name == record) and its source_sha256 == the sha256 of the record tree's
        # global_match.py (the tree check of F-r17-15 (b), by content); (ii) its oracle_dir is the row's own match/ (the same fold products
        # the four-artifact identity below hashes); (iii) the four artifact sha256s below are the row's files it read.  Nothing is re-run.
        rp = os.path.abspath(a.record_result)
        res_r = json.load(open(rp)) if os.path.exists(rp) else None
        ri0 = (res_r or {}).get("impl") if isinstance((res_r or {}).get("impl"), dict) else {}
        tree_gm = os.path.join(report["record_tree"], "verity_capture", "experimental", "cb_a", "global_match.py")
        tree_sha = _sha256_file(tree_gm) if os.path.exists(tree_gm) else None
        odir = os.path.realpath(str((res_r or {}).get("oracle_dir") or ""))
        row_match = os.path.realpath(os.path.join(os.path.abspath(a.row), "match"))
        # the row is usually PULLED to the audit host: the document's oracle_dir then names the row by its directory NAME (…/<row>/match);
        # content identity is the strict compare below (workload sha256, Program digest, X-09 canonical digests are all in the document)
        # or, the dual path proper: the FAST document of record names the same oracle_dir (both checkers ran in place on the pod's row)
        _fd = json.load(open(a.fast_doc)) if a.fast_doc and os.path.exists(a.fast_doc) else {}
        fast_odir = os.path.realpath(str(_fd.get("oracle_dir") or "")) if _fd.get("oracle_dir") else None
        odir_ok = bool(odir) and (odir == row_match or (os.path.basename(odir) == "match" and os.path.basename(os.path.dirname(odir)) == os.path.basename(os.path.abspath(a.row).rstrip("/")))
                                  or (fast_odir is not None and odir == fast_odir))
        rr = {"impl": "record", "external": True, "result_path": rp if res_r is not None else None, "rc": 0 if (res_r or {}).get("verdict") == "PASS" else (3 if res_r is not None else None),
              "wall_s": None, "peak_rss_tree_mb": None, "checker_seconds": (res_r or {}).get("seconds"), "phases": (res_r or {}).get("phases"), "impl_info": ri0 or None,
              "identity": {"impl_name": ri0.get("name"), "source_sha256": ri0.get("source_sha256"), "record_tree_global_match_sha256": tree_sha,
                           "source_equals_record_tree": bool(tree_sha and ri0.get("source_sha256") == tree_sha), "oracle_dir": (res_r or {}).get("oracle_dir"),
                           "oracle_dir_is_the_row": odir_ok,
                           "oracle_dir_match": "path" if odir == row_match else (("name (pulled row)" if os.path.basename(os.path.dirname(odir)) == os.path.basename(os.path.abspath(a.row).rstrip("/")) else "same oracle_dir as the fast document") if odir_ok else None),
                           "utc": (res_r or {}).get("utc")},
              "verdict": (res_r or {}).get("verdict"), "failed_checks": (res_r or {}).get("failed_checks")}
        report["runs"]["record"] = rr
        print(f"[audit] {fx.name} <- record document already produced on the row ({rp}): {rr.get('verdict')} {rr.get('failed_checks') or ''} "
              f"checker {rr.get('checker_seconds')} s; source == record tree {rr['identity']['source_equals_record_tree']}; oracle_dir is the row {rr['identity']['oracle_dir_is_the_row']}", flush=True)
        if res_r is None:
            reasons.append(f"--record-result {rp}: no document")
        else:
            if ri0.get("name", "record") != "record" or not ri0:
                reasons.append(f"--record-result is not a checker-of-record document by its impl block: {ri0 or 'no impl block (a pre-impl-block record cannot be identified by content)'}")
            if not rr["identity"]["source_equals_record_tree"]:
                reasons.append(f"--record-result's checker source {str(ri0.get('source_sha256'))[:16]} != the record tree's global_match.py {str(tree_sha)[:16]} "
                               f"({report['record_tree']}): not the tree named for this audit")
            if not rr["identity"]["oracle_dir_is_the_row"]:
                reasons.append(f"--record-result read oracle_dir {res_r.get('oracle_dir')!r}, not the row's match/ ({row_match}, nor a directory of the row's name, nor the fast document's oracle_dir): not the same preserved artifacts")
    else:
        print(f"[audit] {fx.name} <- record ({report['record_tree']}) ...", flush=True, end="")
        rr = _run("record", a.record_tree)
        report["runs"]["record"] = rr
        print(f" {rr.get('verdict')} rc={rr.get('rc')} wall={rr.get('wall_s')}s rss={rr.get('peak_rss_tree_mb')}MB{' (reused)' if rr.get('reused') else ''}", flush=True)
        res_r = json.load(open(rr["result_path"])) if rr.get("result_path") else None
        if rr.get("rc") not in (0, 3) or res_r is None:
            reasons.append(f"record checker rc={rr.get('rc')}{' (killed at the wall bound)' if rr.get('rc') in (124, -9) else ''}, document {'present' if res_r is not None else 'absent'}")
    # [rev F-r17-15 (a)] the audited document must BE a fast document, by its own impl block (global_match.impl_info: name / source_sha256 /
    # candidate_stage); a record document handed in as --fast-doc would make the audit record-vs-record, GREEN by construction -> RED by name
    res_f = json.load(open(report["fast_doc"])) if report.get("fast_doc") and os.path.exists(report["fast_doc"]) else None
    ri = (res_r or {}).get("impl") if isinstance((res_r or {}).get("impl"), dict) else {}
    report["record_impl"] = {k: ri.get(k) for k in ("name", "source_sha256", "candidate_stage", "module")}
    if res_f is None:
        reasons.append("no fast document to audit against (give --fast-doc or --also-fast)")
    else:
        fi = res_f.get("impl") if isinstance(res_f.get("impl"), dict) else {}
        report["fast_impl"] = {k: fi.get(k) for k in ("name", "source_sha256", "candidate_stage", "module", "config")}
        if fi.get("name") != "fast" or not str(fi.get("candidate_stage") or "").strip():
            reasons.append(f"the audited document is not a MATCH_IMPL=fast document by its impl block: name={fi.get('name')!r} candidate_stage={fi.get('candidate_stage')!r}"
                           + (" (no impl block at all)" if not fi else ""))
        if ri and fi and ri.get("source_sha256") and ri.get("source_sha256") == fi.get("source_sha256"):
            reasons.append("record and fast documents name the SAME checker source (source_sha256 equal): not a differential")
    # [rev F-r17-15 (c)] artifact identity: the four files the record checker READ (sha256, and of the row's originals where the shadow
    # relocates a copy), compared to the fast Match of record's fast_compare.json (exec's tool writes artifacts_sha256 for the same four)
    shadow = os.path.dirname(fx.args[fx.args.index("--oracle") + 1]) if "--oracle" in fx.args else None   # <shadow>/match -> <shadow>
    four = ("match/program.json", "match/instances.jsonl", "build_workload/workload_program.json", "match/capture/tokens.json")
    art = {}
    for rel in four:
        rp, sp = os.path.join(os.path.abspath(a.row), rel), (os.path.join(shadow, rel) if shadow else None)
        row_sha = _sha256_file(rp) if os.path.exists(rp) else None
        read_sha = _sha256_file(sp) if sp and os.path.exists(sp) else None
        if getattr(a, "record_result", None):
            read_sha = row_sha          # the record document was produced on the row's own files (oracle_dir check above), not on a shadow
        art[rel] = {"row_sha256": row_sha, "read_sha256": read_sha, "relocated_copy": bool(read_sha and row_sha and read_sha != row_sha),
                    "bytes": os.path.getsize(rp) if os.path.exists(rp) else None}
    report["artifacts"] = art
    # [R17 mfast n2, F-r17-15 (iii) by CONTENT] since compare.load_oracle hashes the two fold files as it reads them, a document (record or
    # fast) states impl.fold_load[<dir>].{program_sha256, instances_sha256}: those are compared with the row's files here.  A document that
    # states them and names OTHER bytes than the row's is RED (not the same preserved artifacts) -- for a --record-result document this is
    # the identity that the oracle_dir path test above cannot give; a document that predates n2 states none (recorded as `stated: false`,
    # the path/relocation logic above stands alone for it).  Where the checker read a shadow copy (this tool's own runs), the digests must
    # equal the SHADOW's files it read (read_sha256), which the relocation table already ties to the row's originals.
    def _doc_digests(doc: dict | None) -> dict | None:
        fl = _fold_load_of((doc or {}).get("impl")) if doc else {}
        if not fl or not any(isinstance(v, dict) and v.get("instances_sha256") for v in fl.values()):
            return None
        vals = [v for v in fl.values() if isinstance(v, dict) and v.get("instances_sha256")]
        # the row's fold dir is the ONE the document loaded as its oracle (components load through load_derived, never through this path)
        return {"match/program.json": vals[0].get("program_sha256"), "match/instances.jsonl": vals[0].get("instances_sha256"), "dirs": [k for k in fl]}

    stated = {}
    for which, doc, on_shadow in (("record", res_r, not getattr(a, "record_result", None)), ("fast", res_f, bool(a.also_fast) and not a.fast_doc)):
        dd = _doc_digests(doc)
        if dd is None:
            stated[which] = {"stated": False}
            continue
        ent = {"stated": True, "dirs": dd["dirs"], "equal": {}, "expected_source": "shadow (what this tool's run read)" if on_shadow else "the row's files"}
        for rel in ("match/program.json", "match/instances.jsonl"):
            exp = art[rel]["read_sha256"] if on_shadow else art[rel]["row_sha256"]
            ent["equal"][rel] = bool(dd[rel] and exp and dd[rel].lower() == exp.lower())
            ent[rel] = {"document": dd[rel], "expected": exp}
        stated[which] = ent
        bad = [rel for rel, ok in ent["equal"].items() if not ok]
        if bad:
            reasons.append(f"the {which} document states fold-file sha256s that are NOT {ent['expected_source']} for {bad} (impl.fold_load): not the same preserved artifacts")
    if stated.get("record", {}).get("stated") and stated.get("fast", {}).get("stated"):
        same = all(stated["record"][rel]["document"] == stated["fast"][rel]["document"] for rel in ("match/program.json", "match/instances.jsonl"))
        stated["record_fast_same_bytes"] = same
        if not same and not a.allow_fast_read_relocated:
            reasons.append("record and fast documents state DIFFERENT fold-file sha256s (impl.fold_load): the two verdicts are not over the same bytes "
                           "(a store twin needs --allow-fast-read-relocated REASON)")
    report["document_digests"] = stated
    if getattr(a, "record_result", None) and stated.get("record", {}).get("stated") and all(stated["record"]["equal"].values()):
        # content identity holds for the supplied record document: the path test on its oracle_dir is superseded (a pulled row, a renamed
        # directory, a different mount -- the BYTES it read are the row's), the report keeps both facts
        report["runs"]["record"]["identity"]["content_identity"] = True
        reasons[:] = [r for r in reasons if not r.startswith("--record-result read oracle_dir")]
    elif getattr(a, "record_result", None) and report["runs"].get("record", {}).get("identity") is not None:
        # False = it states digests and they are not the row's (RED above); None = a pre-n2 document states none (the path test stands alone)
        report["runs"]["record"]["identity"]["content_identity"] = False if stated.get("record", {}).get("stated") else None
    fc_path = a.fast_compare or (os.path.join(os.path.dirname(report["fast_doc"]), "fast_compare.json") if report.get("fast_doc") else None)
    fc = json.load(open(fc_path)) if fc_path and os.path.exists(fc_path) else None
    report["fast_compare"] = {"path": fc_path if fc else None, "present": bool(fc)}
    if fc:
        theirs = fc.get("artifacts_sha256") or {}
        legacy = False
        if not theirs and isinstance(fc.get("artifacts"), dict):
            # exec's FIRST fast_compare.sh (the A100 llama row, M-0669): basename keys, 16-hex prefixes, tokens.json absent -- read by basename and
            # compared as a PREFIX of the row's sha; named `legacy_schema`; whatever it does not name is still `missing` (RED)
            legacy = True
            theirs = {rel: fc["artifacts"].get(os.path.basename(rel)) for rel in four if fc["artifacts"].get(os.path.basename(rel))}

        def _eq(a_: str | None, b_: str | None) -> bool:
            a_, b_ = (a_ or "").lower(), (b_ or "").lower()
            return bool(a_ and b_ and len(a_) >= 16 and len(b_) >= 16 and (a_ == b_ or (legacy and b_.startswith(a_))))
        mism = {rel: {"fast_compare": theirs.get(rel), "row": art[rel]["row_sha256"]} for rel in four if theirs.get(rel) and not _eq(theirs.get(rel), art[rel]["row_sha256"])}
        missing = [rel for rel in four if not theirs.get(rel)]
        report["fast_compare"].update({"tree": fc.get("tree"), "artifacts_equal": not mism and not missing, "mismatch": mism, "missing": missing,
                                       "legacy_schema": legacy, "fast_document_sha256_16": (fc.get("fast_document") or {}).get("sha256_16")})
        if mism:
            reasons.append(f"artifact identity: the fast Match of record ran on different bytes than the record checker read: {sorted(mism)}")
        if missing:
            reasons.append(f"fast_compare.json names no sha256 for {missing}")
        # [rev M-0774 n1] a HARNESS-written fast_compare.json records what the fast checker actually READ (artifacts_read_sha256: the shadow's
        # files) beside the row's originals; when the two differ the fast verdict was on other bytes than the row's -- named per file as
        # fast_read_relocated; GREEN only for build_workload/workload_program.json (the shadow's path rewrite) or when the operator names the
        # relocation (a STORE twin: progressions instances.jsonl) with --allow-fast-read-relocated REASON; RED otherwise
        read_theirs = fc.get("artifacts_read_sha256") or {}
        if read_theirs:
            frr = {rel: {"fast_read": read_theirs[rel], "row": art[rel]["row_sha256"]} for rel in four
                   if read_theirs.get(rel) and art[rel]["row_sha256"] and read_theirs[rel].lower() != art[rel]["row_sha256"].lower()}
            report["fast_compare"]["fast_read_relocated"] = frr
            bad = sorted(rel for rel in frr if rel != "build_workload/workload_program.json")
            if bad and a.allow_fast_read_relocated:
                report["fast_compare"]["fast_read_relocated_note"] = f"OVERRIDE named by the operator: {a.allow_fast_read_relocated} -- the fast checker read other bytes than the row's for {bad}"
            elif bad:
                reasons.append(f"the fast checker READ other bytes than the row's originals for {bad} (fast_compare.json artifacts_read_sha256 != the row's sha256; "
                               "a store twin or a rewrite beyond workload_program.json's paths -- name --allow-fast-read-relocated REASON if intended)")
    elif a.fast_doc and not a.also_fast:
        report["fast_compare"]["note"] = "no fast_compare.json beside the fast document: the fast verdict's artifact sha256s are not on record (exec's fast_compare.sh writes them); identity rests on the path"
    # [rev F-r17-15 (b)] the record tree must be THE tree the fast verdict ran on: fast_compare.json's tree when present, else this tree
    # for --also-fast, else the row's release commit (attempt.json release_commit / commit/versions.json tree.commit / stages.txt commit=)
    expected, expected_src = None, None
    if fc and fc.get("tree"):
        expected, expected_src = fc["tree"], "fast_compare.json tree"
    elif a.also_fast and not a.fast_doc:
        expected, expected_src = report["this_tree_commit"], "this tree (--also-fast)"
    else:
        expected, expected_src = _row_release_commit(a.row)
    report["tree_check"] = {"record_tree_commit": report["record_tree_commit"], "expected_commit": expected, "expected_source": expected_src,
                            "equal": bool(expected and report["record_tree_commit"] and _same_commit(expected, report["record_tree_commit"])),
                            "override": a.allow_tree_mismatch or None}
    if not report["tree_check"]["equal"]:
        msg = f"record tree {str(report['record_tree_commit'])[:9]} is not the tree the fast verdict ran on ({str(expected)[:9]} from {expected_src})"
        if a.allow_tree_mismatch:
            report["tree_check"]["note"] = f"OVERRIDE named by the operator: {a.allow_tree_mismatch} -- {msg}"
        else:
            reasons.append(msg + " (name --allow-tree-mismatch REASON to audit across trees)")
    if res_r is not None and res_f is not None:
        strict = deep_diff(strip_volatile(res_r), strip_volatile(res_f))
        cls = deep_diff(classes_view(res_r), classes_view(res_f))
        relocated = []
        # exactly ONE of the two documents ran on this tool's shadow of the (pulled) row, the other IN PLACE on the pod's row:
        #   fast external (--fast-doc = the row's fast Match of record) + record re-run here (the R17-37 (2) record audit proper), or
        #   record external (--record-result) + fast re-run here (rev M-0719 "the FAST half against existing record documents").
        shadow_side = None
        if getattr(a, "record_result", None) and a.also_fast and not a.fast_doc:
            shadow_side = "fast"
        elif a.fast_doc and not getattr(a, "record_result", None) and (rr or {}).get("result_path") and not rr.get("external"):
            shadow_side = "record"
        if shadow_side:
            # [rev M-0719; S-r17-37 (3), n2-b symmetric] the in-place document's PATH-VALUED fields (workload.path, requests[i].component.dir)
            # name the row's files at the pod's root; the shadow run's name the shadow's copies under <out>/real/<name>/positive/ -- the same
            # files at two roots.  A difference is set aside as a RELOCATION -- not a verdict difference -- only when both sides are strings of
            # EQUAL BASENAME, the SHADOW side lies under this run's shadow and EXISTS there, and the in-place side does not lie under it;
            # everything else stays a strict difference.  The relocations are listed in the report; the shadow's files are the row's (the
            # four-artifact identity + the workload sha256 in PROVENANCE tie them).
            shadow_root = os.path.realpath(os.path.join(a.out, "real", a.name)) + os.sep

            def _at(doc, path: str):     # the RAW value at a deep_diff path `$.a.b[3].c` (the diff's own rendering is shortened at 200 chars)
                cur = doc
                for tok in re.findall(r"\.([^.\[\]]+)|\[(\d+)\]", path[1:] if path.startswith("$") else path):
                    key, idx = tok
                    try:
                        cur = cur[int(idx)] if idx else cur[key]
                    except (KeyError, IndexError, TypeError):
                        return None
                return cur

            # [rev M-1096 n2-a] ONLY the two named path-valued fields may be set aside -- never any other string field that happens to
            # name an existing file under the shadow
            _RELOC_FIELDS = re.compile(r"^\$\.global_program\.workload\.path$|^\$\.requests\[\d+\]\.component\.dir$")

            def _reloc(d: dict) -> bool:
                path = d.get("path") or ""
                if not _RELOC_FIELDS.match(path):
                    return False
                rv, fv = _at(res_r, path), _at(res_f, path)
                if not (isinstance(rv, str) and isinstance(fv, str) and rv and fv):
                    return False
                # [n2-b] the field's CONTENT sibling must be stated and EQUAL on both sides -- workload.path's `sha256` (the workload document's
                # bytes), component.dir's `digest` (the component's fold digest): a relocation is the same content at two roots, not two files
                # of the same name.  (The shadow's copy of the workload document is named workload.json -- the basename need not agree there.)
                parent = path.rsplit(".", 1)[0]
                sib = "sha256" if path.endswith(".workload.path") else "digest"
                rc, fc = _at(res_r, parent + "." + sib), _at(res_f, parent + "." + sib)
                if not (isinstance(rc, str) and rc and rc == fc):
                    return False
                if sib == "digest" and os.path.basename(rv) != os.path.basename(fv):
                    return False
                sv, pv = (fv, rv) if shadow_side == "fast" else (rv, fv)      # (the shadow run's value, the in-place document's value)
                return os.path.realpath(sv).startswith(shadow_root) and os.path.exists(sv) and not os.path.realpath(pv).startswith(shadow_root)
            relocated = [d for d in strict if _reloc(d)]
            strict = [d for d in strict if d not in relocated]
            cls = [d for d in cls if not _reloc(d)]
        report["compare"] = {"agree_strict": not strict, "agree_class": not cls, "strict_diff": strict[:60], "class_diff": cls[:60],
                             "relocated_paths": relocated[:60], "relocation_shadow_side": shadow_side,
                             "population_record": population_view(res_r), "population_fast": population_view(res_f),
                             "verdicts": {"record": [res_r.get("verdict"), res_r.get("failed_checks")], "fast": [res_f.get("verdict"), res_f.get("failed_checks")]}}
        if strict:
            reasons.append(f"record checker != fast Match of record strictly: {len(strict)} difference(s)" + ("" if cls else f"; {len(cls)} at class level"))
    if a.record_doc and os.path.exists(a.record_doc):   # a document of record from ANOTHER tree (e.g. the row's own GM-01 of record): verdict-level only
        rd = json.load(open(a.record_doc))
        same = rd.get("verdict") == (res_r or {}).get("verdict") and sorted(rd.get("failed_checks") or []) == sorted((res_r or {}).get("failed_checks") or [])
        report["record_doc_compare"] = {"verdict_record_doc": [rd.get("verdict"), rd.get("failed_checks")], "same_verdict": same,
                                        "class_diff": deep_diff(classes_view(rd), classes_view(res_r))[:60] if res_r is not None else None,
                                        "seconds_record_doc": rd.get("seconds"),
                                        "impl_record_doc": (rd.get("impl") or {}).get("name") if isinstance(rd.get("impl"), dict) else None}
        if not same:
            reasons.append("the record checker's verdict differs from the row's document of record (other tree)")
    report["status"] = "GREEN" if not reasons else "RED"
    report["reasons"] = reasons
    # the row's Match cell reads the RECORD checker's document here (rev M-0665: the fast document is the differential, not the verdict)
    report["record_verdict_document"] = {"path": rr.get("result_path"), "verdict": rr.get("verdict"), "failed_checks": rr.get("failed_checks"),
                                         "sha256_16": _sha256_file(rr["result_path"])[:16] if rr.get("result_path") else None}
    _dump(report, os.path.join(a.out, "audit.json"))
    c = report.get("compare") or {}
    rf = report["runs"].get("fast") or {}
    tc = report.get("tree_check") or {}
    md = [f"# match_diff audit -- {a.name} ({report['utc']}, host {report['host']})", "",
          f"row `{report['row']}` · record tree `{report['record_tree']}`" + (f" ({report.get('record_tree_commit')})" if report.get("record_tree_commit") else "")
          + f" · fast document `{report.get('fast_doc')}`", "",
          f"**{report['status']}** -- " + ("; ".join(reasons) if reasons else "record checker == fast Match of record STRICTLY on the same preserved artifacts"), "",
          f"record document of this audit (the verdict a table may read): `{report['record_verdict_document']['path']}` -> {report['record_verdict_document']['verdict']} {report['record_verdict_document']['failed_checks'] or ''} (sha256 {report['record_verdict_document']['sha256_16']})", "",
          f"impl blocks: record `{(report.get('record_impl') or {}).get('name')}` source {str((report.get('record_impl') or {}).get('source_sha256'))[:16]} · fast `{(report.get('fast_impl') or {}).get('name')}` stage `{(report.get('fast_impl') or {}).get('candidate_stage')}` source {str((report.get('fast_impl') or {}).get('source_sha256'))[:16]}"
          + ((" · record half = the document ALREADY PRODUCED on the row (`--record-result`, not re-run): source == record tree's global_match.py **%s**, oracle_dir is the row **%s** (%s), checker %s s, utc %s"
              % (rr['identity']['source_equals_record_tree'], rr['identity']['oracle_dir_is_the_row'], rr['identity'].get('oracle_dir_match'), rr.get('checker_seconds'), rr['identity'].get('utc'))) if rr.get("external") else ""),
          f"tree check: record tree {str(tc.get('record_tree_commit'))[:9]} vs the fast verdict's tree {str(tc.get('expected_commit'))[:9]} ({tc.get('expected_source')}) -> equal **{tc.get('equal')}**" + (f"; override: {tc.get('note')}" if tc.get("note") else ""),
          "artifacts read by the record checker (sha256 of the row's file; `relocated copy` = the shadow rewrote paths in its copy): "
          + "; ".join(f"`{rel}` {str(v.get('row_sha256'))[:16]}{' (relocated copy read)' if v.get('relocated_copy') else ''}" for rel, v in (report.get("artifacts") or {}).items()),
          f"fast_compare.json beside the fast document: {'present (tree ' + str(report['fast_compare'].get('tree'))[:9] + (', LEGACY schema: basename keys / 16-hex prefixes' if report['fast_compare'].get('legacy_schema') else '') + '), artifacts equal ' + str(report['fast_compare'].get('artifacts_equal')) + (', missing ' + str(report['fast_compare'].get('missing')) if report['fast_compare'].get('missing') else '') + (', fast READ relocated: ' + ', '.join(sorted(report['fast_compare']['fast_read_relocated'])) + (' (override: ' + report['fast_compare']['fast_read_relocated_note'] + ')' if report['fast_compare'].get('fast_read_relocated_note') else '') if report['fast_compare'].get('fast_read_relocated') else '') if report.get('fast_compare', {}).get('present') else 'absent (identity by path)'}",
          "fold-file sha256s STATED by the documents (impl.fold_load, n2 -- identity by content): "
          + "; ".join(f"{w} " + ("not stated (pre-n2 checker)" if not (report.get("document_digests") or {}).get(w, {}).get("stated")
                                 else "equal to " + report["document_digests"][w]["expected_source"] + " **" + str(all(report["document_digests"][w]["equal"].values())) + "**"
                                 + f" (instances {str(report['document_digests'][w]['match/instances.jsonl']['document'])[:16]}, program {str(report['document_digests'][w]['match/program.json']['document'])[:16]})")
                     for w in ("record", "fast"))
          + (f"; record and fast over the same bytes **{report['document_digests']['record_fast_same_bytes']}**" if "record_fast_same_bytes" in (report.get("document_digests") or {}) else ""), "",
          "| impl | verdict | failed | wall s | peak RSS MB | phases |", "|---|---|---|---|---|---|",
          f"| record ({'record tree' if a.record_tree else 'this tree'}) | {rr.get('verdict')} | {','.join(rr.get('failed_checks') or [])} | {rr.get('wall_s')} | {rr.get('peak_rss_tree_mb')} | {_fmt_phases(rr.get('phases'))} |"]
    if rf:
        md.append(f"| fast (this tree) | {rf.get('verdict')} | {','.join(rf.get('failed_checks') or [])} | {rf.get('wall_s')} | {rf.get('peak_rss_tree_mb')} | {_fmt_phases(rf.get('phases'))} |")
    if c:
        md += ["", f"agree strict: **{c['agree_strict']}** · agree class: **{c['agree_class']}**", ""]
        for d in c.get("strict_diff") or []:
            md.append(f"- `{d['path']}`: record `{d.get('record')}` | fast `{d.get('fast')}`")
        if c.get("relocated_paths"):
            md += ["", f"path-valued fields set aside as RELOCATIONS of the same files (the {'record' if c.get('relocation_shadow_side') == 'fast' else 'fast'} document was produced in place on the pod's row; the {c.get('relocation_shadow_side')} half ran on this tool's shadow of the pulled row -- equal basenames, the shadow side exists under the shadow): {len(c['relocated_paths'])}"]
            for d in c["relocated_paths"]:
                md.append(f"- `{d['path']}`: record `{d.get('record')}` | fast `{d.get('fast')}`")
        md += ["", "population (record | fast): " + json.dumps({k: [c['population_record'].get(k), c['population_fast'].get(k)] for k in c['population_record'] if k != 'per_request'}, default=str)]
    if report.get("record_doc_compare"):
        rdc = report["record_doc_compare"]
        md += ["", f"document of record (other tree) `{report['record_doc']}`: verdict {rdc['verdict_record_doc']} vs record checker here {c.get('verdicts', {}).get('record')} -> same verdict **{rdc['same_verdict']}**; class-level differences vs it: {len(rdc.get('class_diff') or [])} (named fields of the newer tree expected)"]
        for d in (rdc.get("class_diff") or [])[:25]:
            md.append(f"- `{d['path']}`: record-doc `{d.get('record')}` | record-here `{d.get('fast')}`")
    open(os.path.join(a.out, "audit.md"), "w").write("\n".join(md) + "\n")
    print(json.dumps({"status": report["status"], "reasons": reasons, "record": [rr.get("verdict"), rr.get("wall_s"), rr.get("peak_rss_tree_mb")],
                      "fast": [rf.get("verdict"), rf.get("wall_s"), rf.get("peak_rss_tree_mb")] if rf else None,
                      "agree_strict": c.get("agree_strict"), "agree_class": c.get("agree_class"), "out": os.path.abspath(a.out)}))
    return 0 if report["status"] == "GREEN" else 3


def cmd_ref_tree(a) -> int:
    """`git archive <commit> -- <packages> | tar -x -C out`: the reference tree the record is compared against, with its commit recorded."""
    os.makedirs(a.out, exist_ok=True)
    sha = subprocess.run(["git", "-C", a.repo, "rev-parse", a.commit], capture_output=True, text=True, check=True).stdout.strip()
    ar = subprocess.run(["git", "-C", a.repo, "archive", "--format=tar", sha, "--", *REF_TREE_PATHS], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", a.out], input=ar.stdout, check=True)
    open(os.path.join(a.out, "REF_TREE.json"), "w").write(json.dumps({"commit": sha, "ref": a.commit, "paths": REF_TREE_PATHS,
                                                                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    print(json.dumps({"ref_tree": os.path.abspath(a.out), "commit": sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
