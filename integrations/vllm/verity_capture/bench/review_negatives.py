"""[R10 owner 7] BRIEF §6 binding negatives against a verified sidecar: each mutation must be REJECTED by every verifier it is fed to.

Mutations (each a self-contained sidecar written to <out dir>/neg_<name>.json):
  changed_value            one byte of an opened chunk flipped (the y chunk of obligation 0)
  wrong_position_rank      obligation 0's y rank+1 (same bytes/path)                      -> position rule / fold must fail
  wrong_position_word      obligation 0's y word_index+1                                   -> value at derived position differs
  wrong_root               step tree's bare root := another step's merkle root
  substituted_layout       L1 data relabelled as L2 (layout_id + framing cmt-chunk-fa2h-v7.row1152.s<S>) or L2 data relabelled as L1
  substituted_framing_w    a weights (pos_leaf) tree relabelled with the step chunk framing
  wrong_step_context       tree step:S's framing changed to another step S' (same bytes/paths/root)
  wrong_request_context    context.run_id := another request id (roots/openings unchanged)
  wrong_model_context      roots.weights.tensors: two subtree roots swapped (weights root must not recompute)
  wrong_geo                context.geo_digest flipped
  placeholder_program      context.program_digest := SHA-256("cmt-integ/no-program-digest/"+run_id) (R9's placeholder)
  unrelated_selfconsistent a fresh, internally consistent tree built from the SAME opened chunks (new root/count/paths) replaces the
                           step tree: leaf folds succeed, the wrapper/run-root membership must fail
Verifiers: mine (`review_e2e_verify.main`), owner 3's `layout_verify.py --witness` (if present), owner 6's `e2e_verify.py` (if present).

Usage: review_negatives.py <sidecar.json> <out dir> [--layout-verify path] [--e2e-verify path] [--python exe]
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

from verity_capture.bench import review_e2e_verify as ev
from verity_capture.bench import review_l1_verify as rv


def selfcontained(path: Path) -> dict:
    doc = json.loads(path.read_text())
    rf = doc.get("roots", {}).get("roots_file")
    if rf and (path.parent / rf).exists():
        rj = json.loads((path.parent / rf).read_text())
        for k in ("steps", "weights", "run_root", "weights_root"):
            doc["roots"].setdefault(k, rj[k])
        doc["roots"].setdefault("roots_written_utc", rj["written_utc"])
        doc["roots"].pop("roots_file", None)
    return doc


def mutations(doc: dict) -> dict[str, dict]:
    out = {}
    ob0 = doc["obligations"][0]
    ykey, yrank = ob0["y"]["tree"], int(ob0["y"]["rank"])
    step_trees = [t for t in doc["trees"] if t["id"].startswith("step:")]
    w_trees = [t for t in doc["trees"] if t["id"].startswith("weights:")]

    def tree(d, tid):
        return next(t for t in d["trees"] if t["id"] == tid)

    d = copy.deepcopy(doc)
    lf = next(l for l in tree(d, ykey)["leaves"] if l["rank"] == yrank)
    b = bytearray(bytes.fromhex(lf["bytes_hex"])); b[2 * int(ob0["y"]["word_index"])] ^= 0x01; lf["bytes_hex"] = b.hex()
    out["changed_value"] = d

    d = copy.deepcopy(doc); d["obligations"][0]["y"]["rank"] = yrank + 1
    lf = next(l for l in tree(d, ykey)["leaves"] if l["rank"] == yrank); lf["rank"] = yrank + 1
    out["wrong_position_rank"] = d

    d = copy.deepcopy(doc); d["obligations"][0]["y"]["word_index"] = (int(ob0["y"]["word_index"]) + 1) % 128
    out["wrong_position_word"] = d

    if len(step_trees) >= 2:
        d = copy.deepcopy(doc); a, bb = tree(d, step_trees[0]["id"]), tree(d, step_trees[1]["id"]); a["root"] = bb["root"]
        out["wrong_root"] = d
    else:
        d = copy.deepcopy(doc); t = tree(d, ykey); t["root"] = hashlib.sha256(bytes.fromhex(t["root"])).hexdigest(); out["wrong_root"] = d

    d = copy.deepcopy(doc)
    if doc["layout_id"] == "L1-chunk256":      # present L1 data as L2 (or L2 data as L1): layout_id + step framings relabelled
        d["layout_id"] = "L2-row"; a, b = "cmt-chunk-fa2h-v7.s", "cmt-chunk-fa2h-v7.row1152.s"
    else:
        d["layout_id"] = "L1-chunk256"; a, b = "cmt-chunk-fa2h-v7.row1152.s", "cmt-chunk-fa2h-v7.s"
    for t in d["trees"]:
        if t["id"].startswith("step:"):
            t["framing"] = b + t["framing"][len(a):]
    out["substituted_layout"] = d

    if w_trees:
        d = copy.deepcopy(doc); t = tree(d, w_trees[0]["id"]); t["framing"] = step_trees[0]["framing"]
        out["substituted_framing_w"] = d

    d = copy.deepcopy(doc); t = tree(d, ykey); s = int(ykey.split(":")[1]); pre = t["framing"].rsplit(".s", 1)[0]
    t["framing"] = f"{pre}.s{(s + 1) % max(2, len(d['roots']['steps']))}"
    out["wrong_step_context"] = d

    d = copy.deepcopy(doc); d["context"]["run_id"] = "another-request"; out["wrong_request_context"] = d

    d = copy.deepcopy(doc); ts = d["roots"]["weights"]["tensors"]
    if len(ts) >= 2:
        i, j = 0, 1
        for k, t in enumerate(ts):
            if t["name"] == w_trees[0]["id"].split(":", 1)[1] if w_trees else False:
                i = k
        j = (i + 1) % len(ts)
        ts[i]["subtree_root"], ts[j]["subtree_root"] = ts[j]["subtree_root"], ts[i]["subtree_root"]
        if "subtree_roots" in d["roots"]["weights"]:
            sr = d["roots"]["weights"]["subtree_roots"]; sr[i], sr[j] = sr[j], sr[i]
    out["wrong_model_context"] = d

    d = copy.deepcopy(doc); g = bytearray(bytes.fromhex(d["context"]["geo_digest"])); g[0] ^= 1; d["context"]["geo_digest"] = g.hex(); out["wrong_geo"] = d

    d = copy.deepcopy(doc); d["context"]["program_digest"] = hashlib.sha256(b"cmt-integ/no-program-digest/" + doc["context"]["run_id"].encode()).hexdigest()
    out["placeholder_program"] = d

    d = copy.deepcopy(doc); t = tree(d, ykey); s = int(ykey.split(":")[1])
    chunks = [bytes.fromhex(l["bytes_hex"]) for l in t["leaves"]]
    n = len(chunks); pw = n * 64
    leaves = [rv.l1_leaf(s, i, c, pw) if doc["layout_id"] == "L1-chunk256" else rv.l2_leaf(s, i, c, n) for i, c in enumerate(chunks)]
    lv = rv.fold_levels(leaves)
    t["root"] = lv[-1][0].hex(); t["count"] = n
    remap = {l["rank"]: i for i, l in enumerate(t["leaves"])}
    for i, l in enumerate(t["leaves"]):
        l["rank"] = i; l["path"] = [{"kind": "lift"} if p is None else {"kind": "sibling", "digest": p.hex()} for p in rv.path_in_levels(lv, i)]
    for ob in d["obligations"]:
        for k in ("x", "y"):
            if ob[k]["tree"] == ykey:
                if k == "y":
                    ob["y"]["rank"] = remap[int(ob["y"]["rank"])]
                else:
                    ob["x"]["ranks"] = [remap[r] for r in ob["x"]["ranks"]]
    out["unrelated_selfconsistent"] = d
    return out


def run_mine(path: Path) -> tuple[str, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = ev.main([str(path)])
        except Exception as e:  # noqa: BLE001
            return "REJECTED(exception)", type(e).__name__ + ": " + str(e)[:120]
    res = json.loads(buf.getvalue())
    fails = [k for k, v in res.items() if isinstance(v, dict) and v.get("verdict") == "FAIL"]
    return ("ACCEPTED" if rc == 0 else "REJECTED"), ",".join(fails)


def run_ext(py: str, script: str, args: list[str]) -> tuple[str, str]:
    p = subprocess.run([py, script, *args], capture_output=True, text=True)
    tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
    return ("ACCEPTED" if p.returncode == 0 else "REJECTED"), " | ".join(t[:160] for t in tail)


def main(argv: list[str]) -> int:
    src, outd = Path(argv[0]), Path(argv[1]); outd.mkdir(parents=True, exist_ok=True)
    lv = argv[argv.index("--layout-verify") + 1] if "--layout-verify" in argv else None
    e2 = argv[argv.index("--e2e-verify") + 1] if "--e2e-verify" in argv else None
    py = argv[argv.index("--python") + 1] if "--python" in argv else sys.executable
    doc = selfcontained(src)
    pos = outd / "positive.json"; pos.write_text(json.dumps(doc))
    rows = []
    cases = {"positive": doc, **mutations(doc)}
    for name, d in cases.items():
        p = outd / f"neg_{name}.json" if name != "positive" else pos
        p.write_text(json.dumps(d))
        row = {"case": name, "expected": "ACCEPTED" if name == "positive" else "REJECTED"}
        row["mine"], row["mine_failed_checks"] = run_mine(p)
        if lv and Path(lv).exists():
            row["layout_verify(owner3)"], row["layout_verify_tail"] = run_ext(py, lv, ["--witness", str(p), "--quiet"])
        if e2 and Path(e2).exists():
            row["e2e_verify(owner6)"], row["e2e_verify_tail"] = run_ext(py, e2, [str(p)])
        rows.append(row)
        print(json.dumps(row))
    summary = {"cases": len(rows) - 1,
               "mine_all_as_expected": all(r["mine"].startswith(r["expected"]) for r in rows),
               "owner3_all_as_expected": all(r.get("layout_verify(owner3)", r["expected"]).startswith(r["expected"]) for r in rows) if lv else None,
               "owner6_all_as_expected": all(r.get("e2e_verify(owner6)", r["expected"]).startswith(r["expected"]) for r in rows) if e2 else None}
    (outd / "negatives_summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))
    return 0 if summary["mine_all_as_expected"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
