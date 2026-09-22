import json, hashlib, time, sys
from pathlib import Path
from verity_capture.bench import review_l1_verify as rv
d = Path(sys.argv[1])
rj = json.loads((d/"roots.json").read_text())
pd = bytes.fromhex(rj["program_digest"]); geo = bytes.fromhex(rj["geo_digest"]); run_id = rj["run_id"]
out = {"dir": str(d), "program_digest": rj["program_digest"], "run_id": run_id, "committer": rj["committer"], "roots_written_utc": rj["written_utc"]}
ok_mr = ok_sr = bad = 0; step_roots = []; fails = []
t0 = time.time()
for s in rj["steps"]:
    data = (d/"steps"/f"step_{s['step']}.bin").read_bytes()
    mr = rv.l1_merkle_root(data, s["step"])
    n = len(data)//256
    a = mr.hex() == s["merkle_root"] and n == s["n_leaves"]
    ld = rv.layout_digest(s["layout"])
    b = ld.hex() == s["layout_digest"] and rv.ctx_digest(run_id, s["step"]).hex() == s["ctx_digest"]
    R = rv.l1_step_root(pd, run_id, s["step"], geo, ld, s["n_leaves"], mr)
    c = R.hex() == s["root"]
    ok_mr += a; ok_sr += (b and c); bad += not (a and b and c)
    step_roots.append(bytes.fromhex(s["root"]))
    if not (a and b and c): fails.append((s["step"], a, b, c, n, s["n_leaves"], len(data), s["used_bytes"]))
out.update({"steps": len(rj["steps"]), "merkle_roots_recomputed_from_streams_hashlib": ok_mr, "step_roots_bound_ok": ok_sr, "bad": bad, "fails": fails, "hash_wall_s": round(time.time()-t0, 1)})
out["run_root_match"] = rv.l1_run_root(pd, geo, step_roots).hex() == rj["run_root"]; out["run_root"] = rj["run_root"]
w = rj["weights"]; names = [[t["name"], t["dtype"], t["shape"], t["nbytes"]] for t in w["tensors"]]
subs = [bytes.fromhex(r) for r in w["subtree_roots"]]
out["weights_root_match"] = rv.weights_root(geo, names, subs).hex() == rj["weights_root"]; out["weights_root"] = rj["weights_root"]
out["weights_tensors"] = len(names); out["weights_bytes"] = sum(n[3] for n in names)
out["names_digest_match"] = hashlib.sha256(json.dumps(names, separators=(",",":")).encode()).hexdigest() == w["names_digest"]
out["placeholder_digest_would_be"] = hashlib.sha256(b"cmt-integ/no-program-digest/"+run_id.encode()).hexdigest()
out["bound_digest_is_pinned"] = rj["program_digest"] == "5bb5ac55c1ee110189c8d43e138ff1b41c1e7d48224777f1ef688d27c535edb9"
out["n_leaves_per_step"] = [s["n_leaves"] for s in rj["steps"]]; out["used_bytes_per_step"] = [s["used_bytes"] for s in rj["steps"]]
out["n_tensors_step0"] = rj["steps"][0]["n_tensors"]
out["tokens_equal_control"] = rj["tokens_equal_control"]; out["root_ready_delay_s"] = rj["root_ready_delay_s"]; out["gpu_tree_ms"] = rj["gpu_tree_ms"]
out["serving_wall_s"] = rj["serving_wall_s"]
# weights: recompute ONE tensor's subtree root from the model file? (needs torch/safetensors) -- optional
print(json.dumps(out, indent=1))
