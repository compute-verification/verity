"""J10-R's proof blocks at the frozen shape against the Definition's expected values (`fixtures.py`), bit for bit, on one GPU:

    python check.py --fixtures DIR --out OUT --scratch S [--h1u NAME=PATH ...] [--ref PATH] [--sm] [--jobs 32]
                    [--calls q-4096,...] [--operands-only]

DIR holds fixtures.py's expected.npz and fixtures.json (each array held to its sha256 there). Every build is checked the
same way, none against another:

* **anchors** (small calls, every proof block): each build's receipts (and the fallback's seeds and z) against
  `h1tile.j10r_reference` itself, at 64 x 64 x 512 (m = 2), 64 x 32 x 1024 and 32 x 64 x 4608 (m = 3), with the
  max-magnitude rows of `shape.SPECIAL`;
* **each fixture call** (linear, M): A and B regenerated whole (`shape.whole`, in S, over --jobs processes) and each
  sampled proof block's 32 rows of each held to the digests fixtures.json pins; the served z, `torch._int_mm(A, B^T)`, on every
  sampled proof block's 32 x 32 cells; then each h1u build over the whole call, run under 0xA5 and then 0x5A (receipts, row
  workspace, seeds): no word left unwritten (holding each poison after its run) or differing between the two runs in any
  proof block, every sampled proof block's receipt word for word and its seed_c and sigma_R; with --sm the same call
  by kh_units_multi_j10r_sm, every word equal to kh_units_multi_j10r's; and the fallback (--ref) on the sampled proof blocks:
  receipts, z and seeds.

check.json holds every count; check-got.npz the sampled proof blocks' receipts per build. Exit 4 if any word differs.
--operands-only generates S's operands and exits (run it before taking the GPU).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shape as S
import units as UN

import fixtures as F
from verity.ml.protocols.pouw.circuit import h1tile as HT
from verity.ml.protocols.pouw.circuit import seeds as SD

ANCHORS = ((64, 64, 512), (64, 32, 1024), (32, 64, 4608))


def libs_of(h1u: list[str], ref: str | None, sm: bool) -> dict:
    libs: dict = {}
    for spec in h1u:
        name, _, path = spec.partition("=")
        libs[name] = UN.H1u(path)
        if sm and UN.H1u.has_sm(path):
            libs[f"{name}-sm"] = UN.H1u(path, sm=True)
        elif sm:
            print(json.dumps({"lib": name, "note": "no kh_units_multi_j10r_sm: its served launch is not checked"}))
    if ref:
        libs["ref"] = UN.Ref(ref)
    return libs


def run_build(lib, a, b, salt, roots, base, picks) -> dict:
    """One build on one call, run under each poison (`units.twice`): {receipts of picks (uint32), seed_c and sigma of
    picks (bytes), z of picks (the fallback's), words left unwritten and words unstable between the runs (in every proof block
    the build computed)}."""
    if isinstance(lib, UN.Ref):
        c = UN.RefCall(lib, a, b, salt, roots, base, units=picks)
        tw = UN.twice(c)
        sig = c.sigma.cpu().numpy()
        return {"receipts": c.out.cpu().numpy().view(np.uint32), "z": c.z.cpu().numpy().view(np.uint32),
                "seed_c": [UN.digest(c.seedc.cpu().numpy())] * len(picks), "sigma": [UN.digest(r) for r in sig],
                "poison_left": tw["unwritten"], "unstable": tw["unstable"], "units_run": c.n}
    c = UN.H1uCall(lib, a, b, salt, roots, base)
    tw = UN.twice(c)
    pk = torch.tensor(picks, device="cuda")
    sd = c.seeds[pk].cpu().numpy()
    return {"receipts": c.out[pk].cpu().numpy().view(np.uint32), "seed_c": [UN.digest(r[:8]) for r in sd],
            "sigma": [UN.digest(r[8:]) for r in sd], "poison_left": tw["unwritten"], "unstable": tw["unstable"],
            "units_run": c.units, "_all": c.out}


def compare(got: dict, want_rec: np.ndarray, want_seedc: bytes, want_sigma: list[bytes], want_z: np.ndarray | None) -> dict:
    bad = got["receipts"] != want_rec
    r = {"units": int(want_rec.shape[0]), "words": int(want_rec.size), "bad_words": int(bad.sum()),
         "bad_units": [int(i) for i in np.nonzero(bad.any(axis=1))[0][:8]],
         "seed_c_bad": sum(s != want_seedc for s in got["seed_c"]),
         "sigma_bad": sum(s != w for s, w in zip(got["sigma"], want_sigma)), "poison_left": got["poison_left"],
         "unstable": got["unstable"], "units_run": got["units_run"]}
    if want_z is not None and "z" in got:
        r["z_bad_words"] = int((got["z"] != want_z).sum())
    r["ok"] = (r["bad_words"] == 0 and r["seed_c_bad"] == 0 and r["sigma_bad"] == 0 and r["poison_left"] == 0
               and r["unstable"] == 0 and r.get("z_bad_words", 0) == 0)
    return r


def anchors(libs: dict, rng: random.Random) -> list[dict]:
    res = []
    for m, n, k in ANCHORS:
        g = np.random.default_rng(rng.getrandbits(32))
        a = g.integers(-128, 128, size=(m, k), dtype=np.int8)
        b = g.integers(-128, 128, size=(n, k), dtype=np.int8)
        for i, p in enumerate(S.SPECIAL[:4]):
            a[i] = S._pattern(p, a[i])
            b[n - 1 - i] = S._pattern(p, b[n - 1 - i])
        salt, ra, rb = rng.randbytes(64), rng.randbytes(64), rng.randbytes(64)
        base = rng.getrandbits(40)
        ncb = n // S.T
        us = list(range((m // S.T) * ncb))
        t0 = time.time()
        want_rec, want_z = [], []
        for u in us:
            ri, cj = divmod(u, ncb)
            ac = (a[S.T * ri:S.T * ri + S.T].astype(np.int64) & 0xFF).tolist()
            bc = (b[S.T * cj:S.T * cj + S.T].astype(np.int64) & 0xFF).tolist()
            z, rec = HT.j10r_reference(ac, bc, salt, base + u, (ra, rb), base, HT.j10r_blocks(m, k, (S.T, S.T), ri, cj))
            want_rec.append(rec)
            want_z.append(z)
        wr = np.array(want_rec, np.uint32)
        wz = np.array(want_z, np.uint32).reshape(len(us), S.T, S.T)
        ref_s = time.time() - t0
        sc = SD.call_seed(salt, base, ra, rb)
        sig = [SD.receipt_seed(salt, base + u, ra, rb) for u in us]
        at, bt = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
        for name, lib in libs.items():
            got = run_build(lib, at, bt, salt, (ra, rb), base, us)
            r = {"lib": name, "m": m, "n": n, "k": k, "digits": HT.Layout(S.T, S.T, k).m, "definition_s": round(ref_s, 2),
                 **compare(got, wr, sc, sig, wz if name == "ref" else None)}
            res.append(r)
            print(json.dumps({"anchor": r}), flush=True)
    return res


def operands(linear: str, m: int, scratch: Path, jobs: int) -> tuple[np.ndarray, np.ndarray]:
    return S.whole("a", linear, m, scratch, jobs), S.whole("b", linear, m, scratch, jobs)


def to_gpu(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to("cuda")


def check_call(key: str, c: dict, arrays: dict, libs: dict, scratch: Path, jobs: int, got_out: dict) -> dict:
    lin, m = c["linear"], c["m"]
    ah, bh = operands(lin, m, scratch, jobs)
    picks = [int(u) for u in arrays[f"{key}/units"]]
    ncb = c["n"] // S.T
    op_bad = 0
    for s in c["samples"]:
        ri, cj = s["ri"], s["cj"]
        op_bad += int(S.sha256(ah[S.T * ri:S.T * ri + S.T]) != s["a_sha256"])
        op_bad += int(S.sha256(bh[S.T * cj:S.T * cj + S.T]) != s["b_sha256"])
    r: dict = {"call": key, "linear": lin, "m": m, "n": c["n"], "k": c["k"], "units": c["units"], "picks": len(picks),
               "rows_sampled": c["rows_sampled"], "operand_bad": op_bad, "builds": {}}
    a, b = to_gpu(ah), to_gpu(bh)
    z = torch._int_mm(a, b.t())
    zt = np.stack([z[S.T * (u // ncb):S.T * (u // ncb) + S.T, S.T * (u % ncb):S.T * (u % ncb) + S.T].cpu().numpy()
                   for u in picks]).view(np.uint32)
    del z
    r["served_z_bad_words"] = int((zt != arrays[f"{key}/z"]).sum())
    salt, roots = S.SALT, (bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"]))
    want_rec, want_z = arrays[f"{key}/receipts"], arrays[f"{key}/z"]
    want_sc = arrays[f"{key}/seed_c"].tobytes()
    want_sig = [row.tobytes() for row in arrays[f"{key}/sigma"]]
    full: dict = {}
    for name, lib in libs.items():
        t0 = time.time()
        got = run_build(lib, a, b, salt, roots, c["base"], picks)
        res = compare(got, want_rec, want_sc, want_sig, want_z if name == "ref" else None)
        res["wall_s"] = round(time.time() - t0, 2)
        if "_all" in got:
            base_name = name.removesuffix("-sm")
            if name.endswith("-sm") and base_name in full:
                res["vs_units_bad_words"] = int((got["_all"] != full[base_name]).sum())
                res["ok"] = res["ok"] and res["vs_units_bad_words"] == 0
                del full[base_name]
            elif f"{name}-sm" in libs:
                full[name] = got["_all"]
            del got["_all"]
        got_out[f"{key}/{name}"] = got["receipts"]
        r["builds"][name] = res
        print(json.dumps({"call": key, "build": name, **{x: res[x] for x in res if x != "bad_units"}}), flush=True)
    full.clear()
    del a, b
    torch.cuda.empty_cache()
    r["ok"] = op_bad == 0 and r["served_z_bad_words"] == 0 and all(v["ok"] for v in r["builds"].values())
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--h1u", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--ref")
    ap.add_argument("--sm", action="store_true", help="each h1u build's kh_units_multi_j10r_sm too")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--calls", help="the fixture calls to check (default all), e.g. q-4096,down-16384")
    ap.add_argument("--no-anchors", action="store_true")
    ap.add_argument("--operands-only", action="store_true")
    a = ap.parse_args()
    out, scratch = Path(a.out), Path(a.scratch)
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    doc, arrays = F.load(Path(a.fixtures))
    keys = a.calls.split(",") if a.calls else list(doc["calls"])
    if a.operands_only:
        t0 = time.time()
        for k in keys:
            operands(doc["calls"][k]["linear"], doc["calls"][k]["m"], scratch, a.jobs)
        print(json.dumps({"operands": keys, "wall_s": round(time.time() - t0, 1)}), flush=True)
        return 0
    libs = libs_of(a.h1u, a.ref, a.sm)
    rep: dict = {"schema": "j10r-sm90/check/v1", "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                 "fixtures_npz_sha256": doc["npz_sha256"], "libs": {n: lib.path for n, lib in libs.items()},
                 "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "anchors": [], "calls": []}
    if not a.no_anchors:
        rep["anchors"] = anchors(libs, random.Random(0x10A90))
    (out / "check.json").write_text(json.dumps(rep, indent=1))
    got: dict = {}
    for k in keys:
        try:
            r = check_call(k, doc["calls"][k], arrays, libs, scratch, a.jobs, got)
        except Exception as e:  # noqa: BLE001
            import traceback
            r = {"call": k, "ok": False, "error": "".join(traceback.format_exception(e))[-3000:]}
            print(json.dumps(r), flush=True)
        rep["calls"].append(r)
        (out / "check.json").write_text(json.dumps(rep, indent=1))
    np.savez(out / "check-got.npz", **got)
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rep["verdict"] = {n: all(c.get("builds", {}).get(n, {}).get("ok", False) for c in rep["calls"])
                      and all(x["ok"] for x in rep["anchors"] if x["lib"] == n) for n in libs}
    rep["served_z_ok"] = all(c.get("served_z_bad_words", 1) == 0 for c in rep["calls"])
    rep["ok"] = all(c.get("ok") for c in rep["calls"]) and all(x["ok"] for x in rep["anchors"])
    (out / "check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"verdict": rep["verdict"], "served_z_ok": rep["served_z_ok"], "ok": rep["ok"]}), flush=True)
    return 0 if rep["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
