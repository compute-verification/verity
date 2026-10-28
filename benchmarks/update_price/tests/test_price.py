"""The pricing pipeline on a toy checkpoint: every description it counts decodes to its target (the tasks exit
otherwise), and the tables render."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import price as P

SHAPES = {"model.embed_tokens.weight": (64, 16), "model.layers.0.self_attn.q_proj.weight": (16, 16),
          "model.layers.0.input_layernorm.weight": (16,)}


def toy(root: Path) -> dict:
    """θ, versions 1 and 3 of an Adam-like walk, the moments at step 1, the tokens seen, and a train.json."""
    rng = np.random.Generator(np.random.PCG64(5))
    x = {n: (0.02 * rng.standard_normal(s)).astype(np.float32) for n, s in SHAPES.items()}
    state = {}
    for step in range(4):
        if step in (0, 1, 3):
            (root / str(step)).mkdir(parents=True)
            for n, v in x.items():
                np.save(root / str(step) / f"{n}.npy", v)
        if step == 1:
            (root / "state1").mkdir()
            for n, (ea, es) in state.items():
                np.save(root / "state1" / f"{n}.exp_avg.npy", ea)
                np.save(root / "state1" / f"{n}.exp_avg_sq.npy", es)
        for n, v in x.items():
            g = rng.standard_normal(v.shape).astype(np.float32) + 0.5 * np.sign(v)
            ea, es = state.get(n, (np.zeros_like(v), np.zeros_like(v)))
            ea, es = (0.9 * ea + 0.1 * g).astype(np.float32), (0.95 * es + 0.05 * g * g).astype(np.float32)
            state[n] = (ea, es)
            x[n] = (v - 1e-3 * ea / (np.sqrt(es) + 1e-8)).astype(np.float32)
    for n in (1, 3):
        seen = np.zeros(64, bool)
        seen[: 10 * n] = True
        np.save(root / f"seen{n}.npy", seen)
    steps = [{"step": s, "loss": 3.0 - 0.1 * s, "fwd_s": 0.1, "bwd_s": 0.2, "opt_s": 0.05} for s in range(1, 4)]
    stream = {str(n): {"tokens": 64 * n, "prequential_bits": 200.0 * n, "raw_bits": 384.0 * n, "xz_bits": 300 * n,
                       "unseen_tokens": 64 - 10 * n} for n in (1, 3)}
    return {"model": {"repo": "toy", "revision": "0" * 40}, "data": {"repo": "toy", "revision": "1" * 40, "file": "toy"},
            "config": {"steps": 3, "save": "1,3", "state": "1", "batch": 2, "seq": 32, "lr": 1e-3, "warmup": 1,
                       "min_lr_frac": 0.1, "wd": 0.1, "betas": "0.9,0.95", "eps": 1e-8, "clip": 1.0, "seed": 0},
            "machine": {"cpu": "toy", "threads": 1},
            "train": {"params": [[n, list(s)] for n, s in SHAPES.items()], "steps": steps},
            "token_stream": stream, "replay": {"steps": 1, "match": {"0": True, "1": True}}}


def test_pricing_pipeline_on_a_toy_run(tmp_path):
    ck = tmp_path / "ckpt"
    train = toy(ck)
    names = list(SHAPES)
    tensors = [P.tensor_task({"ckpt": str(ck), "N": n, "fmt": f, "name": nm, "ms": [1], "eps": 1e-8})
               for n in (1, 3) for f in P.FMTS for nm in names]
    by = {(r["N"], r["fmt"], r["name"]): r for r in tensors}
    assert set(by[(3, "fp32", names[1])]["pred"]) == {"0", "1"} and set(by[(1, "fp32", names[1])]["pred"]) == {"0"}
    assert set(by[(3, "fp32", names[1])]["lattice"]) == {"0", "1"}
    sels = [P.selection_task({"ckpt": str(ck), "N": 3, "fmt": f, "names": names, "seed": 0, "m_pred": 1, "eps": 1e-8,
                              "coefs": {nm: by[(3, f, nm)]["pred"]["1"]["coef"] for nm in names},
                              "lattices": {m: {nm: (by[(3, f, nm)]["lattice"][m]["coef"], by[(3, f, nm)]["lattice"][m]["step"])
                                               for nm in names} for m in ("0", "1")}}) for f in P.FMTS]
    assert {"top0.01", "bottom0.1", "theta_top0.1", "random1000", "embed_unseen"} <= set(sels[0]["selections"])
    assert {"ulp", "pred1", "lattice0", "lattice1"} <= set(sels[0]["selections"]["top0.01"])
    gps = [P.gp_task({"ckpt": str(ck), "N": 3, "fmt": "fp32", "layer": 0, "names": names[1:]})]
    res = P.aggregate(train, names, tensors, sels, gps, [1])
    v = res["versions"]["3/fp32"]
    allg = v["groups"]["all"]
    assert allg["n"] == 64 * 16 + 16 * 16 + 16
    assert allg["best"]["replay1"]["bits"] <= allg["best"]["codec"]["bits"] <= allg["methods"]["ulp"]["bits"]
    assert v["replay"]["verified"] is None and res["versions"]["1/fp32"]["replay"]["verified"] is True
    assert set(v["regimes"]) == {"codec", "replay1", "replay_registered", "replay_unregistered"}
    for k in res["versions"]:
        res["versions"][k]["changed"] = 0
    text = P.render(res, train)
    assert "## λ implied" in text and "| 3 | fp32 | replay1 |" in text
    json.dumps(res)
