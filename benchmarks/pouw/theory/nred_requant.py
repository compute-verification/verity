"""N-RED: is a linear's int8 input known before the noise? The premise of A5 (and of A1/A2 on a self-dealer's requests).

A5 tables a row's earliest steps from the linear's input, computed before the call's coins. Under a served-noise design
every linear after layer 0's qkv reads earlier linears' served noise: a noised accumulator that crosses a rounding
boundary of the next int8 quantization, or moves a row's absmax, changes the input. This runs Llama-3.2-1B with every
fused linear in fake int8 (int8-channel127-v1: activations one scale a token, weights one an output row, exact integer
accumulation in float64), once clean and once per design with that design's served noise N_S − β added to every
accumulator (`nred_round2.final_noise`, the call's coins drawn per layer and linear), and compares each linear's int8
input with the clean pass's:

- the share of entries changed;
- the share of rows whose prefix through ssn_1l step τ is unchanged;
- A5's usable share: the saving of `a5_step_greedy`'s tables restricted to each row's unchanged steps, over their
  saving if every input were known;
- top-1 agreement of the next-token logits (clean fake-int8 against fp32, each design against clean).

Run (CPU): python nred_requant.py --model DIR --text wikitext-2-raw-v1-test.json --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nred_attacks as na  # noqa: E402
import nred_round2 as r2  # noqa: E402
import noise_data as nd  # noqa: E402
import noise_sweep as ns  # noqa: E402

DESIGNS = ("ssn_1l", "fix2", "hw")
A5_DESIGNS = ("ssn_1l", "fix2")
TAUS = (1, 2, 4, 8, 13)


class Pass:
    """One forward's design (None: clean), its coin stream, its noise per (layer, linear) and the inputs it saw."""

    def __init__(self, design: str | None, seed: int):
        self.design, self.rng = design, np.random.default_rng(seed)
        self.noise: dict = {}
        self.inputs: dict = {}
        self.noise_std: list = []

    def noise_for(self, li: int, name: str, m: int):
        if self.design is None:
            return None
        if (li, name) not in self.noise:
            n, k = ns.LLAMA[name]
            s = ns.steps1l(k)
            pub = r2.public(self.design, n, s, m, self.rng)
            co = r2.coins(self.design, m, n, s, self.rng)
            nz = r2.final_noise(self.design, pub, co)
            self.noise_std.append(float(nz.std()))
            self.noise[(li, name)] = torch.from_numpy(nz.astype(np.float64))
        return self.noise[(li, name)]


def patch(model, state: list) -> None:
    for li, layer in enumerate(model.model.layers):
        for name, (part, mods) in nd.LINEARS.items():
            offset = 0
            for mod_name in mods:
                lin = getattr(getattr(layer, part), mod_name)
                w = lin.weight.detach().double()
                sw = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) / 127
                wq = torch.round(w / sw).clamp(-127, 127).to(torch.int8)

                def fwd(x, li=li, name=name, offset=offset, wq=wq, sw=sw, first=offset == 0):
                    shp = x.shape
                    x2 = x.reshape(-1, shp[-1]).double()
                    sx = x2.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) / 127
                    xq = torch.round(x2 / sx).clamp(-127, 127)
                    p = state[0]
                    if first:
                        p.inputs[(li, name)] = xq.to(torch.int8).numpy()
                    acc = xq @ wq.double().T
                    nz = p.noise_for(li, name, x2.shape[0])
                    if nz is not None:
                        acc = acc + nz[:, offset:offset + wq.shape[0]]
                    return (acc * sx * sw.T).to(x.dtype).reshape(*shp[:-1], wq.shape[0])

                lin.forward = fwd
                offset += wq.shape[0]


def unchanged_steps(clean: np.ndarray, noisy: np.ndarray, s: int) -> np.ndarray:
    """u_i: the number of leading ssn_1l steps (28 entries, then 31 each) of row i whose entries all agree."""
    diff = clean != noisy
    k = clean.shape[1]
    first = np.where(diff.any(axis=1), diff.argmax(axis=1), k)
    u = np.where(first >= ns.DATA_PER_STEP, 1 + (first - ns.DATA_PER_STEP) // 31, 0)
    return np.where(first == k, s, np.minimum(u, s))


def a5_gain(t: np.ndarray, s: int) -> np.ndarray:
    return np.where(t == s, t, np.maximum(t - 1, 0))


def measure(model, windows: list) -> dict:
    with torch.no_grad():
        fp32_top = [model(x).logits.argmax(-1) for x in windows]
    state: list = [None]
    patch(model, state)
    a5 = r2.a5_step_greedy(A5_DESIGNS)
    acc: dict = {}
    top1: dict = {"clean_vs_fp32": []}
    noise_std: dict = {d: [] for d in DESIGNS}
    for wi, x in enumerate(windows):
        state[0] = Pass(None, 0)
        with torch.no_grad():
            clean_top = model(x).logits.argmax(-1)
        clean = state[0].inputs
        top1["clean_vs_fp32"].append(float((clean_top == fp32_top[wi]).float().mean()))
        for di, design in enumerate(DESIGNS):
            state[0] = Pass(design, 1000 * wi + di + 1)
            with torch.no_grad():
                top = model(x).logits.argmax(-1)
            top1.setdefault(f"{design}_vs_clean", []).append(float((top == clean_top).float().mean()))
            noise_std[design].extend(state[0].noise_std)
            for (li, name), q in state[0].inputs.items():
                n, k = ns.LLAMA[name]
                s = ns.steps1l(k)
                u = unchanged_steps(clean[(li, name)], q, s)
                e = acc.setdefault((design, li, name), {"entries": 0, "changed": 0, "rows": 0,
                                                       "prefix": {t: 0 for t in TAUS + (s,)}, "a5": {}})
                e["entries"] += q.size
                e["changed"] += int((clean[(li, name)] != q).sum())
                e["rows"] += q.shape[0]
                for t in e["prefix"]:
                    e["prefix"][t] += int((u >= t).sum())
                if design in A5_DESIGNS:
                    for m in na.MS:
                        tau = a5[f"{design}/m{m}"]["steps_tabled"][name]
                        got = float(a5_gain(np.minimum(u, tau), s).sum()) * n
                        full = float(a5_gain(np.array(tau), s)) * n * q.shape[0]
                        g = e["a5"].setdefault(m, [0.0, 0.0])
                        g[0] += got
                        g[1] += full
    per, by_name, usable = {}, {}, {}
    for (design, li, name), e in acc.items():
        n, k = ns.LLAMA[name]
        row = {"entry_changed": e["changed"] / e["entries"],
               "rows_unchanged_through_step": {str(t): c / e["rows"] for t, c in e["prefix"].items()}}
        per[f"{design}/{li}.{name}"] = row
        b = by_name.setdefault(f"{design}/{name}", {"entries": 0, "changed": 0, "rows": 0, "prefix": {}})
        b["entries"] += e["entries"]
        b["changed"] += e["changed"]
        b["rows"] += e["rows"]
        for t, c in e["prefix"].items():
            b["prefix"][t] = b["prefix"].get(t, 0) + c
        for m, (got, full) in e["a5"].items():
            u_ = usable.setdefault(f"{design}/m{m}", [0.0, 0.0])
            u_[0] += got
            u_[1] += full
    by_name = {key: {"entry_changed": b["changed"] / b["entries"],
                     "rows_unchanged_through_step": {str(t): c / b["rows"] for t, c in b["prefix"].items()}}
               for key, b in by_name.items()}
    a5_effective = {key: {"usable_share": got / full if full else None,
                          "a5_gap_if_known": a5[key]["gap_share_of_stock"],
                          "a5_gap_effective": (got / full if full else 0.0) * a5[key]["gap_share_of_stock"]}
                    for key, (got, full) in usable.items()}
    std_expect = r2.served_std()
    checks = [
        {"check": "layer 0's qkv input (no noise upstream) is unchanged under every design",
         "pass": all(per[f"{d}/0.qkv"]["entry_changed"] == 0 for d in DESIGNS)},
        {"check": "the clean fake-int8 forward agrees with fp32 on > 90% of top-1 tokens",
         "pass": min(top1["clean_vs_fp32"]) > 0.9, "detail": top1["clean_vs_fp32"]},
        {"check": "every design's noise was added (std > 1) and changed some later linear's input",
         "pass": all(min(noise_std[d]) > 1 and max(v["entry_changed"] for key, v in by_name.items()
                                                   if key.startswith(d + "/")) > 0 for d in DESIGNS),
         "detail": {d: {"noise_std_median": float(np.median(noise_std[d])),
                        "served_std_n256_K2048": std_expect[f"{d}/K2048"]} for d in DESIGNS}},
    ]
    return {"checks": checks, "top1": top1, "by_linear": by_name, "a5_effective": a5_effective, "per_layer": per}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=2)
    args = ap.parse_args(argv)
    import transformers

    torch.manual_seed(0)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    ids = tok("\n\n".join(json.load(open(args.text))), add_special_tokens=False)["input_ids"]
    step = args.ctx - 1
    windows = [torch.tensor([[tok.bos_token_id] + ids[w * step:(w + 1) * step]]) for w in range(args.windows)]
    rep = {"args": vars(args), **measure(model, windows)}
    with open(args.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    checks, top1, by_name, a5_effective = rep["checks"], rep["top1"], rep["by_linear"], rep["a5_effective"]
    for ch in checks:
        print(("PASS " if ch["pass"] else "FAIL ") + ch["check"])
    print(json.dumps({"top1": top1, "by_linear": by_name, "a5_effective": a5_effective}, indent=1))
    return 0 if all(ch["pass"] for ch in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
