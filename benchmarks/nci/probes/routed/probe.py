"""The GPU half of the routed-mixture probes: cut a corpus into windows, score every window with a causal LM, and
confine a Qwen3-MoE's sequences to K experts per layer. ``score``, ``codebook`` and ``selftest`` need torch and
transformers (5.x); ``windows`` and ``analyze`` need only ``analysis.py``. The README has the commands and the order
they run in."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis as A  # noqa: E402

LN2 = math.log(2.0)
#: experts per layer the routing file ranks by prefix mass: the largest K a prefix confinement can ask for
PREFIX_RANKS = 64
MASS_KS = (8, 16, 32, 64)


def encode(tok, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and character offsets, with the tokenizer's BOS (if it has one) first at offset (0, 0)."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    if tok.bos_token_id is not None:
        ids, offs = [tok.bos_token_id, *ids], [(0, 0), *offs]
    return ids, offs


def split(offs: list[tuple[int, int]], prefix_chars: int) -> tuple[int, int]:
    """The first scored token (the first that starts at or after the prefix) and the character the scored span starts
    at: the end of the token before it, so a leading space counts in the span whose token pays for it."""
    j = next((j for j in range(1, len(offs)) if offs[j][0] >= prefix_chars), None)
    if j is None:
        raise ValueError("the window has no token after its prefix")
    return j, offs[j - 1][1]


def window_bits(logits, ids: list[int], j: int) -> tuple[float, float]:
    """Bits of the prefix tokens 1..j-1 and of the scored tokens j.. under next-token ``logits`` (tokens x vocab)."""
    import torch

    target = torch.tensor(ids[1:], device=logits.device)
    nll = torch.nn.functional.cross_entropy(logits[:-1].float(), target, reduction="none") / LN2
    return float(nll[: j - 1].sum()), float(nll[j - 1:].sum())


def routers(model) -> list:
    """The token routers of a Qwen3-MoE-style model in layer order: modules with an (experts x hidden) ``weight``,
    ``top_k`` and ``norm_topk_prob`` whose forward returns (logits, weights, indices), as transformers 5.x builds
    them."""
    import torch

    return [m for m in model.modules() if isinstance(getattr(m, "weight", None), torch.nn.Parameter)
            and hasattr(m, "top_k") and hasattr(m, "norm_topk_prob")]


class Confinement:
    """Each router's forward, replaced by one that can restrict every token of a window to an allowed set of experts
    per layer and can record the routing probabilities every token gives every expert.

    The mask goes on the logits before top-k, so with ``norm_topk_prob`` the k weights are the stock router's top-k
    over the allowed experts. With no mask the forward is the stock one (``selftest`` checks both)."""

    def __init__(self, model):
        self.routers = routers(model)
        if not self.routers:
            raise SystemExit("no Qwen3-MoE-style router (weight, top_k, norm_topk_prob) in this model: confinement and "
                             "routing records support Qwen3-MoE under transformers 5.x only")
        self.mask: list | None = None
        self.record: list[list] | None = None
        for layer, r in enumerate(self.routers):
            r.forward = self._forward(layer, r)

    def _forward(self, layer: int, r):
        import torch
        import torch.nn.functional as F

        def forward(hidden_states):
            logits = F.linear(hidden_states.reshape(-1, r.weight.shape[1]), r.weight)
            if self.record is not None:
                self.record[layer].append(torch.softmax(logits, dim=-1, dtype=torch.float))
            if self.mask is not None:
                logits = logits + self.mask[layer].to(logits.dtype)
            top, idx = torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float), r.top_k, dim=-1)
            if r.norm_topk_prob:
                top = top / top.sum(dim=-1, keepdim=True)
            return logits, top.to(logits.dtype), idx

        return forward

    def allow(self, subsets) -> None:
        """Confine every token to ``subsets[layer]`` (expert indices), or free every router with None."""
        import torch

        if subsets is None:
            self.mask = None
            return
        self.mask = []
        for r, s in zip(self.routers, subsets, strict=True):
            s = torch.as_tensor(s, dtype=torch.long, device=r.weight.device)
            if len(s.unique()) < r.top_k:
                raise ValueError(f"an allowed set needs at least top_k = {r.top_k} experts, got {len(s.unique())}")
            m = torch.full((r.weight.shape[0],), float("-inf"), device=r.weight.device)
            m[s] = 0.0
            self.mask.append(m)


def kmeans(X, k: int, seed: int, iters: int = 30):
    """Spherical k-means with k-means++ seeding on unit rows of ``X``, deterministic for a seed."""
    import torch

    g = torch.Generator().manual_seed(seed)
    C = [X[int(torch.randint(len(X), (1,), generator=g))]]
    for _ in range(1, k):
        d = (1 - (X @ torch.stack(C).T).max(dim=1).values).clamp(min=0)
        C.append(X[int(torch.multinomial(d if d.sum() > 0 else torch.ones_like(d), 1, generator=g))])
    C = torch.stack(C)
    for _ in range(iters):
        a = (X @ C.T).argmax(dim=1)
        for c in range(k):
            if (a == c).any():
                m = X[a == c].mean(0)
                C[c] = m / m.norm()
    return C


def _confine_arg(s: str) -> tuple[str, int]:
    if s == "none":
        return "none", 0
    mode, _, k = s.partition(":")
    if mode not in ("prefix", "codebook") or not k.isdigit():
        raise argparse.ArgumentTypeError("--confine is none, prefix:K or codebook:K")
    return mode, int(k)


def cmd_windows(args) -> None:
    n = 0
    with open(args.corpus) as f, open(args.out, "w") as out:
        for doc in map(json.loads, f):
            for w in A.windows(doc["id"], doc["stratum"], doc["text"], args.size, args.prefix):
                out.write(json.dumps(w) + "\n")
                n += 1
    print(f"{n} windows -> {args.out}")


def cmd_score(args) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mode, K = args.confine
    if args.routing_out and mode != "none":
        raise SystemExit("--routing-out records unconfined routing: drop --confine")
    if mode != "none" and not args.routing:
        raise SystemExit(f"--confine {mode}:{K} needs --routing, the file an unconfined --routing-out run wrote")
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=getattr(torch, args.dtype),
                                                 device_map=args.device).eval()
    device = next(model.parameters()).device
    recording = args.routing_out is not None
    conf = Confinement(model) if mode != "none" or recording else None
    routing = torch.load(args.routing) if mode != "none" else None
    row = {w: i for i, w in enumerate(routing["windows"])} if routing else {}
    codebook = torch.load(args.codebook) if args.codebook else None
    if mode == "codebook" and (codebook is None or routing.get("cluster") is None):
        raise SystemExit("codebook:K needs --codebook and a --routing file recorded with that codebook")
    with open(args.windows) as f:
        wins = [json.loads(line) for line in f]
    limit = getattr(model.config, "max_position_embeddings", None)
    kept: dict[str, list] = {"windows": [], "prefix_rank": [], "profile": [], "cluster": []}
    too_long = []
    part = Path(str(args.out) + ".part")
    with open(part, "w") as out:
        for w in wins:
            ids, offs = encode(tok, w["text"])
            if limit and len(ids) > limit:
                too_long.append(w["id"])
                continue
            j, cut = split(offs, w["prefix_chars"])
            if mode == "prefix":
                conf.allow(routing["prefix_rank"][row[w["id"]], :, :K])
            elif mode == "codebook":
                conf.allow(codebook["subsets"][K][int(routing["cluster"][row[w["id"]]])])
            if recording:
                conf.record = [[] for _ in conf.routers]
            with torch.no_grad():
                logits = model(torch.tensor([ids], device=device)).logits[0]
            prefix_bits, span_bits = window_bits(logits, ids, j)
            rec = {"window": w["id"], "stratum": w["stratum"], "prefix_bits": prefix_bits, "span_bits": span_bits,
                   "span_bytes": len(w["text"][cut:].encode()), "tokens": len(ids)}
            if recording:
                probs = [torch.cat(p) for p in conf.record]
                conf.record = None
                top_k = conf.routers[0].top_k
                prefix_mass = torch.stack([p[:j].sum(0) for p in probs])
                profile = torch.stack([p.mean(0) for p in probs])
                kept["windows"].append(w["id"])
                kept["prefix_rank"].append(prefix_mass.argsort(dim=-1, descending=True)[:, :PREFIX_RANKS]
                                           .to(torch.int16).cpu())
                if args.profiles:
                    kept["profile"].append(profile.flatten().cpu())
                if codebook is not None:
                    unit = profile.flatten() / profile.flatten().norm()
                    kept["cluster"].append(int((codebook["centroids"].to(unit.device) @ unit).argmax()))
                rec["experts_used"] = sum(len(p.topk(top_k, dim=-1).indices.unique()) for p in probs) / len(probs)
                rec["mass_in_top"] = {k: float(profile.sort(dim=-1, descending=True).values[:, :k].sum(-1).mean())
                                      for k in MASS_KS if k <= profile.shape[1]}
            out.write(json.dumps(rec) + "\n")
    if recording:
        torch.save({"windows": kept["windows"], "prefix_rank": torch.stack(kept["prefix_rank"]),
                    "profile": torch.stack(kept["profile"]) if kept["profile"] else None,
                    "cluster": torch.tensor(kept["cluster"]) if kept["cluster"] else None,
                    "layers": len(conf.routers), "experts": conf.routers[0].weight.shape[0]}, args.routing_out)
    meta = {"model": args.model, "revision": args.revision, "dtype": args.dtype, "confine": f"{mode}:{K}",
            "windows": len(wins) - len(too_long), "too_long": too_long, "max_positions": limit,
            "torch": torch.__version__, "transformers": transformers.__version__}
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    part.rename(args.out)
    print(f"{meta['windows']} windows scored, {len(too_long)} longer than {limit} tokens skipped -> {args.out}")


def cmd_codebook(args) -> None:
    import torch

    r = torch.load(args.routing)
    if r.get("profile") is None:
        raise SystemExit("the routing file has no profiles: record it with score --routing-out ... --profiles")
    X = r["profile"].float()
    C = kmeans(X / X.norm(dim=1, keepdim=True), args.paths, args.seed)
    ranked = C.view(args.paths, r["layers"], r["experts"]).argsort(dim=-1, descending=True)
    torch.save({"centroids": C, "subsets": {k: ranked[..., :k].to(torch.int16) for k in args.experts},
                "layers": r["layers"], "experts": r["experts"]}, args.out)
    print(f"{args.paths} paths x K in {args.experts} from {len(X)} windows -> {args.out}")


def cmd_analyze(args) -> None:
    manifest = Path(args.manifest)
    base = Path(args.base) if args.base else manifest.parent
    out = A.analyze(json.loads(manifest.read_text()), base, args.resamples, args.seed)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for sname, s in out["studies"].items():
        if s["ladder_error"]:
            print(f"{sname:10} {s['ladder_error']}")
        for name, c in s["configs"].items():
            iv = "[{:.2f}, {:.2f}]".format(*c["g_interval"]) if c["g_interval"] else "(no interval)"
            print(f"{sname:10} {name:20} {c['bpb']:.4f} bits/byte  g = {c['gains']['besiroglu']['g']:.2f} {iv}")
    v = out["verdict"]
    print(f"verdict: {v['call']} (claim line x{v['claim_line']:.2f}): {v['say']}")


def cmd_selftest(args) -> None:
    """On CPU, a tiny random Qwen3-MoE: a free Confinement reproduces the stock forward, allowing every expert changes
    nothing, allowing K < E keeps every token's experts in the set and changes the output, and prefix plus span bits are
    the window's bits."""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    E, L = 16, 3
    cfg = Qwen3MoeConfig(vocab_size=257, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                         num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=E,
                         num_experts_per_tok=2, norm_topk_prob=True, max_position_embeddings=256)
    model = Qwen3MoeForCausalLM(cfg).eval()
    for r in routers(model):
        torch.nn.init.normal_(r.weight, std=1.0)
    ids = torch.randint(0, cfg.vocab_size, (1, 96))
    with torch.no_grad():
        stock = model(ids).logits
        conf = Confinement(model)
        assert len(conf.routers) == L, f"found {len(conf.routers)} routers, expected {L}"
        assert torch.equal(model(ids).logits, stock), "a free Confinement must be the stock forward"
        conf.allow([list(range(E))] * L)
        assert torch.equal(model(ids).logits, stock), "allowing every expert must change nothing"
        subsets = [list(range(4 * i, 4 * i + 4)) for i in range(L)]
        chosen: list = []
        hooks = [r.register_forward_hook(lambda m, i, o: chosen.append(o[2])) for r in conf.routers]
        conf.allow(subsets)
        confined = model(ids).logits
        for h in hooks:
            h.remove()
        for layer, idx in enumerate(chosen):
            assert set(idx.unique().tolist()) <= set(subsets[layer]), f"layer {layer} left its allowed set"
        assert not torch.allclose(confined, stock), "confinement to 4 of 16 experts should change the output"
    offs = [(0, 0)] + [(i, i + 1) for i in range(95)]
    j, cut = split(offs, 40)
    pre, span = window_bits(stock[0], ids[0].tolist(), j)
    total = float(torch.nn.functional.cross_entropy(stock[0, :-1], ids[0, 1:], reduction="sum")) / LN2
    assert (j, cut) == (41, 40) and abs(pre + span - total) < 1e-3 * total, (j, cut, pre + span, total)
    print(f"selftest passed: {L} routers confined; prefix {pre:.1f} + span {span:.1f} = {total:.1f} bits")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("windows", help="cut a corpus (JSONL of id, stratum, text) into windows")
    p.add_argument("corpus")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=A.WINDOW_BYTES)
    p.add_argument("--prefix", type=int, default=A.PREFIX_BYTES)
    p = sub.add_parser("score", help="bits per window for one model and confinement")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--windows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--confine", type=_confine_arg, default=("none", 0), help="none, prefix:K or codebook:K")
    p.add_argument("--routing", help="a routing file from an unconfined run (needed by --confine)")
    p.add_argument("--routing-out", help="record prefix rankings (and clusters, with --codebook) to this file")
    p.add_argument("--profiles", action="store_true", help="also record each window's routing profile")
    p.add_argument("--codebook", help="a codebook file: assign clusters (with --routing-out) or confine (codebook:K)")
    p = sub.add_parser("codebook", help="cluster recorded routing profiles into expert subsets")
    p.add_argument("--routing", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--paths", type=int, default=16)
    p.add_argument("--experts", type=lambda s: [int(x) for x in s.split(",")], default=list(MASS_KS))
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("analyze", help="g, intervals, D at g and the verdict from a manifest")
    p.add_argument("manifest")
    p.add_argument("--base", help="the directory holding scores/ (default: the manifest's)")
    p.add_argument("--out", required=True)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("selftest", help="CPU check of the confinement on a tiny random Qwen3-MoE")
    args = ap.parse_args(argv)
    {"windows": cmd_windows, "score": cmd_score, "codebook": cmd_codebook, "analyze": cmd_analyze,
     "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
