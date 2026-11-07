"""A production-like plan for the ladder's served window, in serve_tax's plan format, and its replays broken down by kind.

    prod_plan.py build --checkpoint DIR --sharegpt FILE --c4 C4.json.gz --out PLAN.jsonl [--n 8000] [--seed 0]
    prod_plan.py kinds MODEL_DIR

``build`` mixes three kinds of request, each prompt the checkpoint's chat template over its messages:

- ``sharegpt``: vLLM's benchmark_serving ShareGPT set: a conversation's first human turn, capped at its reference reply's
  length, kept when the prompt and the reply each have at least 4 tokens, the prompt at most 1,024 and both at most 2,048;
- ``multiturn``: up to `TURNS` turns of one ShareGPT conversation, turn k's messages every earlier turn with its reference
  reply, then its own, placed `GAP` plan positions apart so that it arrives after the turn before it and its prefix is
  in the cache; capped at its reference reply's length, at most `MULTI_CAP`, with prompt and cap at most `MULTI_CONTEXT`;
- ``long``: C4 documents, from the shard's, joined to a prompt of 8,192 to 32,768 tokens with a request to summarize them,
  capped at `LONG_CAP`.

Each request is greedy or, with probability `SAMPLED`, at temperature 0.7 with its seed by the public seed rule
(`corpus.request_seed` of its id), the rest of its sampling the checkpoint's generation_config; every request stops at its
end of text or its cap. A plan's first requests are a window's: `GAP` keeps a conversation's turns within about a minute of
arrivals at the rates a TP8 node serves. Arrivals are Poisson: each request carries its unit-rate exponential gap
(``gap_draw``, which serve_tax's schedule divides by the load's rate), from a stream of its own so the rest of the plan
doesn't depend on it.

``kinds`` reads a ladder model directory (its plan, window and replays) into kinds.json: per kind, the window's requests,
served tokens and requests whose context passed `TOPK` keys, and per replay the requests whose tokens differ, with each
one's context at its first difference (prompt plus the position) and whether that context had passed `TOPK`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARE = {"long": 0.04, "multiturn": 0.25}
SAMPLED = 0.5
TEMPERATURE = 0.7
TURNS = 4
GAP = (24, 64)
MULTI_CAP = 1024
MULTI_CONTEXT = 16384
LONG = (8192, 32768)
LONG_CAP = 256
LONG_ASK = "\n\nSummarize the main points of each document above."
#: the DSA indexer's top-k (GLM-5.3's index_topk): a query over more keys than this attends to only the top-k the
#: indexer selects, so a difference that starts within it can't come from that selection
TOPK = 2048


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def build(a: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    chat = lambda msgs: enc(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))  # noqa: E731
    rng = random.Random(a.seed)
    convs = [c["conversations"] for c in json.loads(Path(a.sharegpt).read_text())]
    rng.shuffle(convs)
    n = a.n
    want = {k: round(v * n) for k, v in SHARE.items()}
    slots: list[dict | None] = [None] * n

    def alternating(c: list[dict]) -> bool:
        return all(m.get("from") == ("human", "gpt")[j % 2] for j, m in enumerate(c))

    def free_from(p: int) -> int | None:
        while p < n and slots[p] is not None:
            p += 1
        return p if p < n else None

    multi, used, conv_id = 0, set(), 0
    for ci, c in enumerate(convs):
        if multi >= want["multiturn"]:
            break
        if len(c) < 4 or not alternating(c[:2 * TURNS]):
            continue
        turns, msgs = [], []
        for k in range(min(TURNS, len(c) // 2)):
            msgs = msgs + [{"role": "user", "content": c[2 * k]["value"]}]
            ids, cap = chat(msgs), min(MULTI_CAP, len(enc(c[2 * k + 1]["value"])))
            if cap < 4 or len(ids) + cap > MULTI_CONTEXT:
                break
            turns.append((ids, cap))
            msgs = msgs + [{"role": "assistant", "content": c[2 * k + 1]["value"]}]
        if len(turns) < 2:
            continue
        p, placed, prev = rng.randrange(n), [], None
        for k, (ids, cap) in enumerate(turns):
            q = free_from(p)
            if q is None:
                break
            shared = 0 if prev is None else next((j for j, (x, y) in enumerate(zip(prev, ids)) if x != y), min(len(prev), len(ids)))
            slots[q] = {"kind": "multiturn", "conv": conv_id, "turn": k + 1, "prompt_ids": ids, "max_tokens": cap,
                        "shared_with_previous_turn": shared}
            placed.append(q)
            prev, p = ids, q + rng.randint(*GAP)
        if len(placed) < 2:
            for q in placed:
                slots[q] = None
            continue
        multi += len(placed)
        used.add(ci)
        conv_id += 1

    with gzip.open(a.c4, "rt") as f:
        docs = [json.loads(line)["text"] for line in f]
    rng.shuffle(docs)
    d = 0
    for _ in range(want["long"]):
        target = rng.randint(*LONG)
        body: list[int] = []
        while len(body) < target:
            body += enc(docs[d % len(docs)]) + enc("\n\n")
            d += 1
        text = tok.decode(body[:target - len(enc(LONG_ASK)) - 32], skip_special_tokens=True)
        ids = chat([{"role": "user", "content": "Here are some web documents.\n\n" + text + LONG_ASK}])
        q = free_from(rng.randrange(n)) or free_from(0)
        slots[q] = {"kind": "long", "prompt_ids": ids, "max_tokens": LONG_CAP}

    single = (c for ci, c in enumerate(convs) if ci not in used and len(c) >= 2 and alternating(c[:2]))
    for q in range(n):
        while slots[q] is None:
            c = next(single)
            p_ids, out = enc(c[0]["value"]), len(enc(c[1]["value"]))
            if len(p_ids) < 4 or out < 4 or len(p_ids) > 1024 or len(p_ids) + out > 2048:
                continue
            slots[q] = {"kind": "sharegpt", "prompt_ids": chat([{"role": "user", "content": c[0]["value"]}]), "max_tokens": out}

    out_f = Path(a.out)
    gaps = random.Random(f"gap_draw/{a.seed}")
    with open(out_f, "w") as f:
        for i, r in enumerate(slots):
            seeded = rng.random() < SAMPLED
            r |= {"i": i, "prompt_len": len(r["prompt_ids"]), "seeded": seeded,
                  "sampling": {"temperature": TEMPERATURE} if seeded else {"temperature": 0.0},
                  "gap_draw": gaps.expovariate(1.0)}
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    kinds: dict[str, dict] = {}
    for r in slots:
        k = kinds.setdefault(r["kind"], {"requests": 0, "seeded": 0, "prompt_tokens_max": 0, "context_max": 0})
        k["requests"] += 1
        k["seeded"] += r["seeded"]
        k["prompt_tokens_max"] = max(k["prompt_tokens_max"], r["prompt_len"])
        k["context_max"] = max(k["context_max"], r["prompt_len"] + r["max_tokens"])
    meta = {"schema": "verity-bench/replay-ladder/prod-plan/v0", "n": n, "seed": a.seed, "kinds": kinds,
            "checkpoint": str(a.checkpoint), "sharegpt": {"file": Path(a.sharegpt).name, "sha256": sha256(Path(a.sharegpt))},
            "c4": {"file": Path(a.c4).name, "sha256": sha256(Path(a.c4))}, "share": SHARE, "sampled": SAMPLED,
            "temperature": TEMPERATURE, "turns": TURNS, "gap": GAP, "long": LONG, "long_cap": LONG_CAP,
            "multi_cap": MULTI_CAP, "multi_context": MULTI_CONTEXT, "conversations": conv_id}
    out_f.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["kinds"]))
    return 0


def kinds(a: argparse.Namespace) -> int:
    sys.path.insert(0, str(HERE))
    import ladder as L

    d = Path(a.model_dir)
    w = L.R.window(d / "record", L.LOAD, 1)
    plan = {r["i"]: r for r in map(json.loads, (d / "plan.jsonl").read_text().splitlines())}
    client = {}
    for line in (d / "record" / L.LOAD / "rep-1" / "client.jsonl").read_text().splitlines():
        c = json.loads(line)
        client[c["id"]] = c["i"]
    of = {}
    for r in w["requests"]:
        http = r["engine_id"].split("-", 1)[1].rsplit("-", 2)[0]
        of[r["engine_id"]] = plan[client[http]]
    out: dict = {"model_dir": str(d), "kinds": {}, "replays": {}}
    for rid, p in of.items():
        k = out["kinds"].setdefault(p["kind"], {"requests": 0, "seeded": 0, "served_tokens": 0, "context_max": 0,
                                                "past_topk": 0})
        served = next(r["served"] for r in w["requests"] if r["engine_id"] == rid)
        k["requests"] += 1
        k["seeded"] += bool(p.get("seeded"))
        k["served_tokens"] += len(served)
        k["context_max"] = max(k["context_max"], p["prompt_len"] + len(served))
        k["past_topk"] += p["prompt_len"] + len(served) > TOPK
    for f in sorted(d.glob("replay-*/replay.json")):
        r = json.loads(f.read_text())
        rows = []
        for x in r.get("requests_differing") or ():
            p = of.get(x["request"], {})
            at = (p.get("prompt_len") or 0) + x["first_position"]
            rows.append({"request": x["request"], "kind": p.get("kind"), "seeded": p.get("seeded"), "prompt_len": p.get("prompt_len"),
                         "first_position": x["first_position"], "context_at_first": at, "past_topk_at_first": at > TOPK,
                         "positions": x["positions"]})
        by = {}
        for x in rows:
            by.setdefault(x["kind"], 0)
            by[x["kind"]] += 1
        out["replays"][f.parent.name] = {"requests_tokens_differing": r["requests_tokens_differing"], "by_kind": by,
                                         "listed": len(rows), "differing": rows, "bit_exact": r["bit_exact"]}
    (d / "kinds.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"kinds": out["kinds"], "replays": {k: {kk: v[kk] for kk in ("requests_tokens_differing", "by_kind")}
                                                         for k, v in out["replays"].items()}}))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--sharegpt", required=True)
    b.add_argument("--c4", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=8000)
    b.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("kinds")
    k.add_argument("model_dir")
    a = p.parse_args(argv)
    return build(a) if a.cmd == "build" else kinds(a)


if __name__ == "__main__":
    sys.exit(main())
