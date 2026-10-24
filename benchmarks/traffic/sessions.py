"""Fetch public traffic and agent sessions, tokenize them with DeepSeek-V4-Pro's tokenizer, and write one units file
(``units.py``) per source into OUT/units/, with per-source statistics in OUT/sources.json.

    python sessions.py OUT [--sample N] [--only SRC,...]

Sources (class): Mooncake conversation (chat), tool and agent (agentic) and synthetic (rag, long-document QA), by prefix
hashes (``hashtrace.py``); ServeGen's Bailian sessions (chat), DeepSeek-R1 (reasoning) and language m-large (api)
distributions; BurstGPT conversation and API logs (chat, api); Azure 2023 conversation and code (chat, api); SWE-agent
(nebius/SWE-agent-trajectories), OpenHands (nebius/SWE-rebench-openhands-trajectories) and tau-bench (agentic); RAGBench
(rag). Network access is needed; everything is cached under $TRAFFIC_CACHE (default a temp dir).
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import lzma
import os
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hashtrace  # noqa: E402
import units  # noqa: E402

MSG_OVERHEAD = 4  # role and separator tokens per chat message
CACHE = Path(os.environ.get("TRAFFIC_CACHE") or tempfile.mkdtemp(prefix="traffic-"))
MOONCAKE = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/"
SERVEGEN = "https://raw.githubusercontent.com/alibaba/ServeGen/main/data/"
AZURE = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"
TAU = "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/"
RAGBENCH_CONFIGS = ["covidqa", "cuad", "delucionqa", "emanual", "expertqa", "finqa", "hagrid", "hotpotqa", "msmarco",
                    "pubmedqa", "tatqa", "techqa"]


def fetch(url: str) -> Path:
    p = CACHE / url.split("://", 1)[1].replace("/", "_")
    if not p.exists():
        with urllib.request.urlopen(url, timeout=600) as r:
            p.write_bytes(r.read())
    return p


_TOK = None


def ntok(texts: list[str]) -> np.ndarray:
    global _TOK
    if _TOK is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(hf_hub_download("deepseek-ai/DeepSeek-V4-Pro", "tokenizer.json"))
    if not texts:
        return np.zeros(0, dtype=np.int64)
    enc = _TOK.encode_batch([t or "" for t in texts], add_special_tokens=False)
    return np.array([len(e.ids) for e in enc], dtype=np.int64)


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---------------------------------------------------------------------------------------------------------------- sessions

def sessions_to_units(sessions: list[list[tuple[str, str]]], prefix: list[str],
                      group: list[str] | None = None) -> tuple[dict, dict]:
    """``sessions``: each a list of (kind, text), kind in system | user | tool | assistant. ``prefix``: per session the text
    a registered prefix would hold (system prompt plus tool definitions); its registered part is its longest common prefix
    with the most common such text of its ``group``. Every assistant message is one call over everything before it."""
    group = group or [""] * len(sessions)
    modes = {g: Counter(p for p, gg in zip(prefix, group) if gg == g).most_common(1)[0][0] for g in set(group)}
    mode_tok = {g: int(n) for g, n in zip(modes, ntok(list(modes.values())))}
    flat = [t for s in sessions for _, t in s]
    counts = iter(ntok(flat).tolist())
    rows = {f: [] for f in units.FIELDS}
    tool_sizes, user_sizes, comp = [], [], []
    for si, (s, pre, g) in enumerate(zip(sessions, prefix, group)):
        if pre == modes[g]:
            reg_tok = mode_tok[g]
        else:
            k = _lcp(pre, modes[g])
            reg_tok = int(ntok([pre[:k]])[0]) if k else 0
        ctx = 0
        pending = 0  # user and tool tokens since the last call
        turn = 0
        first = True
        new_text = []
        new_tok = 0
        for kind, text in s:
            n = next(counts) + MSG_OVERHEAD
            if kind == "assistant":
                if ctx > 0:
                    rows["sess"].append(si)
                    rows["turn"].append(turn)
                    rows["ctx"].append(ctx)
                    r = min(reg_tok, ctx)
                    rows["reg"].append(r)
                    rows["new"].append(max(0, ctx - r) if first else min(pending, ctx))
                    rows["out"].append(n)
                    turn += 1
                    first = False
                    pending = 0
                ctx += n
            else:
                ctx += n
                if kind != "system":
                    pending += n
                    (tool_sizes if kind == "tool" else user_sizes).append(n)
                    new_text.append(text)
                    new_tok += n - MSG_OVERHEAD
        if si < 400 and new_tok:
            raw = lzma.compress("\n".join(new_text).encode(), format=lzma.FORMAT_RAW,
                                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9}])
            comp.append(8 * len(raw) / new_tok)
    q = lambda a: {p: float(np.percentile(a, p)) for p in (50, 90, 99)} if len(a) else {}
    stats = {"registered_prefix_tokens_mode": mode_tok,
             "sessions_with_mode_prefix": sum(p == modes[g] for p, g in zip(prefix, group)),
             "tool_message_tokens": q(tool_sizes), "user_message_tokens": q(user_sizes),
             "lzma_bits_per_new_token": q(comp)}
    return rows, stats


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text(c.get("text") if isinstance(c, dict) else c) for c in content)
    if isinstance(content, dict):
        return _text(content.get("text") or content.get("content"))
    return str(content)


def _openai_msgs(msgs: list[dict]) -> list[tuple[str, str]]:
    out = []
    for m in msgs:
        role = m.get("role")
        text = _text(m.get("content"))
        if role == "assistant":
            calls = m.get("tool_calls") or []
            text += "".join(json.dumps(c.get("function", c)) for c in calls if isinstance(c, dict))
            out.append(("assistant", text))
        elif role == "system":
            out.append(("system", text))
        elif role in ("tool", "function", "ipython"):
            out.append(("tool", text))
        else:
            out.append(("user", text))
    return out


def _parquet_sample(repo: str, columns: list[str], frac: float, seed: int):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    rng = np.random.default_rng(seed)
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    for f in files:
        pf = pq.ParquetFile(hf_hub_download(repo, f, repo_type="dataset"))
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=columns)
            keep = np.nonzero(rng.random(t.num_rows) < frac)[0]
            if len(keep):
                yield from t.take(keep).to_pylist()


def swe_agent(frac: float):
    sessions, prefix = [], []
    for r in _parquet_sample("nebius/SWE-agent-trajectories", ["trajectory"], frac, 1):
        s = []
        for m in r["trajectory"]:
            role = m.get("role")
            if role == "system":
                s.append(("system", m.get("system_prompt") or m.get("text") or ""))
            elif role in ("ai", "assistant"):
                s.append(("assistant", m.get("text") or ""))
            else:
                s.append(("tool" if s and s[-1][0] == "assistant" else "user", m.get("text") or ""))
        sessions.append(s)
        prefix.append(s[0][1] if s and s[0][0] == "system" else "")
    return sessions, prefix


def openhands(frac: float):
    sessions, prefix = [], []
    for r in _parquet_sample("nebius/SWE-rebench-openhands-trajectories", ["trajectory", "tools"], frac, 2):
        msgs = json.loads(r["trajectory"]) if isinstance(r["trajectory"], str) else r["trajectory"]
        tools = r["tools"] if isinstance(r["tools"], str) else json.dumps(r["tools"])
        s = _openai_msgs(msgs)
        if s and s[0][0] == "system":
            s[0] = ("system", s[0][1] + tools)
            prefix.append(s[0][1])
        else:
            s.insert(0, ("system", tools))
            prefix.append(tools)
        sessions.append(s)
    return sessions, prefix


def tau_bench():
    sessions, prefix, group = [], [], []
    for f in ["gpt-4o-airline", "gpt-4o-retail", "sonnet-35-new-airline", "sonnet-35-new-retail"]:
        for r in json.loads(fetch(TAU + f + ".json").read_text()):
            s = _openai_msgs(r["traj"])
            sessions.append(s)
            prefix.append(s[0][1] if s and s[0][0] == "system" else "")
            group.append(f.split("-")[-1])
    return sessions, prefix, group


# ------------------------------------------------------------------------------------------------------- single requests

def single(ctx, out, reg=None) -> dict:
    ctx = np.asarray(ctx, dtype=np.int64)
    reg = np.zeros_like(ctx) if reg is None else np.minimum(np.asarray(reg, dtype=np.int64), ctx)
    n = len(ctx)
    return {"sess": np.arange(n), "turn": np.zeros(n, dtype=np.int64), "ctx": ctx, "reg": reg, "new": ctx - reg,
            "out": np.asarray(out, dtype=np.int64)}


def ragbench():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    files = [f for f in list_repo_files("galileo-ai/ragbench", repo_type="dataset") if f.endswith(".parquet")
             and "/test" in f and f.split("/")[0] in RAGBENCH_CONFIGS]
    template = ("Use the following documents to answer the question. If the documents do not contain the answer, say "
                "so.\n\nDocuments:\n")
    texts, outs, per = [], [], Counter()
    for f in sorted(files):
        t = pq.read_table(hf_hub_download("galileo-ai/ragbench", f, repo_type="dataset"),
                          columns=["question", "documents", "response"]).to_pylist()
        for r in t:
            docs = "\n\n".join(f"Document {i + 1}: {d}" for i, d in enumerate(r["documents"] or []))
            texts.append(template + docs + "\n\nQuestion: " + (r["question"] or ""))
            outs.append(r["response"] or "")
            per[f.split("/")[0]] += 1
    ctx = ntok(texts) + 2 * MSG_OVERHEAD
    out = ntok(outs) + MSG_OVERHEAD
    reg = np.full(len(ctx), int(ntok([template])[0]) + MSG_OVERHEAD)
    return single(ctx, out, reg), {"per_config": dict(per)}


def burstgpt():
    import pyarrow as pa
    import pyarrow.parquet as pq
    # the hub's parquet conversion concatenates the release's CSVs (with and without failed requests); failed requests
    # (no response tokens) are dropped below, so the duplication weights every kept request equally
    base = "https://huggingface.co/api/datasets/lzzmm/BurstGPT/parquet/default/train/"
    cols = ["Request tokens", "Response tokens", "Log Type"]
    d = pa.concat_tables([pq.read_table(fetch(base + f"{i}.parquet"), columns=cols) for i in (0, 1)]).to_pydict()
    req = np.array(d["Request tokens"], dtype=np.int64)
    resp = np.array(d["Response tokens"], dtype=np.int64)
    kind = np.array(d["Log Type"])
    ok = resp > 0
    out = {}
    for name, cls in [("Conversation log", "chat"), ("API log", "api")]:
        m = ok & (kind == name)
        out[cls] = single(req[m], resp[m])
    return out, {"rows": len(req), "failed_dropped": int((~ok).sum())}


def azure2023():
    import csv
    out = {}
    for name, cls in [("conv", "chat"), ("code", "api")]:
        rows = list(csv.DictReader(io.StringIO(fetch(AZURE + f"AzureLLMInferenceTrace_{name}.csv").read_text())))
        out[cls] = single([int(r["ContextTokens"]) for r in rows], [int(r["GeneratedTokens"]) for r in rows])
    return out


def _servegen_mix(category: str, model: str, chunks: int, per_chunk: int, rng) -> dict:
    """Requests drawn from each client's input and output length distributions (independently: the release has the two
    marginals, not their joint), each client weighted by its mean request rate."""
    ctx, out = [], []
    rates = []
    for c in range(chunks):
        try:
            ds = json.loads(fetch(SERVEGEN + f"{category}/{model}/chunk-{c}-dataset.json").read_text())
            tr = fetch(SERVEGEN + f"{category}/{model}/chunk-{c}-trace.csv").read_text().splitlines()
        except Exception:  # noqa: BLE001 - client ids have gaps
            continue
        dist = {"input_tokens": Counter(), "output_tokens": Counter()}
        for win in ds.values():  # one per 6-hour window, each a dict written as a string, '{}' where the client was idle
            for key, acc in dist.items():
                d = ast.literal_eval(win[key]) if isinstance(win[key], str) else win[key]
                for v, p in d.items():
                    acc[int(v)] += float(p)
        if not dist["input_tokens"] or not dist["output_tokens"]:
            continue
        rates.append(np.mean([float(line.split(",")[1]) for line in tr if line.strip()]))
        ctx.append(dist["input_tokens"])
        out.append(dist["output_tokens"])
    rates = np.array(rates) / np.sum(rates)
    n = rng.multinomial(per_chunk * len(rates), rates)
    cs, os_ = [], []
    for k, ci, co in zip(n, ctx, out):
        if k == 0:
            continue
        vi, pi = np.array(list(ci)), np.array(list(ci.values()), dtype=float)
        vo, po = np.array(list(co)), np.array(list(co.values()), dtype=float)
        cs.append(rng.choice(vi, size=k, p=pi / pi.sum()))
        os_.append(rng.choice(vo, size=k, p=po / po.sum()))
    return single(np.concatenate(cs), np.concatenate(os_))


def servegen_sessions() -> dict:
    data = json.loads(fetch(SERVEGEN + "conversations/conversations_hashed.json").read_text())
    rows = {f: [] for f in units.FIELDS}
    for si, s in enumerate(data):
        prev = None
        for t in sorted(s, key=lambda t: t["turn"]):
            c, o = int(t["input_token_count"]), int(t["output_token_count"])
            if prev is None:
                new = c
            else:
                d = c - prev[0] - prev[1]
                new = d if d >= 0 else max(0, c - prev[0])
            rows["sess"].append(si)
            rows["turn"].append(int(t["turn"]))
            rows["ctx"].append(c)
            rows["reg"].append(0)
            rows["new"].append(min(new, c))
            rows["out"].append(o)
            prev = (c, o)
    return rows


# ------------------------------------------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--frac", type=float, default=0.05, help="fraction of the big agent datasets' rows to sample")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    want = lambda s: not only or s in only
    udir = args.out / "units"
    stats: dict[str, dict] = {}
    rng = np.random.default_rng(0)

    def put(src, cls, rules, note, rows, extra=None):
        units.save(udir / f"{src}.npz", src=src, cls=cls, rules=rules, note=note, **rows)
        stats[src] = {"cls": cls, "rules": rules, "note": note, "calls": len(rows["ctx"]),
                      "sessions": int(len(np.unique(rows["sess"]))), **(extra or {})}
        (args.out / "sources.json").write_text(json.dumps(stats, indent=1))
        print(f"{src}: {stats[src]['calls']} calls, {stats[src]['sessions']} sessions", flush=True)

    def mooncake(name, cls):
        src = f"mooncake-{name}"
        rows = hashtrace.read(fetch(MOONCAKE + f"{name}_trace.jsonl"))
        res = {fm: hashtrace.analyze(rows, fork_min=fm) for fm in (2, 8)}
        stats8 = res[8].pop("stats")
        put(src, cls, ["R1", "R2", "R3"], "Kimi production (one hour), 512-token prefix hashes, capped at 128K",
            res[8], {"fork_min_8": stats8, "fork_min_2": res[2]["stats"]})

    def length_sources(fn, note):
        b, extra = fn() if fn is burstgpt else (fn(), {})
        for cls, rows in b.items():
            put(f"{fn.__name__}-{cls}", cls, ["R1"], note, rows, extra)

    def agent(src, note, fn, *a):
        rows, extra = sessions_to_units(*fn(*a))
        put(src, "agentic", ["R1", "R2", "R3"], note, rows, extra)

    jobs = [
        ("mooncake-conversation", lambda: mooncake("conversation", "chat")),
        ("mooncake-toolagent", lambda: mooncake("toolagent", "agentic")),
        ("mooncake-synthetic", lambda: mooncake("synthetic", "rag")),
        ("servegen-sessions", lambda: put("servegen-sessions", "chat", ["R1", "R3"],
                                          "Bailian multi-turn chat sessions (counts only)", servegen_sessions())),
        ("servegen-r1", lambda: put("servegen-r1", "reasoning", ["R1"],
                                    "Bailian DeepSeek-R1 clients' length distributions, rate-weighted",
                                    _servegen_mix("reason", "deepseek-r1", 200, 400, rng))),
        ("servegen-language", lambda: put("servegen-language", "api", ["R1"],
                                          "Bailian m-large API clients' length distributions, rate-weighted",
                                          _servegen_mix("language", "m-large", 200, 100, rng))),
        ("azure2023", lambda: length_sources(azure2023, "Azure LLM inference trace 2023 (one hour)")),
        ("burstgpt", lambda: length_sources(burstgpt, "BurstGPT, Azure OpenAI GPT service logs")),
        ("ragbench", lambda: put("ragbench", "rag", ["R1", "R2"], "RAGBench test splits: retrieved documents + question",
                                 *ragbench())),
        ("tau-bench", lambda: agent("tau-bench", "tau-bench historical trajectories (no tool schemas)", tau_bench)),
        ("swe-agent", lambda: agent("swe-agent", f"nebius/SWE-agent-trajectories, {args.frac:.0%} sample, full history "
                                    "(no observation collapsing)", swe_agent, args.frac)),
        ("openhands", lambda: agent("openhands", f"nebius/SWE-rebench-openhands-trajectories, {args.frac:.0%} sample, "
                                    "tool schemas in the prefix", openhands, args.frac)),
    ]
    failed = []
    for name, job in jobs:
        if not want(name):
            continue
        try:
            job()
        except Exception as e:  # noqa: BLE001 - one source's failure is reported, the others still run
            failed.append(name)
            stats[f"error:{name}"] = {"error": f"{type(e).__name__}: {e}"}
            (args.out / "sources.json").write_text(json.dumps(stats, indent=1))
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
    if failed:
        print(f"failed sources: {failed}", flush=True)

if __name__ == "__main__":
    main()
