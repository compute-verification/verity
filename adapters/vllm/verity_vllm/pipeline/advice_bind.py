def holdings(adv, io: list[dict]):
    """The certified I/O (``io.json``'s requests, in ingress order) as `derived.Holdings`, and each advice request's
    certified index; or None and why, when the advice's requests aren't the certified ones, each once."""
    users = [r["user"] for r in io]
    ids = [q.id for q in adv.requests]
    if sorted(users) != sorted(ids) or len(set(ids)) != len(ids):
        return None, None, (f"the advice's {len(ids)} requests aren't the {len(users)} certified ones, each once")
    held = D.Holdings(tuple(D.CertifiedRequest(r["user"], tuple(r["prompt"]), r["max_tokens"]) for r in io),
                      tuple(tuple(r["output"]) for r in io))
    return held, tuple(users.index(i) for i in ids), None


def bits(adv, T: int, windows: int, config: dict | None, io: list[dict] | None) -> dict:
    """The advice's bits per egress window of T buckets: v0's, and with ``config`` and an ``io`` it is bound to the
    residual's (the module's docstring); ``io``, the certified I/O, binds the advice to the run, and its tokens are what
    the residual reads and rebuilds the advice from."""
    v0 = I1.bits_by_window(adv, T)
    out = {"T": T, "v0_total": I1.total_bits(adv), "config": config, "bound": None, "public_fields": None,
           "rebuilt": None, "why": None}
    held = order = None
    if io is None:
        out["bound_why"] = "no certified I/O to hold it to: only a varied run's check writes one (io.json)"
    else:
        held, order, out["bound_why"] = holdings(adv, io)
        out["bound"] = held is not None
    dv = ds = None
    if config is None:
        out["why"] = "no registered serving config (--serving-config)"
    elif held is None:
        out["why"] = "the residual reads the certified tokens, and the advice isn't bound to a certified I/O"
    else:
        cfg = D.config_of({"config": config})
        try:
            res = D.residual(adv, cfg, held, order=order)
            shaped = D.residual(adv, cfg, held, order=order, shaped=True)
            walk = D.reconstruct(res, cfg, held)
            dv = D.charge(adv, T, D.derived_records(res, walk, cfg))
            ds = D.charge(adv, T, D.derived_records(shaped, walk, cfg))
        except ValueError as err:
            out["why"], out["public_fields"], out["rebuilt"] = f"the residual under this config: {err}", False, False
        else:
            out["public_fields"] = walk.steps == adv.steps and walk.output_len == tuple(q.output_len for q in adv.requests)
            try:
                out["rebuilt"] = D.rebuild(res, cfg, held, ()) == adv
            except ValueError as err:
                out["why"], out["rebuilt"] = f"the rebuild from the certified I/O: {err}", False

    def total(d, w, fields=None):
        return None if d is None else sum(b for f, b in d.get(w, {}).items() if fields is None or f in fields)

    out["windows"] = [{"window": w, "v0_bits": v0.get(w, 0), "derived_bits": total(dv, w),
                       "derived_shaped_bits": total(ds, w), "forced_bits": total(dv, w, D.FORCED),
                       "in_run": w < windows} for w in sorted(set(range(windows)) | set(v0))]
    return out


# --- the endpoint's own advice ----------------------------------------------------------------------------------------

ENDPOINT = ("advice_trace.jsonl", "requests.jsonl", "windows.jsonl", "serve.json")
SLACK_S = 1.0  # the endpoint's clock against the certifier's: a Commit window opening this soon before the run's origin
COUNTS = ("v0_bits", "derived_bits", "derived_shaped_bits", "forced_bits")
ROW_ORDER = {"runner": "batch", "scheduling": "scheduling"}


def endpoint_config(recs: list[dict], serve: dict, eos: list[int]) -> dict:
    """The serving config `derived.Config` takes, read off the endpoint's records, since it registers none: its scheduler
    trace's header (``recs``, ``advice_trace.jsonl``), ``serve.json``'s ``max_model_len``, the plan's EOS ids, and I1's
    rows in the order the trace's steps record them."""
    head = recs[0]["config"]
    orders = {r.get("order") for r in recs if r["kind"] == "step"}
    if len(orders) != 1 or next(iter(orders)) not in ROW_ORDER:
        raise ValueError(f"the trace's steps record their rows in {sorted(map(str, orders))}, not one of {sorted(ROW_ORDER)}")
    return {"block_size": head["block_size"], "num_gpu_blocks": head["num_gpu_blocks"],
            "max_num_batched_tokens": head["max_num_batched_tokens"],
            "max_model_len": serve["engine"]["engine_kwargs"]["max_model_len"], "eos_token_id": eos[0],
            "stop_token_ids": list(eos[1:]), "scheduler": "async" if head.get("async_scheduling") else "sync",
            "long_prefill_token_threshold": head.get("long_prefill_token_threshold", 0),
            "max_num_seqs": head["max_num_seqs"], "enable_prefix_caching": head["enable_prefix_caching"],
            "row_order": ROW_ORDER[next(iter(orders))]}


def _first_steps(recs: list[dict]) -> dict[int, float]:
    """Each Commit window's first step's clock (unix seconds): the step with the least index from its open to its
    close, as `verity-vllm advice` cuts the windows (its ``windows_of``) and counts their release from."""
    clock = {r["step"]: r["t"] for r in recs if r["kind"] == "step" and "t" in r}
    edge = {(int(r["window"]), r["event"]): int(r["before_step"]) for r in recs if r["kind"] == "window"}
    out = {}
    for (w, event), lo in edge.items():
        hi = edge.get((w, "close"))
        steps = [s for s in clock if s >= lo and (hi is None or s < hi)] if event == "open" else []
        if steps:
            out[w] = clock[min(steps)]
    return out


def served_advice(ep: Path, adv_dir: Path, deployment: dict, io: list[dict], T: int, eos: list[int]) -> tuple[dict, dict]:
    """The endpoint's own advice for the honest run, as `advice` and `bits` report one: ``adv_dir`` is `verity-vllm
    advice`'s over the endpoint's records ``ep`` (`ENDPOINT`), one window advice (`window_advice`) a closed Commit
    window. The run's windows are those whose first step falls in its execution phases (``deployment``'s origin and span,
    `SLACK_S`). A window that carries requests in joins the window before it into a chain (``chain``), joined into one
    I1 v0 advice (`window_chain.join`): its carried requests and their state are the earlier window's, as that window's
    registered advice closed them, never its own string's, so each request and step is charged once. The advice is
    bound to the run when each window's requests are certified ones, every certified request is admitted in exactly one
    window, each window's opening opens its registered root and its registration names the window's Commit
    (``windows.jsonl``), and each chain joins: a window that carries a request in follows the window before it, which
    the run holds, and no window is left owing a carry. Each chain's advice is rebuilt from its certified I/O under the
    config read off the endpoint (`endpoint_config`) by `bits`, charged per bucket (T 1) and placed at its first
    window's first step's bucket, each later window's release counted from its own first step's, so its bits land in
    the execution phases of T buckets I1's rule puts them in, its release at its own width; a chain's bits are on its first
    window's row."""
    recs = _jsonl(ep / "advice_trace.jsonl")
    serve = json.loads((ep / "serve.json").read_text())
    index = json.loads((adv_dir / "windows.json").read_text())
    commits = {int(d["window"]): {"run_id": d.get("run_id"), "run_root": d.get("run_root")}
               for d in _jsonl(ep / "windows.jsonl")}
    first = _first_steps(recs)
    origin, span = deployment["origin_unix"], deployment["windows"] * deployment["window_s"]
    bucket_s = deployment["window_s"] / T
    users = [r["user"] for r in io]
    certified = set(users)
    out = {"T": T, "served": serve.get("run_id"), "config": None, "config_from": "the endpoint's records (its trace's "
           "header, serve.json's max_model_len, the plan's EOS ids), not one it registered", "bound": False,
           "bound_why": None, "public_fields": None, "rebuilt": None, "why": None, "commit_windows": [], "windows": []}
    try:
        out["config"] = config = endpoint_config(recs, serve, eos)
    except (KeyError, ValueError) as err:
        out["why"], config = f"no serving config off the endpoint's records: {err!r}", None
    problems, rows, links, seen, loaded = [], [], {}, {u: 0 for u in users}, []

    def place(charged: dict[int, dict], at: int) -> None:
        for x in charged:
            lw = links.setdefault((x["window"] + at) // T, dict.fromkeys(COUNTS, 0))
            for k in COUNTS:
                lw[k] = None if lw[k] is None or x.get(k) is None else lw[k] + x[k]

    for w in sorted(w for w, t in first.items() if origin - SLACK_S <= t < origin + span):
        if str(w) not in index["windows"]:
            why = (index.get("refused") or {}).get(str(w)) or "still open"
            problems.append(f"Commit window {w} has no advice ({why})")
            rows.append({"window": w, "refused": why})
            continue
        d = adv_dir / f"w{w}"
        reg = json.loads((d / "registration.json").read_text())
        adv = WA.from_json(json.loads((d / "private" / "advice.json").read_text()))
        ids = [q.id for q in adv.requests]
        for i in ids[len(adv.carried):]:
            if i in seen:
                seen[i] += 1
        foreign = [i for i in ids if i not in certified]
        opened = opens(reg, json.loads((d / "private" / "opening.json").read_text()), adv)
        at = math.floor((first[w] - origin) / bucket_s)
        row = {"window": w, "commit": reg.get("commit"), "commit_is_the_windows": reg.get("commit") == commits.get(w),
               "root": reg["binding"]["advice"]["root"], "opened": opened, "requests": len(ids) - len(adv.carried),
               "carried": len(adv.carried), "steps": len(adv.steps), "outputs": len(adv.release),
               "committed_bits": WA.total_bits(adv), "t0_s": round(first[w] - origin, 3), "bucket": at,
               "link_window": at // T, "chain": [w], "bound": None, "public_fields": None, "rebuilt": None, "why": None,
               **dict.fromkeys(COUNTS, 0)}
        rows.append(row)
        if adv.carried and loaded and loaded[-1][-1][0] == w - 1:
            loaded[-1].append((w, adv, row))
        else:
            loaded.append([(w, adv, row)])
        if foreign:
            problems.append(f"Commit window {w} holds {len(foreign)} requests that aren't the run's certified ones "
                            f"({', '.join(foreign[:3])})")
        if not opened:
            problems.append(f"Commit window {w}'s opening doesn't open its registered root to its advice")
        if not row["commit_is_the_windows"]:
            problems.append(f"Commit window {w}'s registration names another Commit than the endpoint's windows.jsonl")
    for chained in loaded:
        at, ws, last = chained[0][2]["bucket"], [w for w, _, _ in chained], {}

        def released(k: int, rid: str, _: int, b: int, chained=chained, at=at, last=last) -> int:
            last[rid] = max(last.get(rid, 0), b + chained[k][2]["bucket"] - at)
            return last[rid]

        try:
            adv = WC.join([(w, a) for w, a, _ in chained], released, WA)
        except I1.MalformedAdvice as err:
            problems.append(str(err))
            for w, a, row in chained:
                place([{"window": b, "v0_bits": n} for b, n in WA.bits_by_window(a, 1).items()], row["bucket"])
                row.update(chain=ws, bound=False, why=str(err), v0_bits=row["committed_bits"],
                           **dict.fromkeys(COUNTS[1:]))
            continue
        b = bits(adv, 1, 0, config, [r for r in io if r["user"] in {q.id for q in adv.requests}])
        place(b["windows"], at)
        for _, _, row in chained:
            row.update(chain=ws, **{k: b[k] for k in ("bound", "public_fields", "rebuilt", "why")})
        chained[0][2].update({k: sum(x[k] for x in b["windows"]) if all(x[k] is not None for x in b["windows"])
                              else None for k in COUNTS})
    missing, twice = [u for u, n in seen.items() if n == 0], [u for u, n in seen.items() if n > 1]
    if not rows:
        problems.append("no Commit window of the endpoint's opened in the run's execution phases")
    if missing:
        problems.append(f"{len(missing)} certified requests are in none of the run's Commit windows ({', '.join(missing[:3])})")
    if twice:
        problems.append(f"{len(twice)} certified requests are in more than one ({', '.join(twice[:3])})")
    ran = [r for r in rows if "refused" not in r]
    out["bound"] = not problems and all(r["bound"] is True for r in ran)
    out["bound_why"] = "; ".join(problems) or next((f"Commit window {r['window']}: {r['why']}" for r in ran
                                                    if r["bound"] is not True), None)
    out["public_fields"] = bool(ran) and all(r["public_fields"] is True for r in ran)
    out["rebuilt"] = bool(ran) and all(r["rebuilt"] is True for r in ran)
    out["why"] = out["why"] or next((f"Commit window {r['window']}: {r['why']}" for r in ran if r["why"]), None)
    out["commit_windows"] = rows
    out["v0_total"] = sum(r["v0_bits"] for r in ran)
    out["chains"] = [[w for w, _, _ in c] for c in loaded if len(c) > 1]
    n = deployment["windows"]
    out["windows"] = [{"window": w, **links.get(w, dict.fromkeys(COUNTS, 0)), "in_run": w < n}
                      for w in sorted(set(range(n)) | set(links))]
    info = {"file": str(adv_dir), "form": "the endpoint's own (verity-vllm advice): window advice, one a Commit "
            "window, a chain of windows that carry requests across their close joined into one I1 v0 advice",
            "served": serve.get("run_id"), "windows": len(ran), "bits": out["v0_total"],
            "requests": sum(r["requests"] for r in ran), "steps": sum(r["steps"] for r in ran),
            "outputs": sum(r["outputs"] for r in ran), "file_sha512": None, "v0_sha512": None}
    return info, out
