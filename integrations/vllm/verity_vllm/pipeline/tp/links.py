"""`tp_links.json`: how a TP rank's Program is bound to the other ranks' (written by the Build, read by the Match and the fold Match),
and the module-path vocabulary its links are named in."""
from __future__ import annotations

import json
import os


def module_in_model_vocabulary(module_path: str | None, model_root: str | None) -> str | None:
    """A derive module path (`<model_root>.<path> (<Class>)`, the traced wrapper's `named_modules()` spelling) in the
    recorder's vocabulary (`model.named_modules()`): the class suffix dropped and the RECORDED model root removed.  `model_root` is data
    the derive wrote (by object identity), never a spelled prefix; without it (a v1 file) the path is returned as spelled, class dropped,
    and consumers compare by `same_module_path` (dotted-suffix identity)."""
    if not isinstance(module_path, str):
        return None
    mp = module_path.split(" (", 1)[0].strip()
    if model_root:
        if mp == model_root:
            return ""
        head = model_root + "."
        if mp.startswith(head):
            return mp[len(head):]
    return mp


def same_module_path(a: str | None, b: str | None) -> bool:
    """Two dotted module paths name the same module when equal, or when one is the other's dotted suffix (the same
    `named_modules()` path read from a wrapper's root vs the model's root).  Structural: no vocabulary prefix is spelled."""
    if a is None or b is None:
        return False
    a, b = a.split(" (", 1)[0].strip(), b.split(" (", 1)[0].strip()
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def write_tp_links(out: str, res: dict, program, rep, tp: int, rank: int, model_root: str | None = None) -> dict:
    """`tp_links.json`: how this rank's Program is bound to the other rank's.  Peer input k (the (k+1)-th user Input after the
    prompt) is NOT a fresh trusted input: it is bound to collective k of rank 1-r — the value that rank's Program computes as its own
    partial (`part_rank{1-r}` of ITS k-th `all_reduce2`/`all_gather2` call), a committed value of that rank.  The Match of rank r binds
    peer input k from rank (1-r)'s committed log and checks its received collective output against `AllReduce2(part_r, part_{1-r})`."""
    params = [(n, str(t)) for n, t in program.fn.body.params]
    user = [n for n, _ in params if n != "weights"]
    probe = res["tp"]["collectives"]["record"]
    peer_ranks = [r for r in range(tp) if r != rank]                       # [lane vllm-tp-n] world-1 peers per collective, in rank order
    n_peer = len(probe) * len(peer_ranks)
    committed_units = list(rep.provenance.get("committed_units") or ())
    n_committed = len(committed_units)
    if n_peer + n_committed > len(user):
        raise RuntimeError(f"tp_links: {n_peer} peer inputs + {n_committed} committed-only inputs vs {len(user)} user inputs")
    peer_stop = len(user) - n_committed
    peer_names = user[peer_stop - n_peer:peer_stop]
    prompt_names = user[:peer_stop - n_peer]
    committed_names = user[peer_stop:]
    expected_committed_names = [str(unit.get("input")) for unit in committed_units]
    if committed_names != expected_committed_names:
        raise RuntimeError(f"tp_links: trailing committed-only inputs {committed_names} != provenance {expected_committed_names}")

    # A listed collective still has the wrapper's peer placeholders, but its output is now a committed-only input and those placeholders
    # are deliberately unused.  Identify its original probe occurrence from the exact peer group in its operand provenance; never infer
    # by count alone.  The remaining rule applications must account for every other probe occurrence in order.
    peer_position = {name: i for i, name in enumerate(peer_names)}
    committed_collectives: dict[int, dict] = {}
    collective_ops = {
        "verity_tp.all_reduce.default": "all_reduce", "verity_tp.all_reduce2.default": "all_reduce",
        "verity_tp.all_gather.default": "all_gather", "verity_tp.all_gather2.default": "all_gather",
    }
    for unit in committed_units:
        kind = collective_ops.get(str(unit.get("op")))
        if kind is None:
            continue
        positions = sorted(peer_position[str(ref.get("node"))] for ref in unit.get("operand_refs") or ()
                           if str(ref.get("node")) in peer_position)
        width = len(peer_ranks)
        if len(positions) != width or positions != list(range((positions[0] // width) * width, (positions[0] // width + 1) * width)):
            raise RuntimeError(f"tp_links: committed collective {unit.get('input')} names peer positions {positions}, expected one group of {width}")
        k = positions[0] // width
        if k in committed_collectives:
            raise RuntimeError(f"tp_links: committed collectives {committed_collectives[k].get('input')} and {unit.get('input')} both claim probe {k}")
        if k >= len(probe) or probe[k]["kind"] != kind:
            got = probe[k]["kind"] if k < len(probe) else None
            raise RuntimeError(f"tp_links: committed collective {unit.get('input')} is {kind} at probe {k}, probe says {got}")
        committed_collectives[k] = unit

    apps = [a for a in rep.provenance["applications"] if a.get("collective")]      # the collective rules tag their applications (kind), in call order
    if len(apps) + len(committed_collectives) != len(probe):
        raise RuntimeError(f"tp_links: {len(apps)} collective rule applications + {len(committed_collectives)} committed collectives "
                           f"vs {len(probe)} probed collectives")
    links = []
    committed_links = []
    app_iter = iter(apps)
    for k, p in enumerate(probe):
        peers = [{"rank": pr, "input": peer_names[k * len(peer_ranks) + j], "operand": f"part_rank{pr}", "collective_k": k} for j, pr in enumerate(peer_ranks)]
        if k in committed_collectives:
            unit = committed_collectives[k]
            committed_links.append({"k": k, "kind": p["kind"], "shape": p["shape"], "dtype": p["dtype"], "peer_collective_k": k,
                                    "peers": peers, "input": unit["input"], "op": unit["op"], "module_path": unit.get("module"),
                                    "module": module_in_model_vocabulary(unit.get("module"), model_root)})
            continue
        a = next(app_iter)
        if a.get("collective") != p["kind"]:
            raise RuntimeError(f"tp_links: collective rule application {a.get('collective')} vs probe {k} kind {p['kind']}")
        ln = {"k": k, "kind": p["kind"], "shape": p["shape"], "dtype": p["dtype"], "peer_collective_k": k, "own_operand": f"part_rank{rank}",
              "peers": peers, "node": a["node"], "module_path": a.get("module_path"), "emitted": a.get("emitted"),
              "module": module_in_model_vocabulary(a.get("module_path"), model_root)}      # recorder vocabulary (`model.named_modules()`)
        if tp == 2:   # the v1 fields, kept verbatim for the two-rank readers (one peer)
            ln.update({"peer_input": peers[0]["input"], "peer_rank": peers[0]["rank"], "peer_operand": peers[0]["operand"]})
        links.append(ln)
    doc = {"schema": "cb-a/tp_links/v2", "world": tp, "rank": rank, "peer_ranks": peer_ranks, "program_digest": res["program"]["digest"], "user_inputs": user,
           "model_root": model_root,
           "module_vocabulary": "link.module_path: the traced wrapper's named_modules() path + ' (Class)'; link.module: the same module in the "
                                "recorder's model.named_modules() vocabulary = module_path minus model_root (recorded by object identity)",
           "prompt_inputs": prompt_names, "peer_inputs": peer_names,
           "peer_input_layout": "peer input k*(world-1)+j of rank r := the j-th other rank's (rank order) partial of collective k",
           "binding": "each peer input of collective k of rank r := part_rank(p) of collective k of rank p (that rank's committed partial); the received "
                      "collective output of rank r is AllReduce/AllGather(part_rank0, .., part_rank{world-1}) in rank order — never a fresh trusted input"
                      + (" (world 2: AllReduce2/AllGather2, the recorded identities)" if tp == 2 else ""),
           "weight_shards": "the `weights` record of this Program holds THIS rank's shards (shapes in result.tp.sharded_params); the rank's "
                            "weight root commits to the shard, the run root binds every rank's weight root (rank-tagged)",
           "links": links}
    if committed_links:
        unused = [peer["input"] for link in committed_links for peer in link["peers"]]
        doc.update({"unverified_inputs": committed_names, "unused_peer_inputs": unused, "committed_collectives": committed_links,
                    "committed_collective_policy": "the listed collective output is the named unverified committed input; its original peer "
                                                   "placeholders remain declared but are unused, and no TP link is asserted for this occurrence"})
    json.dump(doc, open(os.path.join(out, "tp_links.json"), "w"), indent=1)
    summary = {"n_peer_inputs": len(peer_names), "n_prompt_inputs": len(prompt_names), "file": "tp_links.json"}
    if committed_links:
        summary.update({"n_unverified_inputs": len(committed_names), "n_committed_collectives": len(committed_links),
                        "n_unused_peer_inputs": len(doc["unused_peer_inputs"])})
    return summary


def read_tp_links(path: str) -> dict:
    """Load a `tp_links.json` of schema v1 (world 2, one `peer_input`/`peer_rank` per link) or v2 (`peers` per link), returning
    the v2 shape: every link carries `peers = [{rank, input, operand, collective_k}]` and the doc `peer_ranks`.  A v1 file is upgraded in
    memory (world 2: the single peer is rank 1-r); nothing is rewritten on disk."""
    doc = json.load(open(path))
    world, rank = int(doc.get("world") or 2), int(doc.get("rank") or 0)
    doc.setdefault("peer_ranks", [r for r in range(world) if r != rank])
    root = doc.get("model_root")
    for ln in doc.get("links") or []:
        if "module" not in ln:    # v1 / early v2: no recorded model root -> the path as spelled (class dropped); readers use same_module_path
            ln["module"] = module_in_model_vocabulary(ln.get("module_path"), root)
        if "peers" not in ln:
            ln["peers"] = [{"rank": int(ln.get("peer_rank", 1 - rank)), "input": ln.get("peer_input"), "operand": ln.get("peer_operand", f"part_rank{1 - rank}"),
                            "collective_k": int(ln.get("peer_collective_k", ln.get("k", 0)))}]
    return doc


__all__ = ["module_in_model_vocabulary", "read_tp_links", "same_module_path", "write_tp_links"]
