"""The TP Commit's injected negatives: `--serve-shard-from`'s specs parsed against the world, and, after the run, whether each requested
negative was injected on its target rank (`tp-commit` reports one that was not as an error run)."""
from __future__ import annotations


def negatives_applied(flip_partial: str | None, swap_requests: str | None, omit_partial_site: str | None, xrank_pairs: list[dict], made: list[dict]) -> dict[str, dict]:
    """Per requested negative: did the target rank actually inject it?  flip_partial -> the rank's flipped_collective_inputs over the pairs (from
    the cross-rank dumps; None when the pass did not run); swap_requests -> the rank's swap_requests_applied (committer make); omit_partial_site
    -> the rank's tp_partial_source.omit_site.  `applied` is True / False / None(unknown); anything but True is 'NOT APPLIED' (an error run)."""
    applied: dict[str, dict] = {}
    if flip_partial:
        tgt = str(flip_partial.split(":", 1)[0])
        fl = [d_.get("flipped_collective_inputs") for xr_ in xrank_pairs for d_ in (xr_.get("dumps") or []) if str(d_.get("rank")) == tgt]
        applied["flip_partial"] = {"rank": tgt, "flipped_collective_inputs_per_pair": fl, "applied": (any(int(f_ or 0) > 0 for f_ in fl) if fl else None)}
    if swap_requests:
        tgt = str(int(swap_requests.split(":")[0]))
        sw = [m_.get("swap_requests_applied") for m_ in made if str(m_.get("rank")) == tgt]
        applied["swap_requests"] = {"rank": tgt, "swap_requests_applied": sw, "applied": (any(bool(x_) for x_ in sw) if sw else None)}
    if omit_partial_site:
        tgt = str(omit_partial_site.split(":", 1)[0])
        om = [((m_.get("tp_partial_source") or {}).get("omit_site")) for m_ in made if str(m_.get("rank")) == tgt]
        applied["omit_partial_site"] = {"rank": tgt, "omit_site_per_rank": om, "applied": (any(bool(x_) for x_ in om) if om else None)}
    return applied


def parse_serve_shard_from(specs: list[str] | None, world: int) -> dict[str, list[dict]] | None:
    """NEGATIVE: `<rank>:<served param name>:<from_rank>` specs -> {rank: [{"param", "from_rank"}]}; `0:X:1 1:X:0` = the two
    ranks swap their shards of X.  from_rank must be ANOTHER rank of the world (a rank serving its own shard is the positive, refused here)."""
    if not specs:
        return None
    out: dict[str, list[dict]] = {}
    for spec in specs:
        parts_ = str(spec).split(":")
        if len(parts_) != 3 or not parts_[1]:
            raise SystemExit(f"--serve-shard-from <rank>:<served param name>:<from_rank> (got {spec!r})")
        rank, frm = int(parts_[0]), int(parts_[2])
        if not (0 <= rank < world) or not (0 <= frm < world) or frm == rank:
            raise SystemExit(f"--serve-shard-from {spec!r}: rank {rank} must serve ANOTHER rank's shard of a world of {world}")
        out.setdefault(str(rank), []).append({"param": parts_[1], "from_rank": frm})
    return out
