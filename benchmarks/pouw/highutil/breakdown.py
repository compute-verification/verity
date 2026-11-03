"""Where each period's time goes, by resource, from a HIGHUTIL run's stored outputs (no GPU): what pricing the protocol's
own work at the public best's cost needs, to recompute ε_undecl. For each period:

- `gpu`: the blocks' GPU time, split exclusively: `gemm` (any GEMM kernel running: cuBLASLt's, v31's `gemm_kernel`,
  torch._int_mm's `Kernel2`), `protocol` (only the protocol's own kernels running, an instant shared equally among
  them; by kernel and by role: leaf hashing, salts, tree levels, coins, noise, form, openings) and `idle` (no kernel);
  and each protocol kernel's own busy time and its time beside a GEMM, which the exclusive split doesn't charge it;
- `pipes`: the tensor pipe's active and idle time (Nsight Compute's active share of each kernel, kernels serialized,
  times its busy time; idle is an upper bound where a kernel wasn't profiled); the card's other pipes weren't measured;
- `clock`: the blocks' SM clocks (NVML at each block's end) against the card's ceiling, and nvidia-smi's clocks.gr and
  clocks.sm over the period's window;
- `host`: the period's wall time off the GPU (before its first block, after its last, and any time the GPU waited
  between blocks for the host), and outside the period's wall the challenger session's opening with the first two
  blocks' seeds and the verifier's checks;
- `work`: each call's counted work a step from its shape (v31.Fill): MACs, leaves and their bytes, salts, tree nodes,
  dither bits and formed operand bytes, which the public best's rates price.

The kernel times are the Nsight Systems profile's (run.sh's NSYS=1: one period of each arm, `--profile-steps` steps) per
step, scaled so the split sums to the period's own GPU time (`scale`: the period's GPU time a step over the profile's
span a step). A plain period's profile is its baseline's last `--profile-steps` steps of GEMMs in nsys-plain. nvidia-smi's
window is aligned on the challenger session's id (its UTC second): a protocol period's blocks start after the session's
opening, and the plain periods before a protocol period end at its session, so a window is good to about a second and is
trimmed a second at each end. The run-level numbers (the declared utilization, the prover overhead and the unaccounted
share's bounds) are summary.json's.

    breakdown.py OUT [--pipes-from DIR] [--profile-steps 64]      prints one JSON object
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import period as P  # noqa: E402

#: the protocol's own kernels by name prefix, each with its role
ROLES = (("leaf_", "leaf hashing (SHA-512 of the hm96 rows)"), ("salt_", "salts (prf_salt, SHA-512)"),
         ("level_kernel", "tree levels (SHA-512)"), ("top_", "tree levels (SHA-512)"),
         ("bits_kernel", "coins: the dither bits d_A, d_B"), ("perm_kernel", "coins: the permutation pi"),
         ("noise_kernel", "noise: U, V"), ("fill_", "form: X and W' (A + d_A + V, B + d_B + U[pi])"),
         ("form_", "form"), ("wrap", "form"), ("serve_rows_kernel", "form: a served call's A + d_A and B + d_B (O's operands)"),
         ("open_kernel", "openings"), ("pack_kernel", "openings"),
         ("forge_kernel", "a cheat (the negative period's)"))
#: the GEMMs: v31's own, and every kernel that isn't the protocol's (cuBLASLt's sm90_xmma, torch._int_mm's Kernel2); V31_WG's
#: gemm_r4 and gemm_r5 hash a served call's finished rows on warps of their own, so that hashing counts in the GEMM's time
GEMM_OWN = "gemm_kernel"
BASE_KERNELS = {"lt": ("sm90_xmma", "sm100_xmma", "sm120_xmma", "cutlass"), "int_mm": ("Kernel2",)}
PIPES = ("issue", "alu", "imad", "shfl", "f64")


def role(name: str) -> str | None:
    """A kernel's role, or None for a GEMM."""
    if name.startswith(GEMM_OWN):
        return None
    return next((r for p, r in ROLES if name.startswith(p)), None)


def short(name: str) -> str:
    """Nsight Compute's demangled name as Nsight Systems' short name: `void leaf_persist<LtRow, 4>(...)` -> leaf_persist,
    `void cutlass::Kernel2<...>(...)` -> Kernel2."""
    return re.split(r"[<(]", name.removeprefix("void "), maxsplit=1)[0].strip().rsplit("::", 1)[-1]


def kernels(db: Path) -> list[tuple[int, int, str]]:
    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
                           "ON k.shortName = s.id ORDER BY k.start").fetchall()
    finally:
        con.close()


def proto_window(rows: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """The protocol period's kernels: from its first own kernel to its last (the warm-up block before it is a plain one)."""
    own = [(s, e) for s, e, n in rows if role(n) or n.startswith(GEMM_OWN)]
    if not own:
        return []
    lo, hi = own[0][0], max(e for _, e in own)
    return [r for r in rows if r[0] >= lo and r[1] <= hi]


def plain_window(rows: list[tuple[int, int, str]], base: str, launches: int) -> list[tuple[int, int, str]]:
    """A plain period's kernels: its baseline's last `launches` GEMMs and whatever ran among them."""
    mine = [r for r in rows if r[2].startswith(BASE_KERNELS[base])][-launches:]
    if not mine:
        return []
    lo, hi = mine[0][0], max(e for _, e, _ in mine)
    return [r for r in rows if r[0] >= lo and r[1] <= hi]


def split(rows: list[tuple[int, int, str]]) -> dict:
    """The window's span and its exclusive split (ns): GEMM, each protocol kernel (an instant shared equally among the
    protocol kernels running), idle; each kernel's busy time and launches, and each protocol kernel's time beside a GEMM."""
    if not rows:
        return {"span": 0, "gemm": 0, "idle": 0, "excl": {}, "busy": {}, "launches": {}, "beside_gemm": {}}
    ev = sorted([(s, 1, n) for s, _, n in rows] + [(e, -1, n) for _, e, n in rows], key=lambda x: (x[0], x[1]))
    active: dict[str, int] = defaultdict(int)
    excl, beside = defaultdict(float), defaultdict(float)
    gemm = idle = 0.0
    t0, t1 = rows[0][0], max(e for _, e, _ in rows)
    prev = t0
    for t, d, n in ev:
        dt = t - prev
        if dt > 0:
            on = [k for k, c in active.items() if c > 0]
            prot = [k for k in on if role(k)]
            if not on:
                idle += dt
            elif len(prot) < len(on):
                gemm += dt
                for k in prot:
                    beside[k] += dt
            else:
                for k in prot:
                    excl[k] += dt / len(prot)
        active[n] += d
        prev = t
    busy, launches = defaultdict(int), defaultdict(int)
    for s, e, n in rows:
        busy[n] += e - s
        launches[n] += 1
    return {"span": t1 - t0, "gemm": gemm, "idle": idle, "excl": dict(excl), "busy": dict(busy), "launches": dict(launches),
            "beside_gemm": dict(beside)}


def tensor_active(pipes: dict | None) -> dict[str, float]:
    """Each kernel's tensor-pipe active share (Nsight Compute), by short name, time-weighted over its variants."""
    if not pipes:
        return {}
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for k in pipes["kernels"]:
        a = acc[short(k["name"])]
        a[0] += k["tensor_active"] * k["s"]
        a[1] += k["s"]
    return {n: a / s for n, (a, s) in acc.items() if s}


def active_share(name: str, shares: dict[str, float]) -> tuple[float | None, str]:
    """A kernel's tensor-pipe active share: its own, else (a cuBLASLt variant the profile didn't catch) its family's
    time-weighted mean, else None."""
    if name in shares:
        return shares[name], "kernel"
    fam = next((p for ps in BASE_KERNELS.values() for p in ps if name.startswith(p)), None)
    if fam:
        got = [v for n, v in shares.items() if n.startswith(fam)]
        if got:
            return statistics.fmean(got), f"{fam} family"
    return None, "not measured"


def smi_samples(path: Path) -> list[tuple[datetime, dict[str, float]]]:
    if not path.exists():
        return []
    rows = list(csv.reader(path.read_text().splitlines()))
    if not rows:
        return []
    head = [h.strip() for h in rows[0]]
    col = {k: next((i for i, h in enumerate(head) if h.startswith(p)), None)
           for k, p in (("gr", "clocks.current.graphics"), ("sm", "clocks.current.sm"), ("util", "utilization.gpu"))}
    out = []
    for r in rows[1:]:
        if len(r) != len(head) or not re.fullmatch(r"\d{4}/\d\d/\d\d \d\d:\d\d:\d\d\.\d+", r[0].strip()):
            continue
        vals = {k: re.match(r"\s*(\d+(?:\.\d+)?)\b", r[i]) for k, i in col.items() if i is not None}
        if all(vals.values()):   # a sample nvidia-smi couldn't read ([N/A]) is left out
            ts = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)
            out.append((ts, {k: float(m.group(1)) for k, m in vals.items()}))
    return out


def smi_window(samples, lo: datetime, hi: datetime) -> dict | None:
    lo, hi = lo + timedelta(seconds=1), hi - timedelta(seconds=1)
    got = [v for t, v in samples if lo <= t <= hi]
    if not got:
        return None
    med = lambda k: statistics.median(v[k] for v in got) if all(k in v for v in got) else None  # noqa: E731
    return {"from": lo.isoformat(), "to": hi.isoformat(), "samples": len(got), "clocks_gr_mhz": med("gr"),
            "clocks_sm_mhz": med("sm"), "utilization_gpu_pct": med("util")}


def session_time(r: dict) -> datetime | None:
    m = re.match(r"s(\d{8}T\d{6})Z", ((r.get("coins") or {}).get("session")) or "")
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC) if m else None


def windows(results: list[dict]) -> list[tuple[datetime, datetime] | None]:
    """Each period's wall-clock window (see the module docstring), or None where nothing pins it."""
    out: list[tuple[datetime, datetime] | None] = [None] * len(results)
    for i, r in enumerate(results):
        if r["arm"] != "proto" or not (t := session_time(r)):
            continue
        start = t + timedelta(seconds=r["session_open_s"])
        out[i] = (start, start + timedelta(seconds=r["wall_s"]))
        end, j = t, i - 1
        while j >= 0 and results[j]["arm"] == "plain":
            out[j] = (end - timedelta(seconds=results[j]["wall_s"]), end)
            end, j = out[j][0], j - 1
    return out


def work(prep: dict) -> dict:
    """Each call's counted work a step, from its shape."""
    P.configure(prep)
    calls = []
    for name, n, k in P.LAYERS:
        f = P.shape(n, k, prep["row"])
        depth = (f.leaves - 1).bit_length() if f.leaves > 1 else 0
        calls.append({"call": name, "m": P.M, "n": n, "k": k, "lp": f.lp, "L": f.L, "S": f.S, "row_bytes": f.row,
                      "macs": P.M * n * k, "leaves": f.leaves, "leaf_bytes": f.leaves * f.row, "salts": f.leaves,
                      "tree_nodes": (1 << depth) - 1, "committed_words": P.M * n * f.W,
                      "committed_o": P.M * n if f.o else 0,
                      "dither_bits": (P.M + n) * k, "formed_bytes": (P.M + n) * k})
    total = {key: sum(c[key] for c in calls) for key in ("macs", "leaves", "leaf_bytes", "salts", "tree_nodes",
                                                         "committed_words", "committed_o", "dither_bits", "formed_bytes")}
    return {"calls": calls, "step": total}


def period_breakdown(r: dict, prof: dict, prof_steps: int, shares: dict[str, float], card_mhz: float, win, samples) -> dict:
    gpu_s = sum(b["gpu_ms"] for b in r["blocks"]) / 1e3
    span = prof["span"]
    scale = gpu_s / (span * 1e-9 * r["steps"] / prof_steps) if span else None
    to_s = lambda ns: ns * 1e-9 * r["steps"] / prof_steps * scale if scale else None  # noqa: E731
    by_kernel, by_role = [], defaultdict(lambda: {"exclusive_s": 0.0, "busy_s": 0.0})
    tensor_s, unmeasured = 0.0, []
    for name in sorted(prof["busy"], key=lambda n: -prof["busy"][n]):
        ro = role(name)
        share, src = active_share(name, shares)
        busy = to_s(prof["busy"][name])
        if share is None:
            unmeasured.append({"kernel": name, "busy_s": busy})
        elif busy is not None:
            tensor_s += share * busy
        row = {"kernel": name, "role": ro or "GEMM", "launches_per_step": prof["launches"][name] / prof_steps,
               "busy_s": busy, "tensor_active": share, "tensor_active_from": src}
        if ro:
            row.update(exclusive_s=to_s(prof["excl"].get(name, 0.0)), beside_gemm_s=to_s(prof["beside_gemm"].get(name, 0.0)))
            by_role[ro]["exclusive_s"] += row["exclusive_s"] or 0.0
            by_role[ro]["busy_s"] += busy or 0.0
        by_kernel.append(row)
    protocol_s = sum(v["exclusive_s"] for v in by_role.values())
    mhz = [b["sm_mhz"] for b in r["blocks"] if b.get("sm_mhz")]
    out = {"period": r["period"], "arm": r["arm"], "baseline": r.get("baseline", P.OLD_BASELINE) if r["arm"] == "plain" else None,
           "cheat": r.get("cheat"), "accepted": r.get("accepted"), "steps": r["steps"], "macs": r["macs"],
           "credited_macs": r.get("credited_macs", 0), "wall_s": r["wall_s"],
           "gpu": {"s": gpu_s, "gemm_s": to_s(prof["gemm"]), "protocol_s": protocol_s, "idle_s": to_s(prof["idle"]),
                   "protocol_by_role": dict(by_role), "by_kernel": by_kernel, "scale": scale},
           "pipes": {"tensor": {"active_s": tensor_s, "idle_in_gpu_s": gpu_s - tensor_s, "idle_in_wall_s": r["wall_s"] - tensor_s,
                                "bound": "at most (kernels Nsight Compute didn't profile count as idle)" if unmeasured else "exact",
                                "kernels_not_measured": unmeasured} if shares else "not measured (no Nsight Compute run)",
                     **{p: "not measured (Nsight Compute collected the tensor pipe's metrics only)" for p in PIPES}},
           "clock": {"card_mhz": card_mhz, "blocks_sm_mhz": mhz,
                     "below_ceiling": 1 - statistics.median(mhz) / card_mhz if mhz else None,
                     "smi": smi_window(samples, *win) if win else None},
           "host": {"between_blocks_s": r["wall_s"] - gpu_s}}
    if r["arm"] == "proto":
        out["host"].update(gaps_s=r.get("gaps_s"), outside_the_wall={"session_open_s": r.get("session_open_s"),
                                                                     "seeds_ahead_s": r.get("seeds_ahead_s"),
                                                                     "verify_s": r.get("verify_s")})
    return out


def breakdown(out: Path, pipes_from: Path | None = None, profile_steps: int = 64) -> dict:
    prep = json.loads((out / "prep.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    results = [json.loads(x) for x in (out / "periods.jsonl").read_text().splitlines() if x.strip()]
    card = P.load_card(str(out / "card.toml"))
    card_mhz = card["clock"]["sm"]["value"]
    wl = work(prep)
    pdir = pipes_from or out
    pipes = {arm: json.loads(p.read_text()) for arm in ("plain", "proto") if (p := pdir / f"pipes-{arm}.json").exists()}
    shares = {arm: tensor_active(pipes.get(arm)) for arm in ("plain", "proto")}
    rows = {arm: kernels(out / f"nsys-{arm}.sqlite") for arm in ("plain", "proto") if (out / f"nsys-{arm}.sqlite").exists()}
    profs = {"proto": split(proto_window(rows.get("proto", [])))}
    for base in dict.fromkeys(r.get("baseline", P.OLD_BASELINE) for r in results if r["arm"] == "plain"):
        profs[base] = split(plain_window(rows.get("plain", []), base, profile_steps * len(P.LINEARS)))
    samples = smi_samples(out / "smi.csv")
    periods = [period_breakdown(r, profs["proto"] if r["arm"] == "proto" else profs[r.get("baseline", P.OLD_BASELINE)],
                                profile_steps, shares["proto" if r["arm"] == "proto" else "plain"], card_mhz, w, samples)
               for r, w in zip(results, windows(results))]
    keep = ("declared_utilization", "declared_utilization_at_run_clock", "declared_utilization_without_protocol",
            "prover_overhead", "unaccounted_share", "baseline", "credited_macs", "wall_s", "sm_mhz_median", "credited_ands")
    return {"card": {"sku": card["sku"], "clock_mhz": card_mhz, "theta_macs_s": P.theta(card),
                     "rates": {k: v["value"] for k, v in card["rate"].items()}},
            "summary": {k: summary.get(k) for k in keep},
            "baselines": {b: {k: v.get(k) for k in ("prover_overhead", "declared_utilization_without_protocol", "sm_mhz_median")}
                          for b, v in summary.get("baselines", {}).items()},
            "work_per_step": wl, "profile": {"steps": profile_steps, "span_ms_per_step": {
                k: v["span"] / 1e6 / profile_steps for k, v in profs.items() if v["span"]}},
            "tensor_pipe_from": (m.group(0) if (m := re.search(r"r\d{8}-\d{6}-[0-9a-f]{4}", str(pdir))) else str(pdir))
            if pipes else None, "periods": periods}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="a run's out directory")
    ap.add_argument("--pipes-from", help="pipes-{plain,proto}.json from another run of the same kernels (default OUT's)")
    ap.add_argument("--profile-steps", type=int, default=64, help="the Nsight Systems profile's steps (run.sh's)")
    a = ap.parse_args(argv)
    print(json.dumps(breakdown(Path(a.out), Path(a.pipes_from) if a.pipes_from else None, a.profile_steps), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
