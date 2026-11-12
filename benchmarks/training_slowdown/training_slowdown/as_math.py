"""A record that holds a fused attention kernel as one row, at torch's FLOP formula, with that row as the ops the math
kernel computes attention by: the rows the counter writes for the kernel (`flops.Counter._attention`).

    python -m training_slowdown.as_math SRC DST [--partial none|SUBSET:CALLS]

Every run.json under SRC is written under DST with its op logs rewritten, and its steps' counts moved with them. A
fused row [phase, role, op, shapes, calls, MACs] of self-attention (q, k and v of one shape [b, h, s, d]) becomes
these rows, which name no dtype, as the row names none (a counter that names each op's dtype writes the kernel's ops
itself, so a fused row that names one is refused):

  forward, a call    the products QK [[bh, s, d], [bh, d, s]] and PV [[bh, s, s], [bh, s, d]] (the row's MACs) and the
                     scalings of q [[b, h, s, d]] and of k [[b, h, d, s]], numel(q) MACs each, as `fwd theta mul`
                     (what they are where q and k are activations);
  backward, a call   the products of the gradients the call asks for, and their scalings as `bwd <role> mul`:
                       v's          P^T G      [[bh, s, s], [bh, s, d]]
                       q's or k's   G V^T      [[bh, s, d], [bh, d, s]]
                       q's          dS K       [[bh, s, s], [bh, s, d]]   and the scaling [[b, h, s, d]]
                       k's          Q^T dS     [[bh, d, s], [bh, s, s]]   and the scaling [[b, h, d, s]]

A fused row doesn't say which gradients a call asked for; the record's trace does. A trace is taken under the math
kernel, so its backward holds those products, a call: with dp, pv and gk the counts of the products shaped like G V^T,
like P^T G and dS K, and like Q^T dS, every call asks for all three where gk = dp and pv = 2 dp. Otherwise the calls
that ask for fewer ask for one subset (in a model of blocks in sequence they are the lowest trained block's, where
nothing below it trains), and (gk - dp, pv - 2 dp) is (-c, -c) where c of them ask for q's alone, (0, c) for v's
alone, (-c, 0) for q's and v's, and (0, -2c) for k's alone, which 2c calls asking for k's and one of the others'
leave too: those products are the same and only the scalings differ. The ask is settled by this rule, never by a
default:

  - a record whose trace holds attention backward products of the row's shape: the ask is the trace's. `--partial`
    that says otherwise is refused, with both stated; where the trace leaves more than one ask open (k's alone, or
    k's and one other's), `--partial` says which.
  - a record with no such trace rows: refused unless `--partial` says the ask: `none` (every call asks for all three)
    or SUBSET:CALLS (CALLS calls a step ask for the gradients of SUBSET alone, of the letters q, k, v).

A wrong ask is not on the safe side. Asking for more than the step did over-counts |C|: ρ errs low and p high by the
same factor, so S is unchanged where products bind and errs high where no term binds.

A rewritten record says so in its `provenance`. A fused row that isn't self-attention of one shape, a trace whose
backward is no one subset's, an ask of more calls than a row holds, and one of fewer than it holds over several rows
are refused."""
import argparse
import json
import sys
from pathlib import Path

FORWARD = ("_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_for_cpu",
           "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention")
#: (gk - dp, pv - 2 dp) a call that asks for fewer gradients than all three, and the subsets that leave it
SIGNATURES = (((-1, -1), ("q",)), ((0, -2), ("k",)), ((0, 1), ("v",)), ((-1, 0), ("qv",)), ((0, -1), ("qk", "kv")))
ALL = ("", 0)  # every call asks for q's, k's and v's


def parse(flag: str | None) -> tuple[str, int] | None:
    """`--partial`'s value as an ask (subset, calls); `ALL` for `none`; None where it isn't given."""
    if flag is None or flag == "none":
        return None if flag is None else ALL
    subset, _, calls = flag.partition(":")
    letters = "".join(c for c in "qkv" if c in subset)
    if not letters or len(letters) == 3 or sorted(subset) != sorted(letters) or not calls.isdigit() or int(calls) < 1:
        raise ValueError(f"--partial {flag}: not `none` nor SUBSET:CALLS, SUBSET some but not all of q, k, v and CALLS "
                         "a count of calls a step")
    return letters, int(calls)


def said(ask: tuple[str, int]) -> str:
    return "none" if ask == ALL else f"{ask[0]}:{ask[1]}"


def traced(rows: list, n: int, s: int, d: int) -> list[tuple[str, int]] | None:
    """The asks a trace's rows leave open for attention at [n, s, d]: None where they hold no backward product of
    it, or s = d and the three products are one shape."""
    if s == d:
        return None
    kinds = {(s, d, s): "dp", (s, s, d): "pv", (d, s, s): "gk"}
    got = dict.fromkeys(("dp", "pv", "gk"), 0)
    for r in rows:
        dims = r.get("dims") or {}
        kind = kinds.get((dims.get("m"), dims.get("k"), dims.get("n")))
        if (kind and str(r.get("phase")).startswith("b") and dims.get("batch") == n
                and r.get("op") in ("bmm", "baddbmm", "matmul")):
            got[kind] += r["count"]
    if not any(got.values()):
        return None
    a, b = got["gk"] - got["dp"], got["pv"] - 2 * got["dp"]
    if (a, b) == (0, 0):
        return [ALL]
    open_ = [(subset, c) for (x, y), subsets in SIGNATURES for c in [max(abs(a), abs(b)) // max(abs(x), abs(y))]
             if (a, b) == (x * c, y * c) for subset in subsets]
    if not open_:
        raise ValueError(f"the trace's attention backward at [{n}, {s}, {d}] is no one subset's: {got}")
    return open_


def settle(open_: list | None, flag: tuple[str, int] | None, shape: str) -> tuple[tuple[str, int], str]:
    """The ask of the fused backward rows at `shape`, and where it is from: the trace's where it has one, else
    `--partial`'s."""
    if open_ is None:
        if flag is None:
            raise ValueError(f"the record's trace holds no attention backward at {shape} to read the gradients asked "
                             "from: say --partial none (every call asks for q's, k's and v's) or --partial SUBSET:CALLS")
        return flag, "--partial"
    if flag is None and len(open_) > 1:
        raise ValueError(f"the record's trace leaves --partial {' or '.join(map(said, open_))} open at {shape}: say "
                         "which")
    if flag is not None and flag not in open_:
        raise ValueError(f"the record's trace says --partial {' or '.join(map(said, open_))} at {shape} and the "
                         f"command says --partial {said(flag)}")
    return (flag if flag is not None else open_[0]), "its trace"


def backward(asked: str, b: int, h: int, s: int, d: int) -> tuple[dict, dict]:
    """One backward call's products and scalings for the gradients `asked` (of "q", "k", "v"): shapes -> calls."""
    qk, pv, dq = [[b * h, s, d], [b * h, d, s]], [[b * h, s, s], [b * h, s, d]], [[b * h, d, s], [b * h, s, s]]
    bmm, mul = {}, {}
    for when, shapes, scaling in (("v" in asked, pv, None), ("q" in asked or "k" in asked, qk, None),
                                  ("q" in asked, pv, [[b, h, s, d]]), ("k" in asked, dq, [[b, h, d, s]])):
        if when:
            bmm[json.dumps(shapes)] = bmm.get(json.dumps(shapes), 0) + 1
            if scaling:
                mul[json.dumps(scaling)] = 1
    return bmm, mul


def rewrite(log: list, trace: list, flag: tuple[str, int] | None = None) -> tuple[list, dict, dict]:
    """An op log with its fused attention rows as the math kernel's; what a step of it moves by, per `by` class and in
    matmul and elementwise MACs; and each backward shape's ask with where it is from. `trace` is the record's trace
    rows, `flag` what `--partial` says (`parse`)."""
    out, by, moved, asks = {}, {}, {"matmul_macs": 0, "elementwise_macs": 0}, {}

    def add(phase, role, op, shapes, n, macs, counts="", dtype=()):
        row = out.setdefault((phase, role, op, shapes, *dtype), [0, 0])
        row[0], row[1] = row[0] + n, row[1] + macs
        if counts:
            by[f"{phase}/{role}"] = by.get(f"{phase}/{role}", 0) + macs
            moved[counts] += macs

    backwards = [json.loads(r[3])[0] for r in log if "scaled_dot_product" in r[2] and r[2] not in FORWARD]
    for phase, role, op, shapes, n, macs, *dtype in log:
        if "scaled_dot_product" not in op:
            add(phase, role, op, shapes, n, macs, dtype=dtype)
            continue
        if dtype:
            raise ValueError(f"{phase} {op} {shapes}: a fused row that names a dtype ({dtype[0]}); the counter that "
                             f"names dtypes writes the kernel's ops, never the kernel's row")
        operands = json.loads(shapes)
        b, h, s, d = operands[0]
        each, numel = b * h * s * s * d, b * h * s * d
        if not (macs == (2 if op in FORWARD else 5) * n * each and phase == ("fwd" if op in FORWARD else "bwd")
                and operands[:3 if op in FORWARD else 4] == [[b, h, s, d]] * (3 if op in FORWARD else 4)):
            raise ValueError(f"{phase} {op} {shapes}: not self-attention of one shape at torch's formula")
        by[f"{phase}/{role}"] = by.get(f"{phase}/{role}", 0) - macs
        moved["matmul_macs"] -= macs
        if op in FORWARD:
            add("fwd", role, "bmm", json.dumps([[b * h, s, d], [b * h, d, s]]), n, n * each, "matmul_macs")
            add("fwd", role, "bmm", json.dumps([[b * h, s, s], [b * h, s, d]]), n, n * each, "matmul_macs")
            add("fwd", "theta", "mul", json.dumps([[b, h, s, d]]), n, n * numel, "elementwise_macs")
            add("fwd", "theta", "mul", json.dumps([[b, h, d, s]]), n, n * numel, "elementwise_macs")
            continue
        at = json.dumps([b * h, s, d])
        (subset, calls), source = settle(traced(trace, b * h, s, d), flag, at)
        if calls and backwards.count(operands[0]) > 1:
            raise ValueError(f"{calls} calls a step ask for {subset}'s alone at {at}, and the record holds "
                             f"{backwards.count(operands[0])} backward rows of that shape: which holds them isn't in it")
        if calls > n:
            raise ValueError(f"{calls} calls a step ask for {subset}'s alone at {at}, and the record's row holds {n}")
        asks[at] = {"asked": said((subset, calls)), "from": source}
        for asked, times in (("qkv", n - calls), (subset, calls)):
            products, scalings = backward(asked, b, h, s, d)
            for k, c in products.items():
                add("bwd", role, "bmm", k, c * times, c * times * each, "matmul_macs")
            for k, c in scalings.items():
                add("bwd", role, "mul", k, c * times, c * times * numel, "elementwise_macs")
    rows = [[*key[:4], n, macs, *key[4:]] for key, (n, macs) in out.items() if n]
    return rows, {"by": {k: v for k, v in by.items() if v}, **moved, "macs": sum(moved.values())}, asks


def record(rec: dict, flag: tuple[str, int] | None = None) -> int:
    """`rec` rewritten in place, its ask settled by its trace and `flag` (the module docstring's rule); the MACs it
    moved by in all. A step's moved MACs are its largest device's."""
    trace = (rec.get("trace") or {}).get("rows") or []
    moves, asks = {}, {}
    for sid, st in rec.get("structures", {}).items():
        st["log"], moves[sid], how = rewrite(st.get("log", []), trace, flag)
        asks |= how
    total = 0
    for s in [rec.get("setup") or {}, *rec["steps"], rec.get("tail") or {}]:
        mv = moves.get(s.get("struct"))
        if not mv or not (mv["by"] or mv["macs"]):
            continue
        for k, v in mv["by"].items():
            s["by"][k] = s["by"].get(k, 0) + v
        for k in ("matmul_macs", "elementwise_macs", "macs"):
            s[k] += mv[k]
        for devices in (s.get("device_macs"), rec["totals"].get("device_macs")):
            if devices:
                largest = max(devices, key=devices.get)
                devices[largest] += mv["macs"]
        total += mv["macs"]
    rec["totals"]["macs"] += total
    if any(mv["by"] or mv["macs"] for mv in moves.values()):
        was = rec.get("provenance")
        rec["provenance"] = ({} if was is None else was if isinstance(was, dict) else {"said": was}) | {"as_math": {
            "rewritten": "its fused attention rows as the math kernel's ops (training_slowdown.as_math)",
            "gradients_asked": asks}}
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--partial", help="none, or SUBSET:CALLS: the gradients a record's attention backward asks for, "
                                      "where its trace doesn't say")
    a = ap.parse_args(argv)
    if a.dst.exists():
        raise SystemExit(f"{a.dst}: exists")
    done = []
    for path in sorted(a.src.rglob("run.json")):
        rec = json.loads(path.read_text())
        try:
            done.append((path.relative_to(a.src), rec, record(rec, parse(a.partial))))
        except ValueError as e:
            raise SystemExit(f"{path.relative_to(a.src)}: refused: {e}") from None
    for rel, rec, total in done:
        (a.dst / rel).parent.mkdir(parents=True, exist_ok=True)
        (a.dst / rel).write_text(json.dumps(rec))
        asked = ((rec.get("provenance") or {}).get("as_math") or {}).get("gradients_asked") if total else None
        print(f"{rel}: {total} MACs moved" + (f"; gradients asked {json.dumps(asked, sort_keys=True)}" if asked else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
