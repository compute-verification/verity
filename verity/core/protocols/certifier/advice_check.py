"""Advice v0's validity as a Boolean Definition: the first part of the compile unit, "this committed bit string decodes to
a well-formed v0 advice" (`advice`).

    AdviceCheck_v1{L} (bits: Array<L, Value<1>>, length: Array<LW, Value<1>>) -> Value<1>

**Statics.** L is the public cap in bits, a multiple of 40; LW = bitlen(L / 8).

**Inputs.** `bits` is the committed bit string in stream order (bit 8k + j is bit 7 − j of byte k, as `advice.encode`
writes it), and `length` its length in bytes, least significant bit first. Bits past 8·length are read as nothing.

**What 1 means** (`reference`): length ≤ L / 8, and the bytes pass every refusal `advice.decode` makes but three, which
`DEFERRED` names and the circuit doesn't check yet: unique ids, a source's KV computed by its request at those positions
(`advice._covered`), and each request's release buckets non-decreasing. Checked:
- the grammar: version 0, minimal `uvar`s, the widths w_len, w_rel and w_src the least that hold their fields, w_req
  from the request count, no truncation, fewer than 8 padding bits, all zero, and no trailing byte;
- each request: a non-empty UTF-8 id, prompt and output lengths ≥ 1, the outputs' total within the release count;
- each row and source in the stream: n ≥ 1, a row's request at most the requests admitted so far (first-row order), a
  first row that doesn't resume and starts at its hit, a hit only on an admitting row, the hit's sources summing to it,
  adjacent sources of different requests, a request's own KV only on a resume, every source's request admitted by the
  end of its step, and every request admitted by the end;
- each request's replay, rows in order: a continuing row starts where the last ended, no row past the known tokens (so
  at most one row past the last output), `kind` the derived one, no resume of a finished request, no request twice in a
  step, and exactly `output_len` outputs kept.

**The circuit**, three stages:
1. `AdviceParse`, a scan over the bits five at a time: a bit-serial parser whose state is the field being read, its
   bits left and value so far, the header's counts and widths, the counters of the records left, and the in-stream
   checks above. A field of zero width is skipped in the step that ends the field before it. Each block of five bits
   ends at most one request record or row (a row is at least five bits), and the block emits it as an event.
2. Batcher's odd-even merge sort of the L / 5 events by (request, position), one batch of `AdviceSwap` per layer: the
   network on the next power of two without the comparators that reach past the events (`_sort_net`).
3. `AdviceReplay`, a scan over the sorted events: each request's record, then its rows in stream order, replayed with
   one request's state (`advice._replay`'s computed, known and kept).

**Cost** (`size`): ANDs = (L / 5)·(parse + replay) + C(L / 5)·swap + the ends, C(n) the sort's comparators
(`comparators`, about n·log2(n)² / 4).
"""
from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.core.protocols.certifier.advice import FINISHES, VERSION, WIDTH_BITS, MalformedAdvice, _Reader, decode
from verity.core.protocols.certifier.head_check import BIT, _coll, _Consts, _part, _refs, ands

#: the refusals of `advice.decode` the circuit doesn't make yet, by the start of decode's message for each
DEFERRED = {
    "unique ids": "two requests share an id",
    "a source's KV computed there": "computed no KV at positions",
    "release buckets non-decreasing per request": "is released at bucket",
}
BLOCK = 5
VW = 63  # a field's value bits: every width is at most 63


class _Refused(Exception):
    pass


def _need(ok: bool) -> None:
    if not ok:
        raise _Refused


def _check(data: bytes) -> None:
    """`advice.decode`'s refusals but `DEFERRED`'s, as the circuit makes them; raises `_Refused` (or `MalformedAdvice`,
    from the reader's truncation and `uvar` checks)."""
    rd = _Reader(data)
    _need(rd.get(8) == VERSION)
    n_req, n_steps = rd.uvar(), rd.uvar()
    w_len, w_rel, w_src = rd.get(WIDTH_BITS), rd.get(WIDTH_BITS), rd.get(WIDTH_BITS)
    w_req = max(n_req - 1, 0).bit_length()
    lens: list[int] = []
    reqs = []
    for _ in range(n_req):
        size = rd.uvar()
        _need(size >= 1 and 8 * size <= rd.left())
        try:
            bytes(rd.get(8) for _ in range(size)).decode("utf-8")
        except UnicodeDecodeError:
            raise _Refused from None
        P, O = rd.get(w_len), rd.get(w_len)
        FINISHES[rd.get(1)]
        _need(P >= 1 and O >= 1)
        lens += [P, O]
        reqs.append((P, O))
    first = 0
    computed, known, kept, last = [0] * n_req, [0] * n_req, [0] * n_req, [-1] * n_req
    srcs = []
    for s in range(n_steps):
        need = 0
        for _ in range(rd.get(w_req) + 1):
            r, kind = rd.get(w_req), rd.get(1)
            start, n, hit = rd.get(w_len), rd.get(w_len), rd.get(w_len)
            pre = rd.get(1)
            lens += [start, n, hit]
            _need(n >= 1 and r <= first and r < n_req)
            P, O = reqs[r]
            if r == first:
                _need(not pre and start == hit)
                first += 1
                known[r] = P
            else:
                _need(pre or not hit)
                _need(kept[r] < O if pre else start == computed[r])
            _need(not (hit or pre) or start == hit)
            end = start + n
            _need(end <= known[r] and kind == int(n == 1 and start >= P) and last[r] != s)
            last[r], computed[r] = s, end
            if end == known[r] and kept[r] < O:
                kept[r] += 1
                known[r] += 1
            if hit:
                k = rd.get(w_src)
                srcs.append(k)
                total, prev = 0, None
                for _ in range(k + 1):
                    q, m = rd.get(w_req), rd.get(w_len)
                    lens.append(m)
                    _need(m >= 1 and q != prev and (q != r or pre))
                    need, total, prev = max(need, q + 1), total + m, q
                _need(total == hit)
        _need(need <= first)
    _need(first == n_req and all(kept[r] == reqs[r][1] for r in range(n_req)))
    rels = [rd.get(w_rel) for _ in range(sum(O for _, O in reqs))]
    _need(rd.left() < 8 and not rd.get(rd.left()))
    _need(w_len == max(lens, default=0).bit_length() and w_rel == max(rels, default=0).bit_length()
          and w_src == max(srcs, default=0).bit_length())


def reference(data: bytes, L: int) -> bool:
    """`AdviceCheck_v1{L}`'s output on `data`: what the circuit checks, in Python."""
    if not isinstance(data, (bytes, bytearray)) or 8 * len(data) > L:
        return False
    try:
        _check(bytes(data))
    except (_Refused, MalformedAdvice):
        return False
    return True


def decodes(data: bytes) -> tuple[bool, str]:
    """Whether `advice.decode` takes `data`, and its refusal's message when it doesn't."""
    try:
        decode(bytes(data))
    except MalformedAdvice as e:
        return False, str(e)
    return True, ""


# --- the circuit's helpers (bit lists, least significant bit first) --------------------------------------------------

def _pad(x, w):
    return (list(x) + [ZERO] * w)[:w]


def _eqc(C, x, k: int):
    return and_reduce(C, [b if k >> j & 1 else b.inv() for j, b in enumerate(x)]) if k < 1 << len(x) else ZERO


def _eq(C, a, b):
    w = max(len(a), len(b))
    return and_reduce(C, [(x ^ y).inv() for x, y in zip(_pad(a, w), _pad(b, w))])


def _lt(C, a, b):
    w = max(len(a), len(b))
    return sub(C, _pad(a, w), _pad(b, w), w)[1]


def _inc(C, x, c):
    return add(C, x, [c], width=len(x))


def _dec_if(C, x, c):
    return sub(C, x, [c], len(x))[0]


def _x(*terms):
    """The XOR of mutually exclusive terms: their OR."""
    out = ZERO
    for t in terms:
        out = out ^ t
    return out


def _sel(C, s, x):
    return [C.AND(s, b) for b in x]


# --- stage 1: the parser --------------------------------------------------------------------------------------------

MODES = ("VER", "NRW", "NRV", "NSW", "NSV", "WLEN", "WREL", "WSRC", "IDW", "IDV", "IDB", "PL", "OL", "FIN", "RC", "RR",
         "RK", "RS", "RN", "RH", "RP", "SC", "SR", "SN", "REL", "PAD")
M = {name: i for i, name in enumerate(MODES)}
#: the UTF-8 decoder's states, one-hot: accept, then 1, 2 or 3 continuation bytes to go, and the four first bytes whose
#: second byte has a narrower range (E0, ED, F0, F4)
UTF = ("ACC", "N1", "N2", "N3", "E0", "ED", "F0", "F4")


def _widths(L: int) -> SimpleNamespace:
    if type(L) is not int or L < 40 or L % 40:
        raise ValueError(f"L is a positive multiple of 40 bits, not {L!r}")
    nb = L // BLOCK
    return SimpleNamespace(L=L, W=L.bit_length(), LW=(L // 8).bit_length(), REM=(L // 8).bit_length() + 3, NB=nb,
                           P=nb.bit_length())


def _state_layout(Z) -> list[tuple[str, int]]:
    W = Z.W
    return [("rem", Z.REM), ("m", len(MODES)), ("cnt", 6), ("fresh", 1), ("acc", VW), ("nreq", W), ("steps", W),
            ("reqs", W), ("ids", W), ("rows", W), ("srcs", W), ("rels", W), ("wlen", 6), ("wrel", 6), ("wsrc", 6),
            ("seen_len", 1), ("seen_rel", 1), ("seen_src", 1), ("F", W), ("A", VW), ("B", VW), ("H", VW), ("r", W),
            ("kind", 1), ("n1", 1), ("sh_eq", 1), ("hit", 1), ("pre", 1), ("q", W), ("prev", W), ("sfirst", 1),
            ("need", W + 1), ("utf", len(UTF)), ("valid", 1)]


def _event_layout(Z) -> list[tuple[str, int]]:
    """An event, its sort key first (position, request, and no-event as the most significant bit)."""
    return [("pos", Z.P), ("r", Z.W), ("none", 1), ("row", 1), ("A", VW), ("B", VW), ("pre", 1), ("kind", 1),
            ("n1", 1), ("step", Z.W)]


def _size(layout) -> int:
    return sum(w for _, w in layout)


def _unpack(layout, bits) -> dict:
    out, at = {}, 0
    for name, w in layout:
        out[name] = list(bits[at:at + w])
        at += w
    return out


def _pack(layout, d: dict) -> list:
    out = []
    for name, w in layout:
        v = d[name]
        if len(v) != w:
            raise AssertionError(f"{name} has {len(v)} bits, the layout {w}")
        out += v
    return out


def _utf8(C, u, x):
    """One byte x through the strict UTF-8 decoder in one-hot state u: the next state, all zero to refuse."""
    x7, x6, x5, x4, x3, x2, x1, x0 = x[7], x[6], x[5], x[4], x[3], x[2], x[1], x[0]
    s = dict(zip(UTF, u))
    lead = C.AND(x7, x6)
    cont = C.AND(x7, x6.inv())
    lead2 = C.AND(C.AND(lead, x5.inv()), or_reduce(C, [x4, x3, x2, x1]))
    lead3 = C.AND(C.AND(lead, x5), x4.inv())
    low4 = [x0, x1, x2, x3]
    e0, ed = C.AND(lead3, _eqc(C, low4, 0x0)), C.AND(lead3, _eqc(C, low4, 0xD))
    lead4 = C.AND(C.AND(C.AND(lead, x5), x4), x3.inv())
    f0, f4 = C.AND(lead4, _eqc(C, [x0, x1, x2], 0)), C.AND(lead4, _eqc(C, [x0, x1, x2], 4))
    f13 = C.AND(C.AND(lead4, x2.inv()), x1 ^ x0 ^ C.AND(x1, x0))
    acc = s["ACC"]
    return [_x(C.AND(acc, x7.inv()), C.AND(s["N1"], cont)),
            _x(C.AND(acc, lead2), C.AND(s["N2"], cont), C.AND(s["E0"], C.AND(cont, x5)),
               C.AND(s["ED"], C.AND(cont, x5.inv()))),
            _x(C.AND(acc, lead3 ^ e0 ^ ed), C.AND(s["N3"], cont), C.AND(s["F0"], C.AND(cont, x5 ^ x4 ^ C.AND(x5, x4))),
               C.AND(s["F4"], C.AND(cont, C.AND(x5.inv(), x4.inv())))),
            C.AND(acc, f13), C.AND(acc, e0), C.AND(acc, ed), C.AND(acc, f0), C.AND(acc, f4)]


def _bit_step(C, Z, st: dict, b, pos):
    """One bit through the parser: the next state and the event it ends (an emit bit, then the event's fields)."""
    W = Z.W
    m = st["m"]
    md = {name: m[i] for name, i in M.items()}
    live = or_reduce(C, st["rem"])
    consume = C.AND(live, md["PAD"].inv())
    done = C.AND(consume, _eqc(C, st["cnt"], 1))
    D = {name: C.AND(done, md[name]) for name in MODES[:-1]}
    first_bit = C.AND(consume, st["fresh"][0])
    val = [b] + st["acc"][:VW - 1]
    v6 = val[:6]
    v6z = _eqc(C, v6, 0)
    hi_zero = or_reduce(C, val[W:]).inv()
    val_hz = or_reduce(C, val[1:])
    val_nz = val_hz ^ val[0] ^ C.AND(val_hz, val[0])
    val_one = C.AND(val_hz.inv(), val[0])
    bad = []  # bits that refuse

    nm1, nz_borrow = sub(C, st["nreq"], [ONE], W)
    therm = []
    acc_or = ZERO
    for j in reversed(range(W)):
        acc_or = acc_or ^ nm1[j] ^ C.AND(acc_or, nm1[j])
        therm.append(acc_or)
    therm = [C.AND(t, nz_borrow.inv()) for t in reversed(therm)]  # therm[j]: w_req > j
    wreq = [_x(*therm[(1 << i) - 1::1 << i]) for i in range(6)]  # bit i of a count c: the parity of #{m ≥ 1: m·2^i ≤ c}
    wreq_nz = therm[0]
    wlen_nz, wrel_nz, wsrc_nz = (or_reduce(C, st[k]) for k in ("wlen", "wrel", "wsrc"))
    nreq_nz, steps_nz = or_reduce(C, st["nreq"]), or_reduce(C, st["steps"])

    # the header
    bad.append(C.AND(D["VER"], or_reduce(C, val[:8])))
    for name, limit in (("NRW", W), ("NSW", W), ("IDW", W)):
        bad.append(C.AND(C.AND(D[name], v6z.inv()), _lt(C, const_bits(limit, 6), v6)))
    for name in ("NRV", "NSV", "IDV"):
        bad.append(C.AND(C.AND(first_bit, md[name]), b.inv()))
    hdr = D["WSRC"]
    bad.append(C.AND(C.AND(hdr, nreq_nz), wlen_nz.inv()))
    hdr_fail = C.AND(C.AND(hdr, nreq_nz.inv()), steps_nz)
    id_fail = C.AND(D["IDW"], v6z)
    bad += [hdr_fail, id_fail]

    # the requests
    utf_new = _utf8(C, st["utf"], val[:8])
    utf_ok = or_reduce(C, utf_new)
    bad.append(C.AND(D["IDB"], utf_ok.inv()))
    id_last = C.AND(D["IDB"], _eqc(C, st["ids"], 1))
    bad.append(C.AND(id_last, utf_new[0].inv()))
    bad.append(C.AND(C.AND(done, md["PL"] ^ md["OL"] ^ md["RN"] ^ md["SN"]), val_nz.inv()))
    rels_sum_c = add(C, st["rels"], val[:W], width=W, carry_out=True)
    rels_sum, rels_carry = rels_sum_c[:W], rels_sum_c[W]
    bad.append(C.AND(D["OL"], (hi_zero.inv() ^ rels_carry ^ C.AND(hi_zero.inv(), rels_carry))))
    req_last = C.AND(D["FIN"], _eqc(C, st["reqs"], 1))
    ridx = sub(C, st["nreq"], st["reqs"], W)[0]

    # the steps' rows
    r_eff = _sel(C, wreq_nz, st["r"])
    is_first = _eq(C, r_eff, st["F"])
    pre = b
    rp = D["RP"]
    bad.append(C.AND(rp, C.AND(is_first.inv(), _lt(C, st["F"], r_eff))))
    bad.append(C.AND(C.AND(rp, is_first), pre ^ st["sh_eq"][0].inv() ^ C.AND(pre, st["sh_eq"][0].inv())))
    bad.append(C.AND(C.AND(rp, is_first.inv()), C.AND(pre.inv(), st["hit"][0])))
    bad.append(C.AND(C.AND(rp, st["hit"][0] ^ pre ^ C.AND(st["hit"][0], pre)), st["sh_eq"][0].inv()))
    F_new = _inc(C, st["F"], C.AND(rp, is_first))
    sources_start = C.AND(rp, st["hit"][0])

    # the sources
    q_eff = _sel(C, wreq_nz, st["q"])
    sn = D["SN"]
    bad.append(C.AND(C.AND(sn, st["sfirst"][0].inv()), _eq(C, q_eff, st["prev"])))
    bad.append(C.AND(C.AND(sn, st["pre"][0].inv()), _eq(C, q_eff, r_eff)))
    bad.append(C.AND(D["SC"], hi_zero.inv()))
    q1 = add(C, q_eff, [ONE], width=W + 1)
    need_up = mux(C, C.AND(sn, _lt(C, st["need"], q1)), q1, st["need"])
    H_left, H_borrow = sub(C, st["H"], val, VW)
    bad.append(C.AND(sn, H_borrow))
    src_last = C.AND(sn, or_reduce(C, st["srcs"]).inv())
    bad.append(C.AND(src_last, or_reduce(C, H_left)))

    row_end = _x(C.AND(rp, st["hit"][0].inv()), src_last)
    rows_zero = or_reduce(C, st["rows"]).inv()
    step_end = C.AND(row_end, rows_zero)
    bad.append(C.AND(step_end, _lt(C, _pad(F_new, W + 1), need_up)))
    steps_last = C.AND(step_end, _eqc(C, st["steps"], 1))
    after_reqs = req_last
    step_start = _x(C.AND(after_reqs, steps_nz), C.AND(step_end, steps_last.inv()))
    row_start = _x(C.AND(step_start, wreq_nz.inv()), D["RC"], C.AND(row_end, rows_zero.inv()))
    src_start = _x(C.AND(sources_start, wsrc_nz.inv()), D["SC"], C.AND(sn, src_last.inv()))
    to_rel = _x(C.AND(after_reqs, steps_nz.inv()), steps_last,
                C.AND(C.AND(hdr, nreq_nz.inv()), steps_nz.inv()))
    rels_nz = or_reduce(C, st["rels"])
    rel_last = C.AND(D["REL"], _eqc(C, st["rels"], 1))
    enter_rel = _x(C.AND(C.AND(to_rel, rels_nz), wrel_nz), C.AND(D["REL"], rel_last.inv()))
    pad_ok = _x(C.AND(to_rel, C.AND(rels_nz, wrel_nz).inv()), rel_last)
    pad_fail = _x(hdr_fail, id_fail)
    bad.append(C.AND(pad_ok, _lt(C, st["rem"], const_bits(9, Z.REM)).inv()))  # fewer than 8 bits after this one
    bad.append(C.AND(C.AND(live, md["PAD"]), b))

    enter = {
        "NRW": D["VER"], "NRV": C.AND(D["NRW"], v6z.inv()), "NSW": _x(C.AND(D["NRW"], v6z), D["NRV"]),
        "NSV": C.AND(D["NSW"], v6z.inv()), "WLEN": _x(C.AND(D["NSW"], v6z), D["NSV"]), "WREL": D["WLEN"],
        "WSRC": D["WREL"], "IDW": _x(C.AND(hdr, nreq_nz), C.AND(D["FIN"], req_last.inv())),
        "IDV": C.AND(D["IDW"], v6z.inv()), "IDB": _x(D["IDV"], C.AND(D["IDB"], id_last.inv())), "PL": id_last,
        "OL": D["PL"], "FIN": D["OL"], "RC": C.AND(step_start, wreq_nz), "RR": C.AND(row_start, wreq_nz),
        "RK": _x(C.AND(row_start, wreq_nz.inv()), D["RR"]), "RS": D["RK"], "RN": D["RS"], "RH": D["RN"], "RP": D["RH"],
        "SC": C.AND(sources_start, wsrc_nz), "SR": C.AND(src_start, wreq_nz),
        "SN": _x(C.AND(src_start, wreq_nz.inv()), D["SR"]), "REL": enter_rel, "PAD": _x(pad_ok, pad_fail),
    }
    keep = done.inv()
    m_new = [_x(C.AND(md[name], keep) if name != "PAD" else md[name], enter.get(name, ZERO)) for name in MODES]

    groups = [(("NRW", "NSW", "WLEN", "WREL", "WSRC", "IDW"), const_bits(6, 6)), (("NRV", "NSV", "IDV"), v6),
              (("IDB",), const_bits(8, 6)), (("PL", "OL", "RS", "RN", "RH", "SN"), st["wlen"]),
              (("FIN", "RK", "RP"), const_bits(1, 6)), (("RC", "RR", "SR"), wreq), (("SC",), st["wsrc"]),
              (("REL",), st["wrel"])]
    width = [ZERO] * 6
    for names, w in groups:
        g = _x(*(enter[n] for n in names))
        width = [a ^ C.AND(g, x) for a, x in zip(width, w)]
    cnt_new = mux(C, done, width, _dec_if(C, st["cnt"], consume))

    def load(cond, new, old):
        return mux(C, cond, _pad(new, len(old)), old)

    new = dict(st)
    new["rem"] = _dec_if(C, st["rem"], live)
    new["m"], new["cnt"] = m_new, cnt_new
    new["fresh"] = [_x(done, C.AND(st["fresh"][0], consume.inv()))]
    new["acc"] = _sel(C, done.inv(), val)
    new["nreq"] = load(D["NRV"], val[:W], st["nreq"])
    new["steps"] = load(D["NSV"], val[:W], _dec_if(C, st["steps"], step_end))
    new["reqs"] = load(D["NRV"], val[:W], _dec_if(C, st["reqs"], D["FIN"]))
    new["ids"] = load(D["IDV"], val[:W], _dec_if(C, st["ids"], D["IDB"]))
    # a step's rows and a row's sources start with their counter at zero, where the last ended (or never loaded)
    new["rows"] = load(D["RC"], val[:W], _dec_if(C, st["rows"], C.AND(row_end, rows_zero.inv())))
    new["srcs"] = load(D["SC"], val[:W], _dec_if(C, st["srcs"], C.AND(sn, src_last.inv())))
    new["rels"] = mux(C, D["OL"], rels_sum, _dec_if(C, st["rels"], D["REL"]))
    new["wlen"] = load(D["WLEN"], v6, st["wlen"])
    new["wrel"] = load(D["WREL"], v6, st["wrel"])
    new["wsrc"] = load(D["WSRC"], v6, st["wsrc"])
    grp_len = _x(md["PL"], md["OL"], md["RS"], md["RN"], md["RH"], md["SN"])
    top = C.AND(first_bit, b)
    for flag, sel in (("seen_len", grp_len), ("seen_rel", md["REL"]), ("seen_src", md["SC"])):
        f = st[flag][0]
        t = C.AND(top, sel)
        new[flag] = [f ^ t ^ C.AND(f, t)]
    new["F"] = F_new
    new["A"] = load(D["PL"] ^ D["RS"], val, st["A"])
    new["B"] = load(D["RN"], val, st["B"])
    new["H"] = load(D["RH"], val, mux(C, sn, H_left, st["H"]))
    new["r"] = load(D["RR"], val[:W], st["r"])
    new["kind"] = load(D["RK"], [b], st["kind"])
    new["n1"] = load(D["RN"], [val_one], st["n1"])
    new["sh_eq"] = load(D["RH"], [_eq(C, val, st["A"])], st["sh_eq"])
    new["hit"] = load(D["RH"], [val_nz], st["hit"])
    new["pre"] = load(rp, [b], st["pre"])
    new["q"] = load(D["SR"], val[:W], st["q"])
    new["prev"] = load(sn, q_eff, st["prev"])
    new["sfirst"] = [_x(sources_start, C.AND(st["sfirst"][0], sn.inv()))]
    new["need"] = _sel(C, step_end.inv(), need_up)
    utf_acc = const_bits(1, len(UTF))
    new["utf"] = mux(C, D["IDV"], utf_acc, mux(C, D["IDB"], utf_new, st["utf"]))
    new["valid"] = [C.AND(st["valid"][0], or_reduce(C, bad).inv())]

    ev_req = D["OL"]
    ev = {"pos": pos, "r": mux(C, ev_req, ridx, r_eff), "none": [ZERO], "row": [rp], "A": st["A"],
          "B": mux(C, ev_req, val, st["B"]), "pre": [b], "kind": st["kind"], "n1": st["n1"], "step": st["steps"]}
    return new, ev_req ^ rp, ev


def _sig_parse(S):
    Z = _widths(S.L)
    st = Array(_size(_state_layout(Z)), BIT)
    return (("state", st), ("bits", Array(BLOCK, BIT)), ("pos", Array(Z.P, BIT))), \
        Tuple(st, Array(_size(_event_layout(Z)), BIT))


def _build_parse(C, S, state, bits, pos):
    Z = _widths(S.L)
    st = _unpack(_state_layout(Z), state)
    emits, events = [], []
    for b in bits:
        st, emit, ev = _bit_step(C, Z, st, b, list(pos))
        emits.append(emit)
        events.append(_pack(_event_layout(Z), ev))
    two = or_reduce(C, [C.AND(a, c) for i, a in enumerate(emits) for c in emits[i + 1:]])
    st["valid"] = [C.AND(st["valid"][0], two.inv())]
    any_emit = or_reduce(C, emits)
    out = [ZERO] * (len(events[0]) - Z.P - Z.W - 1)
    for e, ev in zip(emits, events):
        out = [o ^ C.AND(e, x) for o, x in zip(out, ev[Z.P + Z.W + 1:])]
    r = [ZERO] * Z.W
    for e, ev in zip(emits, events):
        r = [o ^ C.AND(e, x) for o, x in zip(r, ev[Z.P:Z.P + Z.W])]
    return [_pack(_state_layout(Z), st), list(pos) + r + [any_emit.inv()] + out]


def _sig_parse_end(S):
    Z = _widths(S.L)
    return (("state", Array(_size(_state_layout(Z)), BIT)), ("length", Array(Z.LW, BIT))), BIT


def _build_parse_end(C, S, state, length):
    """The parse ended: on the padding, every request admitted, each width the least, and length ≤ L / 8."""
    Z = _widths(S.L)
    st = _unpack(_state_layout(Z), state)
    oks = [st["valid"][0], st["m"][M["PAD"]], _eq(C, st["F"], st["nreq"]),
           _lt(C, const_bits(Z.L // 8, Z.LW + 1), length).inv()]
    for w, seen in (("wlen", "seen_len"), ("wrel", "seen_rel"), ("wsrc", "seen_src")):
        z = or_reduce(C, st[w])
        oks.append(z.inv() ^ C.AND(z, st[seen][0]))
    return [and_reduce(C, oks)]


# --- stage 2: the sort ----------------------------------------------------------------------------------------------

def _sig_swap(S):
    ev = Array(_size(_event_layout(_widths(S.L))), BIT)
    return (("a", ev), ("b", ev)), Tuple(ev, ev)


def _build_swap(C, S, a, b):
    """The two events in key order (position, request, no-event, the most significant last)."""
    Z = _widths(S.L)
    kw = Z.P + Z.W + 1
    s = _lt(C, b[:kw], a[:kw])
    d = [C.AND(s, x ^ y) for x, y in zip(a, b)]
    return [[x ^ t for x, t in zip(a, d)], [y ^ t for y, t in zip(b, d)]]


def sort_layers(n: int) -> list[list[tuple[int, int]]]:
    """Batcher's odd-even merge sort on n = 2^k wires, layer by layer: each layer's pairs (i, j), i < j, are disjoint."""
    layers, p = [], 1
    while p < n:
        k = p
        while k >= 1:
            layer = []
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        layer.append((i + j, i + j + k))
            layers.append(layer)
            k //= 2
        p *= 2
    return layers


def _sort_net(nb: int) -> list[list[tuple[int, int]]]:
    """The sort of nb events: Batcher's network on the next power of two, less every comparator that reaches a wire ≥ nb.
    Those wires hold +∞, which no comparator (min to the lower wire) moves, so the rest sorts the nb events."""
    n = 1 << max(1, (nb - 1).bit_length())
    return [kept for layer in sort_layers(n) if (kept := [(i, j) for i, j in layer if j < nb])]


def comparators(nb: int) -> int:
    """`_sort_net(nb)`'s comparators, counted in closed form: layer (p, k) compares x with x + k for the x whose offset
    in its block of 2p is in [k % p, 2p − k) and an even number of k-runs past k % p."""

    def runs(y: int, k: int) -> int:
        return 0 if y <= 0 else y // (2 * k) * k + min(y % (2 * k), k)

    n, total, p = 1 << max(1, (nb - 1).bit_length()), 0, 1
    while p < n:
        k = p
        while k >= 1:
            m, o = max(nb - k, 0), k % p
            total += m // (2 * p) * runs(2 * p - k - o, k) + runs(min(m % (2 * p), 2 * p - k) - o, k)
            k //= 2
        p *= 2
    return total


# --- stage 3: the replay --------------------------------------------------------------------------------------------

def _replay_layout(Z) -> list[tuple[str, int]]:
    return [("cur", Z.W), ("group", 1), ("has_row", 1), ("P", VW), ("O", VW), ("computed", VW + 1),
            ("known", VW + 1), ("kept", VW), ("last", Z.W), ("ok", 1)]


def _sig_replay(S):
    Z = _widths(S.L)
    return (("state", Array(_size(_replay_layout(Z)), BIT)), ("event", Array(_size(_event_layout(Z)), BIT))), \
        Tuple(Array(_size(_replay_layout(Z)), BIT), BIT)


def _build_replay(C, S, state, event):
    """One sorted event: a request's record starts its group (closing the last: its outputs all kept), and a row is
    replayed on its request's state as `advice._replay` replays it."""
    Z = _widths(S.L)
    st, e = _unpack(_replay_layout(Z), state), _unpack(_event_layout(Z), event)
    emit = e["none"][0].inv()
    is_row = C.AND(emit, e["row"][0])
    is_req = C.AND(emit, e["row"][0].inv())
    group, has_row = st["group"][0], st["has_row"][0]
    kept_lt = _lt(C, st["kept"], st["O"])
    bad = [C.AND(C.AND(is_req, group), _eq(C, st["kept"], st["O"]).inv())]
    first = has_row.inv()
    pre = e["pre"][0]
    start, n = e["A"], e["B"]
    end = add(C, _pad(start, VW + 1), _pad(n, VW + 1), width=VW + 1)
    bad.append(C.AND(is_row, C.AND(group, _eq(C, e["r"], st["cur"])).inv()))
    bad.append(C.AND(C.AND(is_row, C.AND(first.inv(), pre)), kept_lt.inv()))
    bad.append(C.AND(C.AND(is_row, C.AND(first.inv(), pre.inv())), _eq(C, start, st["computed"]).inv()))
    bad.append(C.AND(is_row, _lt(C, st["known"], end)))
    derived = C.AND(e["n1"][0], _lt(C, start, st["P"]).inv())
    bad.append(C.AND(is_row, e["kind"][0] ^ derived))
    bad.append(C.AND(C.AND(is_row, has_row), _eq(C, e["step"], st["last"])))
    keep = C.AND(C.AND(is_row, _eq(C, end, st["known"])), kept_lt)
    new = {
        "cur": mux(C, is_req, e["r"], st["cur"]),
        "group": [group ^ is_req ^ C.AND(group, is_req)],
        "has_row": [C.AND(is_req.inv(), has_row ^ is_row ^ C.AND(has_row, is_row))],
        "P": mux(C, is_req, e["A"], st["P"]),
        "O": mux(C, is_req, e["B"], st["O"]),
        "computed": mux(C, is_row, end, st["computed"]),
        "known": mux(C, is_req, _pad(e["A"], VW + 1), _inc(C, st["known"], keep)),
        "kept": _sel(C, is_req.inv(), _inc(C, st["kept"], keep)),
        "last": mux(C, is_row, e["step"], st["last"]),
        "ok": [C.AND(st["ok"][0], or_reduce(C, bad).inv())],
    }
    return [_pack(_replay_layout(Z), new), [ZERO]]


def _sig_replay_end(S):
    return (("state", Array(_size(_replay_layout(_widths(S.L))), BIT)),), BIT


def _build_replay_end(C, S, state):
    st = _unpack(_replay_layout(_widths(S.L)), state)
    closed = st["group"][0].inv() ^ C.AND(st["group"][0], _eq(C, st["kept"], st["O"]))
    return [C.AND(st["ok"][0], closed)]


AdviceParse = _part("AdviceParse", ("L",), _sig_parse, _build_parse,
                    "five bits of an advice v0 string through the bit-serial parser: its state, and the event the block "
                    "ends (a request's record or a row)")
AdviceParseEnd = _part("AdviceParseEnd", ("L",), _sig_parse_end, _build_parse_end,
                       "the parse ended on the padding with every request admitted, each width the least and the length "
                       "within L / 8 bytes")
AdviceSwap = _part("AdviceSwap", ("L",), _sig_swap, _build_swap,
                   "two events in order of (request, position), an absent event last")
AdviceReplay = _part("AdviceReplay", ("L",), _sig_replay, _build_replay,
                     "one event of the sorted stream replayed on its request's state")
AdviceReplayEnd = _part("AdviceReplayEnd", ("L",), _sig_replay_end, _build_replay_end,
                        "the replay ended with the last request's outputs all kept")
PARTS = (AdviceParse, AdviceParseEnd, AdviceSwap, AdviceReplay, AdviceReplayEnd)


def _sig(S):
    Z = _widths(S.L)
    return (("bits", Array(Z.L, BIT)), ("length", Array(Z.LW, BIT))), BIT


def _body(B, S, bits, length):
    Z = _widths(S.L)
    K = _Consts(B)
    part = {d.name: bind(d, L=S.L) for d in PARTS}
    lay = _state_layout(Z)
    init = {name: K.bits(0, w) for name, w in lay}
    init["rem"] = K.bits(0, 3) + _refs(length) + K.bits(0, Z.REM - 3 - Z.LW)
    init["m"] = K.bits(1 << M["VER"], len(MODES))
    init["cnt"], init["fresh"], init["valid"] = K.bits(8, 6), K.bits(1, 1), K.bits(1, 1)
    init["utf"] = K.bits(1, len(UTF))
    state_t = Array(_size(lay), BIT)
    blocks = _coll(Array(Z.NB, Array(BLOCK, BIT)), _refs(bits))
    pos = _coll(Array(Z.NB, Array(Z.P, BIT)), [x for i in range(Z.NB) for x in K.bits(i, Z.P)])
    parsed = _refs(B.scan(part["AdviceParse"], _coll(state_t, _pack(lay, init)), [blocks, pos]))
    ns = _size(lay)
    parse_ok = _refs(B.call(part["AdviceParseEnd"], _coll(state_t, parsed[:ns]), length))
    ew = _size(_event_layout(Z))
    events = [parsed[ns + i * ew:ns + (i + 1) * ew] for i in range(Z.NB)]
    ev_t = Array(ew, BIT)
    for layer in _sort_net(Z.NB):
        a = _coll(Array(len(layer), ev_t), [x for i, _ in layer for x in events[i]])
        b = _coll(Array(len(layer), ev_t), [x for _, j in layer for x in events[j]])
        out = _refs(B.batch(part["AdviceSwap"], a, b, axes=(0, 0)))
        for t, (i, j) in enumerate(layer):
            events[i], events[j] = out[2 * t * ew:(2 * t + 1) * ew], out[(2 * t + 1) * ew:(2 * t + 2) * ew]
    rl = _replay_layout(Z)
    rinit = {name: K.bits(0, w) for name, w in rl}
    rinit["ok"] = K.bits(1, 1)
    sorted_events = _coll(Array(Z.NB, ev_t), [x for e in events for x in e])
    replay_t = Array(_size(rl), BIT)
    replayed = _refs(B.scan(part["AdviceReplay"], _coll(replay_t, _pack(rl, rinit)), [sorted_events]))
    replay_ok = _refs(B.call(part["AdviceReplayEnd"], _coll(replay_t, replayed[:_size(rl)])))
    return B.call(BL.and_gate(2), _coll(Array(2, BIT), parse_ok + replay_ok))


AdviceCheck = CompositeDefinition(
    "AdviceCheck", 1, ("L",), _sig, _body, conformance="exact-model-tested",
    doc="1 iff the first `length` bytes of `bits` (L bits at most) are an advice v0 string that `advice.decode` takes, "
        "up to the refusals `advice_check.DEFERRED` names (`advice_check.reference`)")


def unit(L: int):
    return bind(AdviceCheck, L=L)


def arguments(data: bytes, L: int) -> list[list[int]]:
    """The unit's arguments for `data`: its bits in stream order, zero beyond its length, and its length in bytes."""
    Z = _widths(L)
    if 8 * len(data) > L or len(data) >= 1 << Z.LW:
        raise ValueError(f"{len(data)} bytes don't fit L = {L} bits")
    bits = [(byte >> (7 - j)) & 1 for byte in data for j in range(8)]
    return [bits + [0] * (L - len(bits)), [(len(data) >> j) & 1 for j in range(Z.LW)]]


def evaluate(L: int, datas: Sequence[bytes]) -> list[int]:
    """The unit's output on each byte string, in one evaluation."""
    args = [list(rows) for rows in zip(*(arguments(d, L) for d in datas))]
    return [int(b) for b in evaluate_bits(unit(L), *args).reshape(-1).tolist()]


@lru_cache(maxsize=None)
def part_ands(L: int) -> dict[str, int]:
    """Each part's ANDs at L."""
    return {d.name: ands(bind(d, L=L)) for d in PARTS}


def size(L: int) -> dict[str, int]:
    """The unit's ANDs at L, by stage, from its parts' counts: no body is built, so any L is cheap."""
    Z, p = _widths(L), part_ands(L)
    out = {"parse": Z.NB * p["AdviceParse"] + p["AdviceParseEnd"], "sort": comparators(Z.NB) * p["AdviceSwap"],
           "replay": Z.NB * p["AdviceReplay"] + p["AdviceReplayEnd"], "final": 1}
    out["ands"] = sum(out.values())
    return out


__all__ = ["AdviceCheck", "AdviceParse", "AdviceParseEnd", "AdviceReplay", "AdviceReplayEnd", "AdviceSwap", "BLOCK",
           "DEFERRED", "PARTS", "arguments", "comparators", "decodes", "evaluate", "part_ands", "reference", "size",
           "sort_layers", "unit"]
