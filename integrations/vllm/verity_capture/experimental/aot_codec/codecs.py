"""[R8 C-codec] Advice codecs A0-A3: `encode(phi, h) -> bits`, `decode(phi, bits) -> h`, canonical and deterministic.

Each codec is one *grammar* (a single function that both serialises and parses, driven by a Coder in encode or decode mode) over
one *backend*:

  A0  grammar_a0 + BitCoder      explicit typed fields, fixed widths (u8/u16/u32); the first exact upper bound
  A1  grammar_a1 + BitCoder      fields derivable from phi or from already-decoded facts removed; remaining fields at ceil(log2 |range|)
  A2  grammar_a2 + BitCoder      deterministic prediction (sched.py) + residuals: flags, zig-zag deltas, escapes to explicit steps
  A3  grammar_a2 + ArithCoder    the A2 symbol stream under a FIXED probability model fitted on training runs (model.py)

Bits are attributed to categories: framing | request-static | arrival | stop | membership | position | chunk | logits | escape.
"""
from __future__ import annotations

from typing import Any

from .bitio import ArithDecoder, ArithEncoder, BitReader, BitWriter, bits_for
from .facts import FINISH_REASONS, SCHEMA
from .sched import SchedState

U8, U16, U32 = 1 << 8, 1 << 16, 1 << 32


# ---------------------------------------------------------------- coders (backends)

class Coder:
    encoding: bool

    def flag(self, cat: str, key: str, value: int | None = None) -> int: ...
    def ufix(self, cat: str, card: int, value: int | None = None) -> int: ...
    def ugamma(self, cat: str, value: int | None = None) -> int: ...  # value >= 0
    def sym(self, cat: str, key: str, value: int | None = None) -> int: ...  # modelled small non-negative int
    def finish(self) -> list[int]: ...
    def marks(self) -> dict[str, float]: ...


class BitCoder(Coder):
    def __init__(self, encoding: bool, bits: list[int] | None = None) -> None:
        self.encoding = encoding
        self.w = BitWriter() if encoding else None
        self.r = BitReader(bits or []) if not encoding else None

    def flag(self, cat, key, value=None):
        if self.encoding:
            self.w.write(int(bool(value)), 1, cat)
            return int(bool(value))
        return self.r.read(1)

    def ufix(self, cat, card, value=None):
        n = bits_for(card)
        if self.encoding:
            if not (0 <= value < max(card, 1)):
                raise ValueError(f"{cat}: value {value} outside range {card}")
            self.w.write(value, n, cat)
            return value
        return self.r.read(n)

    def ugamma(self, cat, value=None):
        if self.encoding:
            self.w.gamma(value + 1, cat)
            return value
        return self.r.gamma() - 1

    def sym(self, cat, key, value=None):
        return self.ugamma(cat, value)

    def finish(self):
        return list(self.w.bits) if self.encoding else []

    def marks(self):
        return dict(self.w.marks) if self.encoding else {}

    def done(self) -> bool:
        return self.r.done() if not self.encoding else True


class ArithCoder(Coder):
    """Static-model arithmetic coding.  model: dict key -> {"cum": [...], "total": int, "escape": bool}; flags: key -> [n_true, n_false]."""

    def __init__(self, encoding: bool, model: dict, bits: list[int] | None = None) -> None:
        self.encoding = encoding
        self.model = model
        self.enc = ArithEncoder() if encoding else None
        self.dec = ArithDecoder(bits or []) if not encoding else None
        self.ideal_marks: dict[str, float] = {}

    def _code(self, cat: str, cum: list[int], total: int, value: int | None) -> int:
        if self.encoding:
            self.enc.encode(cum[value], cum[value + 1], total, cat)
            return value
        return self.dec.decode(cum, total)

    def flag(self, cat, key, value=None):
        n1, n0 = self.model["flags"][key]
        cum = [0, n0, n0 + n1]  # symbol 0 = false, 1 = true
        return self._code(cat, cum, n0 + n1, None if value is None else int(bool(value)))

    def ufix(self, cat, card, value=None):
        """Uniform over [0, card): exact log2(card) bits, no table (card may be 2**32 for explicit positions)."""
        if card <= 1:
            return (value or 0) if self.encoding else 0
        if self.encoding:
            if not (0 <= value < card):
                raise ValueError(f"{cat}: value {value} outside range {card}")
            self.enc.encode(value, value + 1, card, cat)
            return value
        return self.dec.decode_uniform(card)

    def ugamma(self, cat, value=None):
        """Elias-gamma emulated with p=1/2 binary decisions (same length as the bit code, no model needed)."""
        if self.encoding:
            n = value + 1
            L = n.bit_length()
            for _ in range(L - 1):
                self._code(cat, [0, 1, 2], 2, 0)
            self._code(cat, [0, 1, 2], 2, 1)
            for i in range(L - 2, -1, -1):
                self._code(cat, [0, 1, 2], 2, (n >> i) & 1)
            return value
        z = 0
        while self._code(cat, [0, 1, 2], 2, None) == 0:
            z += 1
        n = 1
        for _ in range(z):
            n = (n << 1) | self._code(cat, [0, 1, 2], 2, None)
        return n - 1

    def sym(self, cat, key, value=None):
        m = self.model["hist"][key]
        cum, total, K = m["cum"], m["total"], len(m["cum"]) - 2  # last symbol = escape
        if self.encoding:
            if value < K:
                self._code(cat, cum, total, value)
            else:
                self._code(cat, cum, total, K)
                self.ugamma(cat, value - K)
            return value
        s = self._code(cat, cum, total, None)
        return s if s < K else K + self.ugamma(cat, None)

    def finish(self):
        return self.enc.finish() if self.encoding else []

    def marks(self):
        return dict(self.enc.marks) if self.encoding else {}


# ---------------------------------------------------------------- helpers

def _has(phi: dict, field: str) -> bool:
    return bool(phi["requests"]) and all(field in q for q in phi["requests"])


def _new_h(phi: dict) -> dict:
    return {"schema": SCHEMA, "requests": [{"prompt_len": q["prompt_len"], "max_tokens": q["max_tokens"]} for q in phi["requests"]],
            "steps": [], "logits": []}


def _fr_index(fr: str) -> int:
    return FINISH_REASONS.index(fr) if fr in FINISH_REASONS else FINISH_REASONS.index("other")


# ---------------------------------------------------------------- A0: explicit

def grammar_a0(c: Coder, phi: dict, h: dict | None) -> dict:
    enc = c.encoding
    out = _new_h(phi)
    N = c.ufix("framing", U16, len(h["requests"]) if enc else None)
    T = c.ufix("framing", U32, len(h["steps"]) if enc else None)
    if enc and N != len(phi["requests"]):
        raise ValueError("hint/phi request count mismatch")
    out["requests"] = []
    for i in range(N):
        q = h["requests"][i] if enc else None
        L = c.ufix("request-static", U32, q["prompt_len"] if enc else None)
        M = c.ufix("request-static", U32, q["max_tokens"] if enc else None)
        a = c.ufix("arrival", U32, q["arrive_step"] if enc else None)
        g = c.ufix("stop", U32, q["n_generated"] if enc else None)
        fr = c.ufix("stop", U8, _fr_index(q["finish_reason"]) if enc else None)
        out["requests"].append({"prompt_len": L, "max_tokens": M, "arrive_step": a, "n_generated": g, "finish_reason": FINISH_REASONS[fr]})
    for t in range(T):
        rows_in = h["steps"][t]["rows"] if enc else None
        n_rows = c.ufix("membership", U16, len(rows_in) if enc else None)
        rows = []
        for k in range(n_rows):
            i = c.ufix("membership", U16, rows_in[k][0] if enc else None)
            nc = c.ufix("position", U32, rows_in[k][1] if enc else None)
            n = c.ufix("chunk", U32, rows_in[k][2] if enc else None)
            rows.append([i, nc, n])
        lg_in = h["logits"][t] if enc else None
        n_lg = c.ufix("logits", U16, len(lg_in) if enc else None)
        lg = []
        for k in range(n_lg):
            i = c.ufix("logits", U16, lg_in[k][0] if enc else None)
            f = c.ufix("logits", U8, lg_in[k][1] if enc else None)
            lg.append([i, f])
        out["steps"].append({"rows": rows})
        out["logits"].append(lg)
    return out


# ---------------------------------------------------------------- shared explicit-step sub-grammars (A1 escape payloads, A2 escapes)

def _rule_logits(rows: list[list[int]], st: SchedState) -> list[list[int]]:
    """is_final = the row reaches the end of the request's known sequence (prompt + tokens sampled so far, per the state)."""
    return [[i, 1 if c + n >= st.L[i] + st.sampled[i] else 0] for i, c, n in rows]


def _explicit_step(c: Coder, phi: dict, st: SchedState, rows_in, lg_in, *, nc_explicit: bool, per_row_nc_flag: bool,
                   logits_explicit_flag: bool) -> tuple[list[list[int]], list[list[int]]]:
    """Explicit step: membership + (position if needed) + chunk sizes; logits by rule unless flagged."""
    enc = c.encoding
    N = len(phi["requests"])
    max_rows = min(int(phi["engine"]["max_num_seqs"]), N)
    n_rows = c.ufix("membership", max_rows + 1, len(rows_in) if enc else None)
    rows = []
    for k in range(n_rows):
        i = c.ufix("membership", N, rows_in[k][0] if enc else None)
        exp_nc = st.expected_nc(i)
        if per_row_nc_flag:
            same = c.flag("position", "nc_eq", (rows_in[k][1] == exp_nc) if enc else None)
            nc = exp_nc if same else c.ufix("position", U32, rows_in[k][1] if enc else None)
        elif nc_explicit:
            nc = c.ufix("position", U32, rows_in[k][1] if enc else None)
        else:
            nc = exp_nc
        # chunk alphabet [1, R]: R = tokens the request is known to hold (prompt + tokens sampled so far) minus its position -- a prefill
        # chunk (R = remaining prompt), a decode row (R = 1, zero bits) and a post-preemption re-prefill of prompt + generated tokens
        # (nc = 0, R = L + sampled; MEASURED gen/preempt4) are all inside it; a row scheduling more is not expressible (raises).
        R = max(st.L[i] + st.sampled[i] - nc, 1)
        n = 1 + c.ufix("chunk", R, (rows_in[k][2] - 1) if enc else None)
        rows.append([i, nc, n])
    rule = _rule_logits(rows, st)
    if logits_explicit_flag:
        same = c.flag("logits", "logits_eq", (lg_in == rule) if enc else None)
        if same:
            return rows, rule
        n_lg = c.ufix("logits", n_rows + 1, len(lg_in) if enc else None)
        lg = []
        for k in range(n_lg):
            i = c.ufix("logits", N, lg_in[k][0] if enc else None)
            f = c.flag("logits", "is_final", lg_in[k][1] if enc else None)
            lg.append([i, f])
        return rows, lg
    return rows, rule


# ---------------------------------------------------------------- A1: remove derivable

def grammar_a1(c: Coder, phi: dict, h: dict | None) -> dict:
    enc = c.encoding
    out = _new_h(phi)
    N = len(phi["requests"])
    E = phi.get("expected_engine_steps")
    if E is not None:  # phi2: declared step bound -> fixed width, with an escape for T > E
        within = c.flag("framing", "T_le_E", (len(h["steps"]) <= E) if enc else None)
        T = c.ufix("framing", E + 1, len(h["steps"]) if enc else None) if within else c.ugamma("framing", len(h["steps"]) if enc else None)
    else:
        T = c.ugamma("framing", len(h["steps"]) if enc else None)
    reqs = []
    for i, q in enumerate(phi["requests"]):
        hq = h["requests"][i] if enc else None
        a = q["arrive_step"] if "arrive_step" in q else c.ufix("arrival", max(T, 1), hq["arrive_step"] if enc else None)
        M = q["max_tokens"]
        g = q["n_generated"] if "n_generated" in q else c.ufix("stop", M + 1, hq["n_generated"] if enc else None)
        fr = "length" if g == M else FINISH_REASONS[c.ufix("stop", len(FINISH_REASONS), _fr_index(hq["finish_reason"]) if enc else None)]
        if enc and g == M and hq["finish_reason"] != "length":
            raise ValueError("finish_reason != length with n_generated == max_tokens is not expressible")
        reqs.append({"prompt_len": q["prompt_len"], "max_tokens": M, "arrive_step": a, "n_generated": g, "finish_reason": fr})
    out["requests"] = reqs
    st = SchedState(phi, reqs)
    # run-level flags for the two rules whose violation would need explicit fields (positions / logits)
    if enc:
        rule_nc = _positions_follow_state(h, SchedState(phi, reqs))
        rule_lg = _logits_follow_rule(h, SchedState(phi, reqs))
    nc_ok = c.flag("escape", "nc_rule", rule_nc if enc else None)
    lg_ok = c.flag("escape", "logits_rule", rule_lg if enc else None)
    for t in range(T):
        rows_in = h["steps"][t]["rows"] if enc else None
        lg_in = h["logits"][t] if enc else None
        rows, rule = _explicit_step(c, phi, st, rows_in, lg_in, nc_explicit=not nc_ok, per_row_nc_flag=False, logits_explicit_flag=False)
        if lg_ok:
            lg = rule
        else:
            n_lg = c.ufix("logits", len(rows) + 1, len(lg_in) if enc else None)
            lg = [[c.ufix("logits", N, lg_in[k][0] if enc else None), c.flag("logits", "is_final", lg_in[k][1] if enc else None)] for k in range(n_lg)]
        out["steps"].append({"rows": rows})
        out["logits"].append(lg)
        st.apply(rows)
    return out


def _logits_follow_rule(h: dict, st: SchedState) -> bool:
    for t, s in enumerate(h["steps"]):
        if h["logits"][t] != _rule_logits(s["rows"], st):
            return False
        st.apply(s["rows"])
    return True


def _positions_follow_state(h: dict, st: SchedState) -> bool:
    """Every row's num_computed equals the state's expectation (its own progress; a derivable prefix-cache hit at admission)."""
    for s in h["steps"]:
        for i, c, n in s["rows"]:
            if st.expected_nc(i) != c:
                return False
        st.apply(s["rows"])
    return True


# ---------------------------------------------------------------- A2: predict + residual   (A3 = same grammar, ArithCoder)

def _zigzag(v: int) -> int:
    return (v << 1) if v >= 0 else ((-v << 1) - 1)


def _unzigzag(z: int) -> int:
    return (z >> 1) if (z & 1) == 0 else -((z + 1) >> 1)


def _a2_facts(phi: dict, h: dict) -> dict:
    """Encoder-side facts behind the hierarchical flags: is the run 'clean' (everything the predictor says, given phi + arrivals)?"""
    fr_rule_ok = all((hq["finish_reason"] == "length") == (hq["n_generated"] == q["max_tokens"]) and hq["finish_reason"] in ("length", "stop")
                     for q, hq in zip(phi["requests"], h["requests"]))
    ngen_ok = all(("n_generated" in q) or hq["n_generated"] == q["max_tokens"] for q, hq in zip(phi["requests"], h["requests"]))
    sim = SchedState(phi, h["requests"])
    T_pred = sim.predict_n_steps()
    T = len(h["steps"])
    matches = []
    sim = SchedState(phi, h["requests"])
    for t in range(T):
        pr, pl = sim.predict()
        matches.append(pr == h["steps"][t]["rows"] and pl == h["logits"][t])
        sim.apply(h["steps"][t]["rows"])
    return {"fr_rule_ok": fr_rule_ok, "ngen_ok": ngen_ok, "T_pred": T_pred, "T_eq": T == T_pred, "matches": matches,
            "clean": fr_rule_ok and ngen_ok and T == T_pred and all(matches)}


def grammar_a2(c: Coder, phi: dict, h: dict | None) -> dict:
    """Hierarchical: one run-level `clean` flag (every request ran to max_tokens unless phi says otherwise, finish rule holds, T and every
    step as predicted) -> 1 bit + arrivals; otherwise the per-fact flags/escapes below."""
    enc = c.encoding
    out = _new_h(phi)
    N = len(phi["requests"])
    f = _a2_facts(phi, h) if enc else None
    clean = c.flag("escape", "clean", f["clean"] if enc else None)
    reqs = []
    prev_a = 0
    # finish_reason rule: "length" iff n_generated == max_tokens else "stop"; one run-level flag, explicit per request only on violation
    fr_rule = 1 if clean else c.flag("stop", "fr_rule", f["fr_rule_ok"] if enc else None)
    for i, q in enumerate(phi["requests"]):
        hq = h["requests"][i] if enc else None
        if "arrive_step" in q:
            a = q["arrive_step"]
        else:  # delta to the previous request's arrival (submission order), zig-zag, modelled
            z = c.sym("arrival", "arrive_delta", _zigzag(hq["arrive_step"] - prev_a) if enc else None)
            a = prev_a + _unzigzag(z)
        prev_a = a
        M = q["max_tokens"]
        if "n_generated" in q:
            g = q["n_generated"]
        elif clean:
            g = M
        else:
            eq = c.flag("stop", "ngen_eq_max", (hq["n_generated"] == M) if enc else None)
            g = M if eq else c.ufix("stop", M + 1, hq["n_generated"] if enc else None)
        if g == M:
            fr = "length"
        elif fr_rule:
            fr = "stop"
        else:
            fr = FINISH_REASONS[c.ufix("stop", len(FINISH_REASONS), _fr_index(hq["finish_reason"]) if enc else None)]
        if enc and g == M and hq["finish_reason"] != "length":
            raise ValueError("finish_reason != length with n_generated == max_tokens is not expressible")
        reqs.append({"prompt_len": q["prompt_len"], "max_tokens": M, "arrive_step": a, "n_generated": g, "finish_reason": fr})
    out["requests"] = reqs
    st = SchedState(phi, reqs)
    T_pred = st.predict_n_steps()
    if clean:
        T, all_ok = T_pred, 1
    else:
        T_eq = c.flag("framing", "T_eq", f["T_eq"] if enc else None)
        T = T_pred if T_eq else c.ugamma("framing", len(h["steps"]) if enc else None)
        all_ok = c.flag("escape", "all_steps_match", all(f["matches"]) if enc else None)
    matches = f["matches"] if enc else None
    # escaped steps are addressed by *gaps*: the number of predicted steps before the next escape (gap == T - t: no further escape),
    # modelled symbol `escape_gap` (histogram + gamma escape) -- k escapes cost k + 1 small integers, not one flag per step (the corpus
    # never escapes, so A3 had no evidence on a per-step flag and paid 1 bit per step of an escaping run: gen/pp4, 157 steps, 1 escape)
    nxt = T if all_ok else None  # index of the next escaped step
    for t in range(T):
        if nxt is None:
            gap = c.sym("escape", "escape_gap", (next((u for u in range(t, T) if not matches[u]), T) - t) if enc else None)
            nxt = t + gap
        pr, pl = st.predict()
        if t < nxt:
            rows, lg = pr, pl
        else:
            rows, lg = _explicit_step(c, phi, st, h["steps"][t]["rows"] if enc else None, h["logits"][t] if enc else None,
                                      nc_explicit=False, per_row_nc_flag=True, logits_explicit_flag=True)
            nxt = None
        out["steps"].append({"rows": rows})
        out["logits"].append(lg)
        st.apply(rows)
    return out


# ---------------------------------------------------------------- public API

class Result:
    def __init__(self, name: str, bits: list[int], marks: dict[str, float], ideal_bits: float | None = None) -> None:
        self.name, self.bits, self.marks, self.ideal_bits = name, bits, marks, ideal_bits

    @property
    def n_bits(self) -> int:
        return len(self.bits)


def encode(name: str, phi: dict, h: dict, model: dict | None = None) -> Result:
    if name == "A0":
        c = BitCoder(True); grammar_a0(c, phi, h)
    elif name == "A1":
        c = BitCoder(True); grammar_a1(c, phi, h)
    elif name == "A2":
        c = BitCoder(True); grammar_a2(c, phi, h)
    elif name == "A3":
        c = ArithCoder(True, model); grammar_a2(c, phi, h)
    else:
        raise KeyError(name)
    bits = c.finish()
    ideal = c.enc.ideal_bits if isinstance(c, ArithCoder) else None
    return Result(name, bits, c.marks(), ideal)


def decode(name: str, phi: dict, bits: list[int], model: dict | None = None) -> dict:
    if name == "A0":
        c = BitCoder(False, bits); h = grammar_a0(c, phi, None)
    elif name == "A1":
        c = BitCoder(False, bits); h = grammar_a1(c, phi, None)
    elif name == "A2":
        c = BitCoder(False, bits); h = grammar_a2(c, phi, None)
    elif name == "A3":
        c = ArithCoder(False, model, bits); h = grammar_a2(c, phi, None)
    else:
        raise KeyError(name)
    if isinstance(c, BitCoder) and not c.done():
        raise ValueError(f"{name}: {len(bits) - c.r.pos} trailing bits not consumed")
    return h


CODECS = ["A0", "A1", "A2", "A3"]
