"""Cases for `pouw-check-difftest`'s lead functions (`PouwCheck.Lead`, `circuit/lead.py`'s window), from a
`random.Random`: an honest input, or one mutation aimed at a refusal or a reject. Each function returns the JSON its port
takes."""
from __future__ import annotations

import random

from verity.core.protocols.pouw.circuit import lead as L

RECORD_NAMES = ["short_root", "short_stream", "long_prover", "long_key_leaf", "no_calls", "too_many_calls",
                "negative_calls", "calls_not_int", "bad_hex"]


def record_instance(rng: random.Random, name: str) -> dict:
    x = {"weights_root": rng.randbytes(64).hex(), "stream": rng.randbytes(64).hex(), "prover": rng.randbytes(64).hex(),
         "key_leaf": rng.randbytes(64).hex(), "calls": rng.choice([1, rng.randrange(1, 300), L.WORDS_POSITION - 1])}
    if name == "short_root":
        x["weights_root"] = x["weights_root"][:-2]
    elif name == "short_stream":
        x["stream"] = x["stream"][:rng.choice([0, 64, 126])]
    elif name == "long_prover":
        x["prover"] += "00"
    elif name == "long_key_leaf":
        x["key_leaf"] += "00"
    elif name == "no_calls":
        x["calls"] = 0
    elif name == "too_many_calls":
        x["calls"] = L.WORDS_POSITION + rng.choice([0, 1, 1 << 70])
    elif name == "negative_calls":
        x["calls"] = -rng.randrange(1, 1 << 70)
    elif name == "calls_not_int":
        x["calls"] = rng.choice(["7", [7], True, None])
    elif name == "bad_hex":
        k = rng.choice(["weights_root", "stream", "prover", "key_leaf"])
        x[k] = "zz" + x[k][2:]
    return x


INSTANCE_NAMES = ["short_salt", "negative_position", "position_past_u64", "position_not_int", "value_not_hex"]


def instance_instance(rng: random.Random, name: str) -> dict:
    calls = rng.randrange(1, 100)
    position = rng.choice([L.a_position(rng.randrange(1000), calls, rng.randrange(calls)),
                           L.words_position(rng.randrange(1 << 20)), 0, L.WORDS_POSITION, (1 << 64) - 1])
    x = {"salt": rng.randbytes(64).hex(), "position": position, "schema": rng.choice([L.WORDS_SCHEMA, L.A_SCHEMA]),
         "values": [rng.randbytes(rng.choice([0, 1, 64, rng.randrange(1024)])).hex()
                    for _ in range(rng.choice([0, 1, 2, 3, 5, 8]))]}
    if name == "short_salt":
        x["salt"] = x["salt"][:rng.choice([0, 2, 126])]
    elif name == "negative_position":
        x["position"] = -rng.randrange(1, 1 << 70)
    elif name == "position_past_u64":
        x["position"] = (1 << 64) + rng.choice([0, 1, 1 << 70])
    elif name == "position_not_int":
        x["position"] = rng.choice(["0", [0], False, None])
    elif name == "value_not_hex":
        x["values"] = [*x["values"], rng.choice(["0g", "abc", 7])]
    return x


UNIT_NAMES = ["bad_bn", "k_zero", "k_past", "no_rows", "cols_past", "other_call_bad", "call_negative", "call_past",
              "off_grid", "row_past", "col_past", "tile_past_max", "block_past", "block_negative", "malformed_shape",
              "call_not_int"]


def _shape(rng: random.Random) -> list[int]:
    if rng.random() < 0.05:
        return [L.MAX_ROWS, L.MAX_COLS, L.MAX_K, rng.choice(L.BNS)]
    return [rng.choice([1, 5, 16, 17, 64, 65, 130, rng.randint(1, 300)]),
            rng.choice([8, 13, 128, 256, 300, 520, rng.randint(1, 2000)]),
            rng.choice([1, 21, 22, 49, 917, 2048, rng.randint(1, 5000)]), rng.choice(L.BNS)]


def unit_instance(rng: random.Random, name: str) -> dict:
    shapes = [_shape(rng) for _ in range(rng.randint(1, 4))]
    call = rng.randrange(len(shapes))
    sh = L.Shape(*shapes[call])
    x = {"shapes": shapes, "call": call, "R0": L.TM * rng.randrange(sh.m // L.TM),
         "C0": L.TN * rng.randrange(sh.n // L.TN), "BLK": rng.randrange(-(-sh.S // L.BLOCK))}
    if name == "bad_bn":
        shapes[call][3] = rng.choice([0, 64, 129, 512])
    elif name == "k_zero":
        shapes[call][2] = rng.choice([0, -21])
    elif name == "k_past":
        shapes[call][2] = L.MAX_K + rng.randint(1, 100)
    elif name == "no_rows":
        shapes[call][0] = rng.choice([0, -3])
    elif name == "cols_past":
        shapes[call][1] = L.MAX_COLS + 1
    elif name == "other_call_bad":
        shapes.append([16, 0, 21, 128])
    elif name == "call_negative":
        x["call"] = -rng.randint(1, len(shapes))
    elif name == "call_past":
        x["call"] = len(shapes) + rng.choice([0, 5])
    elif name == "off_grid":
        x[rng.choice(["R0", "C0"])] += rng.choice([1, 4]) * rng.choice([1, -1])
    elif name == "row_past":
        x["R0"] = sh.m + L.TM * rng.choice([0, 3])
    elif name == "col_past":
        x["C0"] = sh.n + L.TN * rng.choice([0, 3])
    elif name == "tile_past_max":
        x["R0"], x["C0"] = rng.choice([(L.MAX_ROWS, 0), (0, L.MAX_COLS)])
    elif name == "block_past":
        x["BLK"] = -(-sh.S // L.BLOCK) + rng.choice([0, 2])
    elif name == "block_negative":
        x["BLK"] = -rng.randint(1, 3)
    elif name == "malformed_shape":
        shapes[call] = rng.choice([shapes[call][:3], [*shapes[call][:3], "128"], {"rows": 1}])
    elif name == "call_not_int":
        x["call"] = rng.choice([str(call), [call], True, None])
    return x


EGRESS_NAMES = ["held", "over", "bad_shape", "negative_count", "bad_bucket", "release_not_int"]


def egress_instance(rng: random.Random, name: str) -> dict:
    shapes = [[rng.randint(1, 64), rng.choice([128, 4096, rng.randint(1, 20000)]),
               rng.choice([2048, 4096, rng.randint(1, 8192)]), rng.choice(L.BNS)] for _ in range(rng.randint(1, 4))]
    macs = sum(L.Shape(*s).macs for s in shapes)
    cap, fwd = rng.choice([0, 1, 32, rng.randint(1, 1000)]), max(1, macs // rng.choice([1, 2, 5, 64]))
    bits = cap * macs // fwd
    x = {"count": rng.choice([0, bits, bits + 1, rng.randint(0, 2 * bits + 2)]), "shapes": shapes, "cap": cap,
         "fwd_macs": fwd, "release": rng.choice([0, 1, 255, rng.randint(0, 1 << 30)]),
         "buckets": [[rng.choice([0, 0, 1, 3, 255, rng.randint(0, 1 << 20)]),
                      [rng.randint(0, 4096) for _ in range(rng.randint(0, 3))]] for _ in range(rng.randint(0, 6))]}
    if name == "held":
        x["buckets"] = [[0, []] for _ in range(rng.randint(1, 50))]
    elif name == "over":
        x["count"] = bits + rng.randint(1, 1000)
    elif name == "bad_shape":
        shapes[0][3] = 64
    elif name == "negative_count":
        x["count"] = -1
    elif name == "bad_bucket":
        x["buckets"] = [*x["buckets"], rng.choice([[-1, []], [1], [1, [-2]], [1, 2]])]
    elif name == "release_not_int":
        x["release"] = rng.choice(["1", [1], False, None])
    return x


WINDOW_NAMES = ["bad_record", "record_of_other_calls", "record_of_other_prover", "short_root", "no_calls", "a_before_salt",
                "second_salt", "overlap", "closed", "already", "no_position", "bad_shape", "words_not_open",
                "egress_refused", "still_open", "unknown_kind", "malformed_a", "malformed_words", "calls_not_int",
                "events_not_list"]


def _a(f: int, j: int, shape: list[int]) -> dict:
    return {"kind": "a_root", "forward": f, "call": j, "rows": shape[0], "cols": shape[1], "K": shape[2], "bn": shape[3]}


def _w(f: int, verdict: str = L.ACCEPTED) -> dict:
    return {"kind": "words_root", "forward": f, "verdict": verdict}


def window_instance(rng: random.Random, name: str) -> dict:
    """A window of 1 to 6 calls a forward and up to 4 forwards, each call's A root in a shuffled order, then the words."""
    calls = rng.choice([1, 2, 3, 6])
    root, stream, prover, leaf = (rng.randbytes(64) for _ in range(4))
    events: list[dict] = [{"kind": "salt"}]
    for f in range(rng.randint(0, 4)):
        order = list(range(calls))
        rng.shuffle(order)
        events += [_a(f, j, _shape(rng)) for j in order[:rng.randint(1, calls)]] + [_w(f)]
    x = {"record": L.lead_record(root, stream, prover, leaf, calls).hex(), "weights_root": root.hex(),
         "stream": stream.hex(), "prover": prover.hex(), "key_leaf": leaf.hex(), "calls": calls, "events": events}
    a = [k for k, e in enumerate(events) if e["kind"] == "a_root"]
    w = [k for k, e in enumerate(events) if e["kind"] == "words_root"]
    if not a and name not in ("bad_record", "record_of_other_calls", "record_of_other_prover", "short_root", "no_calls",
                              "second_salt", "unknown_kind", "calls_not_int", "events_not_list", "words_not_open"):
        events += [_a(0, 0, _shape(rng)), _w(0)]
        a, w = [len(events) - 2], [len(events) - 1]
    if name == "bad_record":
        x["record"] = rng.randbytes(64).hex()
    elif name == "record_of_other_calls":
        x["record"] = L.lead_record(root, stream, prover, leaf, calls + 1).hex()
    elif name == "record_of_other_prover":
        x["record"] = L.lead_record(root, stream, rng.randbytes(64), leaf, calls).hex()
    elif name == "short_root":
        k = rng.choice(["record", "weights_root", "stream", "prover", "key_leaf"])
        x[k] = rng.randbytes(rng.choice([0, 32, 63, 65])).hex()
    elif name == "no_calls":
        x["calls"] = rng.choice([0, -1, L.WORDS_POSITION])
    elif name == "a_before_salt":
        del events[0]
        events.insert(a[0], {"kind": "salt"})
    elif name == "second_salt":
        events.insert(rng.randint(1, len(events)), {"kind": "salt"})
    elif name == "overlap":
        f = events[a[0]]["forward"]
        events.insert(a[0] + 1, _a(f + 1, 0, _shape(rng)))
    elif name == "closed":
        e = events[a[0]]
        events.insert(w[0] + 1, _a(e["forward"], (e["call"] + 1) % calls, _shape(rng)))
    elif name == "already":
        events.insert(a[0] + 1, dict(events[a[0]]))
    elif name == "no_position":
        events[a[0]]["call"] = rng.choice([calls, -1])
    elif name == "bad_shape":
        k, bad = rng.choice([("rows", [0, L.MAX_ROWS + 1]), ("cols", [0, L.MAX_COLS + 1]), ("K", [0, L.MAX_K + 1]),
                             ("bn", [0, 64, 512])])
        events[a[0]][k] = rng.choice(bad)
    elif name == "words_not_open":
        events.insert(1, _w(rng.randint(0, 3)))
    elif name == "egress_refused":
        events[rng.choice(w)]["verdict"] = rng.choice(["refused: past its bits", "unheld", ""])
    elif name == "still_open":
        del events[w[-1]]
    elif name == "unknown_kind":
        events.insert(rng.randint(0, len(events)), rng.choice([{"kind": "tick"}, {"kind": 1}, {}, [], "salt", None]))
    elif name == "malformed_a":
        e = events[a[0]]
        k = rng.choice(["forward", "call", "rows", "cols", "K", "bn"])
        if rng.random() < 0.5:
            del e[k]
        else:
            e[k] = rng.choice([str(e[k]), [e[k]], True, None])
    elif name == "malformed_words":
        e = events[w[0]]
        e[rng.choice(["forward", "verdict"])] = rng.choice([None, "0", False, ["accepted"]])
    elif name == "calls_not_int":
        x["calls"] = rng.choice([str(calls), [calls], True, None])
    elif name == "events_not_list":
        x["events"] = rng.choice([{"0": {"kind": "salt"}}, "salt", None])
    return x
