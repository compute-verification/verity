"""The pass and draw difftest's cases: `core_check.pass_epoch`'s arguments, ``{"coins", "raw": hex | null, "context",
"recorded"}``, and `core_check.draw`'s, ``{"coins", "recorded", "opening", "stream", "registration", "receipt", "k",
"population", "units"}``.

A pass case is an honest pass (a small workload, its weights' roots, an epoch's coins committed under one of
`core_check.COIN_VERSIONS` and its salt coin opened after the weights' root, or a recorded pass salted by a beacon), then
one mutation (`PASS_MUTATIONS`) aimed at one refusal of `pass_epoch`.  A draw case is an honest draw (the draw coin opened
after a registration's digest, the units `stream_subset` of its key's stream), then one mutation (`DRAW_MUTATIONS`).
`tests/test_difftest.py` draws them for the harness, and `verity/core/protocols/pouw/tests/test_pouw_core_check.py`
holds each mutation to the refusal it aims at.
"""
from __future__ import annotations

import hashlib
import json
import random

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw.audit import EPOCH_SALT_DOMAIN, SALT_DOMAIN
from verity.core.protocols.pouw.protocol import tagged
from verity.ml.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.ml.randomness import derive


class Coins:
    """An epoch's challenger under a coin scheme of `core_check.COIN_VERSIONS` (`coins.Challenger`'s, any version)."""

    def __init__(self, rng: random.Random, epoch: int, scheme: str = Co.SCHEME, early: bool = False) -> None:
        keep = CC.COIN_VERSIONS[scheme][0]
        binding = CC.tagged_sha256(scheme, epoch) if early else tagged(scheme, epoch)[:keep]
        dom = CommitmentDomain(binding, -1, RangeIndexedDomain(len(Co.LABELS)), hash="sha512")
        self.early = early
        self.coins = {label: (rng.randbytes(Co.COIN_BYTES), rng.randbytes(Co.HM96.salt_bytes)) for label in Co.LABELS}
        self.tree = MerkleTree(dom, {i: Co.leaf_value(*self.coins[label]) for i, label in enumerate(Co.LABELS)},
                               lambda _: scheme)
        self.scheme, self.epoch = scheme, epoch
        self.commitment = {"scheme": scheme, "epoch": epoch, "root": self.tree.commitment.root.hex()}

    def open(self, label: str, after: bytes) -> dict:
        coin, salt = self.coins[label]
        path = self.tree.open(Co.LABELS.index(label)).path
        return {"epoch": self.epoch, "label": label, "after": after.hex(), "coin": coin.hex(), "salt": salt.hex(),
                "path": [p.hex() for p in path]}


def _workload(rng: random.Random) -> tuple[list[dict], dict[str, dict]]:
    dims = {w: (rng.randint(1, 9), rng.randint(1, 9)) for w in rng.sample(range(1 << 20), rng.randint(1, 3))}
    us = [{"index": i, "m": rng.randint(1, 9), "k": dims[w][1], "n": dims[w][0], "weight": w, "real_rows": 1}
          for i, w in enumerate([*dims, *rng.choices(list(dims), k=rng.randint(0, 3))])]
    rng.shuffle(us)
    weights = {str(w): {"n": n, "k": k, "root": rng.randbytes(64).hex()} for w, (n, k) in dims.items()}
    return us, weights


def _roots(man: dict) -> dict[int, bytes]:
    return {int(w): bytes.fromhex(v["root"]) for w, v in man["weights"].items()}


def _raw(man: dict, rng: random.Random) -> bytes:
    return json.dumps(man, indent=rng.choice([None, 1]), sort_keys=rng.random() < 0.5).encode()


def honest_pass(rng: random.Random, kind: str | None = None) -> dict:
    """An honest pass: ``coin`` (the current coins), ``recorded-coin`` (the e2e-service passes' coins, in a replay) or
    ``beacon`` (a recorded pass's), as a case with its manifest and the challenger beside it."""
    kind = kind or rng.choice(["coin", "coin", "recorded-coin", "beacon"])
    us, weights = _workload(rng)
    man = {"schema": CC.MANIFEST_SCHEMA, "matmuls": us, "weights": weights, "run": f"r-{rng.randrange(1 << 30):x}"}
    roots = _roots(man)
    if kind == "beacon":
        beacon = rng.randbytes(32)
        salt = derive(beacon, SALT_DOMAIN, {"weights": CC.weights_root(Co.SCHEME, roots)}).stream("salt", 32)
        man.update(schema=CC.MANIFEST_RECORDED, beacon=beacon.hex(), salt=salt.hex())
        ch = Coins(rng, rng.randrange(4))
    else:
        ch = Coins(rng, rng.randrange(4), Co.SCHEME if kind == "coin" else CC.COINS_RECORDED)
        root = CC.weights_root(ch.scheme, roots)
        opening = ch.open("salt", root)
        salt = derive(bytes.fromhex(opening["coin"]), EPOCH_SALT_DOMAIN, {"epoch": ch.epoch, "coins": ch.tree.commitment.root,
                                                                          "weights": root}).stream("salt", 32)
        man.update(coins=ch.commitment, salt_opening=opening, salt=salt.hex())
    case = {"coins": ch.commitment, "recorded": kind != "coin" or rng.random() < 0.3, "run": man["run"]}
    return {"case": case, "man": man, "ch": ch, "kind": kind}


def _flip(h: str) -> str:
    return ("1" if h[0] == "0" else "0") + h[1:]


def _resign(h: dict, rng: random.Random) -> None:
    """The manifest's salt again from its coin or beacon, after a mutation of its workload."""
    man, ch = h["man"], h["ch"]
    roots = _roots(man)
    if "beacon" in man:
        man["salt"] = derive(bytes.fromhex(man["beacon"]), SALT_DOMAIN, {"weights": CC.weights_root(Co.SCHEME, roots)}
                             ).stream("salt", 32).hex()
    elif "salt_opening" in man:
        root = CC.weights_root(ch.scheme, roots)
        man["salt_opening"] = ch.open("salt", root)
        man["salt"] = derive(bytes.fromhex(man["salt_opening"]["coin"]), EPOCH_SALT_DOMAIN,
                             {"epoch": ch.epoch, "coins": ch.tree.commitment.root, "weights": root}).stream("salt", 32).hex()


def _mutate_pass(h: dict, name: str, rng: random.Random) -> bytes | None:
    """Apply mutation `name` to the honest pass `h`; the manifest's bytes, unless the mutation writes its own."""
    man, case = h["man"], h["case"]
    us = man["matmuls"]
    o = man.get("salt_opening")
    match name:
        case "no-pass":
            return None
        case "not-utf8":
            return b"\xff" + _raw(man, rng)
        case "not-json":
            return _raw(man, rng)[:-1]
        case "not-object":
            return json.dumps([man]).encode()
        case "constant":
            return _raw(man, rng)[:-1] + b', "nan": NaN}'
        case "recorded-unread":
            case["recorded"] = False
        case "schema-other":
            man["schema"] = rng.choice(["verity-vllm/pouw-pearl-c-run/v0", "it's", ""])
        case "schema-missing":
            man.pop("schema") if rng.random() < 0.5 else man.update(schema=rng.choice([2, None, ["v2"]]))
        case "matmuls-form":
            u = rng.choice(us)
            k = rng.choice(["index", "m", "k", "n", "weight"])
            u.pop(k) if rng.random() < 0.3 else u.update({k: rng.choice([True, "3", 2.5, None])})
        case "matmuls-not-list":
            man["matmuls"] = rng.choice([{}, None, "x"])
        case "shape":
            rng.choice(us)[rng.choice("mkn")] = rng.choice([0, -3])
        case "ids-64-bit":
            rng.choice(us)[rng.choice(["index", "weight"])] = rng.choice([1 << 64, -1, (1 << 70) + 5])
        case "weights-form":
            w = rng.choice(list(man["weights"]))
            v = man["weights"].pop(w)
            choice = rng.randrange(4)
            if choice == 0:
                man["weights"]["0" + w] = v
            elif choice == 1:
                man["weights"][w] = {**v, "root": "zz" + v["root"]}
            elif choice == 2:
                man["weights"][w] = v["root"]
            else:
                man["weights"][f"w{w}"] = v
        case "weights-not-object":
            man["weights"] = rng.choice([[], None])
        case "no-salt-coin":
            man.pop("salt_opening")
        case "no-verifier-coins":
            case["coins"] = None
        case "verifier-coins":
            case["coins"] = {**case["coins"], "epoch": -1}
        case "opening-form":
            k = rng.choice(["epoch", "label", "after", "coin", "salt", "path"])
            bad = {"epoch": "0", "label": 0, "after": "xy", "coin": None, "salt": 3, "path": "ab"}[k]
            man["salt_opening"] = {**o, k: bad}
            if rng.random() < 0.3:
                man["salt_opening"] = {**o, "path": [p[:64] for p in o["path"]]}
        case "pass-coins-form":
            c = man["coins"]
            man["coins"] = rng.choice([{**c, "scheme": "verity/pouw/epoch-coins/v0"}, {**c, "epoch": -2},
                                       {**c, "root": c["root"][:64]}, {**c, "root": 7}, None])
        case "recorded-scheme":
            case["recorded"] = False
        case "not-the-epochs":
            other = Coins(rng, h["ch"].epoch, h["ch"].scheme).commitment
            man["coins"] = other
        case "indices":
            if len(us) < 2:
                us.append(dict(us[0]))
            us[1]["index"] = us[0]["index"]
        case "dims":
            u = rng.choice(us)
            us.append({**u, "index": max(x["index"] for x in us) + 1, "k": u["k"] + 1})
        case "weights-set":
            if rng.random() < 0.5:
                man["weights"][str((1 << 21) + rng.randrange(9))] = {"root": rng.randbytes(64).hex()}
            else:
                man["weights"].pop(rng.choice(list(man["weights"])))
            _resign(h, rng)
        case "coin-epoch":
            man["salt_opening"] = {**o, "epoch": o["epoch"] + rng.choice([1, -9])}
        case "coin-label":
            man["salt_opening"] = {**o, "label": rng.choice(["draw", "it's", "sa\u00a0lt"])}
        case "coin-size":
            man["salt_opening"] = {**o, rng.choice(["coin", "salt"]): rng.randbytes(rng.choice([31, 33, 191])).hex()}
        case "coin-foreign":
            other = Coins(rng, h["ch"].epoch, h["ch"].scheme)
            man["salt_opening"] = other.open("salt", bytes.fromhex(o["after"]))
        case "coin-tampered":
            man["salt_opening"] = {**o, "coin": _flip(o["coin"])}
        case "coin-path":
            man["salt_opening"] = {**o, "path": [_flip(o["path"][0])]}
        case "coin-early":
            man["salt_opening"] = {**o, "after": rng.randbytes(len(bytes.fromhex(o["after"]))).hex()}
        case "resalted":
            man["salt"] = _flip(man["salt"])
        case "salt-missing":
            man.pop("salt")
        case "beacon-empty":
            man["beacon"] = rng.choice(["", "zz", 5])
        case "no-beacon":
            man.pop("beacon")
        case "not-named":
            raw = _raw(man, rng)
            case["context"] = f"served-zk/{case['run']}/{hashlib.sha512(raw + b' ').hexdigest()}"
            return raw
        case "spaced-hex":
            w = rng.choice(list(man["weights"]))
            r = man["weights"][w]["root"]
            man["weights"][w]["root"] = r[:2] + " " + r[2:].upper()
            if o is not None:
                man["salt_opening"] = {**o, "coin": " ".join(o["coin"][i:i + 2] for i in range(0, 64, 2))}
        case _:
            raise ValueError(name)
    return _raw(man, rng)


#: (mutation, the honest kinds it applies to)
PASS_MUTATIONS = {
    "no-pass": None, "not-utf8": None, "not-json": None, "not-object": None, "constant": None,
    "recorded-unread": ("beacon",), "schema-other": None, "schema-missing": None, "matmuls-form": None,
    "matmuls-not-list": None, "shape": None, "ids-64-bit": None, "weights-form": None, "weights-not-object": None,
    "no-salt-coin": ("coin", "recorded-coin"), "no-verifier-coins": ("coin", "recorded-coin"),
    "verifier-coins": ("coin", "recorded-coin"), "opening-form": ("coin", "recorded-coin"),
    "pass-coins-form": ("coin", "recorded-coin"), "recorded-scheme": ("recorded-coin",),
    "not-the-epochs": ("coin", "recorded-coin"), "indices": None, "dims": None, "weights-set": None,
    "coin-epoch": ("coin", "recorded-coin"), "coin-label": ("coin", "recorded-coin"), "coin-size": ("coin", "recorded-coin"),
    "coin-foreign": ("coin", "recorded-coin"), "coin-tampered": ("coin", "recorded-coin"),
    "coin-path": ("coin", "recorded-coin"), "coin-early": ("coin", "recorded-coin"), "resalted": None, "salt-missing": None,
    "beacon-empty": ("beacon",), "no-beacon": ("beacon",), "not-named": None, "spaced-hex": None,
}
PASS_NAMES = list(PASS_MUTATIONS)


def pass_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `pass_epoch` case: an honest pass, then mutation `name` (on a pass of a kind it applies to)."""
    kinds = PASS_MUTATIONS.get(name)
    h = honest_pass(rng, rng.choice(kinds) if kinds else None)
    case = h["case"]
    raw = _raw(h["man"], rng) if name == "honest" else _mutate_pass(h, name, rng)
    if raw is None:
        context = f"served-zk/{case['run']}/{'0' * 128}"
    else:
        context = case.get("context") or f"served-zk/{case['run']}/{hashlib.sha512(raw).hexdigest()}"
    return {"coins": case["coins"], "raw": None if raw is None else raw.hex(), "context": context,
            "recorded": case["recorded"]}


def pass_args(case: dict) -> tuple:
    return (case["coins"], None if case["raw"] is None else bytes.fromhex(case["raw"]), case["context"], case["recorded"])


def _receipt(rng: random.Random, reg: str) -> dict:
    return {"format": "verity/registration-receipt/v1", "received": str(rng.randrange(9)), "registration": reg,
            "verifier": f"served-zk/challenger/{rng.choice(['window', 'ctl-x'])}"}


def honest_draw(rng: random.Random, scheme: str | None = None, many: bool = False, early: bool | None = None
                ) -> tuple[dict, Coins]:
    """An honest draw under `scheme` (any version by default, `CC.COIN_EARLY`'s binding included, `early`);
    `many` draws at least one unit of at least 50."""
    scheme = scheme or rng.choice([Co.SCHEME, Co.SCHEME, CC.COINS_RECORDED, Co.SCHEME])
    early = (scheme == Co.SCHEME and rng.random() < 0.3) if early is None else early
    ch = Coins(rng, rng.randrange(4), scheme, early)
    reg, stream = rng.randbytes(64).hex(), rng.randbytes(64).hex()
    receipt = _receipt(rng, reg)
    n = rng.choice([50, rng.randint(50, 5000), rng.randint(50, 1 << 40)] + ([] if many else [1, 2, 7]))
    k = rng.randint(1 if many else 0, min(n, 12))
    opening = ch.open("draw", bytes.fromhex(reg))
    ctx, size = CC.draw_key_context(stream, reg, receipt), 2 * k * CC.width(n) + 64
    coin = bytes.fromhex(opening["coin"])
    src = CC.stream_v1(CC.derive_v1(coin, CC.DRAW_DOMAIN, ctx), "draw", size) if ch.early else \
        derive(coin, CC.DRAW_DOMAIN, ctx).stream("draw", size)
    units = CC.stream_subset(k, n, src)
    return {"coins": ch.commitment, "recorded": ch.scheme != Co.SCHEME or ch.early or rng.random() < 0.3, "opening": opening,
            "stream": stream, "registration": reg, "receipt": receipt, "k": k, "population": n, "units": units}, ch


def _mutate_draw(d: dict, ch: Coins, name: str, rng: random.Random) -> None:
    o = d["opening"]
    match name:
        case "verifier-coins":
            d["coins"] = rng.choice([None, {**d["coins"], "root": "00"}, {**d["coins"], "scheme": "x"}])
        case "recorded-scheme" | "early-coins-live":
            d["recorded"] = False
        case "registration":
            d["registration"] = rng.choice([d["registration"].upper(), d["registration"][:-2], None, 5])
        case "stream":
            d["stream"] = rng.choice([None, 3, ["x"]])
        case "receipt":
            k = rng.choice(list(d["receipt"]))
            d["receipt"] = rng.choice([{**d["receipt"], k: rng.choice(['a"b', "a\\b", "\u00e9", 5, "\n"])}, [d["receipt"]], None])
        case "law":
            if rng.random() < 0.5:
                d["k"] = d["population"] + rng.randint(1, 3)
            else:
                d.update(population=rng.choice([0, -1, True, "4"]), k=rng.choice([0, 1]))
        case "opening-form":
            k = rng.choice(["coin", "path", "label"])
            d["opening"] = {**o, k: rng.choice([None, 4] + (["zz"] if k != "label" else []))}
        case "label":
            d["opening"] = ch.open("salt", bytes.fromhex(d["registration"]))
        case "epoch":
            d["opening"] = {**o, "epoch": o["epoch"] + 1}
        case "foreign":
            d["opening"] = Coins(rng, ch.epoch, ch.scheme, ch.early).open("draw", bytes.fromhex(d["registration"]))
        case "tampered":
            d["opening"] = {**o, "salt": _flip(o["salt"])}
        case "early":
            d["opening"] = ch.open("draw", rng.randbytes(64))
        case "units":
            u = d["units"]
            d["units"] = rng.choice([u[:-1], u + [0], list(reversed(u)) if len(u) > 1 else [d["population"]],
                                     [x + 1 for x in u], None, [True] * len(u), "0"])
        case "other-key":
            k = rng.choice(["stream", "registration_digest", "receipt_field"])
            if k == "stream":
                d["stream"] = rng.randbytes(64).hex()
            elif k == "receipt_field":
                d["receipt"] = {**d["receipt"], "received": str(int(d["receipt"]["received"]) + 1)}
            else:
                reg = rng.randbytes(64).hex()
                d.update(registration=reg, opening=ch.open("draw", bytes.fromhex(reg)))
        case _:
            raise ValueError(name)


DRAW_NAMES = ["verifier-coins", "recorded-scheme", "registration", "stream", "receipt", "law", "opening-form", "label",
              "epoch", "foreign", "tampered", "early", "units", "other-key", "early-coins-live"]


def draw_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `draw` case: an honest draw, then mutation `name`."""
    d, ch = honest_draw(rng, CC.COINS_RECORDED if name == "recorded-scheme" else Co.SCHEME if name == "early-coins-live" else None,
                        name in ("units", "other-key"), True if name == "early-coins-live" else None)
    if name != "honest":
        _mutate_draw(d, ch, name, rng)
    return d


def draw_args(d: dict) -> tuple:
    return tuple(d[k] for k in ("coins", "recorded", "opening", "stream", "registration", "receipt", "k", "population", "units"))
