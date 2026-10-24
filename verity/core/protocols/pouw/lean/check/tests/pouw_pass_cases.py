"""The pass and draw difftest's cases: `core_check.pass_epoch`'s arguments, ``{"coins", "raw": hex | null, "context",
"recorded", "reveal", "committed", "stream", "prover"}`` (the last four hex or null), and `core_check.draw`'s,
``{"coins", "recorded", "opening", "stream", "registration", "receipt", "k", "population", "units"}``.

A pass case is an honest pass (a small workload, its weights' roots, an epoch's coins committed under one of
`core_check.COIN_VERSIONS` and its salt coin opened after the weights' root, a recorded pass salted by a beacon, or a
certified pass under a stream and a prover, salted by the network certifier's reveal for the PoUW window its context
names, over the weights' root the certifier recorded with the stream and the prover before it drew that salt), then one
mutation
(`PASS_MUTATIONS`) aimed at one refusal of `pass_epoch`.  A draw case is an honest draw (the draw coin of seeded coins,
or of a live epoch's as a certified window's, opened after a registration's digest, the units `stream_subset` of its
key's stream), then one mutation (`DRAW_NAMES`).
`tests/test_difftest.py` draws them for the harness, and `verity/core/protocols/pouw/tests/test_pouw_core_check.py`
holds each mutation to the refusal it aims at.
"""
from __future__ import annotations

import functools
import hashlib
import json
import random

import pouw_rounds_cases as RC
from verity.ml.crypto import lms
from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import EPOCH_SALT_DOMAIN, SALT_DOMAIN
from verity.core.protocols.pouw.protocol import tagged
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.ml.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.ml.randomness import derive

#: a certified window's verifier's inputs, in `pass_epoch`'s order
CERTIFIED = ("reveal", "committed", "stream", "prover")


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


def live_coins(rng: random.Random, root: bytes, ch: Ch.LiveChallenger, sid: str, salt: bytes | None = None,
               beyond: bool = False) -> tuple[dict, RC.Session, int]:
    """A live Pearl-C pass's coins as its verifier holds them (`core_check.LIVE_PASS`): the session `sid` of challenger
    `ch` the auditor's gate reserved and opened on `root` (`pouw_rounds_cases.weights_session`, its salt coin after
    `salt`, `root` by default, and, when `beyond`, more), the session, and its open's seq in the ledger."""
    s = RC.weights_session(rng, ch, sid, root if salt is None else salt, beyond)
    events = RC.ledger_events(rng, {"root": root.hex()}, s, reserved=True, **RC.weights_end(s))
    ledger = RC.ledger_lines(events, RC._key(0), RC.leaves_of(events, sid))
    seq = next(i for i, e in enumerate(events) if e[:2] == ("open", sid))
    return {"key": s.key(), "ledger": ledger, "main": list(RC.MAIN), "pins": list(RC.PINS), "session": s.files()}, s, seq


def carried(sid: str, seq: int, root: bytes, c: Ch.LiveCommitment, rng: random.Random) -> dict:
    """The gate's answer to the open of session `sid`, without its secret (`pouw_h1_challenger.carried`)."""
    return {"session": sid, "seq": seq, "root": root.hex(), "commitment": c.root.hex(), "host": RC.HOST, "port": 7811,
            "peer": "10.0.0.7", "by": RC.LANE, "opened_by": RC.LANE, "reserve": seq - 1,
            **({"for": RC.LANE} if rng.random() < 0.3 else {})}


def honest_pass(rng: random.Random, kind: str | None = None) -> dict:
    """An honest pass: ``coin`` (the current coins), ``recorded-coin`` (the e2e-service passes' coins, in a replay),
    ``beacon`` (a recorded pass's), ``certified`` (salted by the network certifier's reveal for its PoUW window, which
    the verifier holds) or ``live`` (a live Pearl-C pass's, its salt coin from a session of the auditor's challenger
    opened on its weights' root, which the verifier holds as `core_check.LIVE_PASS`), as a case with its manifest and
    the challenger beside it."""
    kind = kind or rng.choice(["coin", "coin", "recorded-coin", "beacon", "certified", "live"])
    us, weights = _workload(rng)
    man = {"schema": CC.MANIFEST_SCHEMA, "matmuls": us, "weights": weights, "run": f"r-{rng.randrange(1 << 30):x}"}
    roots = _roots(man)
    reveal = window = committed = stream = prover = None
    if kind == "certified":
        reveal, window = rng.randbytes(PS.SALT_BYTES), rng.choice([0, 1, rng.randrange(1 << 64), (1 << 64) - 1])
        stream, prover = rng.randbytes(PS.STREAM_BYTES), rng.randbytes(PS.PROVER_BYTES)
        committed = PS.pouw_record(CC.weights_root(Co.SCHEME, roots), stream, prover)
        man.update(schema=CC.MANIFEST_CERTIFIED, salt=PS.pouw_salt(reveal, window, stream, prover).hex())
        ch = Coins(rng, rng.randrange(4))
    elif kind == "beacon":
        beacon = rng.randbytes(32)
        salt = derive(beacon, SALT_DOMAIN, {"weights": CC.weights_root(Co.SCHEME, roots)}).stream("salt", 32)
        man.update(schema=CC.MANIFEST_RECORDED, beacon=beacon.hex(), salt=salt.hex())
        ch = Coins(rng, rng.randrange(4))
    elif kind == "live":
        root = CC.weights_root(Co.SCHEME, roots)
        ch = Wd.live_challenger(rng.randrange(4), rng.choice([1, 2, 4]), rng.randbytes)
        sid = RC.session_id(rng)
        coins, s, seq = live_coins(rng, root, ch, sid)
        o = s.salt[0]
        salt = derive(o.coin, EPOCH_SALT_DOMAIN, {"epoch": ch.commitment.epoch, "coins": ch.commitment.root, "weights": root}
                      ).stream("salt", 32)
        man.update(coins=ch.commitment.to_json(), salt_opening=o.to_json(), salt=salt.hex(),
                   session=carried(sid, seq, root, ch.commitment, rng))
        case = {"coins": coins, "recorded": rng.random() < 0.3, "run": man["run"], "prefix": f"served-zk/{man['run']}",
                "window": None, **{k: None for k in CERTIFIED}}
        return {"case": case, "man": man, "ch": ch, "kind": kind, "root": root}
    else:
        ch = Coins(rng, rng.randrange(4), Co.SCHEME if kind == "coin" else CC.COINS_RECORDED)
        root = CC.weights_root(ch.scheme, roots)
        opening = ch.open("salt", root)
        salt = derive(bytes.fromhex(opening["coin"]), EPOCH_SALT_DOMAIN, {"epoch": ch.epoch, "coins": ch.tree.commitment.root,
                                                                          "weights": root}).stream("salt", 32)
        man.update(coins=ch.commitment, salt_opening=opening, salt=salt.hex())
    case = {"coins": ch.commitment, "recorded": kind not in ("coin", "certified") or rng.random() < 0.3, "run": man["run"],
            "prefix": f"served-zk/{man['run']}", "window": window,
            **{k: None if v is None else v.hex() for k, v in zip(CERTIFIED, (reveal, committed, stream, prover))}}
    return {"case": case, "man": man, "ch": ch, "kind": kind}


def _context(case: dict, digest: str) -> str:
    """The window's context: the pass's prefix, a certified window's PoUW window (`salt.context`), and the digest."""
    w = case["window"]
    return f"{case['prefix']}/{digest}" if w is None else f"{case['prefix']}/{PS.WINDOW_SEGMENT}/{w}/{digest}"


def _flip(h: str) -> str:
    return ("1" if h[0] == "0" else "0") + h[1:]


def _resign(h: dict, rng: random.Random) -> None:
    """The manifest's salt again from its coin or beacon, after a mutation of its workload."""
    man, ch = h["man"], h["ch"]
    roots = _roots(man)
    if h["kind"] == "live":
        return
    if "beacon" in man:
        man["salt"] = derive(bytes.fromhex(man["beacon"]), SALT_DOMAIN, {"weights": CC.weights_root(Co.SCHEME, roots)}
                             ).stream("salt", 32).hex()
    elif "salt_opening" in man:
        root = CC.weights_root(ch.scheme, roots)
        man["salt_opening"] = ch.open("salt", root)
        man["salt"] = derive(bytes.fromhex(man["salt_opening"]["coin"]), EPOCH_SALT_DOMAIN,
                             {"epoch": ch.epoch, "coins": ch.tree.commitment.root, "weights": root}).stream("salt", 32).hex()


def _relive(h: dict, root: bytes, rng: random.Random, beyond: bool = False) -> None:
    """Live pass `h` salted in a fresh session of its id, opened on `root`, its salt coin after it (and, when `beyond`,
    more coins released), its manifest that session's commitment, salt coin, salt and gate's answer."""
    man, c = h["man"], h["ch"].commitment
    ch, sid = Wd.live_challenger(c.epoch, c.family(Wd.ROUND), rng.randbytes), man["session"]["session"]
    h["case"]["coins"], s, seq = live_coins(rng, root, ch, sid, beyond=beyond)
    o = s.salt[0]
    salt = derive(o.coin, EPOCH_SALT_DOMAIN, {"epoch": c.epoch, "coins": ch.commitment.root, "weights": h["root"]})
    man.update(coins=ch.commitment.to_json(), salt_opening=o.to_json(), salt=salt.stream("salt", 32).hex(),
               session=carried(sid, seq, root, ch.commitment, rng))


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
            case["context"] = _context(case, hashlib.sha512(raw + b' ').hexdigest())
            return raw
        case "certified-unread":
            case["reveal"] = None
        case "uncertified-pass":
            case["reveal"] = rng.randbytes(PS.SALT_BYTES).hex()
        case "reveal-size":
            case["reveal"] = rng.randbytes(rng.choice([0, 32, 63, 65, 128])).hex()
        case "no-window":
            case["window"] = None if rng.random() < 0.5 else rng.choice([f"0{case['window']}", "-1", "x", "", str(1 << 64)])
        case "other-window":
            case["window"] = (case["window"] + rng.choice([1, -1, 1 << 32])) % (1 << 64)
        case "other-reveal":
            case["reveal"] = rng.randbytes(PS.SALT_BYTES).hex()
        case "raw-reveal":
            man["salt"] = case["reveal"]
        case "weights-unrecorded":
            case["committed"] = rng.choice([None, rng.randbytes(rng.choice([0, 32, 63, 65])).hex()])
        case "weights-after-salt":
            if rng.random() < 0.5:
                case["committed"] = rng.randbytes(PS.WEIGHTS_ROOT_BYTES).hex()
            else:
                man["weights"][rng.choice(list(man["weights"]))]["root"] = rng.randbytes(64).hex()
        case "binding-form":
            case[rng.choice(["stream", "prover"])] = rng.choice([None, rng.randbytes(rng.choice([0, 32, 63, 65])).hex()])
        case "stream-other" | "prover-other":
            case[name.split("-")[0]] = rng.randbytes(64).hex()
        case "salted-apart":
            k = rng.choice(["stream", "prover"])
            bound = {"stream": bytes.fromhex(case["stream"]), "prover": bytes.fromhex(case["prover"]), k: rng.randbytes(64)}
            case[k] = bound[k].hex()
            case["committed"] = PS.pouw_record(CC.weights_root(Co.SCHEME, _roots(man)), bound["stream"],
                                               bound["prover"]).hex()
        case "spaced-hex":
            w = rng.choice(list(man["weights"]))
            r = man["weights"][w]["root"]
            man["weights"][w]["root"] = r[:2] + " " + r[2:].upper()
            if o is not None and h["kind"] != "live":
                man["salt_opening"] = {**o, "coin": " ".join(o["coin"][i:i + 2] for i in range(0, 64, 2))}
        case "live-form":
            coins = case["coins"]
            k = rng.choice([x for x in CC.LIVE_PASS if x != "ledger"])
            case["coins"] = rng.choice([{x: v for x, v in coins.items() if x != k}, {**coins, "record": coins["session"]}])
        case "live-in-process":
            seeded = Coins(rng, h["ch"].commitment.epoch)
            root = h["root"]
            opening = seeded.open("salt", root)
            man.update(coins=seeded.commitment, salt_opening=opening, salt=derive(
                bytes.fromhex(opening["coin"]), EPOCH_SALT_DOMAIN,
                {"epoch": seeded.epoch, "coins": seeded.tree.commitment.root, "weights": root}).stream("salt", 32).hex())
        case "live-own-challenger":
            own = Wd.live_challenger(h["ch"].commitment.epoch, h["ch"].commitment.family(Wd.ROUND), rng.randbytes)
            o = own.open(Wd.SALT, 0, h["root"])
            man.update(coins=own.commitment.to_json(), salt_opening=o.to_json(), salt=derive(
                o.coin, EPOCH_SALT_DOMAIN, {"epoch": own.commitment.epoch, "coins": own.commitment.root, "weights": h["root"]}
            ).stream("salt", 32).hex())
        case "live-forged":
            k = rng.choice(["coin", "salt", "after"])
            man["salt_opening"] = {**o, k: _flip(o[k])}
        case "live-not-signed":
            man["coins"] = Wd.live_challenger(h["ch"].commitment.epoch, h["ch"].commitment.family(Wd.ROUND),
                                              rng.randbytes).commitment.to_json()
        case "live-session":
            g = man["session"]
            k = rng.choice(["session", "seq", "root", "commitment", "secret", None])
            man["session"] = None if k is None else {**g, "secret": rng.randbytes(32).hex()} if k == "secret" else {
                **g, k: {"session": RC.session_id(rng), "seq": rng.choice([g["seq"] + 1, str(g["seq"]), True]),
                         "root": rng.randbytes(64).hex(), "commitment": _flip(g["commitment"])}[k]}
        case "live-other-root":
            # the verifier's session was opened on another root than the pass's weights' (H1's window-open root among
            # them), its salt coin after that root
            sid = man["session"]["session"]
            _relive(h, rng.choice([rng.randbytes(64), PS.window_open_root(h["root"], rng.randbytes(64), sid)]), rng)
        case "live-beyond":
            _relive(h, h["root"], rng, beyond=True)
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
    "certified-unread": ("certified",), "uncertified-pass": ("coin", "recorded-coin", "beacon"),
    "reveal-size": ("certified",), "no-window": ("certified",), "other-window": ("certified",),
    "other-reveal": ("certified",), "raw-reveal": ("certified",), "weights-unrecorded": ("certified",),
    "weights-after-salt": ("certified",), "binding-form": ("certified",), "stream-other": ("certified",),
    "prover-other": ("certified",), "salted-apart": ("certified",),
    "live-form": ("live",), "live-in-process": ("live",), "live-own-challenger": ("live",), "live-forged": ("live",),
    "live-not-signed": ("live",), "live-session": ("live",), "live-other-root": ("live",), "live-beyond": ("live",),
}
PASS_NAMES = list(PASS_MUTATIONS)


def pass_instance(rng: random.Random, name: str = "honest", kind: str | None = None) -> dict:
    """One `pass_epoch` case: an honest pass (of `kind`, by default any), then mutation `name` (on a pass of a kind it
    applies to)."""
    kinds = PASS_MUTATIONS.get(name)
    h = honest_pass(rng, kind or (rng.choice(kinds) if kinds else None))
    case = h["case"]
    raw = _raw(h["man"], rng) if name == "honest" else _mutate_pass(h, name, rng)
    if raw is None:
        context = _context(case, "0" * 128)
    else:
        context = case.get("context") or _context(case, hashlib.sha512(raw).hexdigest())
    return {"coins": case["coins"], "raw": None if raw is None else raw.hex(), "context": context,
            "recorded": case["recorded"], **{k: case[k] for k in CERTIFIED}}


def pass_args(case: dict) -> tuple:
    def hex_or_none(k: str) -> bytes | None:
        return None if case.get(k) is None else bytes.fromhex(case[k])

    return (case["coins"], hex_or_none("raw"), case["context"], case["recorded"], *map(hex_or_none, CERTIFIED))


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
            "stream": stream, "registration": reg, "receipt": receipt, "k": k, "population": n, "units": units,
            "key": None}, ch


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
        case "seeded-pinned":
            sk = _challenger_key(rng.randrange(2))
            d["key"] = Ch.ChallengerKey(sk.params.W, sk.params.H, lms.public_key(sk)).to_json()
        case _:
            raise ValueError(name)


@functools.cache
def _challenger_key(seed: int) -> lms.PrivateKey:
    return lms.keygen(lms.Params(64, 4, 2), bytes([seed]) * 64)


def signed(commitment: Ch.LiveCommitment, rng: random.Random, signer: int | None = None, key: int | None = None) -> dict:
    """A live window's coins as its verifier holds them: `commitment`'s challenger record, signed on a host by
    challenger key `signer` (0 or 1, at random), and the verifier's key for key `key` (the signer's by default)."""
    signer = rng.randrange(2) if signer is None else signer
    sk, vk = _challenger_key(signer), _challenger_key(signer if key is None else key)
    host = rng.choice(["vy-nebius-cpu-1", "auditor.local", "h_1"])
    message = Ch.record_message(lms.public_key(sk)[:16], host, commitment)
    record = Ch.ChallengerRecord(commitment, host, lms.sign_at(sk, rng.randrange(4), message).encode())
    return {"key": Ch.ChallengerKey(vk.params.W, vk.params.H, lms.public_key(vk)).to_json(), "record": record.to_json()}


#: the challenger key of a prover's own, which no window's parameters pin (`signed`'s keys 0 and 1 are the verifier's)
PROVERS_KEY = 2


def honest_live_draw(rng: random.Random, many: bool = False) -> tuple[dict, Ch.LiveChallenger]:
    """An honest draw under a live epoch's coins (`window.live_challenger`, a certified window's), held as their signed
    record (`signed`), under the challenger key the window's parameters pin (``key``, the record's): its ``pouw/draw``
    coin 0 opened after the registration's digest."""
    ch = Wd.live_challenger(rng.randrange(4), rng.choice([1, 2, 3]), rng.randbytes)
    reg, stream = rng.randbytes(64).hex(), rng.randbytes(64).hex()
    receipt = _receipt(rng, reg)
    n = rng.choice([50, rng.randint(50, 5000), rng.randint(50, 1 << 40)] + ([] if many else [1, 2, 7]))
    k = rng.randint(1 if many else 0, min(n, 12))
    o = ch.open(Wd.DRAW, 0, bytes.fromhex(reg))
    ctx, size = CC.draw_key_context(stream, reg, receipt), 2 * k * CC.width(n) + 64
    units = CC.stream_subset(k, n, derive(o.coin, CC.DRAW_DOMAIN, ctx).stream("draw", size))
    coins = signed(ch.commitment, rng)
    return {"coins": coins, "recorded": rng.random() < 0.3, "opening": o.to_json(), "stream": stream,
            "registration": reg, "receipt": receipt, "k": k, "population": n, "units": units, "key": coins["key"]}, ch


def _mutate_live_draw(d: dict, ch: Ch.LiveChallenger, name: str, rng: random.Random) -> None:
    o, c = d["opening"], ch.commitment
    coins, rec = d["coins"], d["coins"]["record"]
    match name:
        case "live-coins":
            cj = rec["commitment"]
            bad = rng.choice([{**cj, "root": "00"}, {**cj, "families": "x"}, {**cj, "epoch": -1}, {**cj, "extra": 1}])
            d["coins"] = {**coins, "record": {**rec, "commitment": bad}}
        case "live-not-epoch":
            layout = rng.choice([((Wd.DRAW, 1), (Wd.SALT, 1), (Wd.ROUND, 1)), ((Wd.SALT, 1), (Wd.ROUND, 2)),
                                 ((Wd.SALT, 1), (Wd.DRAW, 2), (Wd.ROUND, 1))])
            d["coins"] = signed(Ch.LiveChallenger(Wd.LIVE_SCHEME, c.epoch, layout, rng.randbytes).commitment, rng)
            d["key"] = d["coins"]["key"]
        case "live-bare":
            d["coins"] = rng.choice([c.to_json(), {**coins, "commitment": c.to_json()}, {"record": rec}])
        case "live-key":
            key = coins["key"]
            d["coins"] = {**coins, "key": rng.choice([{**key, "W": 3}, {**key, "H": 0}, {**key, "public": key["public"][:64]},
                                                      {**key, "extra": 1}, None])}
        case "live-no-record":
            d["coins"] = {**coins, "record": None}
        case "live-unsigned":
            signer = rng.randrange(2)
            d["coins"] = rng.choice([signed(c, rng, signer, 1 - signer),
                                     {**coins, "record": {**rec, "signature": _flip(rec["signature"])}}])
            d["key"] = d["coins"]["key"]
        case "live-own-key":
            d["coins"] = signed(c, rng, PROVERS_KEY)
        case "live-pinned-other":
            pinned = Ch.ChallengerKey.from_json(d["key"])
            d["key"] = rng.choice([signed(c, rng, PROVERS_KEY)["key"], {**d["key"], "H": pinned.H + 1},
                                   {**d["key"], "public": _flip(d["key"]["public"])}])
        case "live-no-pin":
            key = d["key"]
            d["key"] = rng.choice([None, {**key, "W": 3}, {**key, "public": key["public"][:64]}, {**key, "extra": 1}, 7,
                                   "key"])
        case "live-moved":
            other = Wd.live_challenger(c.epoch, c.family(Wd.ROUND), rng.randbytes).commitment.to_json()
            d["coins"] = {**coins, "record": {**rec, "commitment": other}}
        case "live-opening-form":
            k = rng.choice(["coin", "path", "index", "epoch", "label"])
            bad = {"coin": ["zz", None], "path": [None, 4], "index": ["0", None, True], "epoch": [str(c.epoch), None],
                   "label": [5, None]}[k]
            d["opening"] = rng.choice([{**o, k: rng.choice(bad)}, {x: v for x, v in o.items() if x != "index"}])
        case "live-label":
            d["opening"] = ch.open(Wd.SALT, 0, bytes.fromhex(d["registration"])).to_json()
        case "live-epoch":
            d["opening"] = {**o, "epoch": o["epoch"] + 1}
        case "live-foreign":
            other = Wd.live_challenger(c.epoch, c.family(Wd.ROUND), rng.randbytes)
            d["opening"] = other.open(Wd.DRAW, 0, bytes.fromhex(d["registration"])).to_json()
        case "live-tampered":
            d["opening"] = {**o, "salt": _flip(o["salt"])}
        case "live-early":
            d["opening"] = {**o, "after": rng.randbytes(64).hex()}
        case "live-units":
            u = d["units"]
            d["units"] = rng.choice([u[:-1], u + [0], [x + 1 for x in u], None])
        case _:
            raise ValueError(name)


LIVE_DRAW_NAMES = ["live-coins", "live-not-epoch", "live-bare", "live-key", "live-no-record", "live-unsigned",
                   "live-own-key", "live-pinned-other", "live-no-pin", "live-moved", "live-opening-form", "live-label",
                   "live-epoch", "live-foreign", "live-tampered", "live-early", "live-units"]
DRAW_NAMES = ["verifier-coins", "recorded-scheme", "registration", "stream", "receipt", "law", "opening-form", "label",
              "epoch", "foreign", "tampered", "early", "units", "other-key", "early-coins-live", "seeded-pinned",
              *LIVE_DRAW_NAMES]


def draw_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `draw` case: an honest draw (under seeded coins, or live ones a third of the time), then mutation `name` (a
    live one's on a live draw)."""
    if name in LIVE_DRAW_NAMES or (name == "honest" and rng.random() < 1 / 3):
        d, live = honest_live_draw(rng, name == "live-units")
        if name != "honest":
            _mutate_live_draw(d, live, name, rng)
        return d
    d, ch = honest_draw(rng, CC.COINS_RECORDED if name == "recorded-scheme" else Co.SCHEME if name == "early-coins-live" else None,
                        name in ("units", "other-key"), True if name == "early-coins-live" else None)
    if name != "honest":
        _mutate_draw(d, ch, name, rng)
    return d


def draw_args(d: dict) -> tuple:
    return tuple(d[k] for k in ("coins", "recorded", "opening", "stream", "registration", "receipt", "k", "population", "units",
                                "key"))
