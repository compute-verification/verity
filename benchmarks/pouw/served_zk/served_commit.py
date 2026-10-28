"""A served audit window's commitment in the ZK statement's format, from a retained Pearl-C pass, registered, drawn and
judged through the proof system (`sampled_proofs.service`, P1: a `Stream`'s `register`, `select`, `deliver` and
`outcome`), and the check that a hidden tile proof's committed rows are the drawn tile's.

  python served_commit.py params PASS_DIR OUT [--eps E --delta D --entries P --max-tokens T --registration-deadline S]
                                                                              the verifier's own parameters of the window
  python served_commit.py commit PASS_DIR PARAMS OUT [--calls N | --call I ...]   the committer: the window, salted and committed
  python served_commit.py register COMMIT_DIR PARAMS OUT [--extra ITEM=JSON]  the committer: the record, the public items only
  python served_commit.py edit REGISTRATION OUT KEY=JSON ...                  a control's registration, a field changed
  python served_commit.py challenger PARAMS LOG [--lean F --epoch E --state S] the challenger: one JSON request a line on stdin
                                                                              (`handle`), one reply a line on stdout
  python served_commit.py replay PARAMS LOG OUT [--lean F]                    its requests again, from its recorded coins
  (challenger, replay and check take --python-record: PoUW's checks by their Python references alone)
  python served_commit.py coin-variant MANIFEST COINS OPENING OUT CASE        a control's pass, its salt coin CASE
  python served_commit.py check PARAMS REGISTRATION DRAW UNIT PUB OPENING     the verifier: the proof's rows are the unit's
  python served_commit.py shape DIR DIR                                       two staged units' proofs have one shape

- **The window** is SZ_CALLS of the pass's ``o_proj``-shaped calls (k = n = `K`), picked from ``os.urandom`` unless
  ``--call`` names them.  Each call's A rows (the pass's retained activations, BF16 widened to FP32) and its weight's B rows
  are formed on the CPU as `pearl_c` forms them under the pass's epoch (`PearlC.noise`, forming v1, the peel factors), and
  become the statement's fp8-pi rows: fp8-p rows (A′_i ‖ P_A,i of role X, ``hm96-sha512/row-seg/v1``; B̃_j ‖ P_B,j of
  role W, ``hm96-sha512/row/v2``; `leaves.words_fp8_p`), each followed by its place in the call, I or J, as a u32 in one
  128-bit word (`pouw.window.side_row`, `leaves.words_fp8_pi`), which the tile unit reads.  A row whose forming the scheme refuses is no tile's.
- **The tiles** are each call's kept rows' products, then filler tiles up to the padded count: the window's N_real real
  tiles register the grid level of `pouw.window.bucket`, credited g_ℓ <= N_real tiles and padded to N = g_(ℓ+1).  A filler
  tile is a tile of the same template on zero rows (both sides), committed like any row, so its proof has the shape of
  every other tile's.  Unit t is `pouw.window.tile_at` of the advice: the population is
  `population_program(Pc8TileSide{K,1,1,sm120}, N)` under `Q_template_instance`.
- **The commitment** is C-Flock's: each row's b ‖ c is hm96-sha512 over its row digest under a fresh 192-byte salt.  Each
  side of each entry (a call, or the filler) is one frame-v3-sha512 tree of its kept rows only, in order, padded to 2^D
  leaves under the window's side domain (`pouw.window.side_domain` of the window root's domain, one-stage's served domain
  of the window's program, partition and context: one per window and side).  Each entry is one fixed-width entry row
  (`pouw.window.Entry`: kind, tile range, each side's row count and root, the root of the pass's -h3 tile commitment
  (`root_h3_of`), the call's first tile in the pass, ⌈n/64⌉ and m, and the SHA-512 of the call's record, which holds
  the kept rows' places in the call), an
  ``hm96-sha512/row/v2`` row in the **window root** (port ``entries``), padded with
  empty entries to the public bound P (L2).  W, every weight the pass's k = n = K calls read, is
  committed by weight row (BF16, ``hm96-sha512/row/v2``) in **W's root**, beside it; the tiles read the formed B̃ rows of
  each call, not W's rows, until registered rows (P7) read B from W.  ``commitment.json`` is what the committer registers:
  the context, the level, the public items and the two roots.  ``advice.json`` (the entries and their salts) and
  ``prover.npz`` (rows, salts, b ‖ c) are the prover's.
- **The fixed length**: the request is served at a public max_tokens under ignore_eos, so its
  useful work is public.  The verifier's parameters hold its request's max_tokens (``length``), the committer registers
  the runner's record of how it served it (``served.json`` beside the pass, `served_length`), and PoUW's rule refuses an
  EOS-keyed row or a served length other than max_tokens (``window-form``).  A pass with no served record registers no
  length item, which the verifier refuses as registered before the rule.
- **The service** (P1): the committer's `register` writes one-stage's v1 record, scheme `service.SCHEME`, whose window
  holds exactly `pouw.window.REGISTERED`.  Two rules hold it: the service's (`service.check_public` over the `Spec`,
  which declares what one-stage's record shows besides the two salted roots: N, the window item, each root's leaf count
  and domain) and PoUW's inside the window item (`public_problem`: exactly those items, each the verifier's own
  recomputation at the registered level from its own context, ε, δ, the grid, P, |W| and max_tokens).  The `Challenger` holds both,
  then `Stream.register` (R1-R7 and the receipt), `select`, `deliver` (the sampled proof blocks' session records' SHA-512, once)
  and `outcome` (final), and publishes exactly `pouw.window.PUBLIC`.  A Spec fixes one law and K′ is a function of the
  level, so the challenger holds one Stream per level.
- **Of record** (`of_record`): the challenger's checks are Lean's.  PoUW's rule (`core_check.registered`, which
  `public_problem` mirrors for the committer), R1-R6, the pass's epoch, the sample, PoUW's check over the stream's outcome
  (`core_check.case`) and a unit's opening each run Lean's port and Python's reference on one input; Lean's verdict
  decides and is the published items', both go in the window's witness (``outcome.json``'s ``of_record``, the opening
  verdict's ``witness``), and a disagreement refuses as ``lean-python-disagree: <check>``.  The service's public-items
  rule is Lean's too (``public``, `Service.checkPublic` beside `service.check_public`), and `check`'s reading of the
  registration before an opening may refuse only what the opening refuses (``prelude``); the stream's own state stays
  Python's.
- **The coins** are the epoch's (`pouw.coins`): the challenger process that holds the Streams holds the epoch's coin
  challenger.  When the process starts, before it reads a request, that challenger samples the epoch's salt and flip coins
  and their hm96 salts from the Lean verifier of record (`pouw.verifier.coins`: `pouw-check-difftest`'s ``coins``,
  `IO.getRandomBytes`, A3) and commits to them, and the process publishes the commitment (``LOG/coins.json``) before
  any request; with ``--state`` the entropy is instead the file the serving controller wrote from the same Lean coins
  before it served the pass (`e2e.state_entropy`), so the pass's salt coin is one it opens.  No coin is read from
  Python's generator.  A window's registration names its
  pass, and the challenger re-starts the pass's epoch from the pass's manifest under its own commitment
  (`pouw_pearl_c_device.manifest_epoch`): the pass's salt coin must open against it, as the salt coin, after the
  weights' root, and the pass's salt must be that coin's; a pass recorded before epoch coins (a beacon's salt) is read
  only when the request says it is a recorded replay.  `select` opens the epoch's sampling coin after the registration (its
  digest), once, with no retry, and the sample's key is `draw_key` of that coin; Lean ``flock-verify draw --population N
  --k K′ --stream`` samples over the key's stream (`stream_subset` is its Python twin).  A window is received within
  the registration deadline of the salt coin's release and before the epoch's sampling coin is released, or it is refused
  (`late-registration`); parameters without the deadline are refused.  The challenger records each opening
  (``LOG/openings.jsonl``), and `replay` re-runs its requests from those records, byte for byte.
- **A certified window** (`Challenger(certifier=…)`, the class only: the CLI doesn't run one yet) is salted by the network
  certifier instead (`pouw.salt`): its context names its PoUW window p (`salt.context`), and its pass's salt is
  `salt.pouw_salt` of p's certifier salt, the window's stream (`stream_id`, which names the certifier's run: its key id,
  run id and ingress link, `certifier_run`) and its prover (`salt.prover_id`), which the prover recorded with the
  certifier, signed, before p's salt was sampled
  (`salt.pouw_record`).  **No certified window registers today**: a certified window credits only units that read its
  salt (`salt.unit_reason`), and this window's unit, `Pc8TileSide`, reads two committed side rows and no salt, so
  `register` refuses every certified window first (``unsalted``), before the stream receives it or the certifier is
  asked anything.  Past that rule, `register` has the stream receive the window, reads the certifier's `committed` on
  that registration, refuses a pass whose weights, stream or prover it doesn't bind, and only then calls `reveal`,
  failing closed (``certifier``), each through the certifier's client (`RemoteCertifier`: a signed request, a signed
  answer it verifies): the certifier gives p's salt once, within its reveal deadline on its own clock, so its
  parameters carry no ``registration_deadline``.  The window registers once per PoUW window, stream and prover
  (`salt.Registrations`, ``reregistered``), and its sampling coin is a live epoch's ``pouw/draw`` coin 0, checked against
  the commitment the challenger reads from its challenger's signed record (`challenger.trusted_commitment`), which the
  sample's check of record verifies again, both under the challenger key the window's deployment pins (`Deployment`,
  beside the certifier's key and run id; `Deployment.pinned` reads it from infra's published pin, `CHALLENGER_KEY_PIN`,
  and refuses any key whose SHA-512 isn't `CHALLENGER_KEY_SHA512`), so a record the prover signs under its own key is
  refused (``coin-key``).
- **The check** reads a proof's public file (``pub.bin``): instance 0's b ‖ c of its A row and B row must be the unit's rows,
  each by its frame-v3 path to its entry's side root, and the entry by its path to the registered window root.  A
  certified window's units are opened under the salt its pass was checked against (`Challenger.check`,
  `core_check.certified_salt`).  The opening hands the verifier the sampled proof blocks' entries: a development shortcut, until
  position reads in gates (the joint note's step 8) take it over.  Nothing it reads is in the outcome's public items.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from verity.ml.circuits.program import Program
from verity.ml.commitments.identity import identity_digest_sha512, tagged_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest
from verity.ml.circuits.codec import canonical_json
from verity.ml.randomness import Key, derive
from verity.core.protocols.certifier.active import RUN_ID_BYTES
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import verifier as PV
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import leaves as L
from verity.core.protocols.pouw.circuit import pc8
from verity.core.protocols.pouw.coins import Challenger as Coins
from verity.core.protocols.pouw.coins import CoinCommitment, CoinOpening, coin_reason
from verity.core.protocols.pouw.protocol import Context, DomainError, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_kw as P
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import draw as D
from verity.core.protocols.sampled_proofs.one_stage import partition as PT
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.core.protocols.sampled_proofs.one_stage.verdict import Verdict, _Collect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import of_record as OR  # noqa: E402

SCHEMA = "verity-pouw/served-zk-window/v1"
CHALLENGER = "served-zk/challenger"
#: the window sample's key, from the epoch's sampling coin (`draw_key`)
DRAW_DOMAIN = CC.DRAW_DOMAIN
#: the window root's port: the window's entries
PORT = "entries"
#: a window stream's failure budget: its first failure stops it
BUDGET = 1
#: the window's proof deadline (`Check.proof_deadline`), seconds on the challenger's clock from the sample, unless the
#: parameters give ``proof_deadline``: a day
PROOF_DEADLINE = 86400.0
#: the window's registration deadline (`Check.registration_deadline`, `Pouw.AcceptsBy`'s td), seconds on the challenger's
#: clock from the epoch's salt coin's release: the challenger's ``salt`` request, which opens that coin (drawn when the
#: challenger process started, from the Lean verifier's coins or ``--state``) after the weights' root.  Unless
#: ``params --registration-deadline`` gives another: an hour
REGISTRATION_DEADLINE = 3600.0
NO_DEADLINE = ("the window's parameters name no registration_deadline, so the service would accept a window EndToEndWindow "
               "doesn't cover: write them with `served_commit.py params` (REGISTRATION_DEADLINE, or --registration-deadline)")
CERTIFIED_DEADLINE = ("a certified window's time bound is the network certifier's reveal deadline on its own clock "
                      "(`Proxy(reveal_deadline=…)`), so its parameters carry no registration_deadline and it replays none")
#: a certified window's stream as the PoUW salt binds it (`stream_id`)
STREAM_ID = "verity/pouw/served-zk/stream/v3"
#: the frame's owner, as C-Flock's statement domains have it (`verity_flock.ir_frame.OWNER`)
OWNER = -1
SALT_BYTES = 192
K = 4096
TILE = (1, 1)
DEV = "sm120"
SIDES = {"a": ROLE_X, "b": ROLE_W}
#: the rows a forming batch takes at once
CHUNK = 256
ENTRY_SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
W_SCHEMA = L.ROW_SCHEMA["bf16", ROLE_W]
W_DOMAIN = "verity/pouw/served-zk/weights-domain/v0"
#: the public bound on a window's entries (its calls and the filler), and the tile counts a window may register
ENTRIES = 16
RANGE = (1, 1 << 40)
EPS, DELTA = "1/2", "1/10"
#: the max_tokens of the request the verifier issues, `e2e.SERVED_TOKENS`, and the runner's record of how it served it
#: (``{"ignore_eos", "max_tokens", "served"}``, a list per slot), which `e2e.py` writes beside the retained pass
MAX_TOKENS = 4
SERVED_RECORD = "served.json"


# ---- the window's parameters and its spec ---------------------------------------------------------------------------------

def template(k: int):
    return pc8.side_tile(k, TILE, DEV)


def tile_work(params: dict) -> int:
    return params["k"] * TILE[0] * TILE[1]


@lru_cache(maxsize=8)
def population(k: int, n: int):
    """The window's tiles as one-stage's population: (program, query, partition object)."""
    d = template(k)
    prog = PT.population_program(d, n)
    q = PT.template_instance_query(PT.template_id(d))
    return prog, q, PT.partition(prog, q)


def work_at(params: dict, ell: int) -> Wd.Work:
    w = Wd.Work.at(ell, params["grid_bits"])
    lo, hi = params["range"]
    if not (lo <= w.credited and w.n <= hi):
        raise ValueError(f"level {ell} ({w.credited} credited, {w.n} tiles) is outside the window's range {lo}..{hi}")
    return w


def check_item(params: dict) -> dict:
    return {"template": template(params["k"]).id, "tile_work": tile_work(params), "scheme": S.SCHEME, "entries": params["entries"]}


def served_length(served: Mapping[str, Any]) -> dict:
    """The registered length item (`pouw.window.length_item`'s form) of the runner's served record (``served.json``,
    `e2e.py`'s served requests): each slot's max_tokens and served length, and stop fixed when it served under
    ignore_eos, else ``eos``, which the verifier refuses as an EOS-keyed row."""
    return {"max_tokens": list(served["max_tokens"]), "served": list(served["served"]),
            "stop": Wd.STOP_FIXED if served.get("ignore_eos") is True else "eos"}


def items_of(params: dict, ell: int, root_w: str | None, length: Mapping[str, Any] | None = None) -> dict:
    """The registered public items at level `ell` (`pouw.window.REGISTERED`), the rows' length item `length`; with none,
    `pouw.window.LEGACY_REGISTERED`, a window the verifier refuses as registered before the fixed-length rule."""
    w = work_at(params, ell)
    return {"work": w.to_json(tile_work(params)), "check": check_item(params),
            "sampling": Wd.sampling(Fraction(params["eps"]), Fraction(params["delta"]), w, tile_work(params), params["grid_bits"]),
            **({"length": dict(length)} if length is not None else {}), "root_W": root_w}


#: what one-stage's record shows besides the two salted roots, each declared to the service's rule with its reason
LAYOUT = (
    ("population", "structure", "N, the padded tile count g_(ℓ+1): the draw's range, a function of the registered level"),
    ("window", "output", "PoUW's registered items, exactly `pouw.window.REGISTERED` (work, check, sampling, length, root_W), "
                         "each the verifier's own recomputation at the registered level, the length its fixed max_tokens "
                         "(`public_problem`)"),
    (f"leaves:{PORT}", "structure", "P, the public bound on a window's entries"),
    (f"domain:{PORT}", "structure", "the entries' served domain: the window's program, partition and context"),
    ("leaves:W", "structure", "|W|, the deployment's weight rows at k = n = K"),
    ("domain:W", "structure", "W's domain: the template and the deployment"),
)


def law_of(params: dict, ell: int) -> dict:
    return items_of(params, ell, None)["sampling"]["law"]


def spec_of(params: dict, ell: int, predates_deadline: bool = False) -> S.Spec:
    """P1's Spec of a window at level `ell`: the entries and W committed, subset:K′ of the level, the tile template's
    Program and query as the check, and `LAYOUT` declared.  The check carries ``registration_deadline`` from `params`
    (seconds on the challenger's clock from the epoch's salt coin's release, the ``salt`` request that opens it,
    `Pouw.AcceptsBy`'s td: `REGISTRATION_DEADLINE`), and parameters without one raise `ValueError` (`NO_DEADLINE`)
    unless `predates_deadline`, a recorded window's replay, or they are a certified window's (`certified`), which the
    certifier's reveal deadline bounds.  Its proof deadline is ``proof_deadline`` in `params`, else `PROOF_DEADLINE`."""
    if params.get("registration_deadline") is None and not (predates_deadline or certified(params)):
        raise ValueError(NO_DEADLINE)
    d = template(params["k"])
    return S.Spec(values=(S.Value(PORT, ENTRY_SCHEMA, "serve"), S.Value("W", W_SCHEMA, "epoch")),
                  selection=S.Law(f"subset:{law_of(params, ell)['k']}"),
                  check=S.Check(PT.program_sha512(Program(d)), PT.template_instance_query(d),
                                proof_deadline=params.get("proof_deadline", PROOF_DEADLINE),
                                registration_deadline=params.get("registration_deadline")),
                  public=tuple(S.Public(i, k, r) for i, k, r in LAYOUT), context=params["context"].encode())


def certified(params: Mapping[str, Any]) -> bool:
    """Whether `params` are a certified window's: their context names its PoUW window (`salt.context`)."""
    return PS.window_of(params.get("context")) is not None


def certifier_run(certifier: Any) -> dict[str, str]:
    """The network certifier's run as a certified window's stream names it: its `certifier_run`, (key id, run id, ingress
    link), the identity every signed `Revealed` binds, which a `Proxy` and its `RemoteCertifier` give alike (key id and
    run id as hex). A proxy started again draws another run id, and a second link has another name; a proxy without a
    signer has no key id, so no run."""
    try:
        key_id, run_id, link = certifier.certifier_run
    except (AttributeError, TypeError, ValueError) as e:
        raise ValueError(f"a certified window's certifier names its run, (key id, run id, link): {e}") from None
    if not (type(key_id) is bytes and key_id and type(run_id) is bytes and len(run_id) == RUN_ID_BYTES
            and isinstance(link, str) and link):
        raise ValueError(f"a certified window's certifier names its run: a key id, a {RUN_ID_BYTES}-byte run id and its "
                         "ingress link")
    return {"key_id": key_id.hex(), "link": link, "run_id": run_id.hex()}


DEPLOYMENT_FORM = ("a certified window's deployment is the network certifier's public key, its run's "
                   f"{RUN_ID_BYTES}-byte id and the verifier's challenger key (`challenger.ChallengerKey`)")
#: infra's published challenger key, vy-challenger-1's from 2026-10-20T09:45:24Z: a challenger-key/v1 record
CHALLENGER_KEY_PIN = "art:d12f85becc22388f3e9834d3894cbb99eb655bc19484b0bc3abe9b080689a657"
#: the SHA-512 of the public key `CHALLENGER_KEY_PIN` publishes; a record of any other key is refused, whatever it names
CHALLENGER_KEY_SHA512 = "27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3ae065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc"
CHALLENGER_KEY_SCHEMA = "challenger-key/v1"
CHALLENGER_KEY_FORM = (f"the challenger key's pin is a {CHALLENGER_KEY_SCHEMA} record {{H, W, pubkey, pubkey_sha512, "
                       "schema, ...}}: the key's parameters, its public key in lowercase hex and that public key's SHA-512")
NOT_PINNED = (f"the challenger key record's public key is not the one pinned ({CHALLENGER_KEY_PIN}, SHA-512 "
              f"{CHALLENGER_KEY_SHA512[:8]}…)")


def pinned_challenger_key(record: Path | None = None) -> Ch.ChallengerKey:
    """The verifier's challenger key, read from infra's published pin (`CHALLENGER_KEY_PIN`, fetched from the store
    unless `record` is its file): refused (ValueError) unless the record is a challenger-key/v1 record whose public key's
    SHA-512 is `CHALLENGER_KEY_SHA512`, both the one the record names and the one the public key hashes to."""
    if record is None:
        from research.store.local import LocalStore  # noqa: PLC0415  only the verifier's own reading fetches
        record = LocalStore().fetch(CHALLENGER_KEY_PIN)
    try:
        doc = json.loads(Path(record).read_bytes())
    except (OSError, ValueError) as e:
        raise ValueError(f"{CHALLENGER_KEY_FORM}: {e}") from None
    pub = doc.get("pubkey") if isinstance(doc, dict) else None
    if not (isinstance(doc, dict) and doc.get("schema") == CHALLENGER_KEY_SCHEMA and isinstance(pub, str)
            and len(pub) % 2 == 0 and all(c in "0123456789abcdef" for c in pub)):
        raise ValueError(CHALLENGER_KEY_FORM)
    public = bytes.fromhex(pub)
    if not (hashlib.sha512(public).hexdigest() == CHALLENGER_KEY_SHA512 == doc.get("pubkey_sha512")):
        raise ValueError(NOT_PINNED)
    try:
        return Ch.ChallengerKey(doc.get("W"), doc.get("H"), public)
    except Ch.CoinError as e:
        raise ValueError(f"{CHALLENGER_KEY_FORM}: {e}") from None


@dataclass(frozen=True, slots=True)
class Deployment:
    """What a certified window's verifier pins of its deployment: the network certifier's public key and its run's id,
    the only ones whose answers the challenger takes (its client, `certifier.active.RemoteCertifier`, must be for them,
    and refuses any other), and the key of the verifier's own challenger, under which the challenger reads the window's
    coin record and the sample's check of record verifies it (`core_check.draw`, ``coin-key``), so no record a prover
    signs over coins it holds is read.  `pinned` takes the challenger key from infra's published pin."""

    # TODO(NET-50): the auditor's deployment record is to give the certifier's key and run id; nothing writes it yet, so
    # the caller does.
    certifier_key: Any
    run_id: bytes
    challenger_key: Ch.ChallengerKey

    def __post_init__(self) -> None:
        key = self.certifier_key
        if not (type(getattr(key, "key_id", None)) is bytes and callable(getattr(key, "verify", None))
                and type(self.run_id) is bytes and len(self.run_id) == RUN_ID_BYTES
                and isinstance(self.challenger_key, Ch.ChallengerKey)):
            raise ValueError(DEPLOYMENT_FORM)

    @classmethod
    def pinned(cls, certifier_key: Any, run_id: bytes, record: Path | None = None) -> Deployment:
        """The deployment whose challenger key is the pinned one (`pinned_challenger_key`, from `record` or the store)."""
        return cls(certifier_key, run_id, pinned_challenger_key(record))


def stream_id(params: dict, stream: str, certifier: Mapping[str, str]) -> bytes:
    """A certified window's stream as the PoUW salt binds it (`salt.pouw_salt`), 64 bytes: the tagged SHA-512 of the
    deployment, the stream's challenger name and the certifier's run (`certifier_run`), so one stream behind two
    certifier links, or two runs of one, is two streams, and the once-only ledger (`salt.Registrations`), keyed on it,
    is keyed on the certifier run. `Stream.digest` can't be the one bound: its Spec reads the window's context, which
    names the pass's manifest and so its salt, and the registered level, which the traffic fixes."""
    return tagged_sha512(STREAM_ID, canonical_json({"certifier": dict(certifier), "challenger": f"{CHALLENGER}/{stream}",
                                                    "deployment": params["deployment"]}))


def _frame(tag: str, fields: dict, count: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(tag, fields), OWNER, RangeIndexedDomain(0, count), hash="sha512")


def _served(params: dict, ell: int, port: str, schema: str, count: int) -> CommitmentDomain:
    """One-stage's window domain (`registration.window_domain`) of the window's program and partition at level `ell`."""
    _, _, part = population(params["k"], work_at(params, ell).n)
    return _frame(R.WINDOW_DOMAIN, {"program": part["program"], "partition": PT.digest(part), "port": port, "schema": schema,
                                    "leaves": int(count), "run": params["context"]}, count)


def window_domain(params: dict, ell: int) -> CommitmentDomain:
    return _served(params, ell, PORT, ENTRY_SCHEMA, params["entries"])


def side_domain(params: dict, ell: int, side: str) -> tuple[CommitmentDomain, str]:
    """The domain of the window's side-`side` trees at level `ell` and their rows' schema."""
    return Wd.side_domain(window_domain(params, ell).domain_id, side), L.ROW_SCHEMA["fp8-pi", SIDES[side]]


def weights_domain(params: dict) -> CommitmentDomain:
    return _frame(W_DOMAIN, {"template": template(params["k"]).id, "port": "W", "schema": W_SCHEMA,
                             "leaves": params["w_rows"], "deployment": params["deployment"]}, params["w_rows"])


def ports(params: dict, ell: int) -> dict:
    """The verifier's own ports: the window root over P entries, and W's root over |W| rows."""
    return {PORT: {"leaves": params["entries"], "domain": window_domain(params, ell).domain_id.hex()},
            "W": {"leaves": params["w_rows"], "domain": weights_domain(params).domain_id.hex()}}


def level_of(rec: Mapping[str, Any]) -> int:
    try:
        ell = rec["window"]["public"]["work"]["level"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"the registration names no work level: {e}") from None
    if not isinstance(ell, int) or isinstance(ell, bool) or ell < 0:
        raise ValueError(f"the registered level {ell!r} is not a level")
    return ell


def public_problem(params: dict, rec: Any) -> Verdict:
    """PoUW's rule inside the window item, which the service's rule declares as one item: the window is
    ``{"kind": "served", "context", "public"}`` under the verifier's own context, its public items are exactly
    `pouw.window.REGISTERED`, each the verifier's own at the registered level (root_W the registered W root; the length a
    fixed-length window's at the verifier's max_tokens, `pouw.window.length_problem`, else ``window-form``), and N is the
    level's padded count."""
    c = _Collect()
    window = rec.get("window") if isinstance(rec, Mapping) else None
    if not isinstance(window, Mapping) or set(window) != {"kind", "context", "public"}:
        c.fail("public-window", sorted(window) if isinstance(window, Mapping) else window)
        return c.verdict()
    if window["kind"] != "served" or window["context"] != params["context"]:
        c.fail("public-context", {"registered": [window["kind"], window["context"]], "verifier": ["served", params["context"]]})
    items = window["public"]
    if isinstance(items, Mapping) and set(items) == set(Wd.LEGACY_REGISTERED):
        c.fail("window-form", "no-fixed-length: registered before the fixed-length rule (19 Oct), so not of record")
        return c.verdict()
    if not isinstance(items, Mapping) or set(items) != set(Wd.REGISTERED):
        c.fail("public-items", {"registered": sorted(items) if isinstance(items, Mapping) else items,
                                "declared": sorted(Wd.REGISTERED)})
        return c.verdict()
    why = Wd.length_problem(items["length"], params["length"]["max_tokens"])
    if why is not None:
        c.fail("window-form", why.split(": ", 1)[1])
        return c.verdict()
    try:
        ell = level_of(rec)
        w = work_at(params, ell)
    except ValueError as e:
        c.fail("level", str(e))
        return c.verdict()
    roots = {r.get("port"): r.get("root") for r in rec.get("roots") or () if isinstance(r, Mapping)}
    wrong = {k: {"registered": items[k], "verifier": v} for k, v in items_of(params, ell, roots.get("W")).items()
             if items[k] != v}
    if wrong:
        c.fail("public-value", wrong)
    if rec.get("population") != w.n:
        c.fail("population", {"registered": rec.get("population"), "level": w.n})
    return c.verdict()


# ---- rows, trees and openings ------------------------------------------------------------------------------------------------

def f32_rows(data, k: int, idx) -> list[bytes]:
    """Rows `idx` of a BF16 row file as the committed FP32 words (BF16 widened, exact)."""
    out = []
    for i in idx:
        out.append((np.frombuffer(data[2 * k * i:2 * k * (i + 1)], "<u2").astype("<u4") << 16).tobytes())
    return out


def _form(scheme: C.PearlC, ctx: Context, noise: C.Noise, k: int, side: str, rows: list[bytes], idx: list[int]
          ) -> list[list[int]]:
    """The fp8-p words of rows `idx` (FP32 bytes `rows`) on one side, as `PearlC.encode` forms them (A's E lines from its
    rows' own seeds under a row-seeded hashing); raises `P.Reject`."""
    seed_e = noise.seed_a if side == "a" else noise.seed_b
    rows_e = ([C.sample_line(x, 0, 0, 0) for x in scheme.row_seeds(ctx, rows, idx)]
              if side == "a" and scheme.hashing in C.ROW_SEEDED else None)
    f, e = scheme._side(rows, idx, 0 if side == "a" else 1, seed_e, noise.seed_b, k, rows_e)
    peel = C.peel_factors_a if side == "a" else C.peel_factors_b
    p = peel(f.codes, C.scaled_noise(f, e), noise.seed_b, k, scheme.line_norm, scheme.device)
    ok = [P.f32_of_bits(r) * P.f32_of_bits(a) >= P.SIGMA_MIN and C.live_row(x) for r, a, x in zip(f.rho, f.alpha, f.x)]
    return [L.words_fp8_p(c, q) if good else None for c, q, good in zip(f.codes, p, ok)]


def form(scheme: C.PearlC, ctx: Context, noise: C.Noise, k: int, side: str, rows: list[bytes], idx: list[int]
         ) -> list[list[int] | None]:
    """Each row's fp8-p words, or None for a row the scheme refuses (alone, so one refusal leaves its batch's others)."""
    out: list[list[int] | None] = []
    for at in range(0, len(idx), CHUNK):
        r, i = rows[at:at + CHUNK], idx[at:at + CHUNK]
        try:
            out += _form(scheme, ctx, noise, k, side, r, i)
        except P.Reject:
            for one, j in zip(r, i):
                try:
                    out += _form(scheme, ctx, noise, k, side, [one], [j])
                except P.Reject:
                    out.append(None)
    return out


def path(levels, index: int) -> list[str]:
    out = []
    for level in levels[:-1]:
        out.append(level[index ^ 1].hex())
        index >>= 1
    return out


verify_path = CC.verify_path


def side_leaves(dom: CommitmentDomain, schema: str, bcs: list[bytes]) -> list[bytes]:
    return [row_leaf(dom, i, L.tree_leaf(bc), schema) for i, bc in enumerate(bcs)]


entry_bc, entry_leaf = Wd.entry_bc, Wd.entry_leaf


def _salts(n: int) -> list[bytes]:
    return [os.urandom(SALT_BYTES) for _ in range(n)]


def commit_side(dom: CommitmentDomain, schema: str, side: str, words: list[list[int]], salts: list[bytes]) -> tuple[str, dict]:
    bcs = [L.commit("fp8-pi", SIDES[side], w, s) for w, s in zip(words, salts)]
    return Wd.side_levels(dom, side_leaves(dom, schema, bcs))[-1][0].hex(), {"words": np.array(words, dtype=np.uint16),
                                                                             "salts": salts, "bcs": bcs}


def commit_weights(params: dict, rows, salts: list[bytes] | None = None) -> tuple[str, list[bytes]]:
    """W's root: each BF16 weight row (raw bytes, k little-endian words) an ``hm96-sha512/row/v2`` row under a fresh salt."""
    dom = weights_domain(params)
    leaves, used = [], []
    for i, raw in enumerate(rows):
        s = salts[i] if salts else os.urandom(SALT_BYTES)
        used.append(s)
        bc = L.INST.commit_string(L.KEY, sha512_row_v2_digest(bytes(raw), 8 * len(raw)), s)
        leaves.append(row_leaf(dom, i, L.tree_leaf(bc), W_SCHEMA))
    if len(leaves) != params["w_rows"]:
        raise ValueError(f"W has {len(leaves)} rows, not the parameters' {params['w_rows']}")
    return tree_levels(dom, leaves)[-1][0].hex(), used


def filler_shape(fill: int) -> tuple[int, int]:
    """The filler's zero rows (A, B): about the square root of its tiles each, within the side trees' depths, so its tiles
    cover `fill`."""
    cap_a, cap_b = 1 << Wd.DEPTHS["a"], 1 << Wd.DEPTHS["b"]
    b = min(cap_b, max(1, math.isqrt(fill - 1) + 1, -(-fill // cap_a)))
    if -(-fill // b) > cap_a:
        raise ValueError(f"{fill} filler tiles are more than one entry's 2^{Wd.DEPTHS['a']} x 2^{Wd.DEPTHS['b']}")
    return -(-fill // b), b


def commit_window(params: dict, calls: list[dict], root_w: str, root_h3: bytes, salts: dict | None = None,
                  length: Mapping[str, Any] | None = None) -> tuple[dict, dict, dict]:
    """The window's commitment from its calls (each `{"record", "words": {"a", "b"}}`, the record holding the call's
    ``shape`` [m, k, n] and its first tile in the pass ``tile0``, a side's words None where the scheme refused the row) of
    the pass whose -h3 tile commitment's root is `root_h3`: (``commitment.json``, ``advice.json``, the prover's rows).  Each
    side tree holds the kept rows only, each with its place in the call (`Wd.side_row`); the call's record
    (`advice["records"]`, its SHA-512 in the entry) gains those places (``keep``), and the entry its tile0, cols and m.
    `salts` replays the salts of an earlier commit's prover file (`load`).  `length` is the registered length item
    (`served_length` of the pass's served record; none, a window registered before the fixed-length rule)."""
    k = params["k"]
    zero = Wd.side_row([0] * L.ROW_WORDS["fp8-p"](k), 0)
    keeps = [{s: [i for i, w in enumerate(c["words"][s]) if w is not None] for s in SIDES} for c in calls]
    real = sum(len(kp["a"]) * len(kp["b"]) for kp in keeps)
    work = Wd.bucket(real, params["grid_bits"])
    ell = work.level
    work_at(params, ell)
    if len(calls) + 1 > params["entries"]:
        raise ValueError(f"{len(calls)} calls and the filler are more entries than the bound {params['entries']}")
    fa, fb = filler_shape(work.n - real)
    sides = [{s: [Wd.side_row(c["words"][s][i], i) for i in kp[s]] for s in SIDES} for c, kp in zip(calls, keeps)]
    sides.append({"a": [zero] * fa, "b": [zero] * fb})
    records = [{**c["record"], "keep": kp} for c, kp in zip(calls, keeps)] + [None]
    entries, priv, at = [], {}, 0
    for e, (ws, kind) in enumerate(zip(sides, ["call"] * len(calls) + ["filler"])):
        if any(not 1 <= len(ws[s]) <= 1 << Wd.DEPTHS[s] for s in SIDES):
            raise ValueError(f"entry {e}: its sides keep {len(ws['a'])} and {len(ws['b'])} rows, not 1 to 2^D_A and 2^D_B")
        size = len(ws["a"]) * len(ws["b"]) if kind == "call" else work.n - real
        priv[e], roots = {}, {}
        for s in SIDES:
            dom, schema = side_domain(params, ell, s)
            root, priv[e][s] = commit_side(dom, schema, s, ws[s], salts[e][s] if salts else _salts(len(ws[s])))
            roots[s] = bytes.fromhex(root)
        call = kind == "call"
        place = {"tile0": records[e]["tile0"], "cols": -(-records[e]["shape"][2] // Wd.TILE), "m": records[e]["shape"][0]} \
            if call else {}
        entries.append(Wd.Entry(kind, at, at + size, len(ws["a"]), len(ws["b"]), roots["a"], roots["b"],
                                root_h3 if call else bytes(64), Wd.record_digest(records[e]) if call else bytes(64), **place))
        at += size
    entries += [Wd.Entry("none")] * (params["entries"] - len(entries))
    records += [None] * (params["entries"] - len(records))
    indices = [{s: [Wd.side_index(r) for r in ws[s]] for s in SIDES} for ws in sides]
    problem = Wd.window_problem([x.row() for x in entries], work, params["entries"], root_h3, params["grid_bits"],
                                records, indices)
    if problem:
        raise ValueError(f"the window statement fails on the committer's own advice: {problem}")
    dom = window_domain(params, ell)
    esalts = salts["entries"] if salts else _salts(len(entries))
    levels = tree_levels(dom, [entry_leaf(dom, e, entry_bc(x, s)) for e, (x, s) in enumerate(zip(entries, esalts))])
    roots = [{"port": PORT, "root": levels[-1][0].hex(), "leaves": params["entries"], "domain": dom.domain_id.hex()},
             {"port": "W", "root": root_w, "leaves": params["w_rows"], "domain": weights_domain(params).domain_id.hex()}]
    public = {"schema": SCHEMA, "context": params["context"], "level": ell, "items": items_of(params, ell, root_w, length),
              "roots": roots}
    advice = {"schema": SCHEMA, "level": ell, "real": real, "root_h3": root_h3.hex(), "entries": [x.to_json() for x in entries],
              "records": records,
              "entry_salts": [s.hex() for s in esalts], "window_path": [path(levels, e) for e in range(len(entries))]}
    priv["entries"] = esalts
    return public, advice, priv


def entries_of(advice: dict) -> list[Wd.Entry]:
    return [Wd.Entry.from_json(x) for x in advice["entries"]]


def save(p: Path, priv: dict) -> None:
    arrays = {}
    for e, sides in priv.items():
        if e == "entries":
            continue
        for s, x in sides.items():
            arrays[f"e{e}_{s}_words"] = x["words"]
            arrays[f"e{e}_{s}_salts"] = np.frombuffer(b"".join(x["salts"]), np.uint8).reshape(-1, SALT_BYTES)
            arrays[f"e{e}_{s}_bcs"] = np.frombuffer(b"".join(x["bcs"]), np.uint8).reshape(-1, 128)
    np.savez(p, **arrays)


def load(p: Path) -> dict:
    z = np.load(p)
    out: dict = {}
    for key in z.files:
        e, s, what = key.split("_")
        x = out.setdefault(int(e[1:]), {}).setdefault(s, {})
        x[what] = z[key] if what == "words" else [r.tobytes() for r in z[key]]
    return out


def opening(advice: dict, priv: dict, params: dict, unit: int) -> dict:
    """The prover's opening of unit `unit`: its entry with its salt and path to the window root, and each side's row with
    its path to the entry's side root."""
    e, i, j = Wd.tile_at(entries_of(advice), unit)
    out = {"unit": unit, "entry": e, "entry_value": advice["entries"][e], "entry_salt": advice["entry_salts"][e],
           "window_path": advice["window_path"][e]}
    for side, idx in (("a", i), ("b", j)):
        dom, schema = side_domain(params, advice["level"], side)
        levels = Wd.side_levels(dom, side_leaves(dom, schema, priv[e][side]["bcs"]))
        out[side] = {"index": idx, "path": [x.hex() for x in Wd.side_path(dom, levels, idx)]}
    return out


def core_params(params: dict) -> dict:
    """`core_check`'s parameters of the verifier's own window: its check item and its check's query."""
    return OR.core_params(params, check_item(params), R.query_digest(PT.template_instance_query(template(params["k"]))))


def check(params: dict, rec: dict, draw: dict, unit: int, pub: bytes, op: dict, drawn: bool = True,
          record: OR.Record | None = None, salt: bytes | None = None) -> str | None:
    """None if the proof's public file opens unit `unit`'s rows under the registered window root, else why not: PoUW's
    rule on the opening (`core_check.opening`, the unit drawn unless not `drawn`), of record (`record`, Lean's by
    default), beside this reading of the registration (its level the window's, its entries' domain the window's served
    one), which may refuse only what the rule refuses (`Record.prelude`).  `salt` is a certified window's salt, the one
    its pass was checked against (`core_check.certified_salt`, which `Challenger.check` passes); the opening refuses
    every unit of a certified window without it."""
    try:
        ell = level_of(rec)
        work_at(params, ell)
    except ValueError as e:
        pre: str | None = str(e)
    else:
        roots = {r["port"]: r for r in rec["roots"]}
        pre = None if roots.get(PORT, {}).get("domain") == window_domain(params, ell).domain_id.hex() else \
            "the registered window domain is not the window's served domain"
    x = {"params": core_params(params), "commitment": OR.hexed(rec), "units": list(draw.get("units", ())) if drawn else None,
         "unit": unit, "pub": pub.hex(), "opening": op, **({} if salt is None else {"salt": salt.hex()})}
    r = OR.Record() if record is None else record
    try:
        out = r.opening(x)
        return r.prelude(pre, f"opening: {out['refused']}" if isinstance(out, dict) else out)
    except OR.Disagree as e:
        return e.code


# ---- verdicts --------------------------------------------------------------------------------------------------------------

def _session_digest(session: Any) -> str | None:
    """A session record's SHA-512 (its directory's ``session.json``, the bytes `deliver` hashes); None when unreadable."""
    try:
        return hashlib.sha512((Path(session) / "session.json").read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def lean_verdict(path_: Path) -> dict:
    """The Lean verifier's verdict from its output (one `VERDICT {…}` line per session it verified), of record: accepted
    when every session's is, null when it printed none; ``sessions`` the SHA-512 of each session record it covers, in
    order (None for one it names but this process can't read), which `service.Stream.outcome` holds to the delivery."""
    try:
        lines = [json.loads(x[8:]) for x in path_.read_text().splitlines() if x.startswith("VERDICT ")]
    except (OSError, ValueError):
        lines = []
    if not lines:
        return {"of_record": True, "accepted": None, "verifier": "lean flock-verify verify --zk", "sessions": []}
    v = lines[0] if len(lines) == 1 else lines
    return {"of_record": True, "accepted": all(bool(x.get("accepted")) for x in lines),
            "why": next((x.get("why", "") for x in lines if not x.get("accepted")), ""), "verifier": "lean flock-verify verify --zk",
            "digest": hashlib.sha512(canonical_json(v)).hexdigest(), "sessions": [_session_digest(x.get("session")) for x in lines]}


def rust_verdict(path_: Path | None) -> dict | None:
    """The Rust prover-side verifier's verdict of the --zk session (`LIVE\\t{…}` lines), beside the verifier of record."""
    if path_ is None:
        return None
    try:
        lines = [json.loads(x[5:]) for x in path_.read_text().splitlines() if x.startswith("LIVE\t")]
    except (OSError, ValueError):
        lines = []
    v = lines[0].get("verdict", lines[0]) if lines else None
    return {"of_record": False, "accepted": None if v is None else bool(v.get("accepted")), "why": (v or {}).get("why", ""),
            "verifier": "rust flock-circuit --zk (live)"}


def statement(stage: dict, pub: bytes) -> dict:
    """A unit's statement by digest: its template, circuit and class, its public file, its size; nothing of its tile."""
    return {"definition": stage["definition"], "circuit_sha512": stage["circuit_sha512"], "class_sha512": stage["class_sha512"],
            "public_sha512": hashlib.sha512(pub).hexdigest(), "ands": stage["ands"], "k_log": stage["k_log"],
            "binding": stage["binding"]}


# ---- the service ------------------------------------------------------------------------------------------------------------

def _refused(code: str, detail: Any) -> S.Refused:
    return S.Refused(Verdict(False, (code,), {code: detail}))


def register(params: dict, commitment: dict, extra: Mapping[str, Any] = {}) -> dict:
    """The committer's registration of `commitment`: one-stage's v1 record of the level's population, the check's query,
    subset:K′ and `service.SCHEME`, its window the public items (with `extra`, which both rules refuse).  Raises
    `service.Refused` under the service's rule or PoUW's."""
    ell = commitment["level"]
    try:
        prog, q, _ = population(params["k"], work_at(params, ell).n)
    except ValueError as e:
        raise _refused("level", str(e)) from None
    window = {"kind": "served", "context": params["context"], "public": {**commitment["items"], **extra}}
    rec = R.record(prog, q, law_of(params, ell), S.SCHEME, commitment["roots"], window)
    for v in (S.check_public(spec_of(params, ell), rec), public_problem(params, rec)):
        if not v.ok:
            raise S.Refused(v)
    return rec


def _all(units: list[dict], name: str) -> dict[str, Any]:
    """The sampled proof blocks' verdicts by one verifier as one verdict: accepted when every unit's is, null while one hasn't run;
    its sessions those each unit's verdict covers, in the units' order (a verdict of record must cover the delivered ones)."""
    vs = [u["verdicts"].get(name) for u in units]
    if all(v is None for v in vs):
        return {}
    ran = [v for v in vs if v is not None and v.get("accepted") is not None]
    accepted = None if len(ran) < len(vs) else all(v["accepted"] for v in ran)
    return {name: {"of_record": any(bool(v and v.get("of_record")) for v in vs), "accepted": accepted,
                   "why": next((v.get("why", "") for v in ran if not v["accepted"]), ""),
                   "digest": hashlib.sha512(canonical_json([(v or {}).get("digest") for v in vs])).hexdigest(),
                   "units": [(v or {}).get("accepted") for v in vs],
                   "sessions": [d for v in vs for d in (v or {}).get("sessions", ())]}}


#: a sample over the key's stream: (law, N, the stream's bytes) to the units
Sampler = Callable[[Mapping[str, Any], int, bytes], list[int]]


def draw_key(coin: bytes, stream: S.Stream, reg: S.Registration) -> Key:
    """The window sample's key: the epoch's sampling coin, under the stream and the registration and receipt it was opened after."""
    return derive(coin, DRAW_DOMAIN, CC.draw_key_context(stream.digest, reg.digest, dict(reg.receipt)))


def draw_bytes(key: Key, law: Mapping[str, Any], n: int) -> bytes:
    """The bytes a subset:k draw over N reads (Lean's ``2 k · width(N) + 64``)."""
    if law.get("law") != "subset":
        raise _refused("law", f"the window's draw is a subset law's, not {dict(law)}")
    return key.stream("draw", 2 * law["k"] * CC.width(n) + 64)


def python_sampler(law: Mapping[str, Any], n: int, src: bytes) -> list[int]:
    units = CC.stream_subset(law["k"], n, src)
    if units is None:
        raise _refused("sampler", "the draw's stream ran out")
    return units


class Recorded:
    """The epoch's coins as a replay holds them: the published commitment and the openings the challenger recorded.
    Each opens once, and only after what it was recorded opened after; `Challenger.open` checks it against the commitment."""

    def __init__(self, commitment: CoinCommitment, openings: list[CoinOpening]) -> None:
        self.commitment, self._openings, self._opened = commitment, {o.label: o for o in openings}, set()

    def open(self, label: str, after: bytes) -> CoinOpening:
        if label in self._opened:
            raise DomainError(f"epoch {self.commitment.epoch}'s {label!r} coin is already open: a retry never redraws")
        self._opened.add(label)
        o = self._openings.get(label)
        if o is None or o.after != after:
            raise DomainError(f"the record holds no {label!r} coin opened after {after.hex()[:32]}")
        return o


def _signed_coins(path: str | os.PathLike | None, key: Ch.ChallengerKey | None) -> tuple[Ch.LiveCommitment, dict]:
    """A certified window's trusted coin commitment, which the challenger reads itself from the record its challenger
    signed and wrote at `path` under the verifier's `key` (`challenger.trusted_commitment`), and the record with the key
    as the sample's check of record reads them (`core_check.draw` verifies the record again).  Raises `ValueError` when
    there is no such record, or it isn't a live epoch's."""
    if path is None or not isinstance(key, Ch.ChallengerKey):
        raise ValueError("a certified window's challenger reads its coin commitment from its challenger's signed record "
                         "(`coin_record`, the path it was written to) under the challenger key its deployment pins "
                         "(`Deployment.challenger_key`)")
    try:
        trusted = Ch.trusted_commitment(path, key, CC.lms_verify)
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        same = Ch.ChallengerRecord.from_json(doc).commitment == trusted
    except (Ch.CoinError, OSError, ValueError) as e:
        raise ValueError(f"a certified window's draw coin opens against its challenger's signed record: {e}") from None
    if not same:
        raise ValueError(f"the challenger record at {path} changed while the challenger read it")
    if (why := Wd.live_epoch_reason(trusted)) is not None:
        raise ValueError(f"a certified window's draw coin is a live epoch's: {why}")
    return trusted, {"key": key.to_json(), "record": doc}


def _pd():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
    from verity_vllm.protocols.pouw import pearl_c as PD
    return PD


class Challenger:
    """The challenger of one deployment's windows over P1's `Stream`: PoUW's rule (`core_check.registered`), the pass's
    epoch under its own coins, then the stream's, each of PoUW's checks of record in `record` (Lean's, by default).  `coins` is the epoch's coin challenger (`pouw.coins.Challenger`, or
    `Recorded` in a replay); the window's sampling coin keys `sampler` (Lean's sample, or its twin `python_sampler`).  A Spec
    fixes one law and K′ is a function of the registered level, so it holds one Stream per (stream name, level); each
    registration it accepted is held by name.  A control stream `replay`s an earlier draw and opens no coin, so a
    control's verdicts are judged on the honest draw's units.  The window's Check has a proof deadline (`spec_of`): a
    drawn window whose proof isn't delivered by it is `late`.  It has a registration deadline too, and `params` without
    ``registration_deadline`` are refused (`NO_DEADLINE`): the salt coin's opening reads the clock (its release), and
    each stream records the epoch's salt at that release (`Stream.salt_released`) before its first registration, so a
    window received later than the deadline, or before the salt is open, is refused (`late-registration`) and earns no
    credit.  A window received after the epoch's sampling coin was released is refused as `late-registration` too: the
    record's order is each root's receipt before the coin's release.  A control (`replay`) opens no coin and is exempt.
    `predates_deadline` replays a record whose parameters predate the deadline as it ran, untimed.  The clock counts the
    challenger's reads of it unless `clock` is given.

    With `certifier` the window is certified (module docstring): `certifier` is the network certifier's client
    (`certifier.active.RemoteCertifier`), whose `certifier_run` names the stream (`stream_id`) and whose `committed` and
    `reveal` are the certifier's signed answers to the challenger's signed requests, verified; `prover` is its 64-byte
    id (`salt.prover_id`), `coins` the live epoch's challenger (`window.live_challenger`) whose ``pouw/draw`` coin 0 the
    sample opens, `coin_record` the path of the record that challenger signed and wrote on its own host
    (`challenger.write_record`), and `deployment` what the verifier pins of its deployment (`Deployment`): the network
    certifier's key and run id, which `certifier` must be the client for, and its challenger's key, under which the
    challenger reads its trusted commitment itself (`challenger.trusted_commitment`).  Every opening is checked against
    it, and the sample's check of record verifies the record again under the key the deployment pins, not the one the
    record's coins name (``coin-key``).  `registrations` is the deployment's `salt.Registrations`,
    shared by its challengers.  Each registered window's salt, the one its pass was checked against, is held
    (`salts`), and `check` opens its sampled proof blocks under it."""

    def __init__(self, params: dict, coins: Any, sampler: Sampler = python_sampler, budget: int = BUDGET,
                 clock: Callable[[], float] | None = None, record: OR.Record | None = None,
                 predates_deadline: bool = False, certifier: Any = None, prover: bytes | None = None,
                 coin_record: str | os.PathLike | None = None, deployment: Deployment | None = None,
                 registrations: PS.Registrations | None = None) -> None:
        self.trusted: Ch.LiveCommitment | None = None
        self.signed: dict | None = None
        self.deployment: Deployment | None = None
        if certifier is not None:
            # TODO(RT4-INT-7): no Lean statement covers a certified window's timing, and no run measures the commit
            # pipeline the certifier's reveal deadline must be sized against.
            if params.get("registration_deadline") is not None or predates_deadline:
                raise ValueError(CERTIFIED_DEADLINE)
            if type(prover) is not bytes or len(prover) != PS.PROVER_BYTES:
                raise ValueError(f"a certified window's prover is its {PS.PROVER_BYTES}-byte id (`salt.prover_id`)")
            if not certified(params):
                raise ValueError("a certified window's context names its PoUW window (`salt.context`)")
            if not isinstance(deployment, Deployment):
                raise ValueError(f"a certified window's challenger takes its deployment (`Deployment`): {DEPLOYMENT_FORM}")
            self.run = certifier_run(certifier)
            if (getattr(certifier, "certifier_key", None) != deployment.certifier_key
                    or getattr(certifier, "run_id", None) != deployment.run_id):
                raise ValueError("a certified window's certifier client is for the certifier key and run id its "
                                 "deployment pins (`Deployment`)")
            self.trusted, self.signed = _signed_coins(coin_record, deployment.challenger_key)
            self.deployment = deployment
        elif certified(params):
            raise ValueError("the window's context names a PoUW window, so it is certified: its challenger holds the "
                             "network certifier (`certifier`)")
        elif (params.get("registration_deadline") is None) != predates_deadline:
            raise ValueError(NO_DEADLINE if not predates_deadline else
                             "the window's parameters carry a registration_deadline: only a record that predates it replays "
                             "without one")
        self.predates_deadline = predates_deadline
        self.certifier, self.prover = certifier, prover
        self.registrations = PS.Registrations() if registrations is None else registrations
        self.salts: dict[str, bytes] = {}
        self.params, self.coins, self.sampler, self.budget = params, coins, sampler, budget
        self.record = OR.Record() if record is None else record
        self.core = core_params(params)
        self.witness: dict[str, list[dict]] = {}
        self.recorded: dict[str, bool] = {}
        ticks = itertools.count()
        self.clock = clock or (lambda: float(next(ticks)))
        self.streams: dict[tuple[str, int], S.Stream] = {}
        self.regs: dict[str, tuple[S.Stream, S.Registration]] = {}
        self.replays: dict[str, str] = {}
        self.passes: dict[str, str] = {}
        self.manifests: dict[str, Any] = {}   # each registered window's pass manifest, read (`_distinct`)
        self.draws: dict[str, S.Draw] = {}
        self.sessions: dict[str, list[str]] = {}
        self.openings: dict[str, CoinOpening] = {}
        self.salt_released_at: float | None = None
        self._salted: set[tuple[str, int]] = set()
        self._key: Key | None = None

    def _commitment(self) -> Any:
        """The coin commitment the verifier checks openings against: the trusted live one, or the seeded epoch's."""
        return self.coins.commitment if self.trusted is None else self.trusted

    def _coins_json(self) -> dict:
        """The coins the checks of record read: a certified window's challenger record and the key it is signed under
        (`core_check.draw` verifies it), or the seeded epoch's commitment."""
        return self.coins.commitment.to_json() if self.signed is None else self.signed

    def open(self, label: str, after: bytes) -> CoinOpening | Ch.LiveOpening:
        """The epoch's `label` coin, opened once after `after` and checked against the published commitment.  A certified
        window's one coin is its sampling coin, ``pouw/draw`` 0 of its live epoch, checked against the trusted commitment."""
        if self.trusted is not None:
            if label != "draw":
                raise _refused("coin", f"a certified window's salt is the network certifier's: it opens no {label} coin")
            try:
                o = self.coins.open(Wd.DRAW, 0, after)
            except Ch.CoinError as e:
                raise _refused("coin", str(e)) from None
            why = Ch.live_reason(self.trusted, o, Wd.DRAW, 0, after)
        else:
            try:
                o = self.coins.open(label, after)
            except DomainError as e:
                raise _refused("coin", str(e)) from None
            why = coin_reason(self.coins.commitment, o, label, after)
        if why is not None:
            raise _refused("coin", f"the {label}'s {why}")
        self.openings[label] = o
        if label == "salt" and not self.predates_deadline:
            self.salt_released_at = self.clock()
        return o

    def _of_record(self, name: str, check: Callable[..., Any], *args: Any) -> Any:
        """`check`'s verdict of record (`of_record`), both verdicts kept in window `name`'s witness."""
        at = len(self.record.witness)
        try:
            return check(*args)
        except OR.Disagree as e:
            raise _refused(e.code, {"lean": e.lean, "python": e.python}) from None
        finally:
            self.witness.setdefault(name, []).extend(self.record.witness[at:])

    def registered(self, name: str, rec: Mapping[str, Any]) -> None:
        """PoUW's rule over the registration (`core_check.registered`), of record.  Raises `Refused`."""
        try:
            x = {"params": self.core, "commitment": OR.hexed(rec)}
        except TypeError as e:
            raise _refused("registration", f"the registration is not canonical JSON: {e}") from None
        out = self._of_record(name, self.record.registered, x)
        if "refused" in out:
            raise _refused("registration", out["refused"])
        if out["reason"] is not None:
            raise _refused(out["reason"].split(":")[0], out["reason"])

    def pass_epoch(self, raw: bytes | None, recorded: bool = False, name: str = "", **bound: bytes) -> str:
        """How the pass's salt was had (``coin``, ``recorded beacon`` for a recorded replay's, or ``certifier``), its
        manifest `raw` re-started under the challenger's own coin commitment, or a certified window's `bound`
        (`core_check.pass_epoch`'s ``reveal``, ``committed``, ``stream`` and ``prover``), and named by the window's
        parameters, of record.  Raises `Refused`."""
        out = self._of_record(name, self.record.pass_epoch, {
            "coins": self._commitment().to_json(), "raw": None if raw is None else raw.hex(),
            "context": self.params["context"], "recorded": bool(recorded),
            **{k: v.hex() for k, v in bound.items()}})
        if "how" not in out:
            raise _refused(*out["refused"]) if isinstance(out["refused"], list) else _refused("manifest", out["refused"])
        return out["how"]

    def r1_r6(self, name: str, st: S.Stream, rec: Mapping[str, Any], ports_: dict) -> None:
        """One-stage's R1-R6 against the challenger's own copies and the service's public-items rule over the Stream's
        Spec, as `Stream.register` runs them, each of record.  Raises `Refused`."""
        n = rec["population"]
        law = D.parse_law(st.spec.selection.text)
        prog = PT.population_program(st.unit, n)
        q = st.spec.check.query
        codes = self._of_record(name, self.record.registration,
                                OR.registration_params(prog, q, law, S.SCHEME, ports_, list(st.log), self.params["delta"]),
                                dict(rec), lambda: R.check(rec, prog, q, law, S.SCHEME, ports_, None, (), list(st.log), None,
                                                          None, None).codes)
        public = self._of_record(name, self.record.public, OR.public_params(st.spec, rec, S.SCHEME),
                                 lambda: S.check_public(st.spec, rec).to_json())
        codes = list(codes) + [c for c in public.get("codes", ["public"]) if c not in codes]
        if codes:
            raise S.Refused(Verdict(False, tuple(codes), {c: "R1-R6 and the public-items rule, of record" for c in codes}))

    def _keyed(self, law: Mapping[str, Any], n: int) -> list[int]:
        if self._key is None:
            raise _refused("coin", "no draw coin is open for this draw")
        return self.sampler(law, n, draw_bytes(self._key, law, n))

    def stream(self, name: str, ell: int, sampler: Callable[[Mapping[str, Any], int], list[int]]) -> S.Stream:
        if (name, ell) not in self.streams:
            self.streams[name, ell] = S.Stream(spec_of(self.params, ell, self.predates_deadline), template(self.params["k"]),
                                               budget=self.budget,
                                               delta=float(Fraction(self.params["delta"])), challenger=f"{CHALLENGER}/{name}",
                                               clock=self.clock, sampler=sampler)
        return self.streams[name, ell]

    def _held(self, name: str) -> tuple[S.Stream, S.Registration]:
        if name not in self.regs:
            raise _refused("unknown", name)
        return self.regs[name]

    def _drawn(self, name: str) -> S.Draw:
        if name not in self.draws:
            raise _refused("not-drawn", name)
        return self.draws[name]

    def register(self, name: str, rec: Mapping[str, Any], stream: str = "window", replay: str | None = None,
                 manifest: bytes | None = None, recorded: bool = False) -> S.Registration:
        """Receive window `rec` from the pass whose manifest is `manifest` (`pass_epoch`), or a control's (`replay`).  A
        window received after the epoch's sampling coin was released is refused (`late-registration`)."""
        self.witness[name] = []
        if replay is None and not self.predates_deadline and "draw" in self.openings:
            raise _refused("late-registration", f"received after the epoch's draw coin was released (opened after "
                                                f"registration {self.openings['draw'].after.hex()[:32]}): a window "
                                                f"registers before the draw coin")
        self.registered(name, rec)
        if self.certifier is not None:
            return self._certified(name, rec, stream, replay, manifest, recorded)
        how = self.pass_epoch(manifest, recorded, name) if replay is None else None
        ell = level_of(rec)
        sampler = self._keyed if replay is None else (lambda law, n, d=self._drawn(replay): d.units)
        st = self.stream(stream, ell, sampler)
        if not self.predates_deadline and self.salt_released_at is not None and (stream, ell) not in self._salted:
            st.salt_released(self.openings["salt"].coin, self.salt_released_at)
            self._salted.add((stream, ell))
        own = ports(self.params, ell)
        st.ready()
        self.r1_r6(name, st, rec, own)
        reg = st.register(rec, own)
        if replay is None:
            self._distinct(name, manifest)
        self.regs[name] = (st, reg)
        self.recorded[name] = bool(recorded)
        if replay is None:
            self.passes[name] = how
        else:
            self.replays[name] = replay
        return reg

    def _certified(self, name: str, rec: Mapping[str, Any], stream: str, replay: str | None, manifest: bytes | None,
                   recorded: bool) -> S.Registration:
        """A certified window's registration (`register`).  First the window's sampled proof block must read its salt
        (`salt.unit_reason`, ``unsalted``), which this window's `Pc8TileSide` doesn't, so today every certified window is
        refused here, before its stream receives it and before the certifier is asked for anything.  Past that, the
        stream receives it (R1-R7 and the receipt), then the certifier gives what the prover recorded for PoUW window p
        before it sampled p's salt (`salt.committed`), which must bind the pass's weights, the window's stream (`stream_id`)
        and its prover (`core_check.recorded_reason`, ``coin``) before the challenger asks for p's salt
        (`salt.revealed`), since the certifier reveals it once; either refused by the certifier is ``certifier``.  Then
        the pass's epoch under them, of record, and p's entry in the registration ledger (``reregistered``: once per
        stream and prover).  The certifier needs the receipt, so a window refused after it stays in its stream without an
        outcome: its failure is visible and has no retry."""
        if replay is not None or recorded:
            raise _refused("certifier", "a certified window is its own pass's: no control replays it and no recorded "
                                        "pass salts it")
        why = PS.unit_reason(template(self.params["k"]))
        if why is not None:
            raise _refused("unsalted", why)
        p, sid, ell = PS.window_of(self.params["context"]), stream_id(self.params, stream, self.run), level_of(rec)
        st = self.stream(stream, ell, self._keyed)
        own = ports(self.params, ell)
        st.ready()
        self.r1_r6(name, st, rec, own)
        reg = st.register(rec, own)
        try:
            committed = PS.committed(self.certifier, p, reg)
        except PS.Unrevealed as e:
            raise _refused("certifier", str(e)) from None
        why = CC.recorded_reason(manifest, committed, sid, self.prover)
        if why is not None:
            raise _refused("coin", why)
        try:
            reveal = PS.revealed(self.certifier, p, reg)
        except PS.Unrevealed as e:
            raise _refused("certifier", str(e)) from None
        how = self.pass_epoch(manifest, False, name, reveal=reveal, committed=committed, stream=sid, prover=self.prover)
        salt = CC.certified_salt(manifest, self.params["context"], reveal, committed, sid, self.prover)
        self._distinct(name, manifest)
        try:
            self.registrations.admit(p, sid, self.prover, reg.digest)
        except PS.Reregistered as e:
            raise _refused("reregistered", str(e)) from None
        self.regs[name] = (st, reg)
        self.recorded[name] = False
        self.passes[name] = how
        self.salts[name] = salt
        return reg

    def _distinct(self, name: str, manifest: bytes | None) -> None:
        """Window `name`'s pass, credited beside every pass a window registered before it (`manifests`): no call index
        twice under one salt, each below 2^32 (`credit.passes_distinct`), so a pass registered again, on another stream,
        is refused (``repeated-call``).  Raises `Refused`; held once it passes."""
        if manifest is None:
            return
        m = json.loads(manifest)
        why = CR.passes_distinct([*self.manifests.values(), m])
        if why is not None:
            raise _refused(*why)
        self.manifests[name] = m

    def check(self, name: str, unit: int, pub: bytes, op: dict, drawn: bool = True) -> str | None:
        """`check` of window `name`'s unit `unit` against its registration and draw, of record (`record`), under the salt
        its pass was checked against when it is certified (`salts`)."""
        _, reg = self._held(name)
        return check(self.params, dict(reg.record), self._drawn(name).draw, unit, pub, op, drawn, self.record,
                     self.salts.get(name))

    def select(self, name: str) -> S.Draw:
        """The sample, once: a window's opens the epoch's sampling coin after its registration and samples over `draw_key`;
        nothing resamples, and a coin opened once is never opened again (a failed draw's retry is refused)."""
        st, reg = self._held(name)
        if name in self.draws:
            raise _refused("drawn", name)
        if name in self.replays:
            self.draws[name] = st.select(reg)
            return self.draws[name]
        self._key = draw_key(self.open("draw", bytes.fromhex(reg.digest)).coin, st, reg)
        try:
            d = st.select(reg)
        finally:
            self._key = None
        out = self._of_record(name, self.record.draw, {
            "coins": self._coins_json(), "recorded": self.recorded[name],
            "opening": self.openings["draw"].to_json(), "stream": st.digest, "registration": reg.digest,
            "receipt": dict(reg.receipt), "k": reg.record["law"]["k"], "population": reg.record["population"],
            "units": list(d.units),
            "key": None if self.deployment is None else self.deployment.challenger_key.to_json()})
        if out is not None:
            raise _refused(*out["refused"]) if isinstance(out["refused"], list) else _refused("draw", out["refused"])
        self.draws[name] = d
        return d

    def coins_of(self, name: str) -> dict | None:
        """A window's coins, for its outcome: the epoch's commitment, how its pass's salt was had and its sampling coin."""
        if name not in self.passes:
            return None
        draw = self.openings.get("draw")
        return {"commitment": self._commitment().to_json(), "pass": self.passes[name],
                "draw": draw.to_json() if draw is not None and draw.after == bytes.fromhex(self.regs[name][1].digest) else None}

    def deliver(self, name: str, sessions: list[str]) -> S.Outcome | None:
        st, _ = self._held(name)
        o = st.deliver(self._drawn(name), sessions)
        self.sessions[name] = list(sessions)
        return o

    def outcome(self, name: str, units: list[dict]) -> tuple[S.Outcome, dict, dict]:
        """The final outcome from each sampled proof block's ``{"unit", "statement", "verdicts"}``, ascending, and PoUW's public
        items of it (exactly `pouw.window.PUBLIC`) and the stream's record of it.  PoUW's check over the stream's answer
        is of record: the stream's outcome is final once given, so a refusal here (``pouw-check``, or a disagreement)
        leaves the window with no published outcome."""
        st, reg = self._held(name)
        d = self._drawn(name)
        if [u["unit"] for u in units] != D.proved(d.draw):
            raise _refused("units", {"given": [u["unit"] for u in units][:16], "drawn": D.proved(d.draw)[:16]})
        verdicts = {k: v for n in sorted({n for u in units for n in u["verdicts"]}) for k, v in _all(units, n).items()}
        o = st.outcome(d, verdicts, {"units": [{"unit": u["unit"], **u["statement"]} for u in units]})
        public = {**reg.record["window"]["public"], "verdict": o.accepted, "drawn": d.units}
        body = st.records[-1]
        ans = OR.service_answer(o.accepted, list(o.verdict.codes), o.profile.to_json() if o.profile else None, reg.digest,
                                (body.get("audit") or {}).get("draw"), self.params["delta"])
        a = self._of_record(name, self.record.answer, {"params": self.core, "commitment": OR.hexed(reg.record),
                                                       "evidence": OR.hexed(ans)})
        if o.accepted:
            if a["verdict"] != "accept":
                raise _refused("pouw-check", a["reason"])
            public = a["public"]
        if set(public) != {p[0] for p in Wd.PUBLIC}:
            raise _refused("public", sorted(public))
        return o, public, body


def _sessions(units: list[dict]) -> list[str]:
    return [hashlib.sha512(Path(u["session"]).read_bytes()).hexdigest() for u in units]


def _units(rows: list[dict]) -> list[dict]:
    """The outcome's units from the job's rows (each unit's stage, public file, Lean output, opening verdict, live log)."""
    out = []
    for u in rows:
        pub = Path(u["pub"]).read_bytes()
        out.append({"unit": u["unit"], "statement": statement(_json(u["stage"]), pub),
                    "verdicts": {"lean": lean_verdict(Path(u["lean"])), "opening": _json(u["opening"]),
                                 **({"rust": rust_verdict(Path(u["rust"]))} if u.get("rust") else {})}})
    return out


def _put(p: str | None, obj: Any) -> None:
    if p:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(json.dumps(obj, sort_keys=True))


def handle(ch: Challenger, cmd: Mapping[str, Any]) -> tuple[str, dict]:
    """One request to the challenger, as a reply word and its JSON.  Requests: ``coins`` (the epoch's coin commitment),
    ``salt`` (``after`` hex, the weights' root; ``out`` the salt coin's opening), and with a registration's ``name``:
    ``register`` (``registration`` a record's file, ``manifest`` its pass's manifest and ``recorded`` true for a pass
    recorded before epoch coins; ``stream`` and ``replay`` for a control; ``out`` the receipt), ``select`` (``out`` the
    sample; the reply adds the stream's digest the sample read), ``deliver`` (``units``: the job's rows, each with its ``session`` record) and ``outcome`` (``units``; ``out`` a
    directory: outcome.json, audit.json).  Replies: COINS, OPENED, REGISTERED, DRAWN, DELIVERED, ACCEPT, REJECT, LATE,
    PENDING, REFUSED, or ERROR for a request it couldn't read."""
    op, name = cmd.get("op"), cmd.get("name")
    try:
        if op == "coins":
            return "COINS", ch._commitment().to_json()
        if op == "salt":
            o = ch.open("salt", bytes.fromhex(cmd["after"]))
            _put(cmd.get("out"), o.to_json())
            return "OPENED", {"label": o.label, "after": o.after.hex()}
        if op == "register":
            manifest = Path(cmd["manifest"]).read_bytes() if cmd.get("manifest") else None
            reg = ch.register(name, _json(cmd["registration"]), cmd.get("stream", "window"), cmd.get("replay"), manifest,
                              cmd.get("recorded") in (True, "true"))
            _put(cmd.get("out"), reg.receipt)
            return "REGISTERED", {"registration": reg.digest[:32], "order": reg.order, "population": reg.record["population"],
                                  "law": reg.record["law"], "public": sorted(reg.record["window"]["public"])}
        if op == "select":
            d = ch.select(name)
            _put(cmd.get("out"), d.draw)
            return "DRAWN", {**d.draw, "stream": ch._held(name)[0].digest}
        if op == "deliver":
            o = ch.deliver(name, _sessions(_json(cmd["units"])))
            return ("DELIVERED", {"sessions": len(ch.sessions[name])}) if o is None else (o.verdict.codes[0].upper(), {})
        if op == "outcome":
            o, public, body = ch.outcome(name, _units(_json(cmd["units"])))
            if cmd.get("out"):
                _put(f"{cmd['out']}/outcome.json", {"accepted": o.accepted, "verdict": o.verdict.to_json(), "public": public,
                                                    "profile": o.profile.to_json() if o.profile else None,
                                                    "terms": {"sampling": o.terms.sampling}, "record": o.record,
                                                    "coins": ch.coins_of(name), "of_record": ch.witness.get(name)})
                _put(f"{cmd['out']}/audit.json", body)
            audit = body["audit"] or {}
            return "ACCEPT" if o.accepted else o.verdict.codes[0].upper(), {
                "verdicts": {k: v.get("accepted") for k, v in (audit.get("verdicts") or {}).items()},
                "public": sorted(public), "record": o.record[:32]}
        return "REFUSED", {"codes": ["op"], "detail": {"op": op}}
    except S.Refused as e:
        return "REFUSED", {"codes": list(e.verdict.codes), "detail": e.verdict.detail}
    except S.Pending as e:
        return "PENDING", {"why": str(e)}
    except (OSError, KeyError, TypeError, ValueError) as e:
        return "ERROR", {"why": f"{type(e).__name__}: {e}"}


# ---- the CLI ------------------------------------------------------------------------------------------------------------

def _json(p) -> dict:
    return json.loads(Path(p).read_text())


def _pass(pass_dir: Path):
    PD = _pd()
    raw = (pass_dir / "manifest.json").read_bytes()
    man = json.loads(raw)
    if man.get("schema") not in PD.MANIFEST_SCHEMAS:
        raise SystemExit(f"{pass_dir}: not a retained pass ({man.get('schema')!r})")
    calls = [u for u in man["matmuls"] if u["k"] == K and u["n"] == K]
    if not calls:
        raise SystemExit(f"{pass_dir}: no call of k = n = {K}")
    return PD, man, hashlib.sha512(raw).hexdigest(), calls


def root_h3_of(man: dict) -> bytes:
    """The root of the pass's -h3 tile commitment (its manifest's ``commitment``), which each call's entry registers."""
    root = bytes.fromhex((man.get("commitment") or {}).get("root") or "")
    if len(root) != 64:
        raise SystemExit(f"the pass's tile commitment root is {len(root)} bytes, not a SHA-512 (-h3) root's 64")
    return root


def _weights(man: dict, calls: list[dict]) -> list[int]:
    return sorted({int(u["weight"]) for u in calls})


def cmd_params(a) -> int:
    """The verifier's own parameters, from the served request it audits: its run and manifest name the window, the
    model's k = n = K weights give |W|, the request's one slot its fixed length (``--max-tokens``, ignore_eos), and the
    registration deadline is the window's td."""
    _, man, sha, calls = _pass(Path(a.pass_dir))
    rows = sum(next(u["n"] for u in calls if int(u["weight"]) == w) for w in _weights(man, calls))
    deployment = f"served-zk/{man.get('run')}"
    params = {"context": f"{deployment}/{sha}", "deployment": deployment, "k": K, "eps": a.eps, "delta": a.delta,
              "grid_bits": Wd.GRID_BITS, "range": list(RANGE), "entries": a.entries, "w_rows": rows,
              "length": {"max_tokens": [a.max_tokens], "stop": Wd.STOP_FIXED},
              "registration_deadline": a.registration_deadline}
    Path(a.out).write_text(json.dumps(params, sort_keys=True))
    print(json.dumps(params), flush=True)
    return 0


def cmd_commit(a) -> int:
    pass_dir = Path(a.pass_dir)
    PD, man, sha, calls = _pass(pass_dir)
    import fast_tile as FT
    params = _json(a.params)
    scheme = PD.scheme_of(man["scheme"])
    if tuple(scheme.tile) != (Wd.TILE, Wd.TILE):
        raise SystemExit(f"{man['scheme']}: its tiles are {scheme.tile}, not the window's 64 x 64 pass tiles")
    source = os.urandom(32)
    if a.call:
        chosen = [next(x for x in calls if x["index"] == i) for i in a.call]
    else:
        order = sorted(calls, key=lambda u: hashlib.sha256(source + u["index"].to_bytes(8, "little")).digest())
        chosen = sorted(order[:a.calls], key=lambda u: u["index"])
    tile0, at = {}, 0
    for u in man["matmuls"]:
        tile0[u["index"]] = at
        at += -(-u["m"] // Wd.TILE) * -(-u["n"] // Wd.TILE)
    chosen = sorted(chosen, key=lambda u: tile0[u["index"]])
    undo = FT.install(scheme)
    window = []
    try:
        for u in chosen:
            w = man["weights"][str(u["weight"])]
            k, n, real = u["k"], u["n"], u["real_rows"]
            ctx = Context(bytes.fromhex(man["salt"]), Matmul(u["index"], Shape(u["m"], k, n), u["weight"]),
                          bytes.fromhex(u["root_a"]), bytes.fromhex(w["root"]))
            noise = scheme.noise(ctx)
            a_data = (pass_dir / u["rows"]["path"]).read_bytes()
            b_data = np.memmap(pass_dir / w["rows"]["path"], dtype=np.uint8, mode="r")
            words = {"a": form(scheme, ctx, noise, k, "a", f32_rows(a_data, k, range(real)), list(range(real))),
                     "b": form(scheme, ctx, noise, k, "b", f32_rows(b_data, k, range(n)), list(range(n)))}
            record = {"matmul": u["index"], "shape": [u["m"], k, n], "tile0": tile0[u["index"]], "weight": u["weight"],
                      "real_rows": real,
                      "served": {"root_a": u["root_a"], "weight_root": w["root"]}}
            window.append({"record": record, "words": words})
            print(json.dumps({"call": u["index"], "excluded": {s: sum(x is None for x in v) for s, v in words.items()}}), flush=True)
    finally:
        undo()

    def w_rows():
        for wid in _weights(man, calls):
            data = np.memmap(pass_dir / man["weights"][str(wid)]["rows"]["path"], dtype=np.uint8, mode="r")
            n = next(u["n"] for u in calls if int(u["weight"]) == wid)
            for j in range(n):
                yield data[2 * K * j:2 * K * (j + 1)]
    root_w, _ = commit_weights(params, w_rows())
    served = pass_dir / SERVED_RECORD
    length = served_length(_json(served)) if served.is_file() else None
    if length is None:
        print(f"no {SERVED_RECORD} beside the pass: the window registers no fixed length, and the verifier refuses it",
              flush=True)
    public, advice, priv = commit_window(params, window, root_w, root_h3_of(man), length=length)
    advice["provenance"] = {"run": man.get("run"), "manifest_sha512": sha, "scheme": man["scheme"],
                            "transcript_root": man.get("transcript_root"), "commitment": man.get("commitment"),
                            "records_root": man.get("records_root"),
                            "call_draw": {"source": source.hex(), "among": [x["index"] for x in calls]} if not a.call else {"given": a.call}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "commitment.json").write_text(json.dumps(public, sort_keys=True))
    (out / "advice.json").write_text(json.dumps(advice, sort_keys=True))
    save(out / "prover.npz", priv)
    print(json.dumps({"calls": [c["record"]["matmul"] for c in window], "real": advice["real"], "work": public["items"]["work"],
                      "law": public["items"]["sampling"]["law"], "filler": advice["entries"][len(window)]["hi"] - advice["real"],
                      "roots": {r["port"]: r["root"][:16] for r in public["roots"]}}), flush=True)
    return 0


def cmd_register(a) -> int:
    c, params = _json(Path(a.commit_dir) / "commitment.json"), _json(a.params)
    extra = {}
    for x in a.extra or ():
        key, _, value = x.partition("=")
        extra[key] = json.loads(value)
    try:
        rec = register(params, c, extra)
    except S.Refused as e:
        print(f"REFUSED {json.dumps(e.verdict.to_json(), default=str)}", flush=True)
        return 0 if a.expect_refused else 1
    Path(a.out).write_text(json.dumps(rec, sort_keys=True))
    print(f"REGISTERED {json.dumps({'registration': R.digest(rec)[:32], 'population': rec['population'], 'law': rec['law'], 'public': sorted(rec['window']['public'])})}",
          flush=True)
    return 1 if a.expect_refused else 0


def cmd_edit(a) -> int:
    """A control's registration: `registration` with each dotted KEY set to JSON (a public item added, a level raised)."""
    rec = _json(a.registration)
    for x in a.set:
        key, _, value = x.partition("=")
        *up, last = key.split(".")
        at = rec
        for k in up:
            at = at[k]
        at[last] = json.loads(value)
    Path(a.out).write_text(json.dumps(rec, sort_keys=True))
    return 0


def lean_sampler(lean: str, log: Path | None = None) -> Sampler:
    """The window's sample by Lean ``flock-verify draw --stream``: the units of a subset law over N, on the key's stream."""
    def sample(law: Mapping[str, Any], n: int, src: bytes) -> list[int]:
        p = subprocess.run([lean, "draw", "--population", str(n), "--k", str(law["k"]), "--stream", src.hex()],
                           capture_output=True, text=True)
        lines = [x[5:] for x in p.stdout.splitlines() if x.startswith("DRAW ")]
        if p.returncode != 0 or len(lines) != 1:
            raise _refused("sampler", f"flock-verify draw: exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
        if log is not None:
            with open(log, "a") as f:
                f.write(lines[0] + "\n")
        return list(json.loads(lines[0])["units"])
    return sample


def answer(ch: Challenger, line: str) -> str | None:
    """The reply line to one request line; None for ``{"op": "quit"}``."""
    try:
        cmd = json.loads(line)
    except ValueError as e:
        word, body = "REFUSED", {"codes": ["request"], "detail": {"request": str(e)}}
    else:
        if isinstance(cmd, dict) and cmd.get("op") == "quit":
            return None
        word, body = handle(ch, cmd) if isinstance(cmd, dict) else ("REFUSED", {"codes": ["request"], "detail": {}})
    return f"{word} {json.dumps(body, sort_keys=True, default=str)}"


def cmd_challenger(a) -> int:
    """The challenger's process: the epoch's coins, drawn now from the Lean verifier of record (`pouw.verifier.coins`) or
    read from ``--state``, their commitment published to LOG/coins.json before any request; then
    one JSON request a line on stdin (`handle`), one ``WORD {json}`` reply a line on stdout, each also appended to
    LOG/requests.jsonl, each coin it opened to LOG/openings.jsonl, and each request's verdicts of record, Lean's and
    Python's, to LOG/witness.jsonl (a window that ends before its outcome keeps them there); it ends at end of input or
    ``{"op": "quit"}``."""
    log = Path(a.log)
    log.mkdir(parents=True, exist_ok=True)
    if a.state:
        if not Path(a.state).is_file():
            raise SystemExit(f"no challenger state at {a.state}")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
        from e2e import state_entropy
        entropy = state_entropy(a.state)
    else:
        entropy = PV.coins
    try:
        Ch.served_claim(entropy)
    except Ch.CoinError as e:
        raise SystemExit(f"the challenger refused to start: {e}") from None
    coins = Coins(a.epoch, entropy)
    (log / "coins.json").write_text(json.dumps(coins.commitment.to_json(), sort_keys=True))
    ch = Challenger(_json(a.params), coins, lean_sampler(a.lean, log / "draws.jsonl") if a.lean else python_sampler,
                    record=OR.Record(lean=False if a.python_record else None))
    for line in sys.stdin:
        if not line.strip():
            continue
        opened, at = set(ch.openings), len(ch.record.witness)
        reply = answer(ch, line)
        if reply is None:
            break
        with open(log / "openings.jsonl", "a") as f:
            for label in sorted(set(ch.openings) - opened):
                f.write(json.dumps(ch.openings[label].to_json(), sort_keys=True) + "\n")
        with open(log / "requests.jsonl", "a") as f:
            f.write(json.dumps({"request": line.strip(), "reply": reply[:4000]}) + "\n")
        with open(log / "witness.jsonl", "a") as f:
            f.write(json.dumps({"reply": reply.split(" ", 1)[0], "of_record": ch.record.witness[at:]}, sort_keys=True,
                               default=str) + "\n")
        print(reply, flush=True)
    return 0


def _files(p: Path) -> dict[str, bytes]:
    if p.is_file():
        return {"": p.read_bytes()}
    return {q.relative_to(p).as_posix(): q.read_bytes() for q in sorted(p.rglob("*")) if q.is_file()} if p.is_dir() else {}


def replay(params: dict, log: Path, out: Path, sampler: Sampler = python_sampler, record: OR.Record | None = None
           ) -> list[str]:
    """The challenger's requests again (LOG/requests.jsonl) from its records alone: the published commitment and the
    coins it opened (`Recorded`), each request's ``out`` under OUT.  Every difference from the record: a reply, or a file
    a request wrote; empty when the replay is byte-identical.  A record whose parameters predate the registration
    deadline replays as it ran (`Challenger`'s `predates_deadline`): the deadline is in the Spec's digest, which keys the
    sample, so it can't be added to a recorded window."""
    coins = Recorded(CoinCommitment.from_json(_json(log / "coins.json")),
                     [CoinOpening.from_json(json.loads(x)) for x in
                      ((log / "openings.jsonl").read_text().splitlines() if (log / "openings.jsonl").exists() else ()) if x])
    ch = Challenger(params, coins, sampler, record=record, predates_deadline=params.get("registration_deadline") is None)
    diffs = []
    for i, x in enumerate((log / "requests.jsonl").read_text().splitlines()):
        entry = json.loads(x)
        line, was, wrote = entry["request"], entry["reply"], None
        try:
            cmd = json.loads(line)
        except ValueError:
            cmd = None
        if isinstance(cmd, dict) and cmd.get("out"):
            wrote, cmd["out"] = Path(cmd["out"]), str(out / f"{i:04d}" / Path(cmd["out"]).name)
            line = json.dumps(cmd)
        reply = (answer(ch, line) or "")[:4000]
        if reply != was:
            diffs.append(f"request {i}: reply {reply[:120]!r}, recorded {was[:120]!r}")
        if wrote is not None and _files(Path(cmd["out"])) != _files(wrote):
            diffs.append(f"request {i}: {wrote} differs from its replay")
    return diffs


def cmd_replay(a) -> int:
    diffs = replay(_json(a.params), Path(a.log), Path(a.out), lean_sampler(a.lean) if a.lean else python_sampler,
                   OR.Record(lean=False if a.python_record else None))
    n = len((Path(a.log) / "requests.jsonl").read_text().splitlines())
    out = {"requests": n, "diffs": diffs[:8], **({"predates_deadline": True} if "registration_deadline" not in _json(a.params) else {})}
    print(f"{'SAME' if not diffs else 'DIFFERENT'} {json.dumps(out)}", flush=True)
    return 0 if not diffs else 1


#: a control pass's salt coin (`coin_variant`): each is refused at registration, the first four by the coin check
CASES = ("foreign", "early", "mislabelled", "tampered", "resalted")


def coin_variant(man: dict, coins: CoinCommitment, opening: CoinOpening, case: str) -> dict:
    """`man` carrying the epoch's commitment and a salt coin `case` made of `opening`, the challenger's genuine salt coin:
    another challenger's (``foreign``), opened after what the verifier had not recorded (``early``), the epoch's sampling coin
    label (``mislabelled``), a byte of the coin flipped (``tampered``), or the genuine coin in a pass whose salt is not
    that coin's (``resalted``)."""
    o = opening
    if case == "foreign":
        o = Coins(coins.epoch, PV.coins).open("salt", opening.after)
    elif case == "early":
        o = CoinOpening(o.epoch, o.label, bytes(len(o.after)), o.coin, o.salt, o.path)
    elif case == "mislabelled":
        o = CoinOpening(o.epoch, "draw", o.after, o.coin, o.salt, o.path)
    elif case == "tampered":
        o = CoinOpening(o.epoch, o.label, o.after, bytes([o.coin[0] ^ 1]) + o.coin[1:], o.salt, o.path)
    elif case != "resalted":
        raise ValueError(f"a coin variant is one of {', '.join(CASES)}")
    out = {k: v for k, v in man.items() if k != "beacon"}
    return {**out, "schema": _pd().MANIFEST_SCHEMA, "coins": coins.to_json(), "salt_opening": o.to_json()}


def cmd_coin_variant(a) -> int:
    v = coin_variant(_json(a.manifest), CoinCommitment.from_json(_json(a.coins)), CoinOpening.from_json(_json(a.opening)), a.case)
    Path(a.out).write_text(json.dumps(v, sort_keys=True))
    return 0


def cmd_check(a) -> int:
    params, reg, d = _json(a.params), _json(a.registration), _json(a.draw)
    record = OR.Record(lean=False if a.python_record else None)
    why = check(params, reg, d, a.unit, Path(a.pub).read_bytes(), _json(a.opening), drawn=not a.undrawn, record=record)
    record.close()
    if a.out:
        Path(a.out).write_text(json.dumps({"of_record": False, "accepted": why is None, "why": why or "",
                                           "verifier": "served_zk opening check", "witness": record.witness},
                                          sort_keys=True))
    print(f"ACCEPT {a.pub} opens unit {a.unit} of {a.registration}" if why is None else f"REJECT {a.pub}: {why}", flush=True)
    return 0 if (why is None) != a.expect_reject else 1


#: the session record's (``session.json``) fields the verifier writes of its own: its verdict and its timings
OWN = ("verdict",)
#: the session directory's files the verifier writes, not the prover's messages
RECORDS = ("session.json", "verifier.json")


def skeleton(v, top: bool = True):
    """A session record's shape: every integer and boolean as it is, every string by its length (hex or not), every float
    (a timing) only as a float, and the verifier's own fields (`OWN`) dropped."""
    if isinstance(v, dict):
        return {k: skeleton(x, False) for k, x in v.items() if not (top and k in OWN)}
    if isinstance(v, list):
        return [skeleton(x, False) for x in v]
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return ["hex" if all(c in "0123456789abcdef" for c in v) else "str", len(v)]
    return v


def _diffs(a, b, at: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b)) for p in _diffs(a.get(k), b.get(k), f"{at}.{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _diffs(x, y, f"{at}[{i}]")]
    return [] if a == b else [f"{at}: {json.dumps(a)[:60]} / {json.dumps(b)[:60]}"]


def shape(d1: Path, d2: Path) -> dict:
    """What two staged units' proofs show the verifier: circuit, class, the public file's size, each of the prover's
    session files' size, and the session record's `skeleton`."""
    def one(d: Path) -> tuple[dict, object]:
        s = _json(d / "stage.json")
        sess = sorted((d / "sessions").glob("*/")) if (d / "sessions").is_dir() else []
        files = sorted((p.relative_to(sess[-1]).as_posix(), p.stat().st_size) for p in sess[-1].rglob("*")
                       if p.is_file() and p.name not in RECORDS) if sess else []
        rec = skeleton(_json(sess[-1] / "session.json")) if sess else None
        return {"circuit_sha512": s["circuit_sha512"], "class_sha512": s["class_sha512"], "ands": s["ands"], "k_log": s["k_log"],
                "pub_bytes": (d / "pub.bin").stat().st_size, "session_files": files}, rec
    (x, rx), (y, ry) = one(d1), one(d2)
    diffs = [f"{k}: {x[k]} / {y[k]}" for k in x if x[k] != y[k]] + _diffs(rx, ry, "session")
    return {"same": not diffs and bool(x["session_files"]) and rx is not None, "diffs": diffs[:40], "a": x, "b": y}


def cmd_shape(a) -> int:
    v = shape(Path(a.a), Path(a.b))
    print(f"{'SAME' if v['same'] else 'DIFFERENT'} {json.dumps({k: v['a'][k] for k in ('circuit_sha512', 'pub_bytes')})} "
          f"{len(v['a']['session_files'])} proof files {json.dumps(v['diffs'][:6])}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(v, sort_keys=True))
    return 0 if v["same"] else 1


PYTHON_RECORD = ("PoUW's checks by their Python references alone, undoing the switch to Lean's verdict of record "
                 "(`of_record`)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("params")
    p.add_argument("pass_dir")
    p.add_argument("out")
    p.add_argument("--eps", default=EPS)
    p.add_argument("--delta", default=DELTA)
    p.add_argument("--entries", type=int, default=ENTRIES)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                   help="the max_tokens the verifier's request set, each row served to it under ignore_eos")
    p.add_argument("--registration-deadline", type=float, default=REGISTRATION_DEADLINE,
                   help="td: seconds on the challenger's clock from the salt coin's release, when the challenger's `salt` "
                        "request opens it after the weights' root; the coin itself was sampled when the challenger process "
                        "started, from the Lean verifier's coins (pouw.verifier.coins) or --state (default: "
                        "REGISTRATION_DEADLINE)")
    p = sub.add_parser("commit")
    for x in ("pass_dir", "params", "out"):
        p.add_argument(x)
    p.add_argument("--calls", type=int, default=2)
    p.add_argument("--call", type=int, action="append")
    p = sub.add_parser("register")
    for x in ("commit_dir", "params", "out"):
        p.add_argument(x)
    p.add_argument("--extra", action="append", help="ITEM=JSON: one more public item, which register refuses")
    p.add_argument("--expect-refused", action="store_true")
    p = sub.add_parser("edit")
    p.add_argument("registration")
    p.add_argument("out")
    p.add_argument("set", nargs="+", help="KEY=JSON, KEY dotted")
    p = sub.add_parser("challenger")
    p.add_argument("params")
    p.add_argument("log")
    p.add_argument("--lean", help="Lean flock-verify, whose draw over the sampling coin's key is the challenger's "
                                  "(default: its Python twin, stream_subset)")
    p.add_argument("--epoch", type=int, default=0, help="the epoch whose coins the challenger holds (default 0)")
    p.add_argument("--state", help="the file holding the challenger's entropy, the one the serving controller's coins flipped "
                                   "from (serve.sh's $WORK/coin-state/<run id>, written from the Lean verifier's coins "
                                   "before serving); default: fresh from the Lean verifier of record when this process "
                                   "starts (pouw.verifier.coins, IO.getRandomBytes), before it publishes LOG/coins.json")
    p.add_argument("--python-record", action="store_true", help=PYTHON_RECORD)
    p = sub.add_parser("replay")
    for x in ("params", "log", "out"):
        p.add_argument(x)
    p.add_argument("--lean", help="Lean flock-verify, the sample's sampler (default: its Python twin)")
    p.add_argument("--python-record", action="store_true", help=PYTHON_RECORD)
    p = sub.add_parser("coin-variant")
    for x in ("manifest", "coins", "opening", "out"):
        p.add_argument(x)
    p.add_argument("case", choices=CASES)
    p = sub.add_parser("check")
    for x in ("params", "registration", "draw"):
        p.add_argument(x)
    p.add_argument("unit", type=int)
    p.add_argument("pub")
    p.add_argument("opening")
    p.add_argument("--out", help="the opening check's verdict, for the outcome")
    p.add_argument("--undrawn", action="store_true", help="a control: open a unit the sample did not sample")
    p.add_argument("--expect-reject", action="store_true")
    p.add_argument("--python-record", action="store_true", help=PYTHON_RECORD)
    p = sub.add_parser("shape")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out")
    a = ap.parse_args(argv)
    return {"params": cmd_params, "commit": cmd_commit, "register": cmd_register, "edit": cmd_edit,
            "challenger": cmd_challenger, "replay": cmd_replay, "coin-variant": cmd_coin_variant, "check": cmd_check,
            "shape": cmd_shape}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
