"""A certifier's LMS key on disk: the directory `load_key` makes or reads, its signer, and the public key as a
deployment record carries it."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from verity.commitments.crypto import lms
from verity.protocols.accounting.certifier.verifier.certificate import LmsKey, LmsSigner

HEIGHT = 10
#: the processes `load_key` makes a key above `HEIGHT` in (at H = 20, about two minutes on eight cores); at or below
#: it a key takes about a second in one, and a caller on a thread forks nothing
KEYGEN_WORKERS = 8


def load_key(directory: Path, height: int = HEIGHT) -> lms.PrivateKey:
    """The LMS key (N = 64, W = 1) in `directory`, made there first at `height` when absent (above `HEIGHT` in
    `KEYGEN_WORKERS` processes), in the challengers' layout
    (`verity/protocols/compose/device/live_challenger/serve.py`'s `load_key`): `seed` (secret, 0600), `public` (`I ‖ T[1]`), `tree.bin`
    (its tree, checked against `public` and `tree.sha512`), its leaf state `state.json`, and `public.json`, the public
    key as a deployment record carries it."""
    import hashlib

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    seed_p, public_p, tree_p, check_p, state_p = (directory / n for n in ("seed", "public", "tree.bin", "tree.sha512",
                                                                          "state.json"))
    if (directory / "public.json").exists():
        height = json.loads((directory / "public.json").read_text())["params"][2]
    params = lms.Params(64, 1, height)
    if not seed_p.exists():
        fd = os.open(seed_p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, secrets.token_bytes(params.N))
            os.fsync(fd)
        finally:
            os.close(fd)
    seed = seed_p.read_bytes()
    if tree_p.exists() and public_p.exists() and check_p.exists():
        public, raw = public_p.read_bytes(), tree_p.read_bytes()
        tree = (b"",) + tuple(raw[params.N * r:params.N * (r + 1)] for r in range(len(raw) // params.N))
        key = lms.PrivateKey(params, public[:16], seed, tree)
        if (len(tree) != 2 ** (params.H + 1) or lms.public_key(key) != public
                or hashlib.sha512(public + raw).digest() != check_p.read_bytes()):
            raise SystemExit(f"{tree_p} is not the tree of {public_p} at {params}")
    else:
        key = lms.keygen(params, seed, workers=KEYGEN_WORKERS if params.H > HEIGHT else 1)
        raw, public = b"".join(key.tree[1:]), lms.public_key(key)
        for path, data in ((tree_p, raw), (public_p, public), (check_p, hashlib.sha512(public + raw).digest())):
            path.write_bytes(data)
    if not state_p.exists():
        lms.LeafState.create(state_p, public, params.H)
    (directory / "public.json").write_text(json.dumps(public_json(LmsKey(params, public))))
    return key


def signer(directory: Path, height: int = HEIGHT) -> LmsSigner:
    return LmsSigner(load_key(directory, height), directory / "state.json")


def public_json(key: LmsKey) -> dict:
    return {"params": [key.params.N, key.params.W, key.params.H], "public": key.public.hex(), "key_id": key.key_id.hex()}


def read_public(doc: dict) -> LmsKey:
    return LmsKey(lms.Params(*doc["params"]), bytes.fromhex(doc["public"]))
