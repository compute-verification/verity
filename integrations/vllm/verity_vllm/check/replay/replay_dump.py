"""Opt-in diagnostic of the sampled replay: the committed and evaluated words of evaluated rows, written to `VERITY_REPLAY_DUMP_DIR`
(record-neutral; off unless the directory is set)."""
from __future__ import annotations

from typing import Any

import numpy as np

from verity_vllm import settings

_DUMP_N = [0]


def dump_row_words(fam: str, rid: str, step: int, op_path: str, row: int, member: str, have: np.ndarray, want: np.ndarray, eq: bool, *,
                   store: Any = None, elo: int | None = None, ehi: int | None = None) -> None:
    """[DIAGNOSTIC, opt-in, record-neutral] `VERITY_REPLAY_DUMP_DIR=<dir>` + `VERITY_REPLAY_DUMP_FAMILIES=Fam1,Fam2`: write
    the committed words and the evaluated words of every evaluated row of those families (equal or not; <= VERITY_REPLAY_DUMP_MAX files,
    default 1024) so a reader can test, offline, WHICH committed invocation an evaluated row matches (the aliased shared module: layer L's
    rope output vs the canonical path's invocation L).  Off unless the directory is set; nothing on the record changes."""
    import os
    dd = settings.VERITY_REPLAY_DUMP_DIR
    if not dd:
        return
    fams = set(settings.VERITY_REPLAY_DUMP_FAMILIES)
    if fams and fam not in fams:
        return
    if _DUMP_N[0] >= settings.VERITY_REPLAY_DUMP_MAX:
        return
    _DUMP_N[0] += 1
    os.makedirs(dd, exist_ok=True)
    fn = f"{fam}__{rid}__s{int(step)}__{op_path.replace('/', '_')}__row{int(row)}__{member}__{'EQ' if eq else 'NE'}.npz"
    extra: dict[str, np.ndarray] = {}
    if store is not None and elo is not None and hasattr(store, "resolve") and isinstance(getattr(store, "aliases", None), dict):
        # every invocation of the resolved canonical path at this (request, step, member, range): the other layers' words of a shared module
        canon, _inv = store.resolve(op_path)
        paths = {canon: 0} | {a: i for a, (c, i) in store.aliases.items() if c == canon}
        for p, i in sorted(paths.items(), key=lambda t: t[1]):
            w, _why = store.words(rid, step, p, member, elo, ehi)
            if w is not None:
                extra[f"inv{int(i)}__{p}"] = np.asarray(w).reshape(-1)
    np.savez(os.path.join(dd, fn), committed=np.asarray(have).reshape(-1), evaluated=np.asarray(want).reshape(-1), **extra)
