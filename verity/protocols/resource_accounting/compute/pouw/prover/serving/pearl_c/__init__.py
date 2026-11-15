"""Pearl-C on the RTX PRO 6000 (sm_120) and the H100/H200 (sm_90a) at a served linear: the schemes this
device path runs (`SCHEMES`, `scheme_of`), the linears outside them (`refusals`), the kernels' buffers and steps by name, the
prover's hm96 salt key (`new_salt_key`), and the readers a verifier and the option need: the kernel's directory (`Kernel`),
the schedule (`schedule_of`, `pearl_c.lanes`), the manifest (`manifest_epoch`, `epoch_of`), a call's served y
(`served_y`) and a window (`window`).

The run itself, every call computed and committed on the device by GPU 1's kernel pipeline (``PearlCDeviceRun``, its
weights' roots ``weight_roots``), is the workload's, which hands it to the serving option.
"""
from __future__ import annotations

import os
from typing import Any

from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c_work as PW  # noqa: F401 - kept importable here, as the tests read it
from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c_device import (
    DEVICES,
    H100,
    SM120,
    SM120_UNPROMOTED,
)
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import Matmul, Shape
from verity.protocols.resource_accounting.compute.pouw.verifier.serving import SERIAL, SchemeId

from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.kernel import (
    Kernel,
    source_digest,  # noqa: F401 - kept importable here, as callers read it
)
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.lanes import (  # noqa: F401 - the schedule's names, kept importable here, as callers read them
    SCHEDULES,
    SIDE_GROUPS,
    SIDE_IO,
    STEP_GROUPS,
    admitted,
    rotated,
    schedule_of,
    splits,
    whole_options,
)
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.manifest import (  # noqa: F401 - the manifest's readers, kept importable here, as callers read them
    epoch_of,
    manifest_epoch,
    manifest_schema,
    session_head,
)

__all__ = ["CALL_CAPACITY", "CALL_STEPS", "EPOCH_STEPS", "FLAG_ROWS", "JOB_STEPS", "Kernel", "LEAF_CAPACITY", "ROW_ALIGN",
           "SCHEDULES", "SCHEMES", "SERIAL", "epoch_of", "manifest_epoch", "manifest_schema", "new_salt_key", "refusals",
           "rotated", "schedule_of", "scheme_of", "splits"]

#: the Pearl-C schemes this device path runs, by name (`verity.protocols.resource_accounting.compute.pouw.verifier.serving.SchemeId`: forming, device record, hashing format);
#: the device record is GPU 1's pipeline variant too.  v1 is ``sm120`` (G = 4, the panel's ``pearl-c-sm120`` v1); v2 is
#: ``sm120-unpromoted`` (no promotion, the panel's v2, counted once the theory and Lean lanes restate it); ``h100`` (G = 4,
#: a served model's FP8 form on the H200) is pearl_c_sm90's.  The format is -h3, SHA-512 throughout (its seeds, lines,
#: tile leaves and HM96 trees: the one hash), each row's E_A line from its own seed (the work sampling design capping each
#: credited row too); its tree hash comes from `verity.protocols.resource_accounting.compute.pouw.verifier.serving.HASH_FORMATS` through the scheme.
SCHEMES = tuple(C.PearlC(forming="v1", device=d, hashing="h3").name for d in (SM120, SM120_UNPROMOTED, H100))
#: each record's kernels' (k, n) grain past its domain: pearl_c_sm120's (1,024, 128); pearl_c_sm90's (256, 64), 1 KiB row segments, 64-wide tiles
KERNEL_GRAIN = {SM120.name: (1024, 128), SM120_UNPROMOTED.name: (1024, 128), H100.name: (256, 64)}
#: the domain's row granularity: m₁₆ = m rounded up to it, the filler rows zero
ROW_ALIGN = 16
#: tiles one pass's leaf buffer holds (256 MiB)
LEAF_CAPACITY = 1 << 23
#: calls one pass's A-root buffer holds, and rows its screen-flag buffer holds
CALL_CAPACITY = 1 << 16
FLAG_ROWS = 1 << 22
WEIGHT_BUFFERS = ("eb_codes", "fa_codes", "faT32", "fb_codes", "fbT32", "fbT128", "g", "alpha_b", "beta_b", "ia_b", "aux_b",
                  "epb_codes", "bt", "bf", "pb_t", "tree_b", "rowdig_b")
CALL_BUFFERS = ("alpha_a", "beta_a", "ia_a", "aux_a", "epa_codes", "ea_codes", "at", "af", "pa_t", "c", "u", "tree_a", "line_a",
                "seed_a", "rowdig_a", "seeds_a", "lkeys_a", "segdig_a", "segdig_t")
#: per-call buffers only some kernels' pipelines size (`Pipeline.sizes`), each made where its pipeline has it: pearl_c_sm90's
#: A'F_B windows
CALL_BUFFERS_OWN = ("af_win",)
EPOCH_STEPS = ("pad_b", "rowdig_b", "hash_rows_b", "tree_b")
JOB_STEPS = ("lines_eb", "lines_fa", "lines_fb", "stats_b", "eprime_b", "form_b", "bf", "g", "pb")
CALL_STEPS = ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea", "stats_a", "eprime_a", "form_a",
              "af", "pa", "gemm_pearlc", "hash_leaf")
HASH_STEPS = frozenset({"pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea", "hash_leaf", "commit_a",
                        "rows_a", "tree_seed_a"})


def scheme_of(name: str, **instance: Any) -> C.PearlC:
    """The scheme `name` names (`SCHEMES`); `instance` (``k_min``, ``tile``) makes a test instance."""
    if name not in SCHEMES:
        raise KeyError(f"{name!r}: this device path runs {', '.join(SCHEMES)}")
    sid = SchemeId.parse(name)
    s = C.PearlC(forming=sid.forming, device=DEVICES[sid.device], hashing=sid.hashing, **instance)
    if s.name != name:
        raise ValueError(f"the scheme built for {name!r} names itself {s.name!r}")
    return s


def refusals(name: str, shapes: dict[int, tuple[int, int]]) -> list[str]:
    """Every linear outside the scheme's domain or its kernels' (`KERNEL_GRAIN`), by weight id."""
    s, out = scheme_of(name), []
    gk, gn = KERNEL_GRAIN[s.device.name]
    for w, (n, k) in sorted(shapes.items()):
        if k % gk or n % gn:
            out.append(f"linear {w}: (n, k) = ({n}, {k}), and the kernels take k a multiple of {gk:,} and n of {gn}")
            continue
        try:
            s.check(Matmul(0, Shape(ROW_ALIGN, k, n), w))
        except Exception as e:  # noqa: BLE001 - DomainError, named per layer
            out.append(f"linear {w}: {e}")
    return out


def new_salt_key(scheme: str) -> bytes:
    """A fresh prover hm96 salt key for `scheme`'s trees (their salts, kept by the prover and never in a record)."""
    scheme_of(scheme)
    return os.urandom(32)
