"""serve-openai's retention options (its ``retention`` group): which committed values a window keeps off the device
(``--retain-exclude``), where they wait for their draws (``--retain host|device``), and the draw-gated release they wait on
(``--release-after``, `serve_release`); `retention_refusal` says why a combination can't hold, and `retention_record` is
serve.json's record of them."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity_vllm.commit import step_rows
from verity_vllm.config import option

ROWS = step_rows.NAME


@dataclass(frozen=True, kw_only=True)
class RetentionOptions:
    """serve-openai's ``--retain-exclude`` and ``--retain`` (its ``retention`` group, `retention_refusal`, `retention_record`)."""
    retain_exclude: str = option("--retain-exclude", env="VERITY_RETAIN_EXCLUDE", default="", metavar="FAMILIES",
                                 help="rows-sha512: commit these value families without retaining them (sets VERITY_RETAIN_EXCLUDE, as "
                                      "the Commit's --retain-exclude does): the attention interiors only, hidden_m1 (fa2_hidden_m1, "
                                      "fa3_hidden_m1), which no reader of a sealed window opens. Their leaves and roots are the "
                                      "same; each seal's manifest names them committed_not_retained (store_dump.not_retained)")
    retain: str = option("--retain", default="host", choices=("host", "device"),
                         help="rows-sha512: where a window's committed values wait for its draws. host (the default): the pinned host "
                              "pool. device: a device pool of --retain-device-gb (committer.device_pool), which the window's draw-gated "
                              "release empties, keeping on host only the leaves some subprotocol drew; a full pool makes the step wait")
    retain_device_gb: float = option("--retain-device-gb", type=float, default=32.0, metavar="GB",
                                     help="--retain device: the device pool's size, beside the engine and the window ring; it must hold "
                                          "every window awaiting its release and the open one, so at least --window-gb")
    release_after: str = option("--release-after", default="", metavar="PROTOCOLS",
                                help="--retain device --seal: the subprotocols (comma-separated, e.g. pouw) whose draw of each "
                                     "window its release waits for; the union of the leaves their drawn blocks open is kept "
                                     "(serve_release); pouw is PoUW H1's, put by its arm (--protocols), the rest come in "
                                     "through --draws-dir")
    draws_dir: str = option("--draws-dir", default="", metavar="DIR",
                            help="--release-after: where each draw comes in, w<k>.<protocol>.json (default OUT/draws)")


def retention_record(opts: Any, com: Any) -> dict[str, Any]:
    """serve.json's record of ``opts.retention``: the families left unretained, where values wait for their draws, and the pool."""
    r = opts.retention
    return {"retain_exclude": r.retain_exclude or None, "retain": r.retain, "device_pool": getattr(com, "device_pool_record", None),
            "release_after": r.release_after or None, "draws_dir": (r.draws_dir or "OUT/draws") if r.release_after else None}


def retention_refusal(opts: Any) -> str | None:
    """Why ``opts.retention`` can't hold (None: it can): ``--retain-exclude`` names attention interiors only, and the Commit under
    rows-sha512, whose committer leaves them on the device; ``--retain device`` the Commit under rows-sha512 too, with a pool of at
    least ``--window-gb``."""
    r = opts.retention
    if r.retain == "device":
        if not opts.commit or opts.scheme != ROWS:
            return f"--retain device holds the windowed collector's committed values on the device: it needs the Commit under --scheme {ROWS}"
        if r.retain_device_gb < float(getattr(opts, "window_gb", 0) or 0):
            return (f"--retain-device-gb {r.retain_device_gb:g} is under --window-gb {opts.window_gb:g}: the pool must hold the open window "
                    "beside those awaiting their release")
        if getattr(opts, "seal", False) and not getattr(r, "release_after", ""):
            return ("--retain device --seal: a window is sealed only after its draw-gated release, so name the subprotocols whose draws "
                    "it waits for (--release-after)")
    if getattr(r, "release_after", ""):
        if r.retain != "device" or not getattr(opts, "seal", False):
            return "--release-after gates the release of device-held values before each seal: it needs --retain device and --seal"
        from verity_vllm.pipeline.serve_release import protocols_of
        try:
            protocols_of(r.release_after)
        except ValueError as e:
            return str(e)
    spec = r.retain_exclude
    if not spec:
        return None
    from verity_vllm.commit.committer.host_retention import HIDDEN_M1_FAMILIES, retain_exclude_families

    if other := sorted(retain_exclude_families(spec) - HIDDEN_M1_FAMILIES):
        return (f"--retain-exclude {spec}: {other} aren't attention interiors; serve-openai commits without retaining only "
                f"{sorted(HIDDEN_M1_FAMILIES)} (hidden_m1), which no reader of a sealed window opens")
    if not opts.commit or opts.scheme != ROWS:
        return f"--retain-exclude {spec} commits values without retaining them: it needs the Commit under --scheme {ROWS}"
    return None
