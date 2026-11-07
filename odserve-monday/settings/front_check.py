"""The settings job's stand-in for CERTIFIER_FRONT's "$PY": `certifier_piece.py front`'s own code on the front's argv,
remote's counts in it, with the gate's session and the warm-ups alone replaced (`--stop-prompt`'s first warm-up answers
as if it finished on a stop: whether the model's does is the served run's), so every refusal the front makes before it
reaches the gate (the scan's lanes, one link-window, the class's KI and KO, the gate's own `parse_warden` and
`warden_prompts`) is made here, against no gate and no endpoint (`served_session.SSH` is `false`, so a session this
misses can't reach one):

    front_check.py benchmarks/network_traces/certifier_piece.py front PORT SERVE_DIR OUT [flags]

SERVE_DIR and OUT should be a fresh scratch directory's. It prints nothing; the front's refusal, which names the counts,
is on stderr. Exit 0 when the front would have opened its gate session, the front's exit when it refused, 3 when the
argv isn't the front's or the front returned without opening the session.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 3 or Path(argv[1]).name != "certifier_piece.py" or argv[2] != "front":
        print("not certifier_piece.py front", file=sys.stderr)
        return 3
    sys.path.insert(0, str(Path(argv[1]).resolve().parent))
    import certifier_piece as CP

    opened: list[int] = []
    CP.SS.SSH = ("false",)
    CP.SS.session = lambda a, data, ps: opened.append(len(ps)) or 0
    CP.warmup = lambda a, serve_dir, ps=None, **_: [] if ps is None else [
        {"user": "warmup-check-0", "finish_reason": "stop", "seconds": 0.0}]
    rc = CP.front(CP.parser().parse_args(argv[2:]))
    return rc if rc else (0 if opened else 3)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
