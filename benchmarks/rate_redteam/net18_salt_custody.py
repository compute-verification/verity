"""Red-team witness (W5, target 3): NET-18's ingress record as a function of the salt at fixed traffic.

Accounting.Assumptions.BoundaryObs asks that dX's certifiers' records placed during the window be a function of the
block's leak and the boundary's observations. Here the block leaks nothing and the link's traffic is identical in both
runs; only the certifier's salt differs. The proxy (`active.Proxy`, its own salt draw and salt frame) runs its ingress
side for three windows on each salt, and the witness prints whether the records differ, whether any holds the salt or
anything of it, and whether `reveal` gives a window's salt without that window's registration.

Against afe591362 the records differed (each held `salt_commitment(salt)`), and `reveal(w)` took no registration. With
the placeholder in the salt frame's slot and `reveal(p, registration)`, the records are equal and the reveal refuses an
unregistered window. ``uv run python benchmarks/rate_redteam/net18_salt_custody.py``
"""

from verity.core.protocols.certifier.active import Proxy, request_frames
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.core.protocols.sampled_proofs.service import Registration

P = GridParams(buckets=4, allocation=2, frame_bytes=96)
WINDOWS = 3
FIRST = request_frames(b"prompt " * 13, P.frame_bytes)  # two frames
TRAFFIC = [(0, FIRST[: P.frame_bytes]), (2, FIRST[P.frame_bytes :]), (5, request_frames(b"next", P.frame_bytes))]


def registration(window: int) -> Registration:
    record = {"window": {"kind": "served", "window": window}}
    return Registration(record, R.receipt(record, "witness", received=str(window)), R.digest(record), window)


def run(salt: bytes) -> tuple[Proxy, list]:
    proxy = Proxy(("127.0.0.1", 9), P, P, SyncSet((0,), (1,)), [ClockSync(0, 1)] * WINDOWS, origin=0.0,
                  ingress_link="requests", clock=lambda: 0.0, salt=lambda p, n: salt[:n])
    for bucket, frame in TRAFFIC:
        proxy.bucketer.arrive(frame, bucket)
    records = [r for r in (proxy.bucketer.tick()[1] for _ in range(WINDOWS * P.buckets)) if r is not None]
    return proxy, records


def refuses(call) -> bool:
    try:
        call()
    except RuntimeError:
        return True
    return False


s1, s2 = bytes(range(64)), bytes(range(1, 65))
(p1, r1), (p2, r2) = run(s1), run(s2)
frames = [f for r in r1 + r2 for row in r.grid for f in row if f is not None]
print("same traffic, same (empty) leak; salts differ")
print("ingress records of every window equal:", r1 == r2)
print("a record holds a salt:", any(s in f for f in frames for s in (s1, s2)))
print("reveal(0) without a registration refused:", refuses(lambda: p1.reveal(0, None)))
print("reveal(0) on window 1's registration refused:", refuses(lambda: p1.reveal(0, registration(1))))
print("reveal(0) on window 0's registration gives the salt:", p1.reveal(0, registration(0)) == s1)
