"""Configuration 2, recursive zero knowledge: what it is to report, and what of it runs today.

The design (in progress; the ZK server's specification is being written): workers prove the inner statement natively with ZK
off and run the outer proof's arithmetic unmasked; a ZK server, compiled Lean on a CPU node, draws the masks and salts,
checks the outer relations and releases the masked proof; the auditor runs the Lean verifier on what is released. None of
the worker/server split exists yet. Today's nearest pieces, labelled as stand-ins wherever they are used:

- the firewall (``verity/core/service/firewall``, ``verity/firewall-release/v0``): the release decision of one outer session;
- ``flock-circuit prove --zk``: C-Flock's masked prover (M1, ``verity/ml/flock/live/PROTOCOL.md``), CPU only
  (``prove_cpu_zk``), with its own masks rather than a server's.

Each field names its source: ``config1`` is measured by configuration 1 at the same B, ``stand-in`` by one of the pieces
above, ``stub`` by nothing yet.
"""
from __future__ import annotations

FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("worker.vu_per_second", "VU/s", "config1",
     "the worker's inner proof is configuration 1's prover at best B; the unmasked outer arithmetic is not built, so this "
     "is an upper bound"),
    ("zk_server.cpu_s_per_session", "s", "stub", "no ZK server: its relation checks and mask draws don't exist"),
    ("zk_server.cpu_s_per_vu", "s", "stub", "zk_server.cpu_s_per_session / B"),
    ("net.bytes_worker_to_server", "B", "stand-in",
     "configuration 1's transcript.bytes at best B: the session the worker would hand the server to mask, before the outer "
     "proof's own messages"),
    ("proof_bytes.released", "B", "stub",
     "the masked proof the server releases; flock-circuit --zk's session bytes would stand in, on a CPU prover and one "
     "template only"),
    ("zk_servers_per_gpu", "count", "stub",
     "worker.vu_per_second x zk_server.cpu_s_per_vu / cores per server, once the server's CPU time is measured"),
)


def describe() -> str:
    lines = [__doc__.strip(), "", "At configuration 1's best B:"]
    for name, unit, source, how in FIELDS:
        lines.append(f"  {name:<30} [{unit}] {source:<8} {how}")
    return "\n".join(lines)
