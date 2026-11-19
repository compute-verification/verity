"""Salts a test hands in (`rows.commit`'s ``salts_of``) so two commits of one period are one commitment."""
import hashlib

from verity.commitments.periods import rows


def fixed(w, seed: int = 0) -> dict[str, bytes]:
    w = getattr(w, "window", w)
    return {p: hashlib.shake_256(f"{seed} {p}".encode()).digest(n * rows.SALT_BYTES) for p, n in rows.full_ports(w).items()}
