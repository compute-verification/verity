"""The honest responder, the prover's side of PoUS's timed audit: each challenged index answered from the stored encoding.

`server_root` is the root the server reports for its `C` at setup. On a served run the one comparison of that root is
the developer's tie on the prover's case (`period.tie`, and `PousCheck.RunProtocol.tie` with `SetupProtocol.check`, which
hold it to the root of the `C*` setup decoded), in the prover's piece on the developer's node; no auditor-side entry
compares it. `HonestResponder` answers an `audit.simulate` round as an oracle program with no queries, and gives a root-only memory
attester the audit path; `responder` answers a `protocol.TimedAudit` run through its scheme's `respond`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from verity.protocols.accounting.memory.pous.verifier.oracle import Prog, pure
from verity.protocols.accounting.memory.pous.verifier.protocol import Scheme
from verity.protocols.accounting.memory.pous.verifier.scheme import Encoding
from verity.protocols.accounting.memory.pous.verifier.verifier import MerkleTree, tree


@dataclass(frozen=True, slots=True)
class HonestResponder:
    """Answers from `C`, with no queries; `path` gives a root-only memory attester the audit path."""

    encoding: Encoding
    tree: MerkleTree | None = None

    def block(self, i: int) -> bytes:
        return self.encoding.block(i)

    def path(self, i: int) -> tuple[bytes, ...]:
        if self.tree is None:
            raise ValueError("build the responder with the encoding's tree to give paths")
        return self.tree.path(i)

    def answer(self, j: int, i: int) -> Prog[int]:
        return pure(self.encoding.blocks[i])


def server_root(encoding: Encoding) -> bytes:
    """The root the server reports for its `C`."""
    return tree(encoding).root


def responder(scheme: Scheme, block: Callable[[int], bytes]) -> Callable[[int], bytes]:
    """The honest responder: challenge `i` answered by `scheme.respond` from the stored labels."""
    return lambda i: scheme.respond(block, i)


__all__ = ["HonestResponder", "responder", "server_root"]
