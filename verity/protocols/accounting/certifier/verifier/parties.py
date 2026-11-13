"""The network certifier's manifest on the deployment layer (`verity.runtime`): its parties and the two
channels its output value takes, which a deployment configuration's table declares from here. It is the manifest alone.
No party of it runs under the layer's runtime, so this module holds no party and no `ProtocolDef`.

~~~text
network-certifier -> workers    out-opening {link, window}, the output value's witness its body (private)
network-certifier -> verifier   out-record {link, window}, the output value's record its body
~~~

- **The network certifier** (`NETWORK_CERTIFIER`) is the auditor's device on a server's link, one a link. Its machine
  gives it a clock (its grids release each bucket at a time fixed on its own clock), its own randomness (the
  capability `coins`: every salt it hides a record or a row under) and a signing key (`certificate.Signer`, whose
  one-time leaves sign each closed execution phase). Its pacer, its issuer and its link for the challenges
  (`coin_link`) are processes of the one device, not parties.
- **The workers** (`WORKERS`, the developer's) and **the verifier** (`VERIFIER`, the auditor's) are SPOT's parties,
  each declared here as SPOT declares it (`spot.verifier.parties.MANIFEST`).
- The challenger stays SPOT's party. It runs on the network certifier's device, the auditor's, which a
  configuration's table states (`config.Component.colocated`): a manifest names only its own parties.

The two messages are the two parts of the output value the certifier relayed (`out_commit.CertifiedOut`), for the egress
execution phase ``(link, window)`` their heads name:

- ``out-opening`` carries the witness (`CertifiedOut.witness`): the rows' salts in order, `out_commit.SALT_BYTES` bytes
  a row. The certifier hands it to the prover alone, so it is private and never enters a record.
- ``out-record`` carries the record (`CertifiedOut.public`) as canonical JSON: the value's signed head, which the
  verifier reads with `out_commit.check_out`. Its root is a hiding commitment, so it is public. It is a body and not a
  head's fields because its signature alone, in hex, is past a head's bound (`channel.MAX_HEAD`) at the deployed LMS
  parameters.

A head's ``link`` is the link's name as the certificate signs it and its ``window`` the window's index, a u64
(`lms.certificate_message`). The parameters are the core protocol's (`ID`): the ``PARAMS`` its Lean executable's check
reads, the core command line ``network-period check PARAMS COMMITMENT EVIDENCE``.
"""

from __future__ import annotations

from verity.commitments.domains import CERTIFIER_PROTOCOL
from verity.protocol import Channel, Manifest, Role
from verity.channel import Body, Int, Schema, Str
from verity.protocols.proofs.spot.verifier import parties as SP

from . import out_commit as O

ID = CERTIFIER_PROTOCOL
VERSION = 1
NETWORK_CERTIFIER, WORKERS, VERIFIER = "network-certifier", SP.WORKERS, SP.VERIFIER
OUT_OPENING, OUT_RECORD = "out-opening", "out-record"
#: a frame's largest body (`channel.HEADER`: a u32 length) in whole salts: the most rows one opening carries
MAX_OPENING = ((1 << 32) - 1) // O.SALT_BYTES * O.SALT_BYTES
#: a record is its head and one certificate, whose signature is about 70 kB in hex at `lms.DEPLOYED`
MAX_RECORD = 1 << 20
_HEAD = {"link": Str(255), "window": Int(0, (1 << 64) - 1)}
MANIFEST = Manifest(
    ID, VERSION, "network-period check PARAMS",
    parties=(Role(NETWORK_CERTIFIER, "auditor", per="link", needs=("clock", "coins", "signing-key")),
             SP.MANIFEST.role(WORKERS), SP.MANIFEST.role(VERIFIER)),
    channels=(Channel(NETWORK_CERTIFIER, WORKERS, (OUT_OPENING,)),
              Channel(NETWORK_CERTIFIER, VERIFIER, (OUT_RECORD,))),
    messages=(Schema(OUT_OPENING, 1, "private", _HEAD, Body(max=MAX_OPENING, unit=O.SALT_BYTES)),
              Schema(OUT_RECORD, 1, "commitment", _HEAD, Body(max=MAX_RECORD))))

__all__ = ["ID", "MANIFEST", "MAX_OPENING", "MAX_RECORD", "NETWORK_CERTIFIER", "OUT_OPENING", "OUT_RECORD", "VERIFIER",
           "VERSION", "WORKERS"]
