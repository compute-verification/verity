"""The network attester's manifest on the deployment layer (`verity.runtime`): its parties and the two
channels that carry what the device hands another party, which a deployment configuration's table declares from
here. It is the manifest alone: the device's party and the `ProtocolDef` are `../attester/device_party.py`'s.

~~~text
network-certifier -> workers    out-opening {link, window}, the output value's witness its body (private)
                                records {link, window, record}, an execution phase's packed record its body (private)
                                cells-salts {link, window}, the execution phase's rows' salts and wiring's its body (private)
network-certifier -> verifier   out-record {link, window}, the output value's record its body
                                attestation {link, window}, an execution phase's attestation its body
                                registration {link, window}, the execution phase's rows' registration its body
                                deployment {}, the run's key, session, schedule and origin its body (structure)
                                groups {}, a grouped run's requests and execution phases its body (structure)
~~~

- **The network attester** (`NETWORK_ATTESTER`) is the auditor's device on a server's link, one a link. Its machine
  gives it a clock (its grids release each bucket at a time fixed on its own clock), its own randomness (the
  capability `coins`: every salt it hides a record or a row under) and a signing key (`certificate.Signer`, whose
  one-time leaves sign each closed execution phase). Its pacer, its issuer and its link for the challenges
  (`../attester/coin_link.py`) are processes of the one device, not parties.
- **The workers** (`WORKERS`, the developer's) and **the verifier** (`VERIFIER`, the auditor's) are SPOT's parties,
  each declared here as SPOT declares it (`spot.verifier.parties.MANIFEST`).
- The challenger stays SPOT's party. It runs on the network attester's device, the auditor's, which a
  configuration's table states (`config.Component.colocated`): a manifest names only its own parties.

The two messages are the two parts of the output value the attester relayed (`out_commit.AttestedOut`), for the egress
execution phase ``(link, window)`` their heads name:

- ``out-opening`` carries the witness (`AttestedOut.witness`): the rows' salts in order, `out_commit.SALT_BYTES` bytes
  a row. The attester hands it to the prover alone, so it is private and never enters a record.
- ``out-record`` carries the record (`AttestedOut.public`) as canonical JSON: the value's signed head, which the
  verifier reads with `out_commit.check_out`. Its root is a hiding commitment, so it is public. It is a body and not a
  head's fields because its signature alone, in hex, is past a head's bound (`channel.MAX_HEAD`) at the deployed LMS
  parameters.

The device's run hands out the rest, each a message of its kind on the same two channels:

- ``certificate`` carries a line of the run's ``certified.jsonl``: an execution phase's record attested under the link's
  name and direction, or that execution phase refused. Public, as ``out-record`` is: what it signs is a hiding
  commitment.
- ``registration`` carries a line of ``cells.jsonl``: the execution phase's rows' signed head (`network_attester_rows.registration`),
  or that execution phase refused, and its cost (the rows and a row's bytes). Public: its root is a hiding commitment.
  It carries no device timing, which varies with what was committed: the device keeps those with its witness.
- ``deployment`` carries ``deployment.json`` as the check reads it: the device's public key and session, the schedule's
  parameters and their digest, the windows, the origin and the run id. It is protocol structure, so public. It holds no
  address: where the device listened and what it relayed to are a local file beside it, which no party is handed.
- ``groups`` carries a ``--window-per-group`` run's ``groups.json``: each group's request indices and its execution
  phases, which the schedule and the requests' public counts fix. Protocol structure, so public.
- ``records`` carries one of an execution phase's three records (``record``: ``requests``, ``responses`` or ``fails``), packed
  as the device keeps it, and ``cells-salts`` the execution phase's rows' salts and the wiring's (`network_attester_rows.read_witness`).
  Both are the prover's witness: private.

A head's ``link`` is the link's name as the attestation signs it and its ``window`` the window's index, a u64
(`lms.certificate_message`). The parameters are the core protocol's (`ID`): the ``PARAMS`` its Lean executable's check
reads, the core command line ``network-period check PARAMS COMMITMENT EVIDENCE``.
"""

from __future__ import annotations

from verity.commitments.domains import CERTIFIER_PROTOCOL
from verity.protocol import Channel, Manifest, Role
from verity.channel import Body, Int, Schema, Str
from verity.protocols.proofs.spot.verifier import parties as SP

from . import out_commit as O
from . import network_attester_rows as WR

ID = CERTIFIER_PROTOCOL
VERSION = 1
NETWORK_ATTESTER, WORKERS, VERIFIER = "network-certifier", SP.WORKERS, SP.VERIFIER
OUT_OPENING, OUT_RECORD = "out-opening", "out-record"
CERTIFICATE, REGISTRATION, RECORDS, CELLS_SALTS = "certificate", "registration", "records", "cells-salts"
DEPLOYMENT, GROUPS = "deployment", "groups"
#: an execution phase's three records, by the name its ``records`` message's head gives
RECORD_NAMES = ("requests", "responses", "fails")
#: a frame's largest body (`runtime.framing.HEADER`: a u32 length) in whole salts: the most rows one opening carries
MAX_OPENING = ((1 << 32) - 1) // O.SALT_BYTES * O.SALT_BYTES
#: a record is its head and one attestation, whose signature is about 70 kB in hex at `lms.DEPLOYED`
MAX_RECORD = 1 << 20
#: a frame's largest body (`runtime.framing.HEADER`: a u32 length): an execution phase's packed record
MAX_PACKED = (1 << 32) - 1
#: an execution phase's rows' salts and the wiring's, a whole number of the wiring's salt bytes, which a row's are a multiple of
MAX_SALTS = MAX_PACKED // WR.WIRING_SALT_BYTES * WR.WIRING_SALT_BYTES
_HEAD = {"link": Str(255), "window": Int(0, (1 << 64) - 1)}
MANIFEST = Manifest(
    ID, VERSION, "network-period check PARAMS",
    parties=(Role(NETWORK_ATTESTER, "auditor", per="link", needs=("clock", "coins", "signing-key")),
             SP.MANIFEST.role(WORKERS), SP.MANIFEST.role(VERIFIER)),
    channels=(Channel(NETWORK_ATTESTER, WORKERS, (OUT_OPENING, RECORDS, CELLS_SALTS)),
              Channel(NETWORK_ATTESTER, VERIFIER, (OUT_RECORD, CERTIFICATE, REGISTRATION, DEPLOYMENT, GROUPS))),
    messages=(Schema(OUT_OPENING, 1, "private", _HEAD, Body(max=MAX_OPENING, unit=O.SALT_BYTES)),
              Schema(OUT_RECORD, 1, "commitment", _HEAD, Body(max=MAX_RECORD)),
              Schema(RECORDS, 1, "private", _HEAD | {"record": Str(9, "|".join(RECORD_NAMES))}, Body(max=MAX_PACKED)),
              Schema(CELLS_SALTS, 1, "private", _HEAD, Body(max=MAX_SALTS, unit=WR.WIRING_SALT_BYTES)),
              Schema(CERTIFICATE, 1, "commitment", _HEAD, Body(max=MAX_RECORD)),
              Schema(REGISTRATION, 1, "commitment", _HEAD, Body(max=MAX_RECORD)),
              Schema(DEPLOYMENT, 1, "structure", {}, Body(max=MAX_RECORD)),
              Schema(GROUPS, 1, "structure", {}, Body(max=MAX_RECORD))))

__all__ = ["CELLS_SALTS", "CERTIFICATE", "DEPLOYMENT", "GROUPS", "ID", "MANIFEST", "MAX_OPENING", "MAX_PACKED",
           "MAX_RECORD", "MAX_SALTS", "NETWORK_ATTESTER", "OUT_OPENING", "OUT_RECORD", "RECORDS", "RECORD_NAMES", "REGISTRATION",
           "VERIFIER", "VERSION", "WORKERS"]
