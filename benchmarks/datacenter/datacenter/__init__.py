"""A datacenter as Verity accounts for it.

The logical layer (`logical`, `claims`) is where the cryptography happens: a `verity.ml.circuits` Program, the commitments
to its registered inputs and its values, and `audit`, the claims a proof establishes about the prover system. The
physical layer (`physical`, `memory`, `account`) interprets those claims in light of the declared hardware, named
assumptions and trusted devices. The bridge (`binding`) places the Program on the hardware and publishes the placement's
properties (`Profile`); the verifier's `Plan` says what it will check; `observe.run` is the world, executing a placement
and recording what trusted devices see.

    p = place(program, dc, "a.gpu", registration=register(program, weights), deliver="world", x="world", y="b.gpu")
    profile = p.public(plan)
    claims = audit(run(p, {"x": x}, plan), profile)
    print(account(claims, profile))
"""
from .account import (DEDICATED_RACKS, FLEET_CLOSED, SALT_CUSTODY, SERVED_PARTITION, Account, BlockAccount, Credit, Custody,
                      Ledger, account, capacity, double_credit)
from .binding import Crossing, Placement, Profile, crossings, occupancy, place, route, traffic
from .claims import Carried, Computed, Entered, Held, On, Read, Refused, Scheduled, audit, check_records
from .logical import Registration, Value, ands, commit, evaluate, names, read, register, values, work
from .memory import (BLOCK_BITS, Append, Challenge, ReadWindow, Rekey, Release, Store, Wipe, answer_time, read_rate,
                     regen_time, serve_time)
from .observe import Answer, Deviation, Record, Trace, run
from .physical import Assumption, Block, Cadence, Datacenter, Device, Fleet, Link, Port
from .plan import Maintenance, Plan
from .time import Interval

__all__ = [
    "BLOCK_BITS", "DEDICATED_RACKS", "FLEET_CLOSED", "SALT_CUSTODY", "SERVED_PARTITION", "Account", "Answer", "Append",
    "Assumption", "Block", "BlockAccount", "Cadence", "Carried", "Challenge", "Computed", "Credit", "Crossing", "Custody",
    "Datacenter", "Deviation", "Device", "Entered",
    "Fleet", "Held", "Interval", "Ledger", "Link", "Maintenance", "On", "Placement", "Plan", "Port", "Profile", "Read",
    "ReadWindow", "Record", "Refused", "Registration", "Rekey", "Release", "Scheduled", "Store", "Trace", "Value", "Wipe",
    "account", "ands", "answer_time", "audit", "capacity", "check_records", "commit", "crossings", "double_credit", "evaluate",
    "names", "occupancy", "place", "read", "read_rate", "regen_time", "register", "route", "run", "serve_time", "traffic",
    "values", "work",
]
