"""A datacenter as Verity accounts for it.

The logical layer (`logical`, `claims`) is where the cryptography happens: a circuit, the commitment to its values, and
`audit`, the claims a proof establishes about the prover system. The physical layer (`physical`, `memory`, `account`)
interprets those claims in light of the declared hardware, named assumptions and trusted devices. The bridge
(`binding`) places the circuit on the hardware and publishes the placement's properties (`Profile`); the verifier's
`Plan` says what it will check; `observe.run` is the world, executing a placement and recording what trusted devices see.

    profile = p.public(plan)
    claims = audit(run(p, inputs, plan), profile)
    print(account(claims, profile))
"""
from .account import FLEET_CLOSED, Account, BlockAccount, Credit, Ledger, account, credits, double_credit
from .binding import Crossing, Placement, Profile, crossings, occupancy, place, route, traffic
from .claims import Carried, Computed, Entered, Held, On, Read, Refused, Scheduled, audit, check_records
from .logical import Circuit, Commitment, Input, Value, commit, digest, trace
from .memory import Challenge, Reads, Store, Wipe, answer_time, serve_time
from .observe import Answer, Deviation, Record, Trace, run
from .physical import Assumption, Block, Datacenter, Device, Fleet, Grid, Link, Port
from .plan import Plan
from .time import Interval

__all__ = [
    "FLEET_CLOSED", "Account", "Answer", "Assumption", "Block", "BlockAccount", "Carried", "Challenge", "Circuit",
    "Commitment", "Computed", "Credit", "Crossing", "Datacenter", "Deviation", "Device", "Entered", "Fleet", "Grid", "Held",
    "Input", "Interval", "Ledger", "Link", "On", "Placement", "Plan", "Port", "Profile", "Read", "Reads", "Record",
    "Refused", "Scheduled", "Store", "Trace", "Value", "Wipe", "account", "answer_time", "audit", "check_records", "commit", "credits",
    "crossings", "digest", "double_credit", "occupancy", "place", "route", "run", "serve_time", "trace", "traffic",
]
