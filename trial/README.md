# Layout trial

A toy of the proposed layout: remote and onsite Verity end to end on real circuits, with integrity implemented and
the rest standing in for the main repository's protocols:

~~~bash
uv sync
uv run python examples/remote.py [f32_add|bf16_gt_strict]
uv run python examples/onsite.py [f32_add|bf16_gt_strict]
uv run python examples/integrity.py [f32_add|bf16_gt_strict] [replay|transparent|sp1]
uv run python examples/datacenter.py [scenario ...]    # a site over a period of epochs, honest and otherwise
lake build    # the Lean beside the Python
~~~

## The model

The principals are the prover, the verifier, and the trusted hardware: replayers, wardens, pingers. An actor is one
principal's part in one protocol, a transducer over one ordered log of events (`verity.primitives.events`). A send goes
to a principal, and every other actor of that principal receives it; a record (a send to no one) goes to the sender's own
principal, which is how one protocol's verifier reads what another's established. What an actor observes and computes
in between is its own state and never enters the log. `verity.run.run(principals, view)` takes each principal's actors,
plays them against each other and yields one principal's view. The auditor trusts the verifier, so the verifier's view
is the output: what its actors establish are records in it. Replayers, wardens and pingers are trusted too but sealed:
what they see stays their own, and reaches the verifier only as what they send it.

Everything is accounted against one stream (`protocols/accounting/integrity.py`). First, before any coins, the prover
registers its builder: the program that builds each execution's circuit and partition from a bounded bitstring of
advice (the runtime choices no one could fix ahead of time: how requests were batched, which step a session continues,
a preemption, an autotune pick), holding the baseline (weights, constants) as every circuit's first inputs. Then per
execution its advice and a root over the boundary wires of its partition's regions, until the prover closes the stream:
one epoch. A partition has two levels (`primitives/circuits.py`): inference units, which NCI bounds, each split into
regions. An execution may read state an earlier one wrote (as a decode step reads its KV cache): that wire is committed
once, in the writer's root, so the step that reads it reads what the step that wrote it committed, and the stream's
inference units join through it: a session is one inference unit, which reads only its prompt however many steps it
took. Integrity's verifier records the stream, which the other protocols read, and checks units drawn from it. Every
statement over the stream names its digest.

**Verification** establishes that a computation gave some outputs: the prover proves it (`proofs`), or trusted hardware
runs it again (`replay`). Both decide the same claim, that a region of a circuit computes the outputs committed for it
from the inputs committed for it, and record `Verified`; they differ in who sees the opened values. Integrity draws
units of the stream (whole executions, or regions) and reads `Verified` for them; a policy may verify computations of
its own the same way.

**Accounting** says what happened to each resource: **work** (operations performed in a window), **space** (bits held
at an instant) and **communication** (bits that cross the site's boundary, through any channel, side channels too).
Integrity says what ran; each resource's module establishes that resource's line, a cryptographic bound (useful work,
useful space, the warden's audit) wrapped in the physical facts it rests on; and the account sets the lines side by
side. Per resource:
- **capacity**: what the site could have supplied, its devices times their facts;
- **claimed**: what the stream says went to the declared work;
- **observed**: what hardware at the site saw used, by anything (today only communication, by the warden);
- **explained**: the claim as far as verification backs it, a lower bound;
- **headroom**: `min(capacity, observed) - explained`, where anything else could have run.

**Compliance** says whether that was allowed: a policy's parameters are its limits, and its statement says whether the
epoch kept to them. NCI is the first: it reads the stream as integrity verified it, and the account, whose headroom it
limits, since whatever ran there went unchecked. An epoch's verdict says nothing outside its window, so a period of
epochs complies only if each does and their windows leave no gap (`nci.period`).

The two products, `remote` and `onsite`, run the same protocols and differ in who reports each reading. Remotely the
verifier has nothing at the site: its own clock times the work, and the prover reports its devices, so remote shows
that the declared devices were busy with declared work and never that there were no others, and it observes nothing.
Onsite the verifier's hardware counts the devices, times work and space challenges, replays executions and watches the
links, so it also accounts for space and communication, and its device count is a measurement.

A construction is a frozen dataclass of its parameters, with a method per principal it involves that returns that
principal's actor (`Integrity(Replay(), law, delta).verifier()`). A trusted principal's method takes its readings
(`pinger(site, work_window, space_window)`), and an actor that reads one takes the principal it trusts for it
(`Work.verifier(clock_from, site_from)`). Choosing a construction is choosing a value. A product is a construction whose fields
are constructions: its methods return the lists of its parts' actors, and its guarantee is proved from theirs. Actors
have no names and are not wired: anyone can send to anyone, so an actor waits for each kind of event only from the
principal it trusts for it (`receive(kind, sender)`, `gather`, `collect`), and an event keeps in its mailbox until it
waits for it. Nothing orchestrates the actors: each moves when an event it waits for reaches it, so each keeps its own
cadence over the log. Order in the log is logical; physical time enters only as trusted readouts in what a trusted
principal sends.

## Layout

Imports point down only, in Python and in Lean:

~~~text
verity/run.py                 the runner
verity/protocols/             everything with principals
    remote.py, onsite.py      the products: protocols composed, and who reports each reading
    compliance/               whether it was allowed: nci
    accounting/               what happened:
        account.py            each resource's line, side by side
        work.py               one module per resource: a bound, wrapped in the physical facts it rests on
        space.py
        communication.py
        integrity.py          what ran: the stream, and checks of units drawn from it
    verification/             that a computation gave these outputs:
        claim.py              what both establish: a claim on a region's committed boundary, `Verified`
        proofs/               the prover proves it (transparent, sp1/)
        replay.py             trusted hardware runs it again
verity/primitives/            no principals: events, circuits (with partitions and builders), commitments,
                              randomness (coins, and the laws that sample with them), physical (a clock's readings,
                              device types' facts, a site's devices)
~~~

The rules the layout follows, for the next question it raises:
- **Primitives have no principals; protocols do.** A protocol is a construction with a method per principal, and
  records a statement that holds except with the probability it states. A product is a protocol whose parts are
  protocols.
- **The tree follows imports.** A layer is a folder only where a split by purpose is also a split by import order:
  verification, then accounting (what happened), then compliance (was it okay). Below a layer, a folder holds several
  constructions of one statement (`proofs/`: transparent, sp1); anything else is a module.
- **A protocol imports another only for the statement it reads**, never its actors.
- **Who reports a reading is a product's wiring, never a folder.** A reading from the prover is an assumption, one
  from trusted hardware a measurement; the same protocol takes either (`clock_from`, `site_from`).
- **Physical assumptions have three homes.** How the physical world is represented is a primitive (`physical`); the
  facts assumed (the census's, the side channels' bits, a deadline) are a construction's parameters; and the reasoning
  that wraps a cryptographic bound in them is in the resource's module.

`examples/` sit on top: a run of each product, integrity alone, and a datacenter over a period of epochs. An
integration (vLLM) is a prover.

## Conventions

- **Lean sits beside the Python it states.** `verity/primitives/Circuits.lean` sits beside `circuits.py`. There is one
  Lake package at the root (`lakefile.toml`), and a module's name is its path. The wheel excludes `*.lean` and
  `.lake`.
- **`__init__.py` holds only a docstring.** Code goes in named modules, so the files can be dropped (making the
  folders namespace packages) once `packages/verity` is gone and nothing else named `verity` is on the path.
- **Heavy constructions have their own `pyproject.toml`.** It sits beside the source and declares the dependencies
  (`protocols/verification/proofs/sp1/`); the source is imported only by whoever asks for it.
