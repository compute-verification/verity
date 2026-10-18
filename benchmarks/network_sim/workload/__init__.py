"""The simulated datacenter's workload: what runs inside the blocks and what crosses between them.

The logical layer is a Boolean Program and its values; the physical layer is devices in blocks, links, and certifiers on the
blocks' boundaries. They meet in one place, the consistency relation N3
(`note:network-accounting/20261006T0633Z-draft-circuit-certificate-consistency`), checked here in the clear.

    model      the circuit: a tiny integer MLP over a batch of requests, a Boolean Program, and its integer reference
    hardware   the device graph in the shape of Lean's `Accounting.Hardware`: leaves, units (blocks), links, certifiers
    placement  the private placement in the shape of `Accounting.Requirement.Placement` (value -> leaf and step, device
               and step -> tick), its commitment, and the crossing edges it implies
    framing    φ: crossing values to messages, streams and B-byte frames; the declared traffic; the public occupancy profile
    values     each unit's commitment to the values it holds
    block      a block's process: evaluate its gates step by step, send and receive crossing values as framed streams
    n3         the in-the-clear consistency check over what the certified links carried
    run        build, place, run every block (in one process or one process each over TCP) and check
    accounting the Program, hardware and placement as `benchmarks/datacenter`'s objects, for its audit and account

The circuit stays Boolean and untimed. Transfers are derived, not gates: a crossing edge is a value read by a gate placed in
another unit, and the frames that carry it are fixed by φ.
"""
