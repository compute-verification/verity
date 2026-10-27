"""The simulated datacenter's workload: what runs inside the blocks and what crosses between them.

The logical layer is a Program and its values; the physical layer is devices in blocks, links, and certifiers on the
blocks' boundaries.

    model      the circuit: a tiny integer MLP over a batch of requests, a Boolean Program, and its integer reference;
               `build` picks the workload
    served     verity_vllm's serving Program (`Serve_v1`) for a truncated SmolLM2, derived and evaluated on CPU, with the
               MLP's interface
    hardware   the device graph in the shape of Lean's `Accounting.Hardware`: leaves, units (blocks), links, certifiers
    placement  the private placement in the shape of `Accounting.Requirement.Placement` (value -> leaf and step, device
               and step -> tick), its commitment, and the crossing edges it implies
    framing    φ: crossing values to messages, streams and B-byte frames; the declared traffic; the public occupancy profile
    values     each unit's commitment to the values it holds
    block      a block's process: evaluate its gates step by step, send and receive crossing values as framed streams
    run        build, place, run every block (in one process or one process each over TCP) and check the answers
    accounting the hardware and placement as `benchmarks/datacenter`'s objects, whose audit and account read the
               Program itself
    lab        the Program as the network certifier's sessions and server-sent events, for the mock datacenter
               (`../mockdc.py`)
    lab_run    serve a workload on the mock datacenter, each block behind the real certifier's proxy on its uplink, then
               the certifier's audit, N2's served verdict and the answers

The circuit stays untimed (the MLP Boolean, the served Program's values words). Transfers are derived, not gates: a
crossing edge is a value read by a gate placed in another unit, and the frames that carry it are fixed by φ.

The content check N3 (frames are committed crossings) was prototyped here and is retired: the network certifier's
step-indexed scan (`ingress_scan`, `egress`), proved in zero knowledge by `benchmarks/network_traces/scan_zk.py`,
supersedes it.
"""
