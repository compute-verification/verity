"""The simulated datacenter's workload: what runs inside the blocks and what crosses between them.

The logical layer is a Program and its values; the physical layer is devices in blocks, links, and certifiers on the
blocks' boundaries. They meet in one place, the consistency relation N3
(`note:network-accounting/20261006T0633Z-draft-circuit-certificate-consistency`), checked here in the clear.

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
    n3         the in-the-clear consistency check over what the certified links carried
    n3_unit    N3's proof block for one port-window, a Boolean Definition (`N3Window_v3`), the outbox and layout
               commitments and N2's record commitment it opens, and each window's instance on the simulator and the lab
    n3_v4      `N3Window_v4`: v3's unit with its descriptors and rows opened as runs
    n3_v5      `N3Window_v5`: v4's over outbox rows without overlap, with a template, and with frames wider than rows
    n3_cost    N3's ANDs on every link-window: the simulator's and the lab's as built, served rows' at their shape
    run        build, place, run every block (in one process or one process each over TCP) and check
    accounting the hardware and placement as `benchmarks/datacenter`'s objects, whose audit and account read the
               Program itself
    lab        the Program as the network certifier's sessions and server-sent events, for the mock datacenter
               (`../mockdc.py`), and N3 over the records the certifiers publish
    lab_run    serve a workload on the mock datacenter, each block behind the real certifier's proxy on its uplink, then
               the certifier's audit, N3 and the answers

The circuit stays untimed (the MLP Boolean, the served Program's values words). Transfers are derived, not gates: a
crossing edge is a value read by a gate placed in another unit, and the frames that carry it are fixed by φ.
"""
