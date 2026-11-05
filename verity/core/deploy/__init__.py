"""The deployment layer: a protocol-agnostic runtime for parties that exchange messages over channels. `party` is the
interface a protocol's parties implement; `manifest` what a protocol declares (its parties, channels and messages) and the placement check; `transcript`
a party's view and the record, its public projection; `channels` the pipes; `sources` the clocks and coins; `runtime`
what runs them, live, simulated, replayed or mixed (`run`), and `process` a party in a process of its own. It reads no
message's meaning and imports no protocol: the frames are `verity.core.wire`'s.
"""
