"""The deployment layer: a protocol-agnostic runtime for parties that exchange messages over channels. `party` is the
interface a protocol's parties implement; `manifest` what a protocol declares (its parties, channels and messages) and
the placement check; `config` the deployment configuration, the public file a run starts from; `runtime` what runs the
parties, live, simulated, replayed or mixed (`run`); `stream` one party over a byte stream in its protocol's own codec;
`sources` the clocks and coins. Outside code imports from those six, the names the README's Interface section lists.
Internal: `transcript`, a party's view and the record, its public projection; `channels`, the pipes; and `process`, a
party in a process of its own. The layer reads no message's meaning and imports no protocol: the frames are
`verity.configuration.wire`'s.
"""
