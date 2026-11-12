# The ZK server

The ZK server is the developer's only path from its workers to the auditor
([ARCHITECTURE.md](../../../../../ARCHITECTURE.md)'s Glossary), trusted by the developer and by no one else. In a recursive
session the workers prove the statement with Π_in, and the ZK server commits every message they send, masks, checks or
fixes everything else, and releases Π_out's zero-knowledge proofs that V_in, Π_in's verifier, accepts, one per
statement, on a fixed schedule. [Flock's live protocol](../../../flock/prover/live/PROTOCOL.md), §10, says what it computes.

Its Lean is two packages, code and not analysis:

- `Flock.Firewall`, in [Flock's Lean verifier](../../../flock/verifier/README.md) and run as `flock-firewall`: the session loop, each
  rep's frame, the tap and the coin check, which hold every message to §10's send check;
- this package, `core_firewall`, the release side (§10.5): the record `firewall/v0`, one Π_out session's decision over
  `Flock.Firewall.Contract`, that decision as a core protocol (`core-firewall`), and the release side around the session
  loop on the record's files (`core-firewall session`, which `flock-circuit`'s `--firewall-state` runs). Its tests hold
  it to Python's `verity_flock.firewall` and to the frozen cases of the Rust reference it replaced, and the rules it
  runs are proved properties (`security/Properties/Firewall.lean`).

## The reference deployment

Where each party runs and how the links go: the workers on the developer's GPU servers, the ZK server on CPU servers
of its own, and the verifier with the auditor. Each element names the assumption or guarantee premise it realizes, or
says that it realizes none yet. None of them is in place today (Today, below).

~~~text
GPU servers ──data diode──▶ CPU servers ──releases──▶ verifier
     ▲                           ▲                        │
     └─────── live coins ────────┴────── live coins ──────┘
~~~

- **The workers run on the developer's GPU servers, and nobody trusts them.** They hold the values, run the
  computation and prove with Π_in. *Realizes:* nothing needs to. The verifier checks every Π_out proof whatever the
  workers or the ZK server computed, so no soundness statement reads either's honesty
  (`Flock.Firewall.Contract.FirewallComputes` bears on zero knowledge only), and no zero-knowledge statement reads the
  workers' honesty: `Flock.zk_server_blind` holds at every pad-blind worker, and
  `Flock.Guarantees.RecursiveZKLoopAnySum` takes no premise on them.
- **The workers send only through a data diode, to the ZK server.** The one link between the GPU servers and the
  ZK server is a one-way device: bytes cross from the workers to the ZK server, and nothing crosses back.
  *Realizes:* the workers' messages are fixed without the ZK server's salts that commit them: in
  `Flock.Guarantees.RecursiveZK` the auditor's view is the ZK server's commitments to messages that read the auditor's
  tape and the developer's randomness, never the ZK server's salts (`OuterViewZK`'s `msgs a p`). Physically, none yet.
- **The verifier sends the live coins to the workers and to the ZK server directly.** Its challenger sends each coin
  to both at once, after the ZK server's message that the coin answers. *Realizes:* the trusted base's live coins
  (`Claims.trustedBase`: each coin uniform, and hidden from the developer's side until the verifier opens it), and
  `trusted-host` for the challenger (`Verity.Assumptions.TrustedHost`). The direct route is what lets the ZK server
  send the workers nothing (below); today the ZK server relays the coins to them (§10.3, §10.4).
- **The ZK server runs on CPU servers of its own, from another vendor than the GPU servers'.** Physical machines, not
  VMs, that run nothing else, with as many instances side by side as keep up with the windows (Its size, below).
  *Realizes:* A3's scope (`FlockSoundness.Assumptions.UniformRandomBytesTest` doesn't cover a VM whose memory is
  restored more than once:
  [A3's note](../../../../../security/Proofs/Flock/Soundness/assumption-notes/a3-uniform-bytes.md)), and §10.5's public
  release times for honest sessions: on cores of its own, no co-tenant's load can turn one into a missed time and
  spend the stream's failure budget. The other vendor realizes none: no statement names the ZK server's hardware,
  which the developer trusts outside the model.
- **The ZK server is the only exit.** Nothing of the workers' reaches the verifier except in the ZK server's release.
  *Realizes:* the auditor's view in `Flock.Guarantees.RecursiveZK` and `Flock.zk_server_blind`, which is the release
  and the auditor's own coins. None yet physically: the ZK server runs on the workers' machines.
- **The ZK server runs only the compiled Lean** (`flock-firewall`, `core-firewall`), and nothing else runs on its
  servers. *Realizes:* given the trusted base's Lean compiler and runtime, `Flock.Firewall.Contract.FirewallComputes`
  (every value the ZK server reports fixing by its own computation is the specified one), and the hypotheses through
  which `Flock.zk_server_lean_loop` reads the released bytes, not yet proved about the live path
  ([AGENTS.md](../../../../../AGENTS.md), "Lean"). Today Rust and Python run beside it: `flock-circuit`'s host side, and
  `rec_live`'s relay while the workers prove with Π_in.
- **The ZK server mixes two entropy sources**: each random byte it uses is the XOR of the operating system's and an
  independent hardware generator's. *Realizes:* A3 per test, which zero knowledge takes for the ZK server's random
  bytes (`Flock.SecurityProofs.recursiveZK_os_gap`, `Claims.ZKServer.OsTape`), since the mix passes the test if either
  source does. No premise states that either-source form yet, and today the ZK server reads the operating system
  alone.
- **The ZK server never sends to the workers**; the diode is the only link between them. *Realizes:* the pad-blind
  worker of `Flock.zk_server_blind` (`ZkServer.BWorker`: handed only its own part of the ZK server's tape and the
  verifier's answers, never the ZK server's pads, masks and salts or its released messages), here handed no part of that
  tape at all, where the general model's worker (`ZkServer.Worker`) may read the whole of it. None yet: today the tap
  serves the workers' reads from the ZK server's tape (`flock-firewall tap`) and the relay passes them the coins.
- **The ZK server releases fixed-size messages on a fixed schedule**: the public schedule fixes every message's length
  and release time, a failed session's included. *Realizes:* the model's release times (`ZkServer.timed`, each message
  at `ZkServer.sched` of its round alone), which `Flock.zk_server_blind`'s events read, the constant frame skeleton
  (§10.1), and §10.5's rule that proofs leave at public times. Today proofs and `Finish` leave at their public times,
  but the pace of a session's rounds and the time of a cut do not yet, and §10.5's bound doesn't count them.

Two elements are optional:

- **A release gate**: a small device on the ZK server's outbound link that passes only frames of the schedule's
  lengths at the schedule's times, and drops anything else. *Realizes:* no further premise; it holds the previous
  element should the ZK server send off schedule. None yet.
- **A framing filter** at the diode's output, passing the ZK server only well-formed messages of the schedule's kinds
  and lengths. *Realizes:* no premise; the ZK server already refuses a request it can't read (§10.3), and the filter
  narrows what reaches its parser. None yet.

### Its size

The ZK server proves all of Π_out on CPU, so Π_out's CPU time and peak memory per window set its size: as many
instances side by side, each with its vCPUs and memory, as keep up with the GPU servers' windows.
[The recursion benchmark](../../../../../benchmarks/recursion_e2e/README.md) times Π_out's steps on a CPU node. Its
figures are upper bounds where other jobs share the node's vCPUs. A memory total over several instances assumes each
statement's session frees its memory once its proof is released. The workers prove with zero knowledge off, so the
figures leave out V\*'s growth under their `--zk`.

The size depends on where Π_out runs. An option that has the workers prove Π_out, or most of it, with `--zk` on their
GPUs, and leaves the ZK server only the wrap (committing the workers' messages under salts of its own), shrinks it.
That is an open choice, not a decision.

### What stays trusted

- **The ZK server's randomness.** Hash-based proofs can't be re-randomized the way Ganesh–Magri–Venturi 2020
  re-randomize malleable ones, so the ZK server doesn't pass the workers' proof on: it proves with Π_out, with
  randomness of its own, that V_in accepted. Zero knowledge rests on the ZK server keeping that randomness secret.
- **Zero knowledge rests on the ZK server alone.** The workers prove with zero knowledge off, so the ZK server sees
  the values; zero knowledge if either the workers or the ZK server is honest is not a guarantee yet.
- **The stop channel is bounded, not removed.** Whether a release fails is the workers' to choose, so a failing
  session shows that it failed. Within one Π_out session the ZK server defers every stop to its `Finish`, so the
  session reveals one bit (`Flock.zk_server_blind`); a recursive session's stop shows where it fell, at most
  `log₂(1 + Σ_p (M_p + 1))` bits (§10.5, `Flock.Guarantees.RecursiveZKLoopAnySum`), and a stream's failure budget
  bounds their sum. No element above changes that count.

### Today

The ZK server runs on the workers' machines: as `rec_live`'s proxy while the workers prove with Π_in, and as
`flock-circuit`'s host side with the Lean session loop while it proves with Π_out (§10). The coins reach the workers
through it, and the tap serves the workers' reads from its tape. So no element above is in place, and the premises
they realize rest on the ZK server's code and the operating system's process isolation instead.
