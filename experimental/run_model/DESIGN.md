# The run model (draft, not for landing)

For the migration captain and the decision memo of 5 Oct 2026, about 12:00Z. It builds on Daniel's locked points of 05:38Z
(the trusted code is Lean, one program per party over the whole run; Python, Rust and CUDA are the environment and the
workers; canonical bytes between them; the whole run composed in Lean; proofs apart from specs; untrusted code inside its
component; `verity/` is the whole system). It also builds on the proofs coordinator's recursive zero-knowledge
architecture. It adds the three questions the captain asked at 05:45Z: whether Lean drives Python (§2), whether "judge"
is a concept (§3), and where the statements go (§4).

The prototype beside this file backs the claims marked *measured*:
- `exe/` is a Lake package with no dependencies. It holds a party's program as data (`RunModel.Prog`), the one frame
  reader (`RunModel.Frame`), an interpreter and a replay, and a toy sampled audit written both ways.
- `workers/` holds the developer's Python worker and a Python driver.
- `security/` is a Lake package on core's Lean and Mathlib. It states the run's soundness and zero knowledge over the
  party programs.

The log is `art:69981778cc169c07aa583138cff652aabbe175c6033f61c989e3990b01d1dd42`.

## In short

- **The run is two Lean programs, the auditor's and the developer's, and everything else is their environment.** Each
  program asks for bytes and draws coins, and it holds its own state for the whole run. The environment is everything
  else: the other party, every worker, the network, the store and the launcher. It is adversarial for the guarantee in
  question.
- **Lean drives (§2).** Each party's program is the main loop, and the workers serve its requests over frames. When
  Python schedules Lean executables instead, soundness needs the driver to be honest. That contradicts locked point 2.
  In the prototype, a driver that reruns the stateless `draw` got a false claim accepted in 20 runs of 20. When the
  auditor's program drives, the same corrupted table was accepted in 0 of 50.
- **No judge kind (§3).** Two parties, one program each, on hosts that one party, both or neither trusts. The deciding
  functions (C-Flock's `verify`, the draw, PoUS's deadline check, the warden's window) are library functions those
  programs call. Nothing semantic is lost. The rules "judge" carried were there because Python drove.
- **Statements in one `security/` package (§4, option ii).** The end-to-end theorem is stated over party programs,
  which cut across components, and over one model. It needs Mathlib, and the programs must not. Option (ii) is the only
  shape that gives one package per set of `require`s. The main cost is the Lean audit's cache, which must become
  per-module before everything shares one package.

## 1. The run

### Parties, hosts and programs

The two parties are the developer and the auditor (Glossary). A *host* is a machine or process sandbox, trusted by a
party when soundness or zero knowledge can fail if it misbehaves. Trusting a host means it runs exactly its pinned
program, keeps that program's memory and coins to itself, and has no other way out.

| host | trusted by | what runs there |
|---|---|---|
| the auditor's machines | the auditor | **the auditor's program**, and workers (Python tooling, the store client) |
| the developer's CPU tier | the developer | **the developer's program**, and workers (staging, the store client) |
| the GPU rack | neither | workers only: serving, witness regeneration, the inner and outer provers, every kernel |
| an auditor device at the developer's network edge (the warden, PoUS's on-node challenge path) | the auditor, as a named physical assumption | the auditor's program's piece for that device |

Each party has one program over the whole run. A host on the other party's premises runs a piece of it, linked to the
rest over frames.
- **The auditor's program:**
  - takes the registration (the roots);
  - draws the proof units from its own coins, after it holds the roots;
  - runs each session: Hello, a live coin per round, the prover's messages (salted hashes under recursion), then C-Flock's
    `verify` as a function call on what it recorded;
  - runs PoUS's timed challenges and the warden's windows;
  - builds the integrity profile and the verdict, and writes the record.
- **The developer's program:**
  - asks its GPU workers for the roots and each round's messages;
  - salts and hashes each message as the proxy;
  - applies the masks, the send gate and the level-0 recheck;
  - answers draws and challenges.

  It sends the outside nothing its send gate hasn't checked.

### The environment

Each guarantee has its own environment, and every program quantifies over it.

- **Soundness, the auditor's guarantee.** The environment is everything but the auditor's program and its host: the
  developer's whole side, the network, the launcher, the store, and every worker on the auditor's own machines (locked
  point 2). It is one adversary, computationally bounded only where an assumption such as `cr/sha-512` says so.
- **Zero knowledge, the developer's guarantee.** The environment is everything but the developer's program and its host,
  in two parts that meet only through that program:
  - *inner*: the GPU rack and the developer's workers, which hold the witness and answer the program's requests on the
    inner channels;
  - *outer*: the auditor, the network and everything else, which sees only what the program sends it.

  Keeping the two apart is the deployment's egress premise, the warden's subject. Soundness needs no such split.
- **What neither program controls:**
  - timing and aborts, a channel the warden covers;
  - the host itself (isolation, the OS's randomness, the clock);
  - restarts. A run that starts and doesn't end counts as a refusal, so whoever consumes verdicts counts started runs.

### The guarantees

`security/RunModelSecurity.lean` states both guarantees over the programs, and they type-check against core's model:
- `RunSound auditor accept claim δ` is core's `Sound` of `auditor.toGame`. `Prog.toGame` makes each request a `send`
  node whose reply the environment chooses within the request's bound, and each coin a `coin` node. So the environment
  is exactly the prover's strategy in `Verity.Game`.
- `runSound_bind` is the run's soundness from its parts', through core's proved `SoundBind`. Its axioms are `propext`,
  `Classical.choice` and `Quot.sound`. `SoundInterleave` and `SoundBatch` carry over the same way, for sessions the
  environment schedules concurrently.
- `RunZK isInner developer sim ε` is statistical zero knowledge over the split environment. A simulator without the
  witness gives every outer environment, whatever the inner one does, a view within ε of the real program's.

The run's claim is the conjunction of the components' claims (the integrity profile's bounds, PoUW's γ, PoUS's space,
the warden's bits), and δ is the sum of their errors. The assumptions are:
- `cr/sha-512` and `statistical-hiding`;
- the OS's randomness, `uniform/io-getrandombytes`. This replaces A6 (`uniform/flock-coin-server`) and A5
  (`uniform/python-secrets`) once the coins are the programs' own;
- the hosts' named premises;
- the device models and timing for the accounting protocols;
- the interpreter's faithfulness: `Prog.run`, the frame reader and Lean's runtime.

## 2. Lean drives Python, or Python schedules Lean

### A: each party's Lean program is the main loop

~~~lean
def auditor (d k : Nat) : Prog Verdict := do
  let root ← Prog.req "developer" "register".toUTF8 64        -- a frame to the worker, a frame back
  let seed ← Prog.coins 32                                     -- IO.getRandomBytes, after the root is held
  for u in draw seed d k do
    let o ← Prog.req "developer" ("open".toUTF8 ++ u64be u) (8 + 64 * d)
    if let .error e := checkOpening root d u o then return ⟨false, s!"unit {u}: {e}", root, _⟩
  return ⟨true, "", root, _⟩
~~~

That is the prototype's auditor (`exe/RunModel/Audit.lean`); a real one has more steps, not a different shape.
- **The program is data.** `Prog` has three nodes: `ask` (a request with its channel, body and reply bound), `coin` and
  `ret`. The interpreter `Prog.run` turns each request into a frame on the named worker's stdin and reads the reply
  frame from its stdout, refusing a frame longer than the request's bound before reading it. A coin is
  `IO.getRandomBytes`. A timed protocol adds a clock node, `IO.monoNanosNow`, read by the program on either side of the
  request.
- **Bulk data stays out of the frames.** The witness, the 9 GB first-level tree and the inner proof stay in the workers,
  or go by file named by SHA-512, which the program reads and hashes itself, as `flock-verify verify --archive` does
  today.
- **Network.** Lean v4.34's `Std.Async.TCP` lets the program own its socket. A relay worker is just as sound, since it
  can only forward or drop what the program already chose to send.
- **The program starts its own workers.** It starts them from its configuration (`Worker.spawn`), so the launcher
  (`research run`) starts the party program and nothing else.
- **Concurrent sessions are one request.** The program asks a multiplexing worker for the next message from any
  session, so the environment chooses which session advances. That is `Game.interleave`'s semantics, so core's
  `SoundInterleave` and `SoundBatch` apply. Verifying a session is a pure function, so it can run on Lean tasks in
  parallel without changing the semantics.
- **Re-verifying a run is the same program.** `Prog.replay` is the program against its own transcript, a pure function.
  In the prototype, every run's replay reached the same verdict. Like any replay, it certifies nothing (Daniel's ruling
  on replays); it re-checks.

### B: Python schedules the Lean executables (today)

`benchmarks/one_stage/a0.py` runs:
- `flock-circuit serve`, Rust, as the live verifier: registration, the draw, a coin per round, the live verdict;
- `flock-circuit prove`;
- then `flock-verify archive-put`, `flock-verify verify --archive` and `flock-verify draw-test` on the files the serve
  wrote.

It then builds the audit record and the integrity profile in Python (`verity_sampled_proofs.one_stage.audit`). The
auditor's other coins and clocks are also outside Lean:
- sampled proofs' law draws from `os.urandom`;
- PoUW's window secret comes from `secrets.token_bytes` (A5);
- PoUS's timed audit and the warden read their clocks in Python (`perf_counter_ns`, `monotonic`).

### What each implies for the trust boundary

| | A: Lean drives | B: Python drives |
|---|---|---|
| what a theorem is about | each party's program over the whole run | one invocation of one executable; the run needs hypotheses about the schedule |
| coins drawn after the commitments | the program's control flow | the coin server and the driver (A6 says "drawn after the prover's message") |
| custody of the record between the live session and the verdict | none: the program verifies what it recorded itself | `RecordCustodyZK` = `RecordsLiveZK` ∧ `DrawFileZK`, a named assumption about non-Lean code ("nothing in a session record authenticates its writer") |
| drawing again | impossible within a run; a restart is a new run that whoever counts runs sees | invisible: the driver reruns `draw` and keeps the draw it likes |
| verdicts into the profile | the program | Python (`audit.audit_record`), which is then trusted |
| the bridge between the GPU rack and the auditor (zero knowledge) | the developer's program and its send gate | Python, which is then trusted by the developer |
| the boundary itself | the party program's process: its memory, coins and clock, and the one frame reader | each executable's arguments and files, plus every invariant across invocations, held by Python or assumed |

Under A, the trusted base for both guarantees is each party's program plus its host, and the only bytes it trusts are
the ones it reads through one frame reader. Under B, the trusted base also includes whatever keeps the invariants
between invocations: today the Rust coin server, the Python driver and the files between them. Each of those is either
a named custody assumption or trusted non-Lean code.

### What each implies for the model's environment

The environment is adversarial either way, but it can do more under B.
- **Under A,** the environment is a strategy that answers requests: exactly the prover in `Verity.Game`, with a `send`
  node per request and a `coin` node per draw. Both guarantees quantify over every such strategy, and nothing else needs
  modelling (`Prog.toGame`).
- **Under B,** the environment also schedules. It chooses which executable runs, when, how often, on which inputs, and
  which outputs it keeps. So the model must give the adversary oracle access to each executable, with fresh OS coins on
  every call.
  - Soundness then holds only for executables whose outputs authenticate themselves.
  - A coin-drawing executable's output never does, because the adversary picks it by repetition.
  - So under B, locked point 2 (the environment is adversarial) and live coins (no Fiat–Shamir) can't both hold unless
    the driver is trusted, or unless the state is held by a long-lived Lean process, which is A for that piece.
- **The same under both:** the other party and every worker are in the environment, timing and aborts are a channel,
  and the hosts are assumptions.

*Measured* (`workers/drive.py`): a toy table of 2^10 rows, a quarter of them wrong, with 16 units drawn, so a run should
accept with probability at most (3/4)^16 ≈ 1%.

| who drives | runs | accepted |
|---|---|---|
| the auditor's Lean program | 50 | 0 |
| a Python driver, honest | 50 | 0 |
| a Python driver that reruns `draw` until it misses the wrong rows (94–326 draws each) | 20 | 20 |

### What A costs

- **Per request, *measured*:** Lean asking a Python worker over pipes, on a 4-core VM, with 2,000 requests after 20
  warm-up requests.

  | message size | median | p99 | p99.9 |
  |---|---|---|---|
  | 8 B | 9.4 µs | 13 µs | 349 µs |
  | 64 KiB | 37 µs | 44 µs | 89 µs |
  | 2.5 MiB (one inner proof) | 2.9 ms | 4.6 ms | 5.5 ms |

  - That's ample for C-Flock's flows: 1–3 MB/s of inner proof per GPU.
  - PoUS's window is about 0.85 ms (Δ + RTT). A challenge that goes through a worker adds about 13 µs at p99. The
    tail on this VM is 0.35 ms at p99.9, so PoUS's timing loop should own its socket and read its clock in Lean, and be
    measured on a node with isolated cores before PoUS moves.
- **Heavy trusted checks cost the same either way.** The level-0 recheck in Lean is 4–8 times native, but it is trusted
  Lean under A and B alike.
- **Workers become request servers.** The prototype's frame loop is 15 lines of Python. Each party's orchestration,
  today in `a0.py`, `audit.py`, PoUS's audit and the warden, moves into its Lean program. Python keeps everything that
  doesn't decide: launching, staging bulk data, serving, proving and tooling.
- **Proofs.** The theorem is about `Prog`, which is data. What's trusted about IO is the interpreter (`Prog.run` is 3
  cases), the frame reader (30 lines) and Lean's runtime.

### Recommendation

A, one piece at a time, each removing an assumption about non-Lean code:
1. the auditor's coins and draw, and the live sessions' coins, which removes A6, `RecordCustodyZK` and `DrawFileZK`;
2. the integrity profile and the verdict, which removes the trusted Python in `one_stage.audit`;
3. the developer's proxy, send gate and recheck: the recursive design's boundary, already planned as one small Lean
   program;
4. PoUW's window secret (A5), then PoUS's and the warden's timing loops, after the node measurement.

## 3. Is "judge" a concept?

No. The run needs four kinds of thing:
- **parties**: the developer and the auditor;
- **one program per party**, over the run, in Lean;
- **hosts**, each trusted by one party, both or neither;
- **workers**, each on a host, and every one of them in the environment.

The deciding code is a set of library functions over canonical bytes, with no IO: C-Flock's `verify`, the draw, the
integrity profile, PoUS's deadline and answer check, the warden's window decision, the send gate, the proxy's hash and
the frame reader. The party programs call them. Some of those functions decide, but deciding is just what they compute,
not a kind of code.

What "judge" carried, and where each piece goes:

| what "judge" meant (v3 sketch) | in the run model |
|---|---|
| trusted Lean that decides | every line of a party's program is trusted; the deciding functions are ordinary functions in it |
| "takes the claim and its coins from its own pinned configuration and record, never from the driver" | automatic under A: the program holds its configuration and state. The rule existed because Python drove |
| "a judge imports only judges, and never Mathlib" | an executable imports only executables and Lean's core: a rule about executables, not about deciding |
| auditor code at the developer's network edge (the warden) | a host the auditor trusts on the developer's premises, running a piece of the auditor's program, under a named physical assumption |
| re-checking a run offline | the party's program in replay mode (`Prog.replay`) |

Nothing semantic is lost. The README's Glossary already defines the two parties and makes the coin server part of the
verifier, and it has no "judge" entry, so nothing there retires. What goes is the v3 sketch's `Judge/` directories,
which never landed. The PoUS grader (`pous-grader`) checks Lean submissions, so it is a tool in the assurance TCB, not a
party's program.

## 4. Where the statements go

### What the end-to-end theorem reads

The run's soundness and zero knowledge read:
- **both party programs**, which import the components' executables: C-Flock's verifier, the draw, the profile, PoUS's
  deadline, the warden's window and the send gate;
- **one model:** core's `Game` and `Sound`, the split environment, and the probability on Mathlib's `ℝ≥0∞`;
- **the components' guarantees**, as steps to the run's (and as guarantees of their own where a table relies on one, as
  on PoUW's γ);
- **the named assumptions**;
- for C-Flock's part, **ArkLib**.

The prototype has this shape:
- `exe/` builds on the Lean toolchain alone in about 2 s, with C-Flock's dependency-free `Flock.Hash` composed in by
  path.
- `security/` requires core's model and `exe/`, and states `RunSound` and `RunZK` over `exe/`'s programs.

### (i) Statements inline in each component's `lean/`, proofs apart

- **Mathlib in every component.** Each component's statements need core's model, and so Mathlib. Then either the
  component's executables live in a package that fetches Mathlib (Lake fetches every `require` of a package, used or
  not), or each component has two packages: about ten become about twenty.
- **The run's theorem has no component home.** A component's guarantee is stated over that component's functions, but
  the run's is stated over the party programs, which cut across components. The auditor's program calls sampled proofs'
  draw, C-Flock's `verify`, PoUW's tile check, PoUS's deadline and the warden's window. So the run's theorem needs a
  package above every component's spec, and (i) ends with a star into one package anyway.
- **The model is shared across packages.** It must live in core's package, which every component requires; today's
  four game models and two FP32 models are the drift this invites.
- **For it:** a reader of a component sees its guarantee beside its code, and owning a directory owns its statements.

### (ii) Components' `lean/` hold executables; one `security/` package holds the rest

- **One dependency closure.** The components' executables need nothing but the toolchain, and one package on Mathlib
  and ArkLib holds the model, the assumptions, the guarantees (the run's and each component's) and the proofs. Lake
  packages then split exactly where their `require`s differ: executables have none, and `security/` has Mathlib and
  ArkLib. That is principle 2's rule.
- **Building the trusted programs needs only the Lean toolchain**, for both parties.
- **The TCB is two lists:** every executable module (`verity/**/lean/`), and `security/`'s trusted text (its model,
  assumptions and guarantees modules, by the `layers` rule).
- **Costs:**
  - *The audit's cache* is the main velocity risk. `check` reuses a package's audit only while every input it read is
    unchanged, so one package re-audits everything on any change: soundness alone is 45–70 min cold. (ii) needs a
    per-module or per-declaration audit cache first, and the co-change evidence says how often this would bite.
  - *ArkLib for every proof.* It pins the same Mathlib, which is already the one Mathlib.
  - *The statement isn't beside its code.* `security/spec/` can mirror `verity/`'s component tree, with one ownership
    glob per subtree.
  - *Module roots.* `security/` needs roots of its own. The prototype hit this: `RunModel.Security` resolved to the
    executables' package, the verity#1144 rule, so it is `RunModelSecurity` now.

### Which the composition theorem wants

**(ii).** Its statement reads party programs that cut across components, plus one model. It needs Mathlib for
probability, and the programs it is about must not need it. Only (ii) meets both with one package per set of
`require`s. Option (i) arrives at a top-level package for the run anyway, and adds Mathlib to every component's `lean/`
or doubles the packages.

To keep locked point 5 (proofs apart from specs), `security/` has two libraries with roots of their own, in one Lake
package with one build directory:
- `security/spec/` (the model, assumptions and guarantees), which is trusted text;
- `security/proofs/` (every proof and lemma), which is untrusted.

The `layers` rule enforces the line between them. From tonight's tree this is two pure moves, each spec's modules into
`security/spec/<area>/` and each `security_proofs/<area>/` into `security/proofs/`, with module names kept. The
executables arrive as each party program's pieces move to Lean (§2's order). C-Flock's verifier is already one.

## 5. The tree this implies

~~~text
verity/                       the system: each component's Python, Rust and CUDA, and in its lean/ its executable Lean
  primitives/<p>/lean/        frames (the one reader), commitments (SHA-512, Merkle), randomness (the draw), circuits
                              (the canonical reader)
  protocols/<area>/lean/      each component's functions: C-Flock's verify, sampled proofs' draw and profile, PoUS's
                              deadline, the warden's window, the send gate
  <product>/lean/             each party's program over the run, and its interpreter (the onsite and remote protocols:
                              the "runs layer" the memo leaves open)
security/                     one Lake package on Mathlib, ArkLib and the executables
  spec/                       trusted text: the model (game, environment, run), assumptions, guarantees, mirroring verity/
  proofs/                     every proof and lemma (untrusted)
~~~

## 6. What this draft doesn't settle

- No real protocol is ported. The toy's claim (a committed table behind a computationally binding root) isn't stated,
  only the shape of its guarantee.
- `Prog` has no clock node yet, and the split environment doesn't model timing.
- The interpreter's faithfulness is stated in prose, not as a named assumption.
- For Daniel:
  - the order in §2's recommendation;
  - whether the per-module audit cache must land before (ii);
  - the hosts' named premises (isolation, the OS's randomness, the clock, the edge device);
  - whether `security/` sits inside `verity/` or beside it.
