# SP1 backend

D-SP1. Frozen: kept, still in the tables, not developed. What the merge gate runs here: `../README.md`.

Establishes Verity verification obligations with the SP1 zkVM: a generic
checker guest authenticates Merkle openings of the boundary values and
recomputes the gate relations; SP1 proves that execution. The proving system
is unmodified (`sp1-sdk`/`sp1-zkvm`/`sp1-build` pinned `=6.4.0` from
crates.io, `Cargo.lock` committed).

Three crates: `common` (statement/witness codec, Merkle framing, gate
semantics, the kernel), `guest` (the zkVM program: `common` behind
`read_vec`/`commit_slice`), `host` (the CLI `veritor-zk-host`; its `build.rs`
compiles the guest ELF). `check-model` is a dependency-free restatement of
the validator/interpreter used only by `common`'s tests.

Two variants of SP1 itself, each built on a pod: `tcdot/` (its own workspace) is a fork of SP1 v6.4.0 with a tensor-core
dot-product chip (`build_fork.sh`, `sp1-patches/`, `bench.py`); `sec128/` is stock SP1 with the core FRI query count
raised to a 134-bit target (`build.sh`, `run.sh`).

## Using it

Rule: `execute` is light (the K=1536 vector is 10 M cycles, seconds on a
laptop). **Proving is heavy and runs only on the SP1 pod**, never on a laptop
or on the control pod. Building the host takes the SP1 toolchain and ~10 min.

Build (needs `sp1up --version v6.4.0`, i.e. `cargo prove` and the `succinct`
toolchain, plus `clang`/`libclang`, `cmake`, `protobuf-compiler` on Linux):

~~~sh
cd archive/sp1
cargo build --release -p veritor-zk-host                  # CPU prover; also builds the guest ELF
cargo build --release -p veritor-zk-host --features cuda  # GPU prover (Linux + NVIDIA; run with SP1_PROVER=cuda)
./target/release/veritor-zk-host info                     # must print the approved identity below
~~~

Host commands `info | execute | prove | verify` each print one JSON object as
the last `{...}` line of stdout (tracing goes to stdout too under `RUST_LOG`,
so take the last JSON line, as the driver does). Two environment variables
matter: `VERITY_SP1_HOST=<path>` names an installed host binary (default:
`target/release/veritor-zk-host`), `SP1_PROVER=cuda` selects GPU proving.

From Python, `verity_sp1.typed` is the backend boundary: a
`typed-obligation/v0` object in, a judgment out.

~~~python
from verity_sp1.host import Host
from verity_sp1.typed import execute_object, prove_object, verify_object

host = Host(timeout=1800)                      # or Host(binary=Path(...))
report, binding = execute_object(host, obj, witness)      # cycles, public values, binding check
report = prove_object(host, obj, witness, proof_path, mode="core")   # "core" | "compressed"
acceptance = verify_object(host, obj, proof_path)          # .accepted, .reasons()
~~~

`acceptance.accepted` is true only when the proof verifies, the verifying key
is `APPROVED`, the public values are `h_O || h_L || 1` for this object, and
the guest was not a diagnostic build. The guest commits those 33 bytes as its
public values; nothing else about the statement is trusted.

The benchmark workload is `archive/sp1/benchmarks/dot_product/vector_run.py`; run it on
the pod through `research` so the result is recorded and preserved:

~~~sh
uv run research run --on vy-sp1 --project verity --source . --stage sp1.prove --require-result --exclusive \
  --env VERITY_SP1_HOST=/workspace/bin/veritor-zk-host-cuda --env SP1_PROVER=cuda \
  -- /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 \
     archive/sp1/benchmarks/dot_product/vector_run.py --vector accept--packed-word-row --backend sp1-prove --mode core
uv run research fetch <run-id> --all && uv run research inspect <run-id>
~~~

`--source .` ships `git archive HEAD`, so commit first; the snapshot cannot
contain ignored files (`common/src/elem_silu_table.bin` is force-tracked for
that reason). Backends are `reference` (Python re-derivation, runs anywhere),
`sp1-execute`, `sp1-prove`. Baseline recorded runs (2026-09-21, RTX 4090):
K=16 core proof `r20260921-215829-9796` -- 271,289 cycles, 2 shards,
`prove_seconds` 0.89 s after an 18 s one-time key setup, 2.8 MB proof.
K=1536 (`stage2--qkv-proj-decode-0-k1536`) core proof `r20260921-220722-019b`
-- 10,128,736 cycles, 4 shards, `prove_seconds` 3.58 s (17 s setup), 5.9 MB
proof, `verify_wall` 17.5 s; compressed `r20260921-221023-972e` -- 1 shard,
`prove_seconds` 4.46 s, 1.3 MB proof, `verify_wall` 19.6 s; execute-only
`r20260921-221253-d23e` -- same 10,128,736 cycles in 1.26 s.

`archive/sp1/benchmarks/dot_product/baselines.json` freezes these runs with their labels
(`proof_class`, `authentication`) and peak memory; it is the record to compare
against.

Arithmetic-only DIAGNOSTIC (2026-09-21, `r20260921-224405-22dc`): the
`unsound-skip-authentication` guest (built by `SP1_HOST_FEATURES=cuda,unsound-skip-authentication
bash archive/sp1/pod_bootstrap.sh`, run with `vector_run.py --diagnostic`)
proves K=1536 in 9,992,112 cycles, 4 shards, `prove_seconds` 3.66 s (sound:
10,128,736 cycles, 4 shards, 3.58 s; three diagnostic proofs spread 3.48-3.66 s),
peak GPU memory unchanged at 19.5 GiB. What the skip removes is `merkle.auth`
(leaf hashing and path folding), 138,164 cycles, 1.3% of the guest. The guest's
cycles are statement machinery -- `parse.statement` 3.71 M and `merkle.table`
(building the opening table from the statement; no hashing) 4.56 M, together
82% -- then `digest` 0.90 M (9%), the relation `gates` 0.43 M (4.2%). The
relation alone is isolated by the relation-only diagnostic below. A diagnostic
guest commits `h_O || h_L || 0xEE` and is never accepted as sound.

Relation-only DIAGNOSTIC (2026-09-22, `r20260922-050349-3373` and repeats
`r20260922-050209-3eb1`, `r20260922-050533-2010`): the `unsound-relation-only`
guest (built by `SP1_HOST_FEATURES=cuda,unsound-relation-only bash
archive/sp1/pod_bootstrap.sh`, run with `vector_run.py --diagnostic
--relation-only`) runs nothing but the `GemmCoordinate<K>` gate chain: it reads
a compact input `K || x || W` (the two rows of the one obligation, built by
`verity_sp1.relation_only` from the vector's object and witness, authenticated
in Python), evaluates the chain through the kernel's own `GateSet::evaluate`,
and commits `sha256(input) || sha256(output word) || 0xEE`; no statement, no
opening table, no leaf hashing. K=1536: 406,671 cycles (4.0% of the sound
guest's 10,128,736) -- `gates` 346,484, `relation_only.parse` 31,756,
`relation_only.digest` 23,989, `relation_only.io` 363 -- 2 shards,
`prove_seconds` 0.855 / 0.868 / 0.913 s (sound 3.58 s), setup 15.9-16.0 s,
2.8 MB proof, sampled peak GPU memory 7.7-9.9 GiB (sound 19.5 GiB); K=16
(`r20260922-050711-86b5`) 13,226 cycles, 2 shards, 0.68 s; execute-only
(`r20260922-050847-415f`) 406,671 cycles in 0.50 s. The sound guest's `gates`
scope (426,223) also holds the interpreter's per-gate op resolution and
argument gathering over the parsed statement, which the compact path does
without. This is the arithmetic baseline of the zkVM path; its fingerprint
says `statement_machinery: excluded` (the skip-authentication diagnostic says
`included`) and it verifies no statement.

## Tests

~~~sh
uv run pytest archive/sp1/tests                # Python driver over the recorded vectors; execute only
uv run research data fetch-fixtures archive/sp1 fixtures/lean-vectors   # the vectors are fetched fixtures (fixtures/artifacts.json)
cd archive/sp1 && cargo test --release -p veritor-zk-common   # kernel unit tests + cross-language vectors
~~~

`tests/test_typed.py` skips as a module when the host is not built. It checks the
guest identity, that lowering reproduces each vector's recorded statement,
that `execute` reproduces the recorded public values byte for byte for K=16,
K=16 (`@2`) and K=1536, and the negatives (tampered leaf, unbound 33-byte
layout, another object's digest). The prove-and-accept test runs only with
`VERITY_SP1_PROVE=1`, on the pod:

~~~sh
uv run research run --on vy-sp1 --project verity --source . --stage sp1.tests --exclusive \
  --env VERITY_SP1_HOST=/workspace/bin/veritor-zk-host-cuda --env SP1_PROVER=cuda --env VERITY_SP1_PROVE=1 \
  --env PYTHONPATH=.:archive/sp1/python \
  -- /root/.local/bin/uv run --no-project --with pytest --python 3.12 python -m pytest -q archive/sp1/tests
~~~

A host whose `info` is not the approved identity makes the identity test skip
(without `VERITY_SP1_PROVE`) or fail (with it); its `execute` results are
still valid, its proofs are not accepted. `tests/test_relation_only.py` checks
the relation-only compact input without any host (the same sha256 pins as the
Rust unit test in `common/src/relation_only.rs`); its guest tests run only with
`VERITY_SP1_RELATION_ONLY_HOST` pointing at the diagnostic host. `tests/test_ir_call.py` needs no host: it lowers an
IR call's gate list to `veritor.reduce-mufu@1` obligations (`verity_sp1.ir_call`) and checks them against the Python
reference and the IR oracle.

The cross-language vectors under `../../fixtures/` (`proof-format-v3`,
`typed-obligation-v0`, `lean-vectors`) are generated by the Python side and
asserted here; they are what ties the kernel's semantics to `verity.ml.circuits.ops` and
to the statement encoding.

## Guest identity

The approved prover is pinned in `python/verity_sp1/host.py` (`APPROVED`):

~~~text
elf_sha256  cef2b78aef7ae16a3703d972c4cea7ec1cb5fed31797736a77886dc01fa5c567
vk_hash     0x007d9347080c44a23742498ce3b1fb7ca0df111119f60ac2e094c5cd9c2438ab
~~~

`host/build.rs` makes the guest build reproducible without Docker: the
workspace path and the cargo home are remapped to fixed prefixes
(`--remap-path-prefix`) and the build is `--locked`, so the same source tree,
`Cargo.lock` and pinned toolchain (`sp1up --version v6.4.0`) give the same
ELF wherever it is checked out. Verified on 2026-09-21 by building on one
Linux pod from two unrelated paths into two fresh target directories. A host
built before that change carries another identity; `verify_object` refuses
its proofs, and `info` shows which one you have.

## Pods

Proving is heavy and never runs on a laptop or on the small control pod.
`pod_bootstrap.sh` sets up a fresh GPU pod (toolchain, `uv`, the cuda host at
`/workspace/bin/veritor-zk-host-cuda`) and is run as a recorded `research`
run; workloads then set `VERITY_SP1_HOST` to that binary and `SP1_PROVER=cuda`.

## Provenance

Copied from `veritor` `zk/sp1` at commit `eb865ce2` (`common`, `guest`,
`host`, `check-model`, `Cargo.lock`, `rust-toolchain.toml`). Not carried:
the `lean-plan` guest feature and its Lean-generated C dependency
(`lean-guest`), `programs/`, `bench/`, `kernel-results/`. The kernel is
byte-identical; the trimmed `guest/src/main.rs` shifts four panic-location
line numbers in the ELF, so the verifying key is this repository's own and
is recorded by the Python driver's tests, not inherited from veritor's
`lock.json`.

The guest still carries every gate family the veritor checker knew
(`toy-isa`, `word-arithmetic`, `int8-q7`, `ftz`, `reduce`, `embed`, `elem`).
The dot-product slice exercises only `veritor.tc-ampere-bf16@1/@2`; the rest
stays because trimming it would change the verifying key. Crate names keep the
`veritor-zk-*` prefix for the same reason.
