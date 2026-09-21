# SP1 backend

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

## Using it

Rule: `execute` is light (the K=1536 vector is 10 M cycles, seconds on a
laptop). **Proving is heavy and runs only on the SP1 pod**, never on a laptop
or on the control pod. Building the host takes the SP1 toolchain and ~10 min.

Build (needs `sp1up --version v6.4.0`, i.e. `cargo prove` and the `succinct`
toolchain, plus `clang`/`libclang`, `cmake`, `protobuf-compiler` on Linux):

~~~sh
cd backends/sp1
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

The benchmark workload is `benchmarks/dot_product/vector_run.py`; run it on
the pod through `research` so the result is recorded and preserved:

~~~sh
uv run research run --on vy-sp1 --project verity --source . --stage sp1.prove --require-result --exclusive \
  --env VERITY_SP1_HOST=/workspace/bin/veritor-zk-host-cuda --env SP1_PROVER=cuda \
  -- /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 \
     benchmarks/dot_product/vector_run.py --vector accept--packed-word-row --backend sp1-prove --mode core
uv run research fetch <run-id> --all && uv run research inspect <run-id>
~~~

`--source .` ships `git archive HEAD`, so commit first; the snapshot cannot
contain ignored files (`common/src/elem_silu_table.bin` is force-tracked for
that reason). Backends are `reference` (Python re-derivation, runs anywhere),
`sp1-execute`, `sp1-prove`. Baseline recorded runs (2026-09-21, RTX 4090):
K=16 core proof `r20260921-215829-9796` -- 271,289 cycles, 2 shards,
`prove_seconds` 0.89 s after an 18 s one-time key setup, 2.8 MB proof.

## Tests

~~~sh
uv run pytest backends/sp1/tests                # Python driver over the recorded vectors; execute only
cd backends/sp1 && cargo test --release -p veritor-zk-common   # kernel unit tests + cross-language vectors
~~~

The Python tests skip as a module when the host is not built. They check the
guest identity, that lowering reproduces each vector's recorded statement,
that `execute` reproduces the recorded public values byte for byte for K=16,
K=16 (`@2`) and K=1536, and the negatives (tampered leaf, unbound 33-byte
layout, another object's digest). The prove-and-accept test runs only with
`VERITY_SP1_PROVE=1`, on the pod:

~~~sh
uv run research run --on vy-sp1 --project verity --source . --stage sp1.tests --exclusive \
  --env VERITY_SP1_HOST=/workspace/bin/veritor-zk-host-cuda --env SP1_PROVER=cuda --env VERITY_SP1_PROVE=1 \
  --env PYTHONPATH=packages/verity/src:backends/sp1/python \
  -- /root/.local/bin/uv run --no-project --with pytest --python 3.12 python -m pytest -q backends/sp1/tests
~~~

A host whose `info` is not the approved identity makes the identity test skip
(without `VERITY_SP1_PROVE`) or fail (with it); its `execute` results are
still valid, its proofs are not accepted.

The cross-language vectors under `../../fixtures/` (`proof-format-v3`,
`typed-obligation-v0`, `lean-vectors`) are generated by the Python side and
asserted here; they are what ties the kernel's semantics to `verity.ml` and
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
The dot-product slice exercises only `veritor.tc-ampere-bf16@1/@2`; trimming
the rest is a deliberate later change because it changes the verifying key.
Crate names keep the `veritor-zk-*` prefix for the same reason.
