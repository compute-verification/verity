#!/usr/bin/env bash
# One bench-result of the raised-query SP1 variant built by sec128/build.sh: vector_run.py --backend sp1-bare with the
# variant's host, HOME pointing at its source-built GPU server, and a variant file naming what changed.  Afterwards the
# server's sha256 is checked (sp1-sdk replaces a server whose version it rejects) and the stock host, when present, must
# REJECT rep 0's proof.  Run from the source root of a recorded run:
#   research run --on POD --project verity --source . --cwd source --stage sp1.sec128 --require-result -- \
#     bash backends/sp1/sec128/run.sh --instances /workspace/sec128/instances --layout indexed --reps 5 --warmup-vus 4096
set -euo pipefail
SEC_BITS="${SEC_BITS:-134}"
ROOT="${SEC128_ROOT:-/workspace/sec128}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${RESEARCH_RUN_DIR:-$PWD/out-sec128}"; mkdir -p "$OUT"
HOST=/workspace/bin/veritor-zk-host-cuda-relation-bare-sec$SEC_BITS
STOCK=/workspace/bin/veritor-zk-host-cuda-relation-bare
SERVER="$ROOT/home/.sp1/bin/sp1-gpu-server"
PY="${PY:-$(ls /root/.local/share/uv/python/cpython-3.12*-linux-x86_64-gnu/bin/python3.12 | head -1)}"
SEC="$HERE/security-$SEC_BITS.json"
[ -f "$SEC" ] || { echo "no $SEC"; exit 2; }
sha256sum -c "$ROOT/server.sha256"
SERVER_SHA="$(cut -d' ' -f1 "$ROOT/server.sha256")"

"$PY" - "$SEC" "$OUT/variant.json" "$SEC_BITS" "$SERVER_SHA" "$(sha256sum "$HOST" | cut -d' ' -f1)" \
  "$ROOT/prover-source.diff" "$ROOT/verifier-source.diff" "$HOST" "${CUDA_ARCHS:-80}" <<'EOF'
import json, os, sys
sec, out, bits, server_sha, host_sha, pdiff, vdiff, host, archs = sys.argv[1:]
v = json.load(open(sec))
assert v["security_bits"] == int(bits)
q = v["security_parameters"]["fri_queries"]
line = f"core_fri_config(): num_queries unique_decoding_queries(CORE_LOG_BLOWUP) = 124 -> {q} (SP1's formula at a {bits}-bit target); recursion configs unchanged"
v.update({
    "backend_name": f"sp1 relation-bare/v2 (SP1, FRI query count raised to a {bits}-bit target (otherwise unmodified); "
                    f"core STARK, CUDA prover)",
    "label": f"SP1, FRI query count raised to a {bits}-bit target (otherwise unmodified): {q} core queries (stock 124); "
             f"per-shard soundness stays field-bound at 2^{v['security_per_shard_log2']} (security.parameters)",
    "toolchain": {"sp1-primitives": f"6.6.0 with core_fri_config {q} queries (crates.io source patched in place in a copied CARGO_HOME, one line; lock file unchanged, embedded guest ELF/vk = stock)",
                  "sp1-gpu-server": f"6.6.0 source-built: succinctlabs/sp1 v6.6.0 f5a5bbf6 + the same line, CUDA_ARCHS {archs}"},
    "fork": {"base": "succinctlabs/sp1 v6.6.0 f5a5bbf6a1cf6007315b410d88678d96fb399fb5", "changed": [f"crates/primitives/src/fri_params.rs: {line}"],
             "prover_source_diff": open(pdiff).read(), "verifier_source_diff": open(vdiff).read(),
             "harness_changed": ["sp1-cuda 6.4.0 src/client.rs (host-side client, not the proof system): socket connect retries 0..10 -> 0..300 (100 ms apart)"],
             "harness_source_diff": open(os.path.join(os.path.dirname(pdiff), "harness-source.diff")).read(),
             "gpu_server_sha256": server_sha, "host_sha256": host_sha},
    "verifier_name": f"{host} verify (sp1-primitives patched to the same constant)",
})
json.dump(v, open(out, "w"), indent=1)
EOF

rc=0
HOME="$ROOT/home" "$PY" benchmarks/dot_product/vector_run.py --backend sp1-bare --host "$HOST" --variant "$OUT/variant.json" "$@" || rc=$?

post="$(sha256sum "$SERVER" | cut -d' ' -f1)"
cross="{}"
P0="$(ls "$OUT"/proofs/proof-rep0*.bin 2>/dev/null | head -1 || true)"
if [ -n "$P0" ] && [ -x "$STOCK" ]; then
  cross="$(SP1_PROVER=cpu RUST_LOG=error "$STOCK" verify --proof "$P0" --statement "$OUT/proofs/statement.bin" 2>/dev/null | grep '^{' | tail -1 || echo '{}')"
fi
"$PY" - "$OUT/sec128-checks.json" "$SERVER_SHA" "$post" "$cross" "$STOCK" <<'EOF'
import json, sys
out, pre, post, cross, stock = sys.argv[1:]
c = json.loads(cross or "{}")
stock_accepts = bool(c.get("ok")) and c.get("statement_match") is True and bool(c.get("verdict"))
json.dump({"server_sha256_before": pre, "server_sha256_after": post, "server_unchanged": pre == post,
           "stock_host": stock, "stock_host_verdict_on_rep0": c, "stock_host_rejects_rep0": (not stock_accepts) if c else None},
          open(out, "w"), indent=1)
print("sec128 checks:", open(out).read())
EOF
[ "$SERVER_SHA" = "$post" ] || { echo "GPU SERVER REPLACED during the run: the result is not this variant's"; exit 3; }
exit $rc
