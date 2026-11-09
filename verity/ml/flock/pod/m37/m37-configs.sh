#!/usr/bin/env bash
# Measurement only, never landed: Fast100 Ligerito configs for m = 36 and 37, which upstream b684b12 registers only up to
# m = 35. Derives them with upstream's derive_profile (gen_m37.rs) in an unpatched clone beside the patched checkout F,
# copies them into F and registers the two in F's embedded_security_config. Run by 60-circuit.sh before its build.
# usage: m37-configs.sh F OUT
set -euo pipefail
F=$1 O=$2 HERE=$(cd "$(dirname "$0")" && pwd)
U=$(dirname "$F")/flock-derive D=crates/flock-core/configs/ligerito
case $U in */flock-gpu-link/flock-derive) rm -rf "$U" ;; *) echo "m37-configs: refusing to clear $U" >&2; exit 1 ;; esac
git clone -q "$F" "$U" && cd "$U" && git checkout -q b684b12
cp "$HERE/gen_m37.rs" crates/flock-prover/examples/gen_m37.rs
cargo run --release -q -p flock-prover --example gen_m37 -- "$D" 2>&1 | tee "$O/gen_m37.txt"
cp "$D/m36_fast100.toml" "$D/m37_fast100.toml" "$F/$D/"
cp "$D/m36_fast100.toml" "$D/m37_fast100.toml" "$O/"
cd / && rm -rf "$U"
python3 - "$F/crates/flock-core/src/pcs/ligerito.rs" <<'EOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "    EMBEDDED_CONFIGS.iter().find_map(|&(key, toml)| {\n"
assert s.count(a) == 1, "ligerito.rs's embedded_security_config changed upstream"
s = s.replace(a, "    EMBEDDED_CONFIGS.iter().chain(M37_CONFIGS.iter()).find_map(|&(key, toml)| {\n")
s += ('\n// measurement only: Fast100 at m = 36 and 37 (verity/ml/flock/pod/m37/m37-configs.sh)\n'
      'const M37_CONFIGS: &[((usize, LigeritoProfile), &str)] = &[\n'
      '    ((36, LigeritoProfile::Fast100), include_str!("../../configs/ligerito/m36_fast100.toml")),\n'
      '    ((37, LigeritoProfile::Fast100), include_str!("../../configs/ligerito/m37_fast100.toml")),\n'
      '];\n')
p.write_text(s)
print("ligerito.rs: Fast100 registered at m = 36 and 37")
EOF
