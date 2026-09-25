export PATH=$HOME/.cargo/bin:$PATH
S=/workspace/averifier/proofs/r20260922-091240-4bf9/stmt; P=/workspace/averifier/proofs/r20260922-091240-4bf9/proof4096_torch.bin
V=/workspace/cargo-target-a-verifier/release/verity-gkr-verify
which perf || (apt-get install -y linux-tools-generic >/dev/null 2>&1; which perf) || echo "no perf"
perf record -F 999 -g -o $RESEARCH_RUN_DIR/perf.data $V verify --allow-any-circuit --dir $S --proof $P --threads 1 2>&1 | cut -c1-200
perf report -i $RESEARCH_RUN_DIR/perf.data --no-children --stdio 2>/dev/null | grep -v "^$" | head -60
echo "=== callers flat"
perf report -i $RESEARCH_RUN_DIR/perf.data --no-children --stdio -g none --percent-limit 1 2>/dev/null | grep -v "^#" | grep -v "^$" | head -40
rm -f $RESEARCH_RUN_DIR/perf.data
