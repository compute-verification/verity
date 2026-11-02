# pyenv.sh, sourced by the ladder's scripts: P, the node's python, with a private overlay (on PYTHONPATH) of the
# modules the ladder needs and it lacks, installed without dependencies as prep.sh does, so the closure is listed whole.
P=${LADDER_PYTHON:-/workspace/jobs/venv312/bin/python}
[ -x "$P" ] || { echo "no python at $P; set LADDER_PYTHON to one with torch" >&2; exit 2; }
T="${RESEARCH_RUN_DIR:-${OUT:?set OUT or RESEARCH_RUN_DIR}}/pylib"
need=$("$P" - <<'EOF'
import importlib.util
mods = [("pyarrow", "pyarrow"), ("tiktoken", "tiktoken==0.14.0"), ("regex", "regex"), ("safetensors", "safetensors"),
        ("huggingface_hub", "huggingface_hub<1"), ("requests", "requests"), ("urllib3", "urllib3"), ("idna", "idna"),
        ("certifi", "certifi"), ("charset_normalizer", "charset-normalizer"), ("filelock", "filelock"),
        ("fsspec", "fsspec"), ("packaging", "packaging"), ("yaml", "pyyaml"), ("tqdm", "tqdm"),
        ("typing_extensions", "typing-extensions")]
print(" ".join(p for m, p in mods if importlib.util.find_spec(m) is None))
EOF
)
echo "pyenv: $P, missing: ${need:-none}"
# shellcheck disable=SC2086
if [ -n "$need" ]; then uv pip install -q --python "$P" --target "$T" --no-deps $need; fi
export PYTHONPATH="$T${PYTHONPATH:+:$PYTHONPATH}"
